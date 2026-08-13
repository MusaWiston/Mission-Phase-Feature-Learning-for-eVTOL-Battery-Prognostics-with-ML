# Mission-Phase Feature Learning for eVTOL Battery Prognostics (SOC, SOH, RUL) using ML algorithms
[![Dataset: CMU eVTOL (CC-BY-4.0)](https://img.shields.io/badge/Dataset-CMU%20eVTOL-6f42c1.svg)](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Models](https://img.shields.io/badge/Models-LSTM%2FGRU%2FTCN%2FTFT%20%7C%20MoE-6f42c1)]()


**What is this?** An end-to-end, **mission-phase-aware** ML pipeline for lithium-ion battery **prognostics in eVTOL duty cycles**.  
Engineered >50 phase/mission features from the CMU eVTOL dataset and train both tree ensembles and sequence models (LSTM/GRU/BiLSTM/TCN/TFT, plus **Attention-LSTM-MoE**). The enforced **physical consistency** via a **monotone-RUL** penalty and **post-hoc isotonic calibration** across state of health **EOL thresholds (90/85/80%)** of the initial battery capacity.

**Headline (draft)**: Attention-guided sequence models (Temporal Fusion Transformers, Attn-LSTM-MoE) deliver **lowest MAE** and best calibration for RUL@{80,85,90} across mission profile cells; tree models struggle on sequence RUL despite doing fine on SOH estimations.

---

## 1) Why eVTOL battery prognostics?

eVTOL missions demand **high power at takeoff/landing** with no rest during flight, stressing cells thermo-electrically. Mission-aware prognostics are required to plan maintenance, derate missions, and enforce safety margins across **multiple EOL thresholds** (early-warning at 90%, serviceability at 85%, end-of-service at 80%).  

**Data**:**CMU eVTOL Battery Dataset** (22 Sony-Murata VTC6 cells; ~21k cycles) — a widely cited benchmark for eVTOL duty cycles. (ZIP files founds in Release) 
Dataset DOI: `10.1184/R1/14226830.v2`  •  Paper: Scientific Data 10:344 (2023)

---

## 2) Repository structure
├── Attn_lstm.py # Attention-LSTM algorithm (LOCO driver)
├── Attn_lstm_MoE.py # Proposed Attention-LSTM + Mixture-of-Experts, isotonic calibration
├── sequential_baseline.py # GRU / BiLSTM / TCN / TFT agorithms under LOCO
├── baseline_tress.py # (Trees) RF, XGBoost, CatBoost, LightGBM under LOCO
├── battery_common.py # Shared config, dataset builders, sequences, metrics, plotting
├── metrics_helper.py # MAPE + event-based EOL timing error helpers
├── plot_moe_insights.py # Attention heatmaps, gate usage, reliability, latency, etc.
├── feature_dictionary.csv # Glossary of derived features (phase + mission levels)
├── mission_features.csv # Mission-level features per cell-mission
├── phase_features.csv # Phase-level features per cell-mission-segment
├── per_cycle_summary_allcells.csv # (optional) summary export
├── Baseline_mission_profiles.zip / phase_features.rar / eda.zip 
└── README.md

# UPDATE

For feature-table verification and CI only, the smaller dependency set is
sufficient:

```bash
python -m pip install -r requirements-ci.txt
```

First extract `phase_features.rar` so that `phase_features.csv` is available.
Then run:

```bash
python scripts/rebuild_mission_features.py \
  --phase-features phase_features.csv \
  --output /tmp/mission_features.rebuilt.csv \
  --verify-against mission_features.csv
```

The command exits non-zero on a schema, row-count, ordering, or numerical
mismatch. A different test protocol can provide cell-specific CV setpoints with
`--cv-setpoints-json path/to/setpoints.json`.

Verify the checked-in derived files before a rerun with:

```bash
sha256sum --check artifacts.sha256
```

## Canonical raw telemetry pipeline

Use the ignored `data/` directory shown below, or another empty destination,
for the large generated tables.

```bash
python scripts/build_reproducible_pipeline.py \
  --data-dir /path/to/raw_evtol_csv \
  --output-dir data/reproducible \
  --config config/pipeline.json
```

The command is fail-fast by default and writes cycle/RPT audits, separate
feature and target tables, padded sequence samples, deterministic LOCO roles,
and `manifest.json` with the resolved configuration and SHA-256 provenance.
Use `--write-cleaned-telemetry` only when the large canonical row-level files
are needed. See [the complete procedural specification](docs/preprocessing_and_targets.md).

## Run the models

Point the model loaders at the directory containing the new
`phase_features.csv`, `mission_features.csv`, `phase_targets.csv`, and
`mission_targets.csv`; the pipeline also copies the exact feature dictionary
there:

```bash
export BATTERY_DATA_DIR="$PWD/data/reproducible"

python baseline_tress.py --tasks soc soh rul --sweep_rul 80 85 90
python sequential_baseline.py \
  --models gru bilstm tcn tft \
  --tasks soc soh rul \
  --sweep_rul 80 85 90
python Attn_lstm.py
python Attn_lstm_MoE.py
```

The generated `samples/*.npz` files contain unscaled values, masks, lengths,
targets, sample IDs, and the exact predictor list. `loco_folds.csv` identifies
training, validation, and held-out cells. Median imputation and standardisation
must be fitted only on each outer fold's training cells and then applied
unchanged to validation and test cells; `FoldStandardizer` implements this.

## Tests

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## Important reproducibility note

The corrected raw pipeline uses C/5 RPT discharge capacity, reconstructs
contiguous cycle instances instead of globally grouping repeated tester
counters, does not extrapolate SOH, and does not convert censored endpoints
into observed failures..See `docs/reproducibility.md` for the artifact provenance.

## Quickstart Guidelines

```bash
# 0) Clone
git clone https://github.com/MusaWiston/Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML.git
cd Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML

# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
python -m pip install --upgrade pip
pip install -r requirements.txt   # (see "Setup" below)
# ---- Core ----
numpy>=1.26
pandas>=2.2
scipy>=1.11
scikit-learn>=1.7

# ---- Gradient-boosted trees ----
xgboost>=2.1
lightgbm>=4.6
catboost>=1.2

# ---- Sequence models (PyTorch backend) ----
torch>=2.3,<2.5
einops>=0.7

# ---- Plotting & utilities ----
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
pyyaml>=6.0
joblib>=1.3

# ---- Optional: notebooks ----
# jupyterlab>=4.1
# ipykernel>=6.29


# 2) Download dataset (CC-BY-4.0)
#   - Manually from KiltHub (https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830)

# 3) Preprocess + feature engineering
python "Phase labeling Stage.py" --data /path/to/raw_csv
python "Feature engineering stage(features_extractions).py" --data /path/to/raw_csv --out ./data/processed

# 4) Train baselines
python sequential_baseline.py --task SOH --cv LOCO
python baseline_tress.py --task RUL --eol 80 --cv LOCO

# 5) Train Attn-LSTM(-MoE) sequences
python Attn_lstm.py --task RUL --EOL all --tmax 16 --k_window 5
python Attn_lstm_MoE.py --task RUL --eol all --experts 2 --topk 2 --lambda_mono 0.02

# 6) Evaluate + plots
python sequential_baseline.py --eval --dump ./results
