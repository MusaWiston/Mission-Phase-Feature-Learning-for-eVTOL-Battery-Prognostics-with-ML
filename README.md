# Mission-Phase Feature Learning for eVTOL Battery Prognostics (SOC, SOH, RUL) using ML algorithms
[![Data: CMU eVTOL (CC-BY-4.0)](https://img.shields.io/badge/Data-CMU%20eVTOL-6f42c1.svg)](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830)


**What is this?** An end-to-end, **mission-phase-aware** ML pipeline for lithium-ion **prognostics in eVTOL duty cycles**.  
We engineer >50 phase/mission features from the CMU eVTOL dataset and train both tree ensembles and sequence models (LSTM/GRU/BiLSTM/TCN/TFT, plus **Attention-LSTM-MoE**). We enforce **physical consistency** via a **monotone-RUL** penalty and **post-hoc isotonic calibration** across **EOL thresholds (90/85/80%)**.

**Headline (draft)**: Attention-guided sequence models (TFT, Attn-LSTM-MoE) deliver **lowest MAE** and best calibration for RUL@{80,85,90} across cells; tree models struggle on sequence RUL despite doing fine on SOH.

---

## 1) Why eVTOL battery prognostics?

eVTOL missions demand **high power at takeoff/landing** with no rest during flight, stressing cells thermo-electrically. Mission-aware prognostics are required to plan maintenance, derate missions, and enforce safety margins across **multiple EOL thresholds** (early-warning at 90%, serviceability at 85%, end-of-service at 80%).  

**Data**: We use the **CMU eVTOL Battery Dataset** (22 Sony-Murata VTC6 cells; ~21k cycles) — a widely cited benchmark for eVTOL duty cycles.  
Dataset DOI: `10.1184/R1/14226830.v2`  •  Paper: Scientific Data 10:344 (2023)

---

## 2) Repository structure
.
├─ README.md
├─ LICENSE
├─ CITATION.cff                 # How to cite this repo (shown by GitHub) 
├─ pyproject.toml / requirements.txt
├─ Makefile                     # make data | make fe | make train | make eval | make figs
├─ configs/                     # YAML configs for experiments
│  ├─ train_rul.yaml
│  └─ train_soh.yaml
├─ src/evtol_prognosis/         # Installable package (import evtol_prognosis)
│  ├─ __init__.py
│  ├─ utils.py                  # formerly battery_common.py
│  ├─ features/
│  │  ├─ phase_labeling.py      # phase labeling logic
│  │  └─ feature_engineering.py # feature extraction (>50 features)
│  ├─ models/
│  │  ├─ attn_lstm.py
│  │  ├─ attn_lstm_moe.py
│  │  └─ baseline_trees.py      # LightGBM / CatBoost / XGBoost wrappers
│  ├─ training/
│  │  ├─ datamodule.py          # loaders, T_max, k-window
│  │  └─ train.py               # common training loop + LOCO CV
│  └─ evaluation/
│     ├─ metrics.py             # MAE, RMSE, calibration utils
│     └─ plots.py               # RUL@80/85/90, SOH heatmaps, ablations
├─ scripts/                     # CLI entry points (thin wrappers around src/)
│  ├─ download_cmu_evtol.py     # prints CC-BY-4.0 notice and fetches data
│  ├─ preprocess.py             # raw → interim (cleaning, fixes)
│  ├─ extract_features.py       # interim → processed features
│  ├─ train_baselines.py
│  ├─ train_attn_lstm_moe.py
│  └─ evaluate.py
├─ data/                        # (tracked via DVC or LFS; gitignored)
│  ├─ raw/                      # original CMU eVTOL CSVs + impedance
│  ├─ interim/                  # cleaned/validated
│  └─ processed/                # per-cycle & mission-level features, splits
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_checks.ipynb
│  └─ 03_results_figures.ipynb
├─ results/
│  ├─ metrics/                  # CSVs for tables
│  ├─ figs/                     # paper-ready figures
│  └─ tables/                   # LaTeX tables
├─ docs/
│  ├─ data_schema.md            # raw column glossary; impedance headers
│  ├─ feature_dictionary.md     # derived feature definitions (from CSV)
│  └─ api.md                    # brief API for src/ (public functions)
└─ .github/workflows/
   └─ ci.yml                    # lint, unit tests, smoke train (fast)


## 3) Quickstart Guidelines

```bash
# 0) Clone
git clone https://github.com/MusaWiston/Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML.git
cd Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML

# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
python -m pip install --upgrade pip
pip install -r requirements.txt   # (provide this file; see "Setup" below)

# 2) Download dataset (CC-BY-4.0)
#   - Manually from KiltHub (link below), or add a helper script here to fetch VAH**.csv files.

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


