# Mission-Phase Feature Learning for eVTOL Battery Prognostics (SOC, SOH, RUL) using ML algorithms
[![Data: CMU eVTOL (CC-BY-4.0)](https://img.shields.io/badge/Data-CMU%20eVTOL-6f42c1.svg)](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Models](https://img.shields.io/badge/Models-LSTM%2FGRU%2FTCN%2FTFT%20%7C%20MoE-6f42c1)]()


**What is this?** An end-to-end, **mission-phase-aware** ML pipeline for lithium-ion **prognostics in eVTOL duty cycles**.  
We engineer >50 phase/mission features from the CMU eVTOL dataset and train both tree ensembles and sequence models (LSTM/GRU/BiLSTM/TCN/TFT, plus **Attention-LSTM-MoE**). We enforce **physical consistency** via a **monotone-RUL** penalty and **post-hoc isotonic calibration** across **EOL thresholds (90/85/80%)**.

**Headline (draft)**: Attention-guided sequence models (TFT, Attn-LSTM-MoE) deliver **lowest MAE** and best calibration for RUL@{80,85,90} across cells; tree models struggle on sequence RUL despite doing fine on SOH.

---

## 1) Why eVTOL battery prognostics?

eVTOL missions demand **high power at takeoff/landing** with no rest during flight, stressing cells thermo-electrically. Mission-aware prognostics are required to plan maintenance, derate missions, and enforce safety margins across **multiple EOL thresholds** (early-warning at 90%, serviceability at 85%, end-of-service at 80%).  

**Data**: I use the **CMU eVTOL Battery Dataset** (22 Sony-Murata VTC6 cells; ~21k cycles) — a widely cited benchmark for eVTOL duty cycles.  
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


## 3) Quickstart Guidelines

```bash
# 0) Clone
git clone https://github.com/MusaWiston/Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML.git
cd Mission-Phase-Feature-Learning-for-eVTOL-Battery-Prognostics-with-ML

# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
python -m pip install --upgrade pip
pip install -r requirements.txt   # (provide this file; see "Setup" below)
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


