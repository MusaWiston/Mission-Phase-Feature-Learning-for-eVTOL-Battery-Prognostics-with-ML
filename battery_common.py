import os
import os, re, math, json, random, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer

# torch (only needed by LSTM pipelines; safe to import here)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ------------------- PATHS / CONFIG -------------------
# >>> EDIT THESE TO MATCH YOUR MACHINE <<<
DATA_DIR = Path(r"E:\BATTERY\eVTOL_battery_dataset\Feature_engineering")
PHASE_CSV   = DATA_DIR / "phase_features.csv"
MISSION_CSV = DATA_DIR / "mission_features.csv"
DICT_CSV    = DATA_DIR / "feature_dictionary.csv"
#export BATTERY_DATA_DIR=/abs/path/to/Feature_engineering

# output roots per pipeline
OUT_TREES   = Path("./outputs/baselines_trees")
OUT_LSTM    = Path("./outputs/baseline_attn_lstm")
OUT_MOE     = Path("./attn_lstm_moe")
OUT_ROOT    = Path("./outputs/attn_lstm_moe2")

# globals
SEED     = 1337
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
RNG      = np.random.default_rng(SEED)

# sequence caps
MAX_STEPS_SOC  = 16   # segments in SOC windows
MAX_STEPS_MISS = 20   # missions in SOH/RUL windows

# SOH horizon to predict ΔSOH over K missions (avoids quantized 0s)
SOH_HORIZON_K  = 5

# RUL thresholds
EOL_SWEEP      = [80.0, 85.0, 90.0]

# LSTM defaults (used by LSTM & MoE pipelines)
D_HID    = 96
N_LAYERS = 1
DROPOUT  = 0.15
BATCH_SZ = 64
EPOCHS   = 60
LR       = 2e-3
W_DECAY  = 1e-4
PATIENCE = 8

# MoE config (used by proposed pipeline)
N_EXPERTS   = 6
TOPK        = 2
LAMBDA_GATE = 0.02

# ------------------- UTILS -------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: dict):
    ensure_dir(path.parent)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def one_hot(series: pd.Series, prefix: Optional[str] = None) -> pd.DataFrame:
    return pd.get_dummies(series.astype("category"), prefix=prefix, dummy_na=False)

def try_torch_load_state_dict(model, model_path: Path):
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)

def summarize_fleet_metrics(per_cell_csv: Path, out_json: Path):
    if not per_cell_csv.exists(): return
    df = pd.read_csv(per_cell_csv)
    if df.empty:
        return
    fleet = {
        "num_cells": int(df["cell_id"].nunique()),
        "total_samples": int(df["N"].sum()),
        "MAE_mean": float(df["MAE"].mean()),
        "MAE_median": float(df["MAE"].median()),
        "RMSE_mean": float(df["RMSE"].mean()),
        "RMSE_median": float(df["RMSE"].median()),
    }
    save_json(out_json, fleet)

# ------------------- LOADERS -------------------
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [PHASE_CSV, MISSION_CSV, DICT_CSV]:
        if not p.exists(): raise FileNotFoundError(p)
    ph = pd.read_csv(PHASE_CSV)
    ms = pd.read_csv(MISSION_CSV)
    fd = pd.read_csv(DICT_CSV)
    ph.columns = [c.strip() for c in ph.columns]
    ms.columns = [c.strip() for c in ms.columns]
    fd.columns = [c.strip() for c in fd.columns]
    return ph, ms, fd

# ------------------- FEATURE SPACE -------------------
def features_from_dict(fd: pd.DataFrame, level: str, present: set) -> List[str]:
    if not {"feature","level"}.issubset(fd.columns):
        raise ValueError("feature_dictionary.csv must have columns: 'feature', 'level'")
    feats = fd.loc[fd["level"].str.lower()==level.lower(), "feature"].astype(str).tolist()
    feats = [f for f in feats if f in present]
    return feats

def drop_meta(cols: List[str]) -> List[str]:
    meta = {
        "cell_id","mission_id","cycleNumber","seg_id",
        "phase_name","phase_family","t_start_s","t_end_s",
        "RUL_missions_after_phase","RUL_missions_censored",
        "SOH_end_pct","SOH_next_pct","RUL_missions","RUL_missions85",
        "dis_dSOC","dis_dSOC_baseline","delta_SOC","SOC_delta","dSOC"
    }
    return [c for c in cols if c not in meta]

def append_optional_conditions(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    extra = []
    for cand in ["cell_condition", "test_condition"]:
        if cand in df.columns:
            extra.append(one_hot(df[cand], prefix=cand))
    return pd.concat([X] + extra, axis=1) if extra else X

# ------------------- DATASET / COLLATE -------------------
class SequenceScaler:
    def __init__(self): self.scaler = StandardScaler()
    def fit(self, Xseq: List[np.ndarray]):
        X = np.concatenate([x for x in Xseq], axis=0) if Xseq else np.zeros((1,1))
        self.scaler.fit(X)
    def transform(self, Xseq: List[np.ndarray]) -> List[np.ndarray]:
        return [self.scaler.transform(x) for x in Xseq]
    def save(self, path: Path):
        import joblib; ensure_dir(path.parent); joblib.dump(self.scaler, path)

class SeqDataset(Dataset):
    def __init__(self, Xseq, y, lengths, ids):
        self.Xseq = Xseq; self.y = np.asarray(y, dtype=np.float32)
        self.L = lengths; self.ids = ids
    def __len__(self): return len(self.Xseq)
    def __getitem__(self, i):
        return (torch.tensor(self.Xseq[i], dtype=torch.float32),
                torch.tensor([self.y[i]], dtype=torch.float32),
                int(self.L[i]),
                self.ids[i])

def collate_pad(batch):
    Ls = torch.tensor([b[2] for b in batch], dtype=torch.long)
    D  = batch[0][0].shape[-1]; Tm = int(Ls.max().item())
    xs = torch.zeros(len(batch), Tm, D, dtype=torch.float32)
    ys = torch.zeros(len(batch), 1, dtype=torch.float32)
    mask = torch.zeros(len(batch), Tm, dtype=torch.bool)
    ids = []
    for i,(x,y,L,idx) in enumerate(batch):
        xs[i,:L,:] = x; ys[i] = y; mask[i,:L] = True; ids.append(idx)
    return xs, ys, Ls, mask, ids

# ------------------- MODELS (applicable to LSTM layers pipelines) -------------------
class AttnLSTMBase(nn.Module):
    def __init__(self, d_in, d_hid=D_HID, n_layers=N_LAYERS, dropout=DROPOUT, nonneg_head: bool=False):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers>1 else 0.0)
        self.q = nn.Linear(d_hid, d_hid); self.k = nn.Linear(d_hid, d_hid); self.v = nn.Linear(d_hid, d_hid)
        self.head = nn.Sequential(nn.Linear(2*d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hid,1))
        self.nonneg = nonneg_head
    def forward(self, x, lengths, mask):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        H, (hT, _) = self.lstm(packed); H,_ = pad_packed_sequence(H, batch_first=True)
        if H.size(1) != mask.size(1): mask = mask[:, :H.size(1)]
        hT = hT[-1]
        Q = self.q(hT).unsqueeze(1); K = self.k(H); V = self.v(H)
        logits = (Q*K).sum(-1)/math.sqrt(K.size(-1)); logits = logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(logits, dim=1)
        ctx   = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)
        feats = torch.cat([ctx, hT], dim=-1)
        yhat  = self.head(feats)
        if self.nonneg: yhat = F.softplus(yhat)
        return yhat, alpha

class AttnLSTMMoE(nn.Module):
    def __init__(self, d_in, d_hid=D_HID, n_layers=N_LAYERS, dropout=DROPOUT,
                 n_experts=N_EXPERTS, topk: Optional[int]=TOPK, nonneg_head: bool=False):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers>1 else 0.0)
        self.q = nn.Linear(d_hid, d_hid); self.k = nn.Linear(d_hid, d_hid); self.v = nn.Linear(d_hid, d_hid)
        self.gate = nn.Linear(2*d_hid, n_experts)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(2*d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hid,1)
        ) for _ in range(n_experts)])
        self.topk   = topk; self.nonneg = nonneg_head
    def _apply_nonneg(self, y): return F.softplus(y) if self.nonneg else y
    def forward(self, x, lengths, mask):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        H, (hT, _) = self.lstm(packed); H,_ = pad_packed_sequence(H, batch_first=True)
        if H.size(1) != mask.size(1): mask = mask[:, :H.size(1)]
        hT = hT[-1]
        Q = self.q(hT).unsqueeze(1); K = self.k(H); V = self.v(H)
        logits = (Q*K).sum(-1)/math.sqrt(K.size(-1)); logits = logits.masked_fill(~mask, -1e9)
        alpha = torch.softmax(logits, dim=1)
        ctx   = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)
        feats = torch.cat([ctx, hT], dim=-1)

        gate_logits = self.gate(feats)
        if self.topk is None:
            pi = torch.softmax(gate_logits, dim=-1)
            outs = torch.stack([self._apply_nonneg(e(feats)) for e in self.experts], dim=-1)
            yhat = (outs * pi.unsqueeze(1)).sum(-1)
        else:
            topv, topi = torch.topk(gate_logits, self.topk, dim=-1)
            pi = torch.softmax(topv, dim=-1)
            parts = []
            for j in range(self.topk):
                idx = topi[:, j]
                outj = torch.zeros(feats.size(0), 1, device=feats.device)
                for eix in idx.unique():
                    m = (idx==eix); outj[m] = self._apply_nonneg(self.experts[int(eix)](feats[m]))
                parts.append(pi[:, j:j+1] * outj)
            yhat = torch.stack(parts, dim=-1).sum(-1)
        return yhat, alpha, gate_logits

# ------------------- METRICS -------------------
def mae_np(y, yhat):  return float(np.mean(np.abs(np.asarray(yhat) - np.asarray(y)))) if len(y) else np.nan
def rmse_np(y, yhat): return float(np.sqrt(np.mean((np.asarray(yhat) - np.asarray(y))**2))) if len(y) else np.nan

# ------------------- LEAKAGE GUARD -------------------
SUSPICIOUS_PATTERNS = [
    r"rul", r"eol", r"remaining", r"remain", r"soh_next", r"future",
    r"label", r"target", r"mission_id", r"cycle", r"age", r"elapsed",
    r"life", r"percent_remaining"
]

def name_based_drop(feat_names: List[str]) -> List[str]:
    keep = []
    for f in feat_names:
        low = f.lower()
        if any(re.search(p, low) for p in SUSPICIOUS_PATTERNS):
            continue
        keep.append(f)
    return keep

def corr_based_drop(X_tr: np.ndarray, y_tr: np.ndarray, feat_names: List[str], thr: float=0.98) -> List[int]:
    """Return indices of columns to keep (|corr| < thr or nan). Computed on TRAIN ONLY."""
    if X_tr.size == 0: return list(range(len(feat_names)))
    keep = []
    y = y_tr.ravel()
    for j in range(X_tr.shape[1]):
        x = X_tr[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            keep.append(j); continue
        cj = np.corrcoef(x[ok], y[ok])[0,1]
        if not np.isfinite(cj) or abs(cj) < thr:
            keep.append(j)
    return keep

# ------------------- BUILDERS -------------------
def pick_soc_target(ph: pd.DataFrame) -> str:
    for c in ["dis_dSOC", "delta_SOC", "SOC_delta", "dSOC", "dis_dSOC_baseline"]:
        if c in ph.columns: return c
    raise ValueError("No SOC delta target found (dis_dSOC / delta_SOC / SOC_delta / dSOC / dis_dSOC_baseline).")

def build_soc_sequences(ph: pd.DataFrame, fd: pd.DataFrame, max_steps: int):
    ph = ph.copy().sort_values(["cell_id","mission_id","seg_id"] if "seg_id" in ph else ["cell_id","mission_id"])
    feats = drop_meta(features_from_dict(fd, "phase", set(ph.columns)))
    X_frame = append_optional_conditions(ph, ph[feats].fillna(0.0))
    if "dis_is_flight_phase" in ph.columns:
        ph = ph[ph["dis_is_flight_phase"] == 1]
        X_frame = X_frame.loc[ph.index]
    tgt_col = pick_soc_target(ph)

    Xseq, y, Ls, groups, ids = [], [], [], [], []
    for (cell, mid), g in ph.groupby(["cell_id","mission_id"]):
        idxs = g.index.values
        Xg = X_frame.loc[idxs].values
        yg = g[tgt_col].astype(float).values
        seg = g["seg_id"].values if "seg_id" in g.columns else np.arange(len(g))
        for t in range(1, len(g)):
            s = max(0, t - max_steps + 1)
            Xseq.append(Xg[s:t+1]); y.append(float(yg[t])); Ls.append(int(t - s + 1))
            groups.append(str(cell)); ids.append((str(cell), int(mid) if not pd.isna(mid) else -1, int(seg[t])))
    d_in = Xseq[0].shape[1] if Xseq else X_frame.shape[1]
    return Xseq, y, Ls, groups, ids, d_in, X_frame.columns.tolist(), tgt_col

def build_soh_sequences(ms: pd.DataFrame, fd: pd.DataFrame, max_steps: int, horizon_k: int=1):
    ms = ms.copy().sort_values(["cell_id","mission_id"])
    feats = drop_meta(features_from_dict(fd, "mission", set(ms.columns)))
    X_frame = append_optional_conditions(ms, ms[feats].fillna(0.0))
    if "SOH_end_pct" not in ms.columns: raise ValueError("mission_features.csv must contain SOH_end_pct")

    ms["SOH_future"] = ms.groupby("cell_id")["SOH_end_pct"].shift(-horizon_k)
    df = ms[ms["mission_id"] <= (ms.groupby("cell_id")["mission_id"].transform("max") - horizon_k)].copy()
    df["dSOH_to_future"] = (df["SOH_end_pct"] - df["SOH_future"]).clip(lower=0.0)

    Xseq, y, Ls, groups, ids, soh_curr = [], [], [], [], [], []
    for cell, g in df.groupby("cell_id"):
        Xi = X_frame.loc[g.index].values
        yi = g["dSOH_to_future"].astype(float).values
        mids = g["mission_id"].values
        soh_now = g["SOH_end_pct"].astype(float).values
        for t in range(len(g)):
            s = max(0, t - max_steps + 1)
            Xseq.append(Xi[s:t+1]); y.append(float(yi[t])); Ls.append(int(t - s + 1))
            groups.append(str(cell)); ids.append((str(cell), int(mids[t]) if not pd.isna(mids[t]) else -1))
            soh_curr.append(float(soh_now[t]))
    d_in = Xseq[0].shape[1] if Xseq else X_frame.shape[1]
    return Xseq, y, Ls, groups, ids, np.array(soh_curr), d_in, X_frame.columns.tolist()

def build_rul_sequences(ms: pd.DataFrame, fd: pd.DataFrame, eol_pct: float, max_steps: int):
    ms = ms.copy().sort_values(["cell_id","mission_id"])
    feats = drop_meta(features_from_dict(fd, "mission", set(ms.columns)))
    X_frame = append_optional_conditions(ms, ms[feats].fillna(0.0))
    if "SOH_end_pct" not in ms.columns: raise ValueError("mission_features.csv must contain SOH_end_pct")

    RUL = []
    for cell, g in ms.groupby("cell_id"):
        g = g.sort_values("mission_id")
        below = np.where(g["SOH_end_pct"].values < eol_pct)[0]
        eol_idx = below[0] if len(below) else None
        for i in range(len(g)):
            RUL.append(max(0, eol_idx - i) if eol_idx is not None else (len(g)-1 - i))
    ms[f"RUL_missions_{int(eol_pct)}"] = np.array(RUL, dtype=float)

    Xseq, y, Ls, groups, ids = [], [], [], [], []
    for cell, g in ms.groupby("cell_id"):
        Xi = X_frame.loc[g.index].values
        yi = g[f"RUL_missions_{int(eol_pct)}"].astype(float).values
        mids = g["mission_id"].values
        for t in range(len(g)):
            s = max(0, t - max_steps + 1)
            Xseq.append(Xi[s:t+1]); y.append(float(yi[t])); Ls.append(int(t - s + 1))
            groups.append(str(cell)); ids.append((str(cell), int(mids[t]) if not pd.isna(mids[t]) else -1))
    d_in = Xseq[0].shape[1] if Xseq else X_frame.shape[1]
    return Xseq, y, Ls, groups, ids, d_in, X_frame.columns.tolist()

# ------------------- CLASSICAL FEATURES -------------------
def pool_last_step(Xseq: List[np.ndarray]) -> np.ndarray:
    return np.stack([x[-1] for x in Xseq], axis=0) if Xseq else np.zeros((0,1))

# ------------------- SPLITS -------------------
def inner_val_split(train_groups: List[str]):
    groups_arr = np.array(train_groups)
    n_splits = min(3, max(2, len(np.unique(groups_arr))))
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups_arr))
    tr_idx, va_idx = next(gkf.split(idx, groups=groups_arr))
    return tr_idx, va_idx

# ------------------- PLOTTING -------------------
def make_comparison_plots(out_root: Path):
    import matplotlib.pyplot as plt
    tasks = ["soc_phase_delta","soh_next_mission","rul_mission_80","rul_mission_85","rul_mission_90"]
    model_order = ["rf","xgb","catboost","lightgbm","attn_lstm","attn_lstm_moe"]
    def _load(task_dir: Path):
        rows, percell = [], {}
        for m in model_order:
            per_csv = task_dir / m / "per_cell_metrics.csv"
            if per_csv.exists():
                df = pd.read_csv(per_csv)
                if not df.empty:
                    rows.append({"model": m,
                                 "MAE_mean": float(df["MAE"].mean()),
                                 "RMSE_mean": float(df["RMSE"].mean()),
                                 "N_cells": int(df["cell_id"].nunique()),
                                 "N_total": int(df["N"].sum())})
                    percell[m] = df["MAE"].values
        return (pd.DataFrame(rows), percell)
    plots_dir = out_root / "plots"; ensure_dir(plots_dir)
    summary_rows = []
    for task in tasks:
        task_dir = out_root / task
        if not task_dir.exists(): continue
        summ, percell = _load(task_dir)
        if summ.empty: continue
        summ = summ.set_index("model").reindex(model_order).dropna().reset_index()
        labels = summ["model"].tolist()
        # Fleet MAE
        import matplotlib
        plt.figure(figsize=(8,4)); plt.bar(labels, summ["MAE_mean"].values)
        plt.title(f"{task} — Fleet MAE (lower is better)"); plt.ylabel("MAE"); plt.xticks(rotation=20, ha="right")
        plt.tight_layout(); plt.savefig(plots_dir / f"{task}_fleet_MAE_bar.png"); plt.close()
        # Fleet RMSE
        plt.figure(figsize=(8,4)); plt.bar(labels, summ["RMSE_mean"].values)
        plt.title(f"{task} — Fleet RMSE (lower is better)"); plt.ylabel("RMSE"); plt.xticks(rotation=20, ha="right")
        plt.tight_layout(); plt.savefig(plots_dir / f"{task}_fleet_RMSE_bar.png"); plt.close()
        # Per-cell box
        data = [percell[m] for m in labels if m in percell]
        plt.figure(figsize=(9,4)); plt.boxplot(data, labels=labels, showfliers=False)
        plt.title(f"{task} — Per-cell MAE distribution"); plt.ylabel("MAE per cell"); plt.xticks(rotation=20, ha="right")
        plt.tight_layout(); plt.savefig(plots_dir / f"{task}_percell_MAE_box.png"); plt.close()
        for _, r in summ.iterrows(): summary_rows.append({"task": task, **r.to_dict()})
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(plots_dir / "summary_fleet_metrics.csv", index=False)
        print("Saved comparison plots & summary at:", plots_dir)

# ------------------- RUL MONOTONICITY CHECK -------------------
def check_rul_monotonicity(model_dir_root: Path, model_name: str) -> Dict[str, float]:
    """Loads all_cells_predictions.csv for rul_80/85/90 and reports violation rates."""
    def load(task): 
        p = model_dir_root / f"rul_mission_{task}" / model_name / "all_cells_predictions.csv"
        return pd.read_csv(p) if p.exists() else None
    df80, df85, df90 = load(80), load(85), load(90)
    if df80 is None or df85 is None or df90 is None: 
        return {}
    # harmonize keys and columns
    keys = [c for c in ["cell_id","mission_id"] if c in df80.columns and c in df85.columns and c in df90.columns]
    pred_col = "y_pred_pp" if "y_pred_pp" in df80.columns else "y_pred"
    df = df80[keys+[pred_col]].merge(df85[keys+[pred_col]], on=keys, suffixes=("_80","_85"))
    df = df.merge(df90[keys+[pred_col]].rename(columns={pred_col: f"{pred_col}_90"}), on=keys)
    p80, p85, p90 = df[f"{pred_col}_80"].values, df[f"{pred_col}_85"].values, df[f"{pred_col}_90"].values
    v1 = np.mean(p85 > p80)  # should be <=
    v2 = np.mean(p90 > p85)
    v_any = np.mean((p85 > p80) | (p90 > p85))
    out = {"viol_85_gt_80": float(v1), "viol_90_gt_85": float(v2), "viol_any": float(v_any), "N": int(len(df))}
    save_json(model_dir_root / "rul_monotonicity.json", out)
    print(f"[Monotonicity] {model_name}: any-violation={out['viol_any']:.3f} over N={out['N']}")
    return out
