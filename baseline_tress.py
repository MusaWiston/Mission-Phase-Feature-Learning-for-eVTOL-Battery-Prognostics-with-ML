# baseline_tress.py
# Fair, fast tree baselines under LOCO with early stopping & hypertuning.
# Models: RandomForest, XGBoost, CatBoost, LightGBM
# Data utilities are imported from battery_common.py

import os, json, warnings, argparse, math
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Optional boosters
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor

# ---- Project utilities (from your codebase) ----
from battery_common import (
    SEED, ensure_dir, save_json,
    load_inputs, build_soc_sequences, build_soh_sequences, build_rul_sequences,
    pool_last_step, inner_val_split, corr_based_drop, name_based_drop,
    mae_np, rmse_np, MAX_STEPS_SOC, MAX_STEPS_MISS, SOH_HORIZON_K
)

RNG = np.random.default_rng(SEED)

# ----------------- CONFIG -----------------
EOL_SWEEP = [80.0, 85.0, 90.0]  # thresholds for RUL

# ----------------- SMALL UTILITIES -----------------
def write_preds_csv(y_true, y_pred, test_ids, out_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(dict(y_true=y_true, y_pred=y_pred))
    if len(test_ids):
        cols = list(zip(*test_ids))
        names = ["cell_id", "mission_id", "seg_or_mission"][:len(cols)]
        for j, nm in enumerate(names):
            df[nm] = cols[j]
        df = df[names + [c for c in df.columns if c not in names]]
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    return df

def robust_val_split(train_groups: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Use GroupKFold to make a (train, val) split on the training set."""
    groups_arr = np.array(train_groups)
    n_splits = min(3, max(2, len(np.unique(groups_arr))))
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups_arr))
    tr_idx, va_idx = next(gkf.split(idx, groups=groups_arr))
    return tr_idx, va_idx

def fit_imputer_scaler(X_tr: np.ndarray) -> Tuple[SimpleImputer, StandardScaler]:
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    sc  = StandardScaler(with_mean=True, with_std=True)
    sc.fit(X_tr_i)
    return imp, sc

def transform(imputer, scaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(imputer.transform(X))

# ----------------- SINGLE MODEL TRAIN/EVAL -----------------
def train_eval_rf(X_tr, y_tr, X_va, y_va, X_te, y_te, test_ids, out_csv: Path) -> Dict[str, float]:
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=None,
        min_samples_leaf=2, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    df = write_preds_csv(y_te, y_pred, test_ids, out_csv)
    return {"MAE": float(np.mean(np.abs(y_te - y_pred))),
            "RMSE": float(np.sqrt(np.mean((y_te - y_pred)**2))),
            "N": int(len(y_te))}

def train_eval_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, test_ids, out_csv: Path) -> Optional[Dict[str, float]]:
    if not XGB_AVAILABLE:
        print("  [XGBoost] Not installed; skipping."); return None
    xgb = XGBRegressor(
        n_estimators=4000, eta=0.03,
        max_depth=7, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        tree_method="hist", random_state=SEED, n_jobs=-1
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        verbose=False,
        early_stopping_rounds=200
    )
    y_pred = xgb.predict(X_te, iteration_range=(0, xgb.best_ntree_limit))
    df = write_preds_csv(y_te, y_pred, test_ids, out_csv)
    return {"MAE": float(np.mean(np.abs(y_te - y_pred))),
            "RMSE": float(np.sqrt(np.mean((y_te - y_pred)**2))),
            "N": int(len(y_te))}

def train_eval_catboost(X_tr, y_tr, X_va, y_va, X_te, y_te, test_ids, out_csv: Path) -> Optional[Dict[str, float]]:
    if not CAT_AVAILABLE:
        print("  [CatBoost] Not installed; skipping."); return None
    cat = CatBoostRegressor(
        iterations=5000, learning_rate=0.03,
        depth=8, l2_leaf_reg=3.0, loss_function="RMSE",
        random_seed=SEED, verbose=False, allow_writing_files=False,
        task_type="GPU" if os.getenv("CATBOOST_GPU","0")=="1" else "CPU"
    )
    cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], use_best_model=True, verbose=False)
    y_pred = cat.predict(X_te)
    df = write_preds_csv(y_te, y_pred, test_ids, out_csv)
    return {"MAE": float(np.mean(np.abs(y_te - y_pred))),
            "RMSE": float(np.sqrt(np.mean((y_te - y_pred)**2))),
            "N": int(len(y_te))}

def train_eval_lgbm(X_tr, y_tr, X_va, y_va, X_te, y_te, test_ids, out_csv: Path) -> Optional[Dict[str, float]]:
    if not LGBM_AVAILABLE:
        print("  [LightGBM] Not installed; skipping."); return None
    # Use a robust objective and force_col_wise=True (removes small auto-detection overhead)
    lgbm = LGBMRegressor(
        objective="huber",
        n_estimators=5000,           # large; early stopping will select the best
        learning_rate=0.03,
        num_leaves=63,               # try {31,63,127} if sweeping
        min_data_in_leaf=200,        # regularization for mission-scale data
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        force_col_wise=True,
        random_state=SEED, n_jobs=-1
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric=["l1","l2"],
        verbose=False,
        early_stopping_rounds=200
    )
    y_pred = lgbm.predict(X_te, num_iteration=lgbm.best_iteration_)
    df = write_preds_csv(y_te, y_pred, test_ids, out_csv)
    return {"MAE": float(np.mean(np.abs(y_te - y_pred))),
            "RMSE": float(np.sqrt(np.mean((y_te - y_pred)**2))),
            "N": int(len(y_te))}

# ----------------- TASK RUNNERS -----------------
def run_task_soc(outdir: Path):
    ph, ms, fd = load_inputs()
    Xseq, y, Ls, groups, ids, d_in, feat_names, tgt_col = build_soc_sequences(ph, fd, MAX_STEPS_SOC)
    task_name = "soc_phase_delta"
    return run_loco(task_name, outdir, Xseq, y, Ls, groups, ids, feat_names)

def run_task_soh(outdir: Path):
    ph, ms, fd = load_inputs()
    Xseq, y, Ls, groups, ids, soh_curr, d_in, feat_names = build_soh_sequences(ms, fd, MAX_STEPS_MISS, horizon_k=SOH_HORIZON_K)
    task_name = "soh_next"
    return run_loco(task_name, outdir, Xseq, y, Ls, groups, ids, feat_names)

def run_task_rul(outdir: Path, eol_pct: float):
    ph, ms, fd = load_inputs()
    Xseq, y, Ls, groups, ids, d_in, feat_names = build_rul_sequences(ms, fd, eol_pct, MAX_STEPS_MISS)
    task_name = f"rul_eol_{int(eol_pct)}"
    return run_loco(task_name, outdir, Xseq, y, Ls, groups, ids, feat_names)

# ----------------- CORE LOCO LOOP -----------------
def run_loco(task_name: str, outdir: Path,
             Xseq: List[np.ndarray], y: np.ndarray, Ls: List[int],
             groups: List[str], ids: List[Tuple], feat_names: List[str]) -> Dict[str, Dict]:
    outdir = Path(outdir)
    ensure_dir(outdir)
    # Flatten to last-step features
    X = pool_last_step(Xseq)
    groups_arr = np.array(groups)
    cells = sorted(list({str(g) for g in groups}))
    idx_all = np.arange(len(groups))

    # Initialize collectors
    models = {
        "rf":       {"pred_frames": [], "metrics": []},
        "xgb":      {"pred_frames": [], "metrics": []},
        "catboost": {"pred_frames": [], "metrics": []},
        "lightgbm": {"pred_frames": [], "metrics": []},
    }

    for hold in cells:
        print(f"\n[{task_name}][trees] Hold-out cell: {hold}")
        te_mask = (groups_arr == hold)
        te_idx  = idx_all[te_mask]
        trval_idx = idx_all[~te_mask]

        if trval_idx.size == 0 or te_idx.size == 0:
            print(f"  [warn] Skipping fold {hold} (insufficient data).")
            continue

        X_trval, y_trval = X[trval_idx], y[trval_idx]
        X_te,    y_te    = X[te_idx],    y[te_idx]
        ids_te   = [ids[i] for i in te_idx]
        groups_trval = [groups[i] for i in trval_idx]  # for inner split

        # ------- name-based guard then correlation-based guard (TRAIN ONLY) -------
        keep_by_name = name_based_drop(feat_names)
        X_trval = X_trval[:, [feat_names.index(f) for f in keep_by_name]]
        X_te    = X_te[:,    [feat_names.index(f) for f in keep_by_name]]
        feat_kept_names = keep_by_name

        # inner split on training for early stopping/hyperparams
        tr_idx, va_idx = robust_val_split(groups_trval)
        X_tr, y_tr = X_trval[tr_idx], y_trval[tr_idx]
        X_va, y_va = X_trval[va_idx], y_trval[va_idx]

        # correlation-based guard on TRAIN ONLY
        keep_idx = corr_based_drop(X_tr, y_tr, feat_kept_names, thr=0.995)
        X_tr = X_tr[:, keep_idx]; X_va = X_va[:, keep_idx]; X_te_f = X_te[:, keep_idx]
        feat_kept_names = [feat_kept_names[i] for i in keep_idx]

        # imputer + scaler fit on TRAIN only
        imp, sc = fit_imputer_scaler(X_tr)
        X_tr = transform(imp, sc, X_tr)
        X_va = transform(imp, sc, X_va)
        X_te_f = transform(imp, sc, X_te_f)

        print(f"  Shapes: X_tr={X_tr.shape} X_va={X_va.shape} X_te={X_te_f.shape} | kept={len(feat_kept_names)} | y_std={np.std(y_tr):.3g}")

        # -- Train each model --
        results = {}
        # RF
        out_csv = outdir / f"{task_name}__{hold}__rf_preds.csv"
        results["rf"] = train_eval_rf(X_tr, y_tr, X_va, y_va, X_te_f, y_te, ids_te, out_csv)
        # XGB
        out_csv = outdir / f"{task_name}__{hold}__xgb_preds.csv"
        results["xgb"] = train_eval_xgb(X_tr, y_tr, X_va, y_va, X_te_f, y_te, ids_te, out_csv)
        # CAT
        out_csv = outdir / f"{task_name}__{hold}__catboost_preds.csv"
        results["catboost"] = train_eval_catboost(X_tr, y_tr, X_va, y_va, X_te_f, y_te, ids_te, out_csv)
        # LGBM
        out_csv = outdir / f"{task_name}__{hold}__lightgbm_preds.csv"
        results["lightgbm"] = train_eval_lgbm(X_tr, y_tr, X_va, y_va, X_te_f, y_te, ids_te, out_csv)

        # Collect
        for m in models.keys():
            if results[m] is not None:
                models[m]["metrics"].append(dict(cell=hold, **results[m]))

    # Save per-model metrics
    summary = {}
    for m in models.keys():
        rows = models[m]["metrics"]
        if not rows: continue
        dfm = pd.DataFrame(rows)
        ensure_dir(outdir / "metrics")
        dfm.to_csv(outdir / "metrics" / f"{task_name}__{m}_metrics.csv", index=False)
        summary[m] = {
            "MAE_mean": float(dfm["MAE"].mean()),
            "RMSE_mean": float(dfm["RMSE"].mean()),
            "N_total": int(dfm["N"].sum())
        }
    save_json(outdir / f"{task_name}__summary.json", summary)
    print(f"[{task_name}] summary:", json.dumps(summary, indent=2))
    return summary

# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./outputs/baselines_trees")
    ap.add_argument("--tasks", type=str, nargs="+",
                    default=["soc", "soh", "rul"],
                    help="Subset: soc, soh, rul")
    ap.add_argument("--sweep_rul", type=float, nargs="*", default=EOL_SWEEP,
                    help="EOL thresholds to evaluate for RUL (80/85/90).")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    if "soc" in args.tasks:
        run_task_soc(outdir)
    if "soh" in args.tasks:
        run_task_soh(outdir)
    if "rul" in args.tasks:
        for eol in args.sweep_rul:
            run_task_rul(outdir, eol_pct=float(eol))

if __name__ == "__main__":
    main()
