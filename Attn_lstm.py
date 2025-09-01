import warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from battery_common import (
    OUT_LSTM, ensure_dir, set_seed, save_json, load_inputs,
    build_soc_sequences, build_soh_sequences, build_rul_sequences,
    inner_val_split, SequenceScaler, SeqDataset, collate_pad,
    AttnLSTMBase, try_torch_load_state_dict,
    mae_np, rmse_np, summarize_fleet_metrics, make_comparison_plots,
    SOH_HORIZON_K, EOL_SWEEP, DEVICE, SEED, BATCH_SZ, EPOCHS, LR, W_DECAY, PATIENCE
)

def train_seq(model_name: str, model: nn.Module,
              X_tr, y_tr, L_tr, ids_tr,
              X_va, y_va, L_va, ids_va,
              outdir: Path) -> Path:
    ensure_dir(outdir)
    scaler = SequenceScaler(); scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va)
    ds_tr = SeqDataset(X_tr, y_tr, L_tr, ids_tr); ds_va = SeqDataset(X_va, y_va, L_va, ids_va)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH_SZ, shuffle=True, collate_fn=collate_pad)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_pad)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=W_DECAY)
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    best = {"epoch": -1, "val_mae": 1e9}; es = 0

    for ep in range(1, EPOCHS+1):
        model.train(); tr_sum=0.0; trN=0
        for xb, yb, Lb, mb, _ in dl_tr:
            xb,yb,Lb,mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
            yhat,_ = model(xb, Lb, mb)
            loss = loss_fn(yhat, yb)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_sum += float(loss.item())*xb.size(0); trN += xb.size(0)

        # val
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for xb,yb,Lb,mb,_ in dl_va:
                xb,yb,Lb,mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
                yhat,_ = model(xb, Lb, mb)
                y_true.append(yb.cpu().numpy().ravel()); y_pred.append(yhat.cpu().numpy().ravel())
        y_true = np.concatenate(y_true) if y_true else np.array([])
        y_pred = np.concatenate(y_pred) if y_pred else np.array([])
        v_mae, v_rmse = mae_np(y_true,y_pred), rmse_np(y_true,y_pred)
        sche.step(v_mae)
        print(f"[{model_name}] Epoch {ep:03d} | train Huber {tr_sum/max(1,trN):.4f} | val MAE {v_mae:.4f} RMSE {v_rmse:.4f}")

        if v_mae + 1e-6 < best["val_mae"]:
            best.update({"epoch": ep, "val_mae": v_mae})
            torch.save(model.state_dict(), outdir / f"{model_name}_best.pt")
            scaler.save(outdir / f"{model_name}_scaler.joblib")
            es=0
        else:
            es += 1
            if es >= PATIENCE:
                print(f"[{model_name}] Early stopping at {ep}. Best={best['epoch']} (MAE={best['val_mae']:.4f})")
                break
    return outdir / f"{model_name}_best.pt"

def infer_seq(model_name: str, model: nn.Module, pt_path: Path, scaler_path: Path,
              X_te, y_te, L_te, ids_te, out_csv: Path) -> Dict[str, float]:
    import joblib
    scaler = SequenceScaler(); scaler.scaler = joblib.load(scaler_path)
    X_te = scaler.transform(X_te)
    ds_te = SeqDataset(X_te, y_te, L_te, ids_te)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_pad)
    try_torch_load_state_dict(model, pt_path); model.eval()

    import time
    y_true_all, y_pred_all, id_rows = [], [], []
    n_samples = 0
    t0 = time.time()
    with torch.no_grad():
        for xb,yb,Lb,mb,ids in dl_te:
            xb,yb,Lb,mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
            yhat,_ = model(xb, Lb, mb)
            y_true_all.append(yb.cpu().numpy().ravel()); y_pred_all.append(yhat.cpu().numpy().ravel()); id_rows.extend(ids)
            n_samples += xb.size(0)
    t1 = time.time()
    avg_ms_per_sample = 1000.0 * (t1 - t0) / max(1, n_samples)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    res = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if len(id_rows):
        cols = list(zip(*id_rows)); names = ["cell_id","mission_id","seg_or_mission"][:len(cols)]
        for j,nm in enumerate(names): res[nm] = cols[j]
        res = res[names + [c for c in res.columns if c not in names]]
    ensure_dir(out_csv.parent); res.to_csv(out_csv, index=False)
    return {"MAE": mae_np(y_true, y_pred), "RMSE": rmse_np(y_true, y_pred), "N": int(len(y_true)), "avg_ms_per_sample": float(avg_ms_per_sample)}

def loco_attn(task_name: str, Xseq,y,Ls,groups,ids, d_in, post=None, soh_payload=None, nonneg=False):
    base_dir = OUT_LSTM / task_name; ensure_dir(base_dir)
    cells = sorted(list({str(g) for g in groups}))
    groups_arr = np.array(groups); idx_all = np.arange(len(groups))
    pred_frames=[]; metrics=[]
    for hold in cells:
        print(f"\n[{task_name}][attn] Hold-out cell: {hold}")
        te_mask = (groups_arr==hold); te_idx = idx_all[te_mask]; tr_idx = idx_all[~te_mask]
        if len(te_idx)==0 or len(tr_idx)<50: 
            print(f"  Skip {hold}"); continue
        tr_groups = [groups[i] for i in tr_idx]
        tr2, va2 = inner_val_split(tr_groups)
        tr_final, va_final = tr_idx[tr2], tr_idx[va2]
        cell_dir = base_dir / "attn_lstm" / hold; ensure_dir(cell_dir)

        model = AttnLSTMBase(d_in=d_in, nonneg_head=nonneg).to(DEVICE)
        name  = f"{task_name}_attn_lstm"
        pt = train_seq(name, model,
                       [Xseq[i] for i in tr_final], [y[i] for i in tr_final], [Ls[i] for i in tr_final], [ids[i] for i in tr_final],
                       [Xseq[i] for i in va_final], [y[i] for i in va_final], [Ls[i] for i in va_final], [ids[i] for i in va_final],
                       cell_dir)
        metrics_i = infer_seq(name, AttnLSTMBase(d_in=d_in, nonneg_head=nonneg).to(DEVICE),
                              pt, cell_dir / f"{name}_scaler.joblib",
                              [Xseq[i] for i in te_idx], [y[i] for i in te_idx], [Ls[i] for i in te_idx], [ids[i] for i in te_idx],
                              cell_dir / f"{name}_test_{hold}.csv")
        if post:
            df = pd.read_csv(cell_dir / f"{name}_test_{hold}.csv")
            y_true_pp, y_pred_pp = post(df, [ids[i] for i in te_idx], soh_payload(hold, te_idx) if soh_payload else None)
            metrics_i = {"MAE": mae_np(y_true_pp, y_pred_pp), "RMSE": rmse_np(y_true_pp, y_pred_pp), "N": len(y_true_pp)}
            df["y_true_pp"]=y_true_pp; df["y_pred_pp"]=y_pred_pp; df.to_csv(cell_dir / f"{name}_test_{hold}.csv", index=False)
        metrics_i["cell_id"]=hold; metrics.append(metrics_i)
        df = pd.read_csv(cell_dir / f"{name}_test_{hold}.csv"); df["model"]="attn_lstm"; pred_frames.append(df)

    mdir = base_dir / "attn_lstm"; ensure_dir(mdir)
    if pred_frames: pd.concat(pred_frames, ignore_index=True).to_csv(mdir / "all_cells_predictions.csv", index=False)
    if metrics:
        met = pd.DataFrame(metrics).sort_values("cell_id")
        met.to_csv(mdir / "per_cell_metrics.csv", index=False)
        summarize_fleet_metrics(mdir / "per_cell_metrics.csv", mdir / "fleet_metrics.json")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(SEED); ensure_dir(OUT_LSTM)
    ph, ms, fd = load_inputs()

    # SOC (signed) -> nonneg=False
    X_soc, y_soc, L_soc, g_soc, id_soc, d_in_soc, soc_feats, soc_tgt = build_soc_sequences(ph, fd, max_steps=16)
    loco_attn("soc_phase_delta", X_soc, y_soc, L_soc, g_soc, id_soc, d_in_soc, post=None, nonneg=False)

    # SOH (Δ over K missions) -> nonneg=True, with absolute postprocess
    X_soh, y_soh, L_soh, g_soh, id_soh, soh_curr, d_in_soh, soh_feats = build_soh_sequences(ms, fd, max_steps=20, horizon_k=SOH_HORIZON_K)
    def soh_post(df_preds: pd.DataFrame, ids_test, payload):
        sc = payload["soh_curr_test"]; T=len(df_preds)
        y_true_abs = np.clip(sc[:T]-df_preds["y_true"].values, 0.0, 100.0)
        y_pred_abs = np.clip(sc[:T]-df_preds["y_pred"].values, 0.0, 100.0)
        return y_true_abs, y_pred_abs
    def soh_payload(cell, te_idx): return {"soh_curr_test": soh_curr[te_idx]}
    loco_attn("soh_next_mission", X_soh, y_soh, L_soh, g_soh, id_soh, d_in_soh, post=soh_post, soh_payload=soh_payload, nonneg=True)

    # RUL thresholds (nonneg=True)
    for eol in [80.0,85.0,90.0]:
        X_rul, y_rul, L_rul, g_rul, id_rul, d_in_rul, rul_feats = build_rul_sequences(ms, fd, eol_pct=eol, max_steps=20)
        loco_attn(f"rul_mission_{int(eol)}", X_rul, y_rul, L_rul, g_rul, id_rul, d_in_rul, post=None, nonneg=True)

    make_comparison_plots(OUT_LSTM)
    print(" Plain Attention-LSTM baseline done →", OUT_LSTM.resolve())

if __name__ == "__main__":
    main()
