import math, json, warnings, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ======== Import shared utilities / data builders from battery_common ========
from battery_common import (
    DEVICE, SEED, OUT_ROOT, EOL_SWEEP,
    set_seed, ensure_dir, save_json,
    load_inputs,
    build_soc_sequences, build_soh_sequences, build_rul_sequences,
    SequenceScaler, SeqDataset, collate_pad,
    mae_np, rmse_np, inner_val_split,
)

# ========================= Hyperparameters (MoE) =============================
# (You can tweak here; the ablation harness below can sweep some of these)
D_HID        = 96
N_LAYERS     = 1
DROPOUT      = 0.15
N_EXPERTS    = 6
TOPK         = 2           # set to None for dense gating, or 1 for hard routing
BATCH_SZ     = 64
EPOCHS       = 60
LR           = 2e-3
W_DECAY      = 1e-4
PATIENCE     = 8
LAMBDA_GATE  = 0.02        # gate entropy regularization
LAMBDA_MONO  = 0.02         # RUL monotonic penalty (set >0 for monotonic encouragement)

# ============================ Model Definitions =============================
class AttnLSTMMoE(nn.Module):
    """
    Attention-LSTM with Mixture-of-Experts head.
    - LSTM on sequence
    - dot-product attention over time
    - gating over expert MLP heads
    """
    def __init__(self, d_in: int, d_hid: int = D_HID, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT, n_experts: int = N_EXPERTS,
                 topk: Optional[int] = TOPK, nonneg_head: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.q = nn.Linear(d_hid, d_hid)
        self.k = nn.Linear(d_hid, d_hid)
        self.v = nn.Linear(d_hid, d_hid)

        self.gate = nn.Linear(2 * d_hid, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_hid, d_hid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hid, 1)
            ) for _ in range(n_experts)
        ])

        self.topk = topk
        self.nonneg = nonneg_head

    def _apply_nonneg(self, y):
        return F.softplus(y) if self.nonneg else y

    def forward(self, x, lengths, mask):
        # x: [B, T, D], lengths: [B], mask: [B, T] True for valid steps
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        H, (hT, _) = self.lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)   # [B, T', H]
        if H.size(1) != mask.size(1):
            mask = mask[:, :H.size(1)]

        hT = hT[-1]                                       # [B, H]
        Q = self.q(hT).unsqueeze(1)                       # [B, 1, H]
        K = self.k(H)                                     # [B, T, H]
        V = self.v(H)                                     # [B, T, H]

        logits = (Q * K).sum(-1) / math.sqrt(K.size(-1))  # [B, T]
        logits = logits.masked_fill(~mask, -1e9)
        alpha  = torch.softmax(logits, dim=1)             # [B, T]
        ctx    = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # [B, H]

        feats  = torch.cat([ctx, hT], dim=-1)             # [B, 2H]
        gate_logits = self.gate(feats)                    # [B, E]

        if self.topk is None:
            # dense mixture
            pi = torch.softmax(gate_logits, dim=-1)
            outs = torch.stack([self._apply_nonneg(e(feats)) for e in self.experts], dim=-1)  # [B, 1, E]
            yhat = (outs * pi.unsqueeze(1)).sum(-1)      # [B, 1]
        else:
            # top-k sparse mixture
            topv, topi = torch.topk(gate_logits, self.topk, dim=-1)  # [B, k]
            pi = torch.softmax(topv, dim=-1)                         # [B, k]
            parts = []
            for j in range(self.topk):
                idx = topi[:, j]
                outj = torch.zeros(feats.size(0), 1, device=feats.device)
                for eix in idx.unique():
                    m = (idx == eix)
                    outj[m] = self._apply_nonneg(self.experts[int(eix)](feats[m]))
                parts.append(pi[:, j:j+1] * outj)
            yhat = torch.stack(parts, dim=-1).sum(-1)    # [B, 1]

        return yhat, alpha, gate_logits

# =========================== Train / Inference =============================
def _safe_load_state_dict(model: nn.Module, model_path: Path):
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)  # PyTorch >=2.3
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)                     # older PyTorch version
    model.load_state_dict(state)

def _gate_entropy(pi_logits: torch.Tensor) -> torch.Tensor:
    """Normalized gate entropy in [0,1], high=more uniform."""
    pi = torch.softmax(pi_logits, dim=-1) + 1e-8
    ent = - (pi * pi.log()).sum(dim=-1) / math.log(pi.size(-1))
    return ent.mean()

def _rul_monotonic_penalty(ids_batch: List, yhat: torch.Tensor) -> torch.Tensor:
    """
    Encourages RUL non-increasing with mission index for each cell (soft constraint).
    yhat: [B,1]
    ids_batch: list of tuples (cell_id, mission_id) or (cell_id, mission_id, seg)
    """
    if yhat.numel() <= 1:
        return torch.zeros((), device=yhat.device)

    # group indices by cell_id
    by_cell: Dict[str, List[Tuple[int,int]]] = {}
    for i, rec in enumerate(ids_batch):
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            cell, mission = str(rec[0]), int(rec[1])
        else:
            # if ids are not as expected, bail out
            return torch.zeros((), device=yhat.device)
        by_cell.setdefault(cell, []).append((i, mission))

    penalty_terms = []
    for cell, pairs in by_cell.items():
        if len(pairs) < 2:
            continue
        # sort by mission_id
        pairs.sort(key=lambda x: x[1])
        idxs = [p[0] for p in pairs]
        yh = yhat[idxs, 0]  # [m]
        # forward differences; penalize increases: max(0, y_{t+1} - y_t)
        diffs = yh[1:] - yh[:-1]
        penalty_terms.append(F.relu(diffs).mean())

    if not penalty_terms:
        return torch.zeros((), device=yhat.device)
    return torch.stack(penalty_terms).mean()

def train_moe_seq_model(
    run_name: str,
    model: nn.Module,
    X_tr: List[np.ndarray], y_tr: List[float], L_tr: List[int], ids_tr: List,
    X_va: List[np.ndarray], y_va: List[float], L_va: List[int], ids_va: List,
    d_in: int, outdir: Path,
    nonneg: bool = False,
    epochs: Optional[int] = None,
    lr: float = LR,
    lambda_gate: float = LAMBDA_GATE,
    lambda_mono: float = LAMBDA_MONO,
    apply_mono: bool = False,
) -> Tuple[Path, Path, list]:
    """
    Trains MoE with SmoothL1 + gate entropy encouragement - lambda_mono * monotonic penalty (RUL only).
    Saves best scaler & weights by val MAE; returns their paths.
    """
    ensure_dir(outdir)
    # scale
    scaler = SequenceScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)

    ds_tr = SeqDataset(X_tr, y_tr, L_tr, ids_tr)
    ds_va = SeqDataset(X_va, y_va, L_va, ids_va)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SZ, shuffle=True,  collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_pad)

    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=W_DECAY)
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best = {"epoch": -1, "val_mae": 1e9}
    es_wait = 0
    E = epochs or EPOCHS
    gate_sched_log = []

    for epoch in range(1, E + 1):
        curr_lambda_gate = lambda_gate_schedule(epoch, EPOCHS)
        gate_sched_log.append((int(epoch), float(curr_lambda_gate)))
        model.train()
        tr_sum, tr_N = 0.0, 0

        for xb, yb, Lb, mb, ids_b in dl_tr:
            xb, yb, Lb, mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
            yhat, _, gate_logits = model(xb, Lb, mb)

            task_loss = loss_fn(yhat, yb)
            ent = _gate_entropy(gate_logits)
            loss = task_loss - curr_lambda_gate * ent

            if apply_mono and lambda_mono > 0.0:
                # small, soft penalty to discourage RUL increases
                mono = _rul_monotonic_penalty(ids_b, yhat)
                loss = loss + lambda_mono * mono

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += float(task_loss.item()) * xb.size(0)
            tr_N   += xb.size(0)

        # validation
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for xb, yb, Lb, mb, _ in dl_va:
                xb, yb, Lb, mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
                yh, _, _ = model(xb, Lb, mb)
                y_true.append(yb.cpu().numpy().ravel())
                y_pred.append(yh.cpu().numpy().ravel())
            y_true = np.concatenate(y_true) if y_true else np.array([])
            y_pred = np.concatenate(y_pred) if y_pred else np.array([])
            v_mae, v_rmse = mae_np(y_true, y_pred), rmse_np(y_true, y_pred)

        sche.step(v_mae)
        print(f"[{run_name}] Epoch {epoch:03d} | train Huber {tr_sum/max(1,tr_N):.4f} | val MAE {v_mae:.4f} RMSE {v_rmse:.4f}")

        if v_mae + 1e-6 < best["val_mae"]:
            best.update({"epoch": epoch, "val_mae": v_mae})
            torch.save(model.state_dict(), outdir / f"{run_name}_best.pt")
            scaler.save(outdir / f"{run_name}_scaler.joblib")
            es_wait = 0
        else:
            es_wait += 1
            if es_wait >= PATIENCE:
                print(f"[{run_name}] Early stopping at {epoch}. Best={best['epoch']} (MAE={best['val_mae']:.4f})")
                break

    save_json(outdir / f"{run_name}_best.json", best)
    return outdir / f"{run_name}_best.pt", outdir / f"{run_name}_scaler.joblib", gate_sched_log



def infer_moe_seq_model(
    run_name: str, model: nn.Module, model_path: Path, scaler_path: Path,
    X_te: List[np.ndarray], y_te: List[float], L_te: List[int], ids_te: List,
    d_in: int, out_csv: Path
) -> Dict[str, float]:
    import joblib, time
    import numpy as np
    import pandas as pd

    # scale
    scaler = SequenceScaler()
    scaler.scaler = joblib.load(scaler_path)
    X_te = scaler.transform(X_te)

    ds_te = SeqDataset(X_te, y_te, L_te, ids_te)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_pad)

    _safe_load_state_dict(model, model_path)
    model.eval()

    y_true_all, y_pred_all, id_rows = [], [], []
    attn_summ_rows = []   # (argmax, maxval, entropy)
    gate_summ_rows = []   # (top1_idx, top1_pi, top2_idx, top2_pi)
    attn_full = []
    gate_logits_full = []
    n_samples = 0

    t0 = time.time()
    with torch.no_grad():
        for xb, yb, Lb, mb, ids_b in dl_te:
            xb, yb, Lb, mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
            yh, alpha, gate_logits = model(xb, Lb, mb)

            y_true_all.append(yb.cpu().numpy().ravel())
            y_pred_all.append(yh.cpu().numpy().ravel())
            id_rows.extend(ids_b)

            alpha_np = alpha.cpu().numpy()
            for i in range(alpha_np.shape[0]):
                a = alpha_np[i]
                Li = int(Lb[i].cpu().item())
                a = a[:Li]
                attn_full.append(a)
                if Li > 0:
                    amax_idx = int(a.argmax())
                    amax_val = float(a[amax_idx])
                    ap = a / (a.sum() + 1e-12)
                    ent = float(- (ap * np.log(ap + 1e-12)).sum() / np.log(max(1, Li)))
                else:
                    amax_idx, amax_val, ent = -1, float('nan'), 0.0
                attn_summ_rows.append((amax_idx, amax_val, ent))

            gl = gate_logits.cpu().numpy()
            gate_logits_full.append(gl)
            pi = np.exp(gl - gl.max(axis=1, keepdims=True))
            pi = pi / (pi.sum(axis=1, keepdims=True) + 1e-12)
            top2_idx = np.argsort(-pi, axis=1)[:, :2]
            top2_val = np.take_along_axis(pi, top2_idx, axis=1)
            for i in range(pi.shape[0]):
                gate_summ_rows.append((int(top2_idx[i,0]), float(top2_val[i,0]), int(top2_idx[i,1]), float(top2_val[i,1])))

            n_samples += xb.size(0)
    t1 = time.time()
    avg_ms_per_sample = 1000.0 * (t1 - t0) / max(1, n_samples)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])

    res = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if len(id_rows):
        cols = list(zip(*id_rows))
        names = ["cell_id", "mission_id", "seg_or_mission"][:len(cols)]
        for j, nm in enumerate(names):
            res[nm] = cols[j]
        res = res[names + [c for c in res.columns if c not in names]]

    if len(attn_summ_rows) == len(res):
        a_idx, a_max, a_ent = zip(*attn_summ_rows)
        res["attn_argmax"] = a_idx
        res["attn_max"]    = a_max
        res["attn_entropy"]= a_ent
    if len(gate_summ_rows) == len(res):
        t1_idx, t1_pi, t2_idx, t2_pi = zip(*gate_summ_rows)
        res["gate_top1"]   = t1_idx
        res["gate_pi_top1"]= t1_pi
        res["gate_top2"]   = t2_idx
        res["gate_pi_top2"]= t2_pi

    ensure_dir(out_csv.parent)
    res.to_csv(out_csv, index=False)

    # Save full arrays (variable-length attention lists and gate logits) in NPZ
    extras_path = out_csv.with_suffix(".extras.npz")
    np.savez(extras_path, attn_list=np.array(attn_full, dtype=object), gate_logits=np.array(gate_logits_full, dtype=object), allow_pickle=True)

    return {"MAE": mae_np(y_true, y_pred), "RMSE": rmse_np(y_true, y_pred), "N": int(len(y_true)), "avg_ms_per_sample": float(avg_ms_per_sample)}


# ==================== Postprocess helper (ΔSOH → SOH) ======================
def _apply_postprocess_and_overwrite_csv(
    df_preds: pd.DataFrame, out_csv: Path, ids_test: List,
    postprocess_fn, payload: Dict
) -> Dict[str, float]:
    y_true_pp, y_pred_pp = postprocess_fn(df_preds, ids_test, payload)
    df_preds["y_true_pp"] = y_true_pp
    df_preds["y_pred_pp"] = y_pred_pp
    df_preds.to_csv(out_csv, index=False)
    return {"MAE": mae_np(y_true_pp, y_pred_pp), "RMSE": rmse_np(y_true_pp, y_pred_pp), "N": int(len(y_true_pp))}

# ============================= LOCO (MoE (Mixture of experts) only) =============================
def loco_moe(
    task_name: str,
    Xseq: List[np.ndarray], y: List[float], Ls: List[int], groups: List[str], ids: List,
    d_in: int, feat_space: List[str],
    *, nonneg_head: bool,
    postprocess_fn=None,
    extra_payload: Optional[dict] = None,
    aux_vector: Optional[np.ndarray] = None,  # i.e, SOH_curr for ΔSOH reconstruction
    apply_mono: bool = False,
    lambda_mono: float = LAMBDA_MONO,
    n_experts: int = N_EXPERTS,
    topk: Optional[int] = TOPK,
    d_hid: int = D_HID,
    dropout: float = DROPOUT,
    lr: float = LR,
    epochs: Optional[int] = None,
    lambda_gate: float = LAMBDA_GATE,
):
    base_task_dir = OUT_ROOT / task_name
    base_dir = base_task_dir / "attn_lstm_moe"
    ensure_dir(base_dir)
    save_json(base_task_dir / "feature_space.json", {"features": feat_space, "task": task_name})

    cells = sorted(list({str(g) for g in groups}))
    groups_arr = np.array(groups)
    idx_all = np.arange(len(groups))

    per_cell_rows, pred_frames = [], []

    for hold in cells:
        print(f"\n[{task_name}][moe] Hold-out cell: {hold}")
        te_mask = (groups_arr == hold)
        te_idx  = idx_all[te_mask]
        tr_idx  = idx_all[~te_mask]
        if len(te_idx) == 0 or len(tr_idx) < 50:
            print(f"  Skipping {hold} (test N={len(te_idx)}, train N={len(tr_idx)})")
            continue

        tr_groups = [groups[i] for i in tr_idx]
        tr2_idx, va2_idx = inner_val_split(tr_groups)
        tr_final, va_final = tr_idx[tr2_idx], tr_idx[va2_idx]

        cell_dir = base_dir / hold
        ensure_dir(cell_dir)

        model = AttnLSTMMoE(
            d_in=d_in, d_hid=d_hid, n_layers=N_LAYERS, dropout=dropout,
            n_experts=n_experts, topk=topk, nonneg_head=nonneg_head
        ).to(DEVICE)

        run_name = f"{task_name}_attn_lstm_moe"
        best_pt, best_scaler, gate_sched_log = train_moe_seq_model(
            run_name, model,
            [Xseq[i] for i in tr_final], [y[i] for i in tr_final], [Ls[i] for i in tr_final], [ids[i] for i in tr_final],
            [Xseq[i] for i in va_final], [y[i] for i in va_final], [Ls[i] for i in va_final], [ids[i] for i in va_final],
            d_in, cell_dir, nonneg=nonneg_head,
            epochs=epochs, lr=lr, lambda_gate=lambda_gate,
            lambda_mono=lambda_mono, apply_mono=apply_mono
        )

        # Test
        ids_te = [ids[i] for i in te_idx]
        out_csv = cell_dir / f"{task_name}_attn_lstm_moe_test_{hold}.csv"
        metrics = infer_moe_seq_model(
            run_name,
            AttnLSTMMoE(d_in=d_in, d_hid=d_hid, n_layers=N_LAYERS, dropout=dropout,
                        n_experts=n_experts, topk=topk, nonneg_head=nonneg_head).to(DEVICE),
            best_pt, best_scaler,
            [Xseq[i] for i in te_idx], [y[i] for i in te_idx], [Ls[i] for i in te_idx], ids_te,
            d_in, out_csv
        )
        dfp = pd.read_csv(out_csv)

        if postprocess_fn is not None:
            payload = dict(extra_payload) if extra_payload else {}
            if aux_vector is not None:
                payload["soh_curr_test"] = np.asarray(aux_vector)[te_idx]  # <-- critical fix for SOH
            metrics = _apply_postprocess_and_overwrite_csv(dfp, out_csv, ids_te, postprocess_fn, payload)
            dfp = pd.read_csv(out_csv)  # reload with pp cols

        metrics["cell_id"] = hold
        per_cell_rows.append(metrics)

        dfp["model"] = "attn_lstm_moe"
        pred_frames.append(dfp)

        # free GPU mem between cells
        # write gate schedule
        try:
            import pandas as _pd
            _gs = _pd.DataFrame(gate_sched_log, columns=["epoch","lambda_gate"])
            _gs.to_csv(cell_dir / f"{run_name}_gate_schedule.csv", index=False)
        except Exception as _e:
            print("[warn] gate schedule save skipped:", _e)
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Save aggregates
    if pred_frames:
        all_pred = pd.concat(pred_frames, ignore_index=True)
        all_pred.to_csv(base_dir / "all_cells_predictions.csv", index=False)
    if per_cell_rows:
        met = pd.DataFrame(per_cell_rows).sort_values("cell_id")
        met.to_csv(base_dir / "per_cell_metrics.csv", index=False)
        save_json(base_dir / "fleet_metrics.json", {
            "num_cells": int(met["cell_id"].nunique()),
            "total_samples": int(met["N"].sum()),
            "MAE_mean": float(met["MAE"].mean()),
            "MAE_median": float(met["MAE"].median()),
            "RMSE_mean": float(met["RMSE"].mean()),
            "RMSE_median": float(met["RMSE"].median()),
        })

# ======================= Task-specific postprocess ==========================

def soh_postprocess(df_preds: pd.DataFrame, ids_test: List, payload: Dict):
    """
    Reconstruct absolute SOH_next from ΔSOH predictions:
      SOH_next = clip(SOH_curr - ΔSOH, 0..100).
    Requires payload["soh_curr_test"] aligned with df_preds rows.
    """
    if "soh_curr_test" not in payload:
        raise KeyError("soh_postprocess requires payload['soh_curr_test'] aligned to the test rows.")
    soh_curr_test = np.asarray(payload["soh_curr_test"])
    T = len(df_preds)
    assert len(soh_curr_test) >= T, "Length mismatch: soh_curr_test shorter than predictions."
    assert np.all(np.isfinite(soh_curr_test[:T])), "NaNs/Infs found in soh_curr_test[:T]."

    y_true_abs = np.clip(soh_curr_test[:T] - df_preds["y_true"].values, 0.0, 100.0)
    y_pred_abs = np.clip(soh_curr_test[:T] - df_preds["y_pred"].values, 0.0, 100.0)
    return y_true_abs, y_pred_abs


# ============================== Ablation ====================================
def ablate_moe_quick(
    task_name: str,
    Xseq, y, Ls, groups, ids, d_in, feat_space,
    *,  # required keywords:
    nonneg_head: bool,
    apply_mono: bool = False,
    aux_vector: Optional[np.ndarray] = None,
    cells_limit: int = 3,
    epochs_quick: int = 12,
    grid_N_EXPERTS: List[int] = (2, 4, 6),
    grid_TOPK: List[Optional[int]] = (1, 2),
    grid_LAMBDA_MONO: List[float] = (0.0, 0.01, 0.02),
    lr: float = 2e-3,
    d_hid: int = 96,
    dropout: float = 0.15,
    lambda_gate: float = 0.02,
) -> Path:
    """
    Minimal ablation to quickly sweep (N_EXPERTS, TOPK, LAMBDA_MONO) on a few cells and dump a table.
    Use for SOH or RUL (for SOC, LAMBDA_MONO is ignored).
    """
    print("\n[Ablation] Quick sweep starting ...")
    base_task_dir = OUT_ROOT / task_name
    results = []
    cells = sorted(list({str(g) for g in groups}))[:cells_limit]
    groups_arr = np.array(groups)
    idx_all = np.arange(len(groups))

    for ne in grid_N_EXPERTS:
        for tk in grid_TOPK:
            for lm in grid_LAMBDA_MONO:
                per_cell_rows = []
                for hold in cells:
                    te_mask = (groups_arr == hold)
                    te_idx  = idx_all[te_mask]
                    tr_idx  = idx_all[~te_mask]
                    if len(te_idx) == 0 or len(tr_idx) < 50:
                        continue
                    tr_groups = [groups[i] for i in tr_idx]
                    tr2_idx, va2_idx = inner_val_split(tr_groups)
                    tr_final, va_final = tr_idx[tr2_idx], tr_idx[va2_idx]

                    cell_dir = base_task_dir / "ablation_tmp"
                    ensure_dir(cell_dir)
                    model = AttnLSTMMoE(d_in=d_in, d_hid=d_hid, n_layers=N_LAYERS, dropout=dropout,
                                        n_experts=ne, topk=tk, nonneg_head=nonneg_head).to(DEVICE)
                    run_name = f"{task_name}_ablation_ne{ne}_tk{tk}_lm{lm}"
                    best_pt, best_scaler, gate_sched_log = train_moe_seq_model(
                        run_name, model,
                        [Xseq[i] for i in tr_final], [y[i] for i in tr_final], [Ls[i] for i in tr_final], [ids[i] for i in tr_final],
                        [Xseq[i] for i in va_final], [y[i] for i in va_final], [Ls[i] for i in va_final], [ids[i] for i in va_final],
                        d_in, cell_dir, nonneg=nonneg_head,
                        epochs=epochs_quick, lr=lr, lambda_gate=lambda_gate,
                        lambda_mono=lm, apply_mono=apply_mono
                    )
                    # test quickly
                    ids_te = [ids[i] for i in te_idx]
                    out_csv = cell_dir / f"{run_name}_test_{hold}.csv"
                    metrics = infer_moe_seq_model(
                        run_name,
                        AttnLSTMMoE(d_in=d_in, d_hid=d_hid, n_layers=N_LAYERS, dropout=dropout,
                                    n_experts=ne, topk=tk, nonneg_head=nonneg_head).to(DEVICE),
                        best_pt, best_scaler,
                        [Xseq[i] for i in te_idx], [y[i] for i in te_idx], [Ls[i] for i in te_idx], ids_te,
                        d_in, out_csv
                    )
                    dfp = pd.read_csv(out_csv)
                    if task_name == "soh_next_mission":
                        payload = {}
                        if aux_vector is not None:
                            payload["soh_curr_test"] = np.asarray(aux_vector)[te_idx]
                        metrics = _apply_postprocess_and_overwrite_csv(dfp, out_csv, ids_te, soh_postprocess, payload)

                    metrics["cell_id"] = hold
                    per_cell_rows.append(metrics)

                    # write gate schedule
        try:
            import pandas as _pd
            _gs = _pd.DataFrame(gate_sched_log, columns=["epoch","lambda_gate"])
            _gs.to_csv(cell_dir / f"{run_name}_gate_schedule.csv", index=False)
        except Exception as _e:
            print("[warn] gate schedule save skipped:", _e)
        del model
        try:
                  torch.cuda.empty_cache()
        except Exception:
                        pass

        if per_cell_rows:
                    met = pd.DataFrame(per_cell_rows)
                    results.append({
                        "N_EXPERTS": ne, "TOPK": tk, "LAMBDA_MONO": lm,
                        "MAE_mean": float(met["MAE"].mean()),
                        "RMSE_mean": float(met["RMSE"].mean()),
                        "cells": len(met["cell_id"].unique()),
                        "N_total": int(met["N"].sum()),
                    })
                    print(f"[Ablation] ne={ne} tk={tk} lm={lm} -> MAE={results[-1]['MAE_mean']:.4f} RMSE={results[-1]['RMSE_mean']:.4f}")

    out_csv = base_task_dir / "moe_ablation_summary.csv"
    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print("Ablation summary saved to:", out_csv)
    else:
        print("Ablation produced no rows (insufficient samples?).")
    return out_csv

def isotonic_calibrate_rul_across_thresholds(base_dir: Path, model_name: str = "attn_lstm_moe") -> None:
    import pandas as pd, numpy as np, json
    from pathlib import Path

    def _load(task):
        p = base_dir / f"rul_mission_{task}" / model_name / "all_cells_predictions.csv"
        return pd.read_csv(p) if p.exists() else None

    df80, df85, df90 = _load(80), _load(85), _load(90)
    if df80 is None or df85 is None or df90 is None:
        print("[Isotonic] Missing one or more RUL outputs; skip calibration.")
        return

    keys = [c for c in ["cell_id","mission_id"] if c in df80.columns and c in df85.columns and c in df90.columns]
    pred_col = "y_pred_pp" if "y_pred_pp" in df80.columns else "y_pred"
    df = df80[keys+[pred_col]].merge(df85[keys+[pred_col]], on=keys, suffixes=("_80","_85"))
    df = df.merge(df90[keys+[pred_col]].rename(columns={pred_col: f"{pred_col}_90"}), on=keys)

    x = np.array([80.0, 85.0, 90.0], dtype=float)

    try:
        from sklearn.isotonic import IsotonicRegression
        use_sklearn = True
    except Exception:
        use_sklearn = False

    y_cal = np.zeros((len(df), 3), dtype=float)
    viol_before = np.mean((df[f"{pred_col}_85"].values > df[f"{pred_col}_80"].values) | (df[f"{pred_col}_90"].values > df[f"{pred_col}_85"].values))

    for i, r in df.iterrows():
        y = np.array([r[f"{pred_col}_80"], r[f"{pred_col}_85"], r[f"{pred_col}_90"]], dtype=float)
        if use_sklearn:
            ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
            y_fit = ir.fit_transform(x, y)
        else:
            y_fit = y.copy()
            y_fit[1] = min(y_fit[1], y_fit[0])
            y_fit[2] = min(y_fit[2], y_fit[1])
        y_cal[i] = y_fit

    viol_after = np.mean((y_cal[:,1] > y_cal[:,0]) | (y_cal[:,2] > y_cal[:,1]))

    cal_cols = pd.DataFrame({**{k: df[k] for k in keys}, "y_pred_cal_80": y_cal[:,0], "y_pred_cal_85": y_cal[:,1], "y_pred_cal_90": y_cal[:,2]})
    def _write(task, colname):
        p = base_dir / f"rul_mission_{task}" / model_name / "all_cells_predictions.csv"
        outp = base_dir / f"rul_mission_{task}" / model_name / "all_cells_predictions_calibrated.csv"
        d = pd.read_csv(p)
        d = d.merge(cal_cols[keys+[colname]], on=keys, how="left")
        d.rename(columns={colname: "y_pred_cal"}, inplace=True)
        d.to_csv(outp, index=False)

    _write(80, "y_pred_cal_80")
    _write(85, "y_pred_cal_85")
    _write(90, "y_pred_cal_90")

    summary = {"viol_before": float(viol_before), "viol_after": float(viol_after), "N": int(len(df)), "used_sklearn": bool(use_sklearn)}
    with open(base_dir / "rul_isotonic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Isotonic] Calibration finished. violations: before={viol_before:.3f}, after={viol_after:.3f}")


# ---- Gate entropy schedule (simple linear decay) ----
def lambda_gate_schedule(epoch: int, max_epochs: int, start: float = 0.02, end: float = 0.005) -> float:
    """
    Linearly decays lambda_gate from `start` to `end` over the course of training.
    """
    if max_epochs <= 1:
        return end
    t = max(0.0, min(1.0, epoch / float(max_epochs - 1)))
    return start + (end - start) * t


# ================================= MAIN =====================================
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(SEED)
    ensure_dir(OUT_ROOT)

    ph, ms, fd = load_inputs()

    # ------------------ 1) SOC (phase-level ΔSOC) ------------------
    print("\n>>> Building SOC (phase-level) sequences")
    X_soc, y_soc, L_soc, g_soc, id_soc, d_in_soc, soc_feats, soc_tgt = build_soc_sequences(ph, fd, max_steps=16)
    # ΔSOC can be signed; nonneg_head=False, no postprocess
    loco_moe(
        task_name="soc_phase_delta",
        Xseq=X_soc, y=y_soc, Ls=L_soc, groups=g_soc, ids=id_soc,
        d_in=d_in_soc, feat_space=soc_feats,
        nonneg_head=False,
        postprocess_fn=None,
        aux_vector=None,
        apply_mono=False,           # monotonic not relevant for ΔSOC
        lambda_mono=0.0
    )

    # ------------------ 2) SOH_next (mission-level ΔSOH) -----------
    print("\n>>> Building SOH_next (mission-level) sequences as ΔSOH")
    X_soh, y_soh, L_soh, g_soh, id_soh, soh_curr, d_in_soh, soh_feats = build_soh_sequences(ms, fd, max_steps=20)
    loco_moe(
        task_name="soh_next_mission",
        Xseq=X_soh, y=y_soh, Ls=L_soh, groups=g_soh, ids=id_soh,
        d_in=d_in_soh, feat_space=soh_feats,
        nonneg_head=True,                  # ΔSOH >= 0
        postprocess_fn=soh_postprocess,    # reconstruct absolute SOH_next
        aux_vector=soh_curr,               # critical for postprocess
        apply_mono=False,                  # monotonic not relevant for ΔSOH
        lambda_mono=0.0
    )

    # ------------------ 3) RUL @ multiple EOL thresholds ----------
    print("\n>>> Building RUL mission-level sequences (EOL sweep)")
    for eol in EOL_SWEEP:
        X_rul, y_rul, L_rul, g_rul, id_rul, d_in_rul, rul_feats = build_rul_sequences(ms, fd, eol_pct=eol, max_steps=20)
        loco_moe(
            task_name=f"rul_mission_{int(eol)}",
            Xseq=X_rul, y=y_rul, Ls=L_rul, groups=g_rul, ids=id_rul,
            d_in=d_in_rul, feat_space=rul_feats,
            nonneg_head=True,              # RUL >= 0
            postprocess_fn=None,
            aux_vector=None,
            apply_mono=True,               # encourage non-increasing RUL
            lambda_mono=LAMBDA_MONO
        )

    # ------------------ 4) Tiny ablation (optional) ---------------
    # running a quick sweep on RUL@85 for a few cells and short epochs
    try:
        X_rul85, y_rul85, L_rul85, g_rul85, id_rul85, d_in_rul85, rul85_feats = build_rul_sequences(ms, fd, eol_pct=85.0, max_steps=20)
        ablate_moe_quick(
            "rul_mission_85",
            X_rul85, y_rul85, L_rul85, g_rul85, id_rul85, d_in_rul85, rul85_feats,
            nonneg_head=True, apply_mono=True,
            aux_vector=None,
            cells_limit=3, epochs_quick=12,
            grid_N_EXPERTS=[2, 4, 6],
            grid_TOPK=[1, 2],
            grid_LAMBDA_MONO=[0.0, 0.01, 0.02],
            lr=LR, d_hid=D_HID, dropout=DROPOUT, lambda_gate=LAMBDA_GATE
        )
    except Exception as e:
        # Safe to ignore if dataset is too small or if you don't want the sweep in long runs
        print("[Ablation] skipped due to:", e)

    # Post-hoc isotonic calibration across RUL thresholds
    try:
        isotonic_calibrate_rul_across_thresholds(OUT_ROOT)
    except Exception as e:
        print("[Isotonic] skipped due to:", e)

    print("\n MoE LOCO sweep finished. Outputs →", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()

