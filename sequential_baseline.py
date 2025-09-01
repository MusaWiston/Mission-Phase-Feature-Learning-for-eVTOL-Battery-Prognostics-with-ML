
# sequential_baseline.py

import math, warnings, argparse, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from battery_common import (
    DEVICE, SEED, set_seed, ensure_dir, save_json,
    load_inputs,
    build_soc_sequences, build_soh_sequences, build_rul_sequences,
    SequenceScaler, SeqDataset, collate_pad,
    mae_np, rmse_np,
    OUT_LSTM, EOL_SWEEP, SOH_HORIZON_K,
    D_HID, N_LAYERS, DROPOUT, BATCH_SZ, EPOCHS, LR, W_DECAY, PATIENCE
)

# ----------------------------- Shared attention & heads -----------------------------

class _TimeAttentionPool(nn.Module):
    """Single-query dot-product attention pooling over time with masking."""
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)

    def forward(self, H: torch.Tensor, q_vec: torch.Tensor, mask: torch.Tensor):
        # H: (B, T, H), q_vec: (B, H), mask: (B, T) True for valid
        Q = self.q(q_vec).unsqueeze(1)        # (B, 1, H)
        K = self.k(H)                          # (B, T, H)
        V = self.v(H)                          # (B, T, H)
        logits = torch.einsum("bij,bij->bi", Q.expand_as(K), K) / math.sqrt(K.size(-1))
        logits = logits.masked_fill(~mask, -1e9)
        alpha  = torch.softmax(logits, dim=1)  # (B, T)
        ctx    = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # (B, H)
        return ctx, alpha

class _Head(nn.Module):
    """Two-layer MLP head with optional non-negativity via softplus."""
    def __init__(self, d_in: int, d_hid: int, dropout: float, nonneg: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, 1)
        )
        self.nonneg = nonneg

    def forward(self, x):
        y = self.net(x)
        return F.softplus(y) if self.nonneg else y

# ----------------------------- Baseline models -----------------------------

class AttnGRUBase(nn.Module):
    """GRU + time attention + MLP head; forward matches AttnLSTMBase signature."""
    def __init__(self, d_in, d_hid: int = D_HID, n_layers: int = N_LAYERS, dropout: float = DROPOUT, nonneg_head: bool=False):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hid, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
        self.pool = _TimeAttentionPool(d_hid)
        self.head = _Head(d_in=2*d_hid, d_hid=d_hid, dropout=dropout, nonneg=nonneg_head)

    def forward(self, x, lengths, mask):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        H, hT = self.gru(packed)
        H, _  = pad_packed_sequence(H, batch_first=True)   # (B, T*, H)
        if H.size(1) != mask.size(1): mask = mask[:, :H.size(1)]
        hT = hT[-1]                                        # (B, H)
        ctx, alpha = self.pool(H, hT, mask)                # (B, H), (B, T*)
        feats = torch.cat([ctx, hT], dim=-1)               # (B, 2H)
        yhat = self.head(feats)
        return yhat, alpha

class AttnBiLSTMBase(nn.Module):
    """Bidirectional LSTM + time attention + MLP head."""
    def __init__(self, d_in, d_hid: int = D_HID, n_layers: int = N_LAYERS, dropout: float = DROPOUT, nonneg_head: bool=False):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers>1 else 0.0, bidirectional=True)
        self.pool = _TimeAttentionPool(2*d_hid)
        self.head = _Head(d_in=4*d_hid, d_hid=2*d_hid, dropout=dropout, nonneg=nonneg_head)

    def forward(self, x, lengths, mask):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        H, (hT, _) = self.lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)    # (B, T*, 2H)
        if H.size(1) != mask.size(1): mask = mask[:, :H.size(1)]
        # take last layer's forward/backward states
        h_f = hT[-2]; h_b = hT[-1]                         # (B, H) each
        h_cat = torch.cat([h_f, h_b], dim=-1)              # (B, 2H)
        ctx, alpha = self.pool(H, h_cat, mask)             # (B, 2H)
        feats = torch.cat([ctx, h_cat], dim=-1)            # (B, 4H)
        yhat = self.head(feats)
        return yhat, alpha

# ---- TCN ----
class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):
        return x if self.chomp_size==0 else x[:, :, :-self.chomp_size].contiguous()

class _TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding); self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.down = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class _TCN(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, kernel_size=3, dropout=0.0, causal=True):
        super().__init__()
        layers = []
        c_in = d_in
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation if causal else (kernel_size - 1) // 2
            layers.append(_TemporalBlock(c_in, d_hid, kernel_size, dilation=dilation, padding=padding, dropout=dropout))
            c_in = d_hid
        self.net = nn.Sequential(*layers)

    def forward(self, x):         # x: (B, T, D)
        y = self.net(x.transpose(1,2))   # (B, H, T)
        return y.transpose(1,2)          # (B, T, H)

class AttnTCNBase(nn.Module):
    def __init__(self, d_in, d_hid: int = D_HID, n_layers: int = N_LAYERS, dropout: float = DROPOUT, kernel_size: int = 3, nonneg_head: bool=False):
        super().__init__()
        self.tcn = _TCN(d_in, d_hid, n_layers, kernel_size=kernel_size, dropout=dropout, causal=True)
        self.pool = _TimeAttentionPool(d_hid)
        self.head = _Head(d_in=2*d_hid, d_hid=d_hid, dropout=dropout, nonneg=nonneg_head)

    def forward(self, x, lengths, mask):
        H = self.tcn(x)                                 # (B, T, H)
        if H.size(1) != mask.size(1): mask = mask[:, :H.size(1)]
        idx = torch.clamp(lengths - 1, min=0)
        hT = H[torch.arange(H.size(0), device=H.device), idx]  # (B, H)
        ctx, alpha = self.pool(H, hT, mask)
        feats = torch.cat([ctx, hT], dim=-1)
        yhat = self.head(feats)
        return yhat, alpha

# ---- TFT-lite ----
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x):  # (B, T, d)
        return x + self.pe[:, :x.size(1), :]

class AttnTFTLite(nn.Module):
    def __init__(self, d_in, d_hid: int = D_HID, n_layers: int = N_LAYERS, dropout: float = DROPOUT, nhead: int = 4, dim_ff_mult: int = 4, nonneg_head: bool=False):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hid)
        self.pos  = _PositionalEncoding(d_hid)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_hid, nhead=nhead,
                                               dim_feedforward=dim_ff_mult*d_hid,
                                               dropout=dropout, batch_first=True,
                                               activation="relu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = _TimeAttentionPool(d_hid)
        self.head = _Head(d_in=2*d_hid, d_hid=d_hid, dropout=dropout, nonneg=nonneg_head)

    def forward(self, x, lengths, mask):
        H = self.proj(x)                 # (B, T, d)
        H = self.pos(H)
        key_pad = ~mask                  # True where PAD
        H = self.encoder(H, src_key_padding_mask=key_pad)
        idx = torch.clamp(lengths - 1, min=0)
        hT = H[torch.arange(H.size(0), device=H.device), idx]
        ctx, alpha = self.pool(H, hT, mask)
        feats = torch.cat([ctx, hT], dim=-1)
        yhat = self.head(feats)
        return yhat, alpha

# ----------------------------- Train / Infer loops -----------------------------

def train_seq(model_name: str, model: nn.Module,
              X_tr, y_tr, L_tr, ids_tr,
              X_va, y_va, L_va, ids_va,
              outdir: Path) -> Path:
    ensure_dir(outdir)
    scaler = SequenceScaler(); scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va)
    ds_tr = SeqDataset(X_tr, y_tr, L_tr, ids_tr); ds_va = SeqDataset(X_va, y_va, L_va, ids_va)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SZ, shuffle=True,  collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_pad)

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
    dl_te = DataLoader(ds_te, batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_pad)

    # load weights (compat w/ torch <2.3 and >=2.3)
    try:
        state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(pt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    y_true_all, y_pred_all, id_rows = [], [], []
    n_samples = 0
    t0 = time.time()
    with torch.no_grad():
        for xb,yb,Lb,mb,ids in dl_te:
            xb,yb,Lb,mb = xb.to(DEVICE), yb.to(DEVICE), Lb.to(DEVICE), mb.to(DEVICE)
            yhat,_ = model(xb, Lb, mb)
            y_true_all.append(yb.cpu().numpy().ravel())
            y_pred_all.append(yhat.cpu().numpy().ravel())
            id_rows.extend(ids); n_samples += xb.size(0)
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

# ----------------------------- LOCO runner for a single model -----------------------------

def loco_seq_model(task_name: str, model_key: str, model_ctor, d_in: int,
                   Xseq, y, Ls, groups, ids,
                   out_root: Path, nonneg: bool, post=None, soh_payload=None):
    base_dir = out_root / task_name; ensure_dir(base_dir)
    cells = sorted(list({str(g) for g in groups}))
    groups_arr = np.array(groups); idx_all = np.arange(len(groups))
    pred_frames=[]; metrics=[]

    for hold in cells:
        print(f"\n[{task_name}][{model_key}] Hold-out cell: {hold}")
        te_mask = (groups_arr==hold); te_idx = idx_all[te_mask]; tr_idx = idx_all[~te_mask]
        if len(te_idx)==0 or len(tr_idx)<50:
            print(f"  Skip {hold}"); continue

        tr_groups = [groups[i] for i in tr_idx]
        # inner val split (GroupKFold) — reuse battery_common.inner_val_split via Attn_lstm, but inline here:
        from battery_common import inner_val_split
        tr2, va2 = inner_val_split(tr_groups)
        tr_final, va_final = tr_idx[tr2], tr_idx[va2]

        cell_dir = base_dir / model_key / hold; ensure_dir(cell_dir)

        # train
        model = model_ctor(d_in=d_in, nonneg_head=nonneg).to(DEVICE)
        name  = f"{task_name}_{model_key}"
        pt = train_seq(name, model,
                       [Xseq[i] for i in tr_final], [y[i] for i in tr_final], [Ls[i] for i in tr_final], [ids[i] for i in tr_final],
                       [Xseq[i] for i in va_final], [y[i] for i in va_final], [Ls[i] for i in va_final], [ids[i] for i in va_final],
                       cell_dir)

        # test
        metrics_i = infer_seq(name, model_ctor(d_in=d_in, nonneg_head=nonneg).to(DEVICE),
                              cell_dir / f"{name}_best.pt", cell_dir / f"{name}_scaler.joblib",
                              [Xseq[i] for i in te_idx], [y[i] for i in te_idx], [Ls[i] for i in te_idx], [ids[i] for i in te_idx],
                              cell_dir / f"{name}_test_{hold}.csv")

        # optional postprocess (SOH Δ→absolute)
        if post:
            df = pd.read_csv(cell_dir / f"{name}_test_{hold}.csv")
            y_true_pp, y_pred_pp = post(df, [ids[i] for i in te_idx], soh_payload(hold, te_idx) if soh_payload else None)
            metrics_i = {"MAE": mae_np(y_true_pp, y_pred_pp), "RMSE": rmse_np(y_true_pp, y_pred_pp), "N": len(y_true_pp)}
            df["y_true_pp"]=y_true_pp; df["y_pred_pp"]=y_pred_pp; df.to_csv(cell_dir / f"{name}_test_{hold}.csv", index=False)

        metrics_i["cell_id"]=hold; metrics.append(metrics_i)
        df = pd.read_csv(cell_dir / f"{name}_test_{hold}.csv"); df["model"]=model_key; pred_frames.append(df)

    mdir = base_dir / model_key; ensure_dir(mdir)
    if pred_frames: pd.concat(pred_frames, ignore_index=True).to_csv(mdir / "all_cells_predictions.csv", index=False)
    if metrics:
        met = pd.DataFrame(metrics).sort_values("cell_id")
        met.to_csv(mdir / "per_cell_metrics.csv", index=False)
        from battery_common import summarize_fleet_metrics
        summarize_fleet_metrics(mdir / "per_cell_metrics.csv", mdir / "fleet_metrics.json")

# ----------------------------- Task drivers -----------------------------

MODEL_REGISTRY = {
    "gru":    AttnGRUBase,
    "bilstm": AttnBiLSTMBase,
    "tcn":    AttnTCNBase,
    "tft":    AttnTFTLite,
}

def run_soc(models: List[str], out_root: Path):
    ph, ms, fd = load_inputs()
    X, y, Ls, groups, ids, d_in, feat_names, tgt_col = build_soc_sequences(ph, fd, max_steps=16)
    for m in models:
        loco_seq_model("soc_phase_delta", m, MODEL_REGISTRY[m], d_in, X,y,Ls,groups,ids, out_root, nonneg=False, post=None)

def run_soh(models: List[str], out_root: Path):
    ph, ms, fd = load_inputs()
    X, y, Ls, groups, ids, soh_curr, d_in, feat_names = build_soh_sequences(ms, fd, max_steps=20, horizon_k=SOH_HORIZON_K)
    def soh_post(df_preds: pd.DataFrame, ids_test, payload):
        sc = payload["soh_curr_test"]; T=len(df_preds)
        y_true_abs = np.clip(sc[:T]-df_preds["y_true"].values, 0.0, 100.0)
        y_pred_abs = np.clip(sc[:T]-df_preds["y_pred"].values, 0.0, 100.0)
        return y_true_abs, y_pred_abs
    def soh_payload(cell, te_idx): return {"soh_curr_test": soh_curr[te_idx]}
    for m in models:
        loco_seq_model("soh_next_mission", m, MODEL_REGISTRY[m], d_in, X,y,Ls,groups,ids, out_root, nonneg=True, post=soh_post, soh_payload=soh_payload)

def run_rul(models: List[str], out_root: Path, eol_list: List[float]):
    ph, ms, fd = load_inputs()
    for eol in eol_list:
        X, y, Ls, groups, ids, d_in, feat_names = build_rul_sequences(ms, fd, eol_pct=float(eol), max_steps=20)
        task = f"rul_mission_{int(eol)}"
        for m in models:
            loco_seq_model(task, m, MODEL_REGISTRY[m], d_in, X,y,Ls,groups,ids, out_root, nonneg=True, post=None)

# ----------------------------- Main -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+", default=["gru","bilstm","tcn","tft"],
                    help="Subset of baselines to run. Options: gru, bilstm, tcn, tft")
    ap.add_argument("--tasks", type=str, nargs="+", default=["soc","soh","rul"],
                    help="Subset of tasks to run. Options: soc, soh, rul")
    ap.add_argument("--sweep_rul", type=float, nargs="*", default=EOL_SWEEP,
                    help="EOL thresholds (80/85/90) if --tasks includes rul")
    ap.add_argument("--out_root", type=str, default=str(OUT_LSTM),
                    help="Root output directory (defaults to OUT_LSTM)")
    return ap.parse_args()

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(SEED)
    args = parse_args()
    out_root = Path(args.out_root); ensure_dir(out_root)

    if "soc" in args.tasks:
        run_soc(args.models, out_root)
    if "soh" in args.tasks:
        run_soh(args.models, out_root)
    if "rul" in args.tasks:
        run_rul(args.models, out_root, args.sweep_rul)

    print("Sequential baselines finished →", out_root.resolve())

if __name__ == "__main__":
    main()
