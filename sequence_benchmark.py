#!/usr/bin/env python3
"""Leakage-safe outer-LOCO benchmark for the two prespecified sequence models.

The script uses the corrected 20-feature mission manifest, 20-mission causal
histories, deterministic sample-balanced inner cell folds, and training-only
imputation/scaling.  It writes every held-out prediction and fold log so that
aggregate numbers are auditable rather than manuscript-only claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from baseline_gate import MANIFESTS


SEED = 1337
MAX_STEPS = 20
BATCH_SIZE = 64
HIDDEN = 96
DROPOUT = 0.15
MAX_EPOCHS = 60
EARLY_PATIENCE = 8
LR = 2e-3
WEIGHT_DECAY = 1e-4
MOE_BALANCE_WEIGHT = 0.02


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(9, max(1, torch.get_num_threads())))


def task_target(task: str) -> tuple[str, bool, float | None]:
    if task == "soh":
        return "SOH_m_plus_5_pct", False, 100.0
    if task in {"rul90", "rul85", "rul80"}:
        return f"RUL{task[-2:]}_missions", True, None
    raise KeyError(task)


def build_sequences(
    missions: pd.DataFrame,
    features: list[str],
    task: str,
) -> dict[str, np.ndarray]:
    target, _, _ = task_target(task)
    xs, masks, ys, cells, mission_ids = [], [], [], [], []
    for cell, cell_frame in missions.groupby("cell_id", sort=True):
        cell_frame = cell_frame.sort_values("mission_id").reset_index(drop=True)
        raw_x = cell_frame[features].to_numpy(np.float32)
        for index in np.flatnonzero(cell_frame[target].notna().to_numpy()):
            start = max(0, index - MAX_STEPS + 1)
            sequence = raw_x[start : index + 1]
            length = len(sequence)
            padded = np.full((MAX_STEPS, len(features)), np.nan, dtype=np.float32)
            mask = np.zeros(MAX_STEPS, dtype=bool)
            # Right padding permits packed LSTM evaluation while preserving the
            # same observed chronological window as a left-padded artifact.
            padded[:length] = sequence
            mask[:length] = True
            xs.append(padded)
            masks.append(mask)
            ys.append(float(cell_frame.iloc[index][target]))
            cells.append(str(cell))
            mission_ids.append(int(cell_frame.iloc[index]["mission_id"]))
    return {
        "x": np.stack(xs),
        "mask": np.stack(masks),
        "y": np.asarray(ys, dtype=np.float32),
        "cell": np.asarray(cells),
        "mission_id": np.asarray(mission_ids, dtype=np.int64),
    }


def balanced_inner_folds(cells: np.ndarray, train_indices: np.ndarray) -> dict[str, int]:
    counts = pd.Series(cells[train_indices]).value_counts().to_dict()
    totals = [0, 0, 0]
    assignment: dict[str, int] = {}
    for cell, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(3), key=lambda value: (totals[value], value))
        assignment[str(cell)] = fold
        totals[fold] += int(count)
    return assignment


class FoldScaler:
    def fit(self, x: np.ndarray, mask: np.ndarray) -> "FoldScaler":
        valid = x[mask]
        self.median = np.nanmedian(valid, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(valid), valid, self.median)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std = np.where(self.std > 1e-8, self.std, 1.0)
        return self

    def transform(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(x), x, self.median)
        scaled = (filled - self.mean) / self.std
        scaled[~mask] = 0.0
        return scaled.astype(np.float32)

    def as_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class SequenceDataset(Dataset):
    def __init__(self, x, mask, y, indices):
        self.x = torch.from_numpy(x[indices])
        self.mask = torch.from_numpy(mask[indices])
        self.y = torch.from_numpy(y[indices])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.mask[index], self.y[index]


class TimeAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)

    def forward(self, hidden, query, mask):
        q = self.q(query).unsqueeze(1)
        k = self.k(hidden)
        v = self.v(hidden)
        logits = (q * k).sum(dim=-1) / math.sqrt(k.shape[-1])
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        return context, weights


class AttentionLSTMMoE(nn.Module):
    def __init__(self, d_in: int, nonnegative: bool):
        super().__init__()
        self.lstm = nn.LSTM(d_in, HIDDEN, num_layers=1, batch_first=True)
        self.attention = TimeAttention(HIDDEN)
        self.gate = nn.Linear(2 * HIDDEN, 6)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * HIDDEN, HIDDEN),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(HIDDEN, 1),
                )
                for _ in range(6)
            ]
        )
        self.nonnegative = nonnegative

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).to(torch.int64)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden_packed, (last, _) = self.lstm(packed)
        hidden, _ = pad_packed_sequence(
            hidden_packed, batch_first=True, total_length=x.shape[1]
        )
        last = last[-1]
        context, attention = self.attention(hidden, last, mask)
        fused = torch.cat([context, last], dim=-1)
        gate_logits = self.gate(fused)
        full_prob = torch.softmax(gate_logits, dim=-1)
        top_value, top_index = torch.topk(full_prob, k=2, dim=-1)
        sparse = torch.zeros_like(full_prob).scatter(1, top_index, top_value)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expert_output = torch.cat([expert(fused) for expert in self.experts], dim=1)
        output = (sparse * expert_output).sum(dim=1)
        if self.nonnegative:
            output = F.softplus(output)
        return output, attention, gate_logits


class PositionalEncoding(nn.Module):
    def __init__(self, width: int, max_steps: int = MAX_STEPS):
        super().__init__()
        position = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
        )
        values = torch.zeros(max_steps, width)
        values[:, 0::2] = torch.sin(position * divisor)
        values[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("values", values.unsqueeze(0))

    def forward(self, x):
        return x + self.values[:, : x.shape[1]]


class MissionPhaseTransformer(nn.Module):
    def __init__(self, d_in: int, nonnegative: bool):
        super().__init__()
        self.projection = nn.Linear(d_in, HIDDEN)
        self.position = PositionalEncoding(HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=4,
            dim_feedforward=384,
            dropout=DROPOUT,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.attention = TimeAttention(HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )
        self.nonnegative = nonnegative

    def forward(self, x, mask):
        hidden = self.position(self.projection(x))
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        lengths = mask.sum(dim=1).to(torch.int64)
        last = hidden[torch.arange(len(hidden), device=hidden.device), lengths - 1]
        context, attention = self.attention(hidden, last, mask)
        output = self.head(torch.cat([context, last], dim=-1)).squeeze(-1)
        if self.nonnegative:
            output = F.softplus(output)
        return output, attention, None


def make_model(name: str, d_in: int, nonnegative: bool) -> nn.Module:
    if name == "attention_lstm_moe":
        return AttentionLSTMMoE(d_in, nonnegative)
    if name == "mission_phase_transformer":
        return MissionPhaseTransformer(d_in, nonnegative)
    raise KeyError(name)


def predict(model, loader, upper_bound, target_mean: float, target_std: float):
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            output, _, _ = model(x, mask)
            output = output * target_std + target_mean
            y = y * target_std + target_mean
            output = torch.clamp(output, min=0.0, max=upper_bound)
            truth.append(y.numpy())
            pred.append(output.cpu().numpy())
    return np.concatenate(truth), np.concatenate(pred)


def train_fold(
    model_name: str,
    task: str,
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: Path,
) -> tuple[nn.Module, list[dict], dict]:
    set_seed()
    _, _, upper_bound = task_target(task)
    # Absolute SOH and RUL occupy very different numeric ranges.  A target
    # transform fitted only on the inner training cells prevents Huber's
    # saturated gradient from turning scale into an architecture confound.
    target_mean = float(y[train_indices].mean())
    target_std = float(y[train_indices].std())
    if target_std < 1e-8:
        target_std = 1.0
    y_scaled = ((y - target_mean) / target_std).astype(np.float32)
    model = make_model(model_name, x.shape[-1], False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_function = nn.SmoothL1Loss(beta=1.0)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        SequenceDataset(x, mask, y_scaled, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        SequenceDataset(x, mask, y_scaled, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    best_mae = math.inf
    best_epoch = -1
    waiting = 0
    log = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    start_training = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        count = 0
        for batch_x, batch_mask, batch_y in train_loader:
            output, _, gate_logits = model(batch_x, batch_mask)
            task_loss = loss_function(output, batch_y)
            loss = task_loss
            if gate_logits is not None:
                probability = torch.softmax(gate_logits, dim=-1)
                mean_assignment = probability.mean(dim=0)
                uniform_assignment = torch.full_like(
                    mean_assignment, 1.0 / mean_assignment.numel()
                )
                balance_loss = torch.sum((mean_assignment - uniform_assignment) ** 2)
                loss = loss + MOE_BALANCE_WEIGHT * balance_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(task_loss.detach()) * len(batch_y)
            count += len(batch_y)
        val_true, val_pred = predict(
            model, val_loader, upper_bound, target_mean, target_std
        )
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        val_rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
        scheduler.step(val_mae)
        row = {
            "epoch": epoch,
            "train_huber": running / max(1, count),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        log.append(row)
        if val_mae + 1e-6 < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            waiting = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
            if waiting >= EARLY_PATIENCE:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    info = {
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_mae),
        "epochs_run": int(len(log)),
        "training_seconds": float(time.perf_counter() - start_training),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    return model, log, info


def evaluate_model_task(
    samples: dict[str, np.ndarray],
    features: list[str],
    task: str,
    model_name: str,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_raw, mask, y = samples["x"], samples["mask"], samples["y"]
    cells, mission_ids = samples["cell"], samples["mission_id"]
    _, _, upper_bound = task_target(task)
    all_index = np.arange(len(y))
    prediction_rows, fold_rows, log_rows = [], [], []
    for fold_number, test_cell in enumerate(sorted(np.unique(cells)), start=1):
        fold_dir = output_dir / task / model_name / test_cell
        fold_predictions_path = fold_dir / "predictions.csv"
        fold_record_path = fold_dir / "fold_record.json"
        fold_log_path = fold_dir / "training_log.csv"
        if (
            fold_predictions_path.exists()
            and fold_record_path.exists()
            and fold_log_path.exists()
        ):
            prediction_rows.append(pd.read_csv(fold_predictions_path))
            fold_rows.append(json.loads(fold_record_path.read_text(encoding="utf-8")))
            cached_log = pd.read_csv(fold_log_path)
            log_rows.extend(cached_log.to_dict(orient="records"))
            print(
                f"{task} {model_name} {fold_number:02d}/{len(np.unique(cells))} "
                f"{test_cell}: resumed cached fold",
                flush=True,
            )
            continue
        test_indices = all_index[cells == test_cell]
        outer_train = all_index[cells != test_cell]
        assignment = balanced_inner_folds(cells, outer_train)
        val_cells = sorted(cell for cell, fold in assignment.items() if fold == 0)
        train_cells = sorted(cell for cell, fold in assignment.items() if fold != 0)
        train_indices = outer_train[np.isin(cells[outer_train], train_cells)]
        val_indices = outer_train[np.isin(cells[outer_train], val_cells)]

        scaler = FoldScaler().fit(x_raw[train_indices], mask[train_indices])
        x = scaler.transform(x_raw, mask)
        model, log, info = train_fold(
            model_name,
            task,
            x,
            mask,
            y,
            train_indices,
            val_indices,
            fold_dir,
        )
        test_loader = DataLoader(
            SequenceDataset(
                x,
                mask,
                ((y - info["target_mean"]) / info["target_std"]).astype(np.float32),
                test_indices,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        start_inference = time.perf_counter()
        true, pred = predict(
            model,
            test_loader,
            upper_bound,
            info["target_mean"],
            info["target_std"],
        )
        inference_seconds = time.perf_counter() - start_inference
        error = pred - true
        fold_mae = float(np.mean(np.abs(error)))
        fold_rmse = float(np.sqrt(np.mean(error**2)))
        fold_predictions = pd.DataFrame(
            {
                "task": task,
                "model": model_name,
                "test_cell": test_cell,
                "mission_id": mission_ids[test_indices],
                "y_true": true,
                "y_pred": pred,
                "error": error,
            }
        )
        prediction_rows.append(fold_predictions)
        fold_row = {
            "task": task,
            "model": model_name,
            "fold": fold_number,
            "test_cell": test_cell,
            "train_cells": ";".join(train_cells),
            "validation_cells": ";".join(val_cells),
            "train_n": int(len(train_indices)),
            "validation_n": int(len(val_indices)),
            "test_n": int(len(test_indices)),
            "test_mae": fold_mae,
            "test_rmse": fold_rmse,
            "inference_ms_per_sample": float(1000.0 * inference_seconds / len(test_indices)),
            **info,
        }
        fold_rows.append(fold_row)
        fold_log_rows = [
            {"task": task, "model": model_name, "test_cell": test_cell, **row}
            for row in log
        ]
        log_rows.extend(fold_log_rows)
        fold_predictions.to_csv(fold_predictions_path, index=False)
        fold_record_path.write_text(json.dumps(fold_row, indent=2), encoding="utf-8")
        pd.DataFrame(fold_log_rows).to_csv(fold_log_path, index=False)
        (fold_dir / "scaler.json").write_text(json.dumps(scaler.as_dict()), encoding="utf-8")
        print(
            f"{task} {model_name} {fold_number:02d}/{len(np.unique(cells))} {test_cell}: "
            f"MAE={fold_mae:.4f} RMSE={fold_rmse:.4f} "
            f"epoch={info['best_epoch']} time={info['training_seconds']:.1f}s",
            flush=True,
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    logs = pd.DataFrame(log_rows)
    cell_mae = predictions.groupby("test_cell").apply(
        lambda group: float(np.mean(np.abs(group["error"]))), include_groups=False
    )
    metrics = {
        "task": task,
        "model": model_name,
        "cells": int(predictions["test_cell"].nunique()),
        "n": int(len(predictions)),
        "mae": float(np.mean(np.abs(predictions["error"]))),
        "rmse": float(np.sqrt(np.mean(predictions["error"] ** 2))),
        "macro_mae": float(cell_mae.mean()),
        "p90_cell_mae": float(np.percentile(cell_mae, 90)),
        "worst_cell_mae": float(cell_mae.max()),
        "features": features,
        "history": MAX_STEPS,
        "seed": SEED,
        "inner_validation": "three sample-balanced cell folds; fold 0 validation",
        "training": {
            "optimizer": "Adam",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_PATIENCE,
            "loss": "SmoothL1 beta=1",
            "moe_load_balance_weight": MOE_BALANCE_WEIGHT,
            "gradient_clip": 1.0,
            "target_transform": "training-cell mean/std; inverted before scoring",
        },
    }
    return metrics, predictions, folds, logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", choices=sorted(MANIFESTS), default="manuscript20")
    parser.add_argument(
        "--tasks", nargs="+", choices=["soh", "rul90", "rul85", "rul80"], default=["soh"]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["attention_lstm_moe", "mission_phase_transformer"],
        default=["attention_lstm_moe", "mission_phase_transformer"],
    )
    args = parser.parse_args()
    set_seed()
    missions = pd.read_csv(args.input)
    features = MANIFESTS[args.manifest]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, all_predictions, all_folds, all_logs = [], [], [], []
    for task in args.tasks:
        samples = build_sequences(missions, features, task)
        print(
            f"Prepared {task}: N={len(samples['y'])} cells={len(np.unique(samples['cell']))}",
            flush=True,
        )
        for model in args.models:
            metrics, predictions, folds, logs = evaluate_model_task(
                samples, features, task, model, args.output_dir
            )
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            all_folds.append(folds)
            all_logs.append(logs)
            print(json.dumps(metrics, indent=2), flush=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output_dir / "sequence_predictions.csv", index=False
    )
    pd.concat(all_logs, ignore_index=True, sort=False).to_csv(
        args.output_dir / "sequence_training_logs.csv", index=False
    )
    pd.concat(all_folds, ignore_index=True, sort=False).to_csv(
        args.output_dir / "sequence_fold_metrics.csv", index=False
    )
    (args.output_dir / "sequence_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
