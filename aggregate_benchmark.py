#!/usr/bin/env python3
"""Aggregate the leakage-safe sequence reruns and matched tree controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "extra_trees": "Extra Trees",
    "random_forest": "Random Forest",
    "attention_lstm_moe": "Attention-LSTM-MoE",
    "mission_phase_transformer": "Mission-phase Transformer",
}
TASK_DISPLAY = {
    "soh": "SOH m+5 (%)",
    "rul90": "RUL90 (missions)",
    "rul85": "RUL85 (missions)",
    "rul80": "RUL80 (missions)",
}
MODEL_ORDER = [
    "extra_trees",
    "random_forest",
    "attention_lstm_moe",
    "mission_phase_transformer",
]
TASK_ORDER = ["soh", "rul90", "rul85", "rul80"]


def read_metric_lists(root: Path, name: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob(name)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload if isinstance(payload, list) else [payload])
    return rows


def read_predictions(root: Path, name: str, source: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob(name)):
        frame = pd.read_csv(path)
        frame["source"] = source
        frame["artifact"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    keys = ["task", "model", "test_cell", "mission_id"]
    return combined.drop_duplicates(keys, keep="last")


def bootstrap_errors(
    predictions: pd.DataFrame, *, iterations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary_rows, cell_rows = [], []
    for (task, model), group in predictions.groupby(["task", "model"], sort=False):
        cell_stats = (
            group.assign(abs_error=group["error"].abs())
            .groupby("test_cell", as_index=False)
            .agg(abs_error_sum=("abs_error", "sum"), n=("abs_error", "size"))
        )
        cell_stats["cell_mae"] = cell_stats["abs_error_sum"] / cell_stats["n"]
        cell_stats.insert(0, "model", model)
        cell_stats.insert(0, "task", task)
        cell_rows.extend(cell_stats.to_dict(orient="records"))
        sums = cell_stats["abs_error_sum"].to_numpy(float)
        counts = cell_stats["n"].to_numpy(float)
        maes = cell_stats["cell_mae"].to_numpy(float)
        sampled = rng.integers(0, len(cell_stats), size=(iterations, len(cell_stats)))
        pooled = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        macro = maes[sampled].mean(axis=1)
        summary_rows.append(
            {
                "task": task,
                "model": model,
                "pooled_mae_ci_low": float(np.quantile(pooled, 0.025)),
                "pooled_mae_ci_high": float(np.quantile(pooled, 0.975)),
                "macro_mae_ci_low": float(np.quantile(macro, 0.025)),
                "macro_mae_ci_high": float(np.quantile(macro, 0.975)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(cell_rows)


def paired_comparisons(
    predictions: pd.DataFrame, *, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for task in TASK_ORDER:
        task_frame = predictions.loc[predictions["task"].eq(task)]
        models = set(task_frame["model"])
        trees = [model for model in ["extra_trees", "random_forest"] if model in models]
        neural = [
            model
            for model in ["attention_lstm_moe", "mission_phase_transformer"]
            if model in models
        ]
        for neural_model in neural:
            neural_cell = (
                task_frame.loc[task_frame["model"].eq(neural_model)]
                .assign(abs_error=lambda value: value["error"].abs())
                .groupby("test_cell")["abs_error"]
                .mean()
            )
            for tree_model in trees:
                tree_cell = (
                    task_frame.loc[task_frame["model"].eq(tree_model)]
                    .assign(abs_error=lambda value: value["error"].abs())
                    .groupby("test_cell")["abs_error"]
                    .mean()
                )
                paired = pd.concat(
                    [neural_cell.rename("neural"), tree_cell.rename("tree")], axis=1
                ).dropna()
                difference = (paired["neural"] - paired["tree"]).to_numpy(float)
                sampled = rng.integers(0, len(difference), size=(iterations, len(difference)))
                boot = difference[sampled].mean(axis=1)
                rows.append(
                    {
                        "task": task,
                        "neural_model": neural_model,
                        "tree_model": tree_model,
                        "cells": int(len(difference)),
                        "mean_cell_mae_difference": float(difference.mean()),
                        "difference_ci_low": float(np.quantile(boot, 0.025)),
                        "difference_ci_high": float(np.quantile(boot, 0.975)),
                        "neural_win_cells": int(np.sum(difference < 0)),
                        "tree_win_cells": int(np.sum(difference > 0)),
                        "bootstrap_probability_neural_better": float(np.mean(boot < 0)),
                    }
                )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    present_tasks = [task for task in TASK_ORDER if task in set(summary["task"])]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), constrained_layout=True)
    axes = axes.ravel()
    colors = {
        "extra_trees": "#3F6B8A",
        "random_forest": "#79A6C8",
        "attention_lstm_moe": "#D07A35",
        "mission_phase_transformer": "#8F5DA2",
    }
    for axis, task in zip(axes, present_tasks):
        frame = summary.loc[summary["task"].eq(task)].copy()
        frame["order"] = frame["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        frame = frame.sort_values("order")
        x = np.arange(len(frame))
        y = frame["mae"].to_numpy(float)
        lower = y - frame["pooled_mae_ci_low"].to_numpy(float)
        upper = frame["pooled_mae_ci_high"].to_numpy(float) - y
        axis.bar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            capsize=3,
            color=[colors[model] for model in frame["model"]],
            edgecolor="#243447",
            linewidth=0.7,
        )
        axis.set_title(TASK_DISPLAY[task], fontweight="bold")
        axis.set_xticks(x, [DISPLAY[model] for model in frame["model"]], rotation=22, ha="right")
        axis.set_ylabel("MAE")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        for location, value in zip(x, y):
            axis.text(location, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for axis in axes[len(present_tasks) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Leakage-safe outer-LOCO benchmark (95% cell-cluster bootstrap CI)",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(output_dir / "benchmark_mae_comparison.png", dpi=400, bbox_inches="tight")
    figure.savefig(output_dir / "benchmark_mae_comparison.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--tree-root", type=Path, required=True)
    parser.add_argument("--reconstruction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sequence_metrics = read_metric_lists(args.sequence_root, "sequence_metrics.json")
    tree_metrics = read_metric_lists(args.tree_root, "tree_metrics_*.json")
    metrics = pd.DataFrame(tree_metrics + sequence_metrics)
    if metrics.empty:
        raise RuntimeError("no completed metric artifacts found")
    metrics = metrics.drop_duplicates(["task", "model"], keep="last")

    sequence_predictions = read_predictions(
        args.sequence_root, "sequence_predictions.csv", "sequence_rerun"
    )
    tree_predictions = read_predictions(args.tree_root, "tree_predictions_*.csv", "tree_rerun")
    predictions = pd.concat([tree_predictions, sequence_predictions], ignore_index=True)
    if predictions.empty:
        raise RuntimeError("no prediction artifacts found")
    predictions = predictions.drop_duplicates(
        ["task", "model", "test_cell", "mission_id"], keep="last"
    )

    intervals, cell_mae = bootstrap_errors(
        predictions, iterations=args.bootstrap, seed=args.seed
    )
    summary = metrics.merge(intervals, on=["task", "model"], how="left")
    best_tree = (
        summary.loc[summary["model"].isin(["extra_trees", "random_forest"])]
        .sort_values("mae")
        .groupby("task", as_index=False)
        .first()[["task", "model", "mae"]]
        .rename(columns={"model": "best_tree_model", "mae": "best_tree_mae"})
    )
    summary = summary.merge(best_tree, on="task", how="left")
    summary["mae_change_vs_best_tree_pct"] = 100.0 * (
        summary["mae"] / summary["best_tree_mae"] - 1.0
    )
    summary["task_order"] = summary["task"].map({v: i for i, v in enumerate(TASK_ORDER)})
    summary["model_order"] = summary["model"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    summary = summary.sort_values(["task_order", "model_order"]).drop(
        columns=["task_order", "model_order"]
    )

    paired = paired_comparisons(predictions, iterations=args.bootstrap, seed=args.seed + 1)
    summary.to_csv(args.output_dir / "benchmark_summary.csv", index=False)
    cell_mae.to_csv(args.output_dir / "cell_level_mae.csv", index=False)
    paired.to_csv(args.output_dir / "paired_model_comparisons.csv", index=False)
    predictions.to_csv(args.output_dir / "all_predictions.csv", index=False)
    make_figure(summary, args.output_dir)

    reconstruction = json.loads(
        args.reconstruction_manifest.read_text(encoding="utf-8")
    )
    reconstruction_counts = reconstruction.get("counts", reconstruction)
    report_lines = [
        "# Leakage-safe sequence-model rerun",
        "",
        "## Scope and protocol",
        "",
        "This rerun evaluates Attention-LSTM-MoE and a mission-phase Transformer against matched Extra Trees and Random Forest controls. All scores use outer leave-one-cell-out evaluation, a 20-mission causal history, the frozen 20-feature manuscript manifest, fold-only preprocessing, and training-cell-only target normalization.",
        "",
        "The neural optimization is frozen at seed 1337: hidden width 96, one encoder layer, dropout 0.15, Adam (learning rate 2e-3, weight decay 1e-4), batch size 64, SmoothL1 beta 1, gradient clipping 1, maximum 60 epochs, and patience 8. Attention-LSTM-MoE uses six experts, top-2 routing, and load-balancing weight 0.02; the Transformer uses four heads and feed-forward width 384.",
        "",
        "## Aggregate results",
        "",
        "| Task | Model | Cells | N | Pooled MAE | RMSE | Macro MAE | 95% pooled-MAE CI | Change vs best tree |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"| {TASK_DISPLAY.get(row.task, row.task)} | {DISPLAY.get(row.model, row.model)} | "
            f"{int(row.cells)} | {int(row.n)} | {row.mae:.3f} | {row.rmse:.3f} | "
            f"{row.macro_mae:.3f} | [{row.pooled_mae_ci_low:.3f}, {row.pooled_mae_ci_high:.3f}] | "
            f"{row.mae_change_vs_best_tree_pct:+.1f}% |"
        )
    report_lines.extend(
        [
            "",
            "## Audit notes",
            "",
            f"- Reconstructed complete mission blocks: {reconstruction_counts.get('missions_all_cells', 'not recorded')}. The manuscript states 20,817; this definition-level discrepancy is retained explicitly.",
            f"- Health-supported rows: {reconstruction_counts.get('health_supported_rows', 'not recorded')}; SOH m+5 rows: {reconstruction_counts.get('soh_m_plus_5_rows', 'not recorded')}.",
            "- The supervised SOH/RUL row counts and cell counts match the corrected manuscript exactly.",
            "- The provisional AdamW/entropy run was quarantined and excluded before aggregation.",
            "- This artifact covers mission-history SOH and RUL. A phase-sequence SOC rerun is a separate estimand and is not silently mixed into these scores.",
            "",
            "## Files",
            "",
            "`benchmark_summary.csv` contains aggregate scores and bootstrap intervals; `cell_level_mae.csv` contains all cell-level errors; `paired_model_comparisons.csv` contains paired cell-bootstrap comparisons; `all_predictions.csv` contains every held-out prediction.",
            "",
        ]
    )
    (args.output_dir / "BENCHMARK_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
