#!/usr/bin/env python3
"""Reproduce outer-LOCO tree controls on reconstructed mission features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


MANIFESTS = {
    # Literal interpretation of the corrected manuscript's 20 mission-level
    # descriptors (current and C-rate are end-of-mission causal summaries).
    "manuscript20": [
        "chg_total_dur_s",
        "chg_energy_Wh",
        "chg_mean_I_A",
        "chg_mean_C_rate",
        "chg_cc_dur_s",
        "chg_cc_energy_Wh",
        "chg_cv_dur_s",
        "chg_cv_energy_Wh",
        "chg_cv_fraction",
        "cv_setpoint_V",
        "dis_total_dur_s",
        "dis_flight_Wh_abs",
        "dis_mean_C_rate",
        "dis_max_C_rate",
        "dis_vi_R_ohm_med",
        "temp_mean_C",
        "temp_max_C_max",
        "gap_count",
        "chg_cum_Wh",
        "dis_cum_Wh",
    ],
    # Causal subset obtained by removing the four target/context fields from
    # the repository's original 24-row mission feature dictionary.
    "dictionary20": [
        "chg_total_dur_s",
        "chg_cc_dur_s",
        "chg_cv_dur_s",
        "chg_energy_Wh",
        "chg_cv_energy_Wh",
        "chg_cv_fraction",
        "cv_setpoint_V",
        "chg_cv_mean_I_A_proxy",
        "chg_cum_Wh",
        "dis_total_dur_s",
        "dis_flight_Wh_abs",
        "dis_mean_C_rate_med",
        "dis_max_C_rate_p90",
        "dis_vi_R_ohm_med",
        "dis_vi_R_ohm_p90",
        "dis_cum_Wh",
        "temp_mean_C_med",
        "temp_max_C_max",
        "temp_th_gain_med",
        "temp_th_gain_max",
    ],
}

PUBLISHED = {
    "soh": {
        "extra_trees": {"mae": 0.610, "rmse": 0.825, "n": 18513, "cells": 19},
        "random_forest": {"mae": 0.635, "rmse": 0.887, "n": 18513, "cells": 19},
    },
    "rul90": {
        "extra_trees": {"mae": 24.934, "rmse": 35.552, "n": 5814, "cells": 19},
        "random_forest": {"mae": 24.698, "rmse": 35.980, "n": 5814, "cells": 19},
    },
    "rul85": {
        "extra_trees": {"mae": 44.149, "rmse": 56.765, "n": 9595, "cells": 16},
        "random_forest": {"mae": 47.294, "rmse": 64.394, "n": 9595, "cells": 16},
    },
    "rul80": {
        "extra_trees": {"mae": 119.172, "rmse": 142.264, "n": 8257, "cells": 8},
        "random_forest": {"mae": 118.905, "rmse": 144.550, "n": 8257, "cells": 8},
    },
}


def task_frame(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str, tuple[float, float | None]]:
    if task == "soh":
        target = "SOH_m_plus_5_pct"
        bounds = (0.0, 100.0)
    elif task in {"rul90", "rul85", "rul80"}:
        target = f"RUL{task[-2:]}_missions"
        bounds = (0.0, None)
    else:
        raise KeyError(task)
    return frame.loc[frame[target].notna()].copy(), target, bounds


def make_model(name: str, trees: int):
    common = dict(
        n_estimators=trees,
        min_samples_leaf=2,
        max_depth=None,
        random_state=1337,
        n_jobs=-1,
    )
    if name == "extra_trees":
        return ExtraTreesRegressor(**common)
    if name == "random_forest":
        return RandomForestRegressor(**common)
    raise KeyError(name)


def evaluate(
    frame: pd.DataFrame,
    features: list[str],
    task: str,
    model_name: str,
    trees: int,
) -> tuple[dict, pd.DataFrame]:
    data, target, bounds = task_frame(frame, task)
    cells = sorted(data["cell_id"].unique())
    predictions = []
    fold_times = []
    for fold, test_cell in enumerate(cells, start=1):
        train = data.loc[data["cell_id"].ne(test_cell)]
        test = data.loc[data["cell_id"].eq(test_cell)]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(train[features])
        x_test = imputer.transform(test[features])
        model = make_model(model_name, trees)
        start = time.perf_counter()
        model.fit(x_train, train[target].to_numpy(float))
        fit_seconds = time.perf_counter() - start
        pred = model.predict(x_test)
        pred = np.maximum(pred, bounds[0])
        if bounds[1] is not None:
            pred = np.minimum(pred, bounds[1])
        fold_times.append(fit_seconds)
        predictions.append(
            pd.DataFrame(
                {
                    "task": task,
                    "model": model_name,
                    "test_cell": test_cell,
                    "mission_id": test["mission_id"].to_numpy(int),
                    "y_true": test[target].to_numpy(float),
                    "y_pred": pred,
                    "error": pred - test[target].to_numpy(float),
                    "fit_seconds": fit_seconds,
                }
            )
        )
        print(
            f"{task} {model_name} {fold:02d}/{len(cells)} {test_cell}: "
            f"N={len(test)} MAE={mean_absolute_error(test[target], pred):.4f} "
            f"fit={fit_seconds:.2f}s",
            flush=True,
        )
    pred = pd.concat(predictions, ignore_index=True)
    cell_mae = pred.groupby("test_cell").apply(
        lambda g: mean_absolute_error(g["y_true"], g["y_pred"]),
        include_groups=False,
    )
    result = {
        "task": task,
        "model": model_name,
        "cells": int(pred["test_cell"].nunique()),
        "n": int(len(pred)),
        "mae": float(mean_absolute_error(pred["y_true"], pred["y_pred"])),
        "rmse": float(mean_squared_error(pred["y_true"], pred["y_pred"]) ** 0.5),
        "macro_mae": float(cell_mae.mean()),
        "p90_cell_mae": float(np.percentile(cell_mae, 90)),
        "worst_cell_mae": float(cell_mae.max()),
        "mean_fit_seconds": float(np.mean(fold_times)),
        "trees": int(trees),
    }
    published = PUBLISHED.get(task, {}).get(model_name)
    if published:
        result["published"] = published
        result["delta_mae"] = result["mae"] - published["mae"]
        result["delta_rmse"] = result["rmse"] - published["rmse"]
        result["support_match"] = result["n"] == published["n"] and result["cells"] == published["cells"]
    return result, pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", choices=sorted(MANIFESTS), default="manuscript20")
    parser.add_argument("--tasks", nargs="+", choices=sorted(PUBLISHED), default=["soh"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["extra_trees", "random_forest"],
        default=["extra_trees"],
    )
    parser.add_argument("--trees", type=int, default=600)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    features = MANIFESTS[args.manifest]
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing features: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, predictions = [], []
    for task in args.tasks:
        for model in args.models:
            result, pred = evaluate(frame, features, task, model, args.trees)
            result["manifest"] = args.manifest
            result["features"] = features
            results.append(result)
            predictions.append(pred)
            print(json.dumps(result, indent=2), flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / f"tree_predictions_{args.manifest}_{args.trees}.csv", index=False
    )
    (args.output_dir / f"tree_metrics_{args.manifest}_{args.trees}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
