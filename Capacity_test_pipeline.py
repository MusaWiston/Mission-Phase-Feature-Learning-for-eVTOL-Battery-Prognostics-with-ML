from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


class PipelineError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _first_column(df: pd.DataFrame, candidates: Any, required: bool = True) -> str | None:
    if candidates is None:
        if required:
            raise PipelineError("A required column mapping is null")
        return None
    if isinstance(candidates, str):
        candidates = [candidates]
    exact = {str(c): str(c) for c in df.columns}
    folded = {str(c).strip().lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        hit = folded.get(str(candidate).strip().lower())
        if hit:
            return hit
    if required:
        raise PipelineError(f"None of {candidates!r} found. Available columns: {list(df.columns)!r}")
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Schema:
    cycle: str
    time: str
    current: str
    voltage: str
    charge_capacity: str
    phase: str | None
    step: str | None
    discharge_capacity: str | None


def resolve_schema(df: pd.DataFrame, cfg: dict[str, Any]) -> Schema:
    c = cfg["columns"]
    return Schema(
        cycle=_first_column(df, c["cycle"]),
        time=_first_column(df, c["time"]),
        current=_first_column(df, c["current"]),
        voltage=_first_column(df, c["voltage"]),
        charge_capacity=_first_column(df, c["charge_capacity"]),
        phase=_first_column(df, c.get("phase"), required=False),
        step=_first_column(df, c.get("step"), required=False),
        discharge_capacity=_first_column(df, c.get("discharge_capacity"), required=False),
    )


def load_cell_csv(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, Schema, str]:
    df = pd.read_csv(path, low_memory=False)
    schema = resolve_schema(df, cfg)
    cell_col = _first_column(df, cfg["columns"].get("cell"), required=False)
    cell = str(df[cell_col].dropna().iloc[0]) if cell_col and df[cell_col].notna().any() else path.stem
    cell = cell.split("_")[0].upper()
    required_numeric = [schema.cycle, schema.time, schema.current, schema.voltage, schema.charge_capacity]
    for col in required_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[schema.cycle, schema.time]).copy()
    df["_source_row"] = np.arange(len(df), dtype=int)
    df = df.sort_values([schema.cycle, schema.time, "_source_row"], kind="stable")
    return df, schema, cell


def _phase_mask(series: pd.Series | None, pattern: str, fallback: pd.Series) -> pd.Series:
    if series is None:
        return fallback.fillna(False)
    text = series.fillna("").astype(str)
    labelled = text.str.contains(pattern, case=False, regex=True)
    return labelled if labelled.any() else fallback.fillna(False)


def audit_cycle(group: pd.DataFrame, schema: Schema, cfg: dict[str, Any]) -> dict[str, Any]:
    d = cfg["detection"]
    sign = int(d["charge_current_sign"])
    signed_current = sign * group[schema.current]
    charge_fallback = signed_current > float(d["current_deadband_a"])
    discharge_fallback = signed_current < -float(d["current_deadband_a"])
    phase = group[schema.phase] if schema.phase else None
    charge_mask = _phase_mask(phase, d["charge_phase_regex"], charge_fallback) & charge_fallback
    flight_mask = _phase_mask(phase, d["flight_phase_regex"], discharge_fallback) & discharge_fallback
    charge = group.loc[charge_mask]
    flight = group.loc[flight_mask]
    tail_n = max(1, int(d["full_charge_tail_points"]))
    vmax = float(charge[schema.voltage].max()) if len(charge) else np.nan
    tail_voltage = float(charge[schema.voltage].tail(tail_n).median()) if len(charge) else np.nan
    voltage_limit = float(d["full_charge_voltage_v"]) * float(d["full_charge_min_fraction"])
    capacity = float(charge[schema.charge_capacity].max()) if len(charge) else np.nan
    charge_end = float(charge[schema.time].max()) if len(charge) else np.nan
    flight_start = float(flight[schema.time].min()) if len(flight) else np.nan
    checks = {
        "enough_charge_rows": len(charge) >= int(d["min_charge_rows"]),
        "enough_discharge_rows": len(flight) >= int(d["min_discharge_rows"]),
        "fully_charged": np.isfinite(tail_voltage) and tail_voltage >= voltage_limit,
        "capacity_in_range": np.isfinite(capacity) and float(d["min_capacity_ah"]) <= capacity <= float(d["max_capacity_ah"]),
        "charge_precedes_flight": np.isfinite(charge_end) and np.isfinite(flight_start) and charge_end <= flight_start,
        "flight_label_present": len(flight) > 0,
    }
    required = ["enough_charge_rows", "fully_charged", "capacity_in_range"]
    if d.get("require_charge_then_discharge", True):
        required += ["enough_discharge_rows", "charge_precedes_flight"]
    if d.get("require_flight_label", True):
        required += ["flight_label_present"]
    accepted = all(checks[k] for k in required)
    failed = [k for k in required if not checks[k]]
    return {
        "cycle": group[schema.cycle].iloc[0],
        "accepted": accepted,
        "rejection_reasons": ";".join(failed),
        "available_capacity_ah": capacity,
        "max_charge_voltage_v": vmax,
        "tail_charge_voltage_v": tail_voltage,
        "charge_rows": int(len(charge)),
        "flight_rows": int(len(flight)),
        "charge_end_time": charge_end,
        "flight_start_time": flight_start,
        **checks,
    }


def detect_capacity_tests(df: pd.DataFrame, schema: Schema, cell: str, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [audit_cycle(group, schema, cfg) for _, group in df.groupby(schema.cycle, sort=True)]
    audit = pd.DataFrame(rows)
    audit.insert(0, "cell", cell)
    accepted = audit["accepted"].sort_values("cycle")
    if len(accepted) < int(cfg["labels"]["minimum_tests_per_cell"]):
        audit["cell_warning"] = f"fewer than {cfg['labels']['minimum_tests_per_cell']} accepted tests"
    else:
        audit["cell_warning"] = ""
    return audit


def build_mission_index(df: pd.DataFrame, schema: Schema, cell: str, cfg: dict[str, Any]) -> pd.DataFrame:
    d = cfg["detection"]
    signed_current = int(d["charge_current_sign"]) * df[schema.current]
    phase = df[schema.phase] if schema.phase else None
    flight = _phase_mask(phase, d["flight_phase_regex"], signed_current < -float(d["current_deadband_a"]))
    records: list[dict[str, Any]] = []
    mission_no = 0
    for cycle, group in df.groupby(schema.cycle, sort=True):
        mask = flight.loc[group.index]
        if not mask.any():
            continue
        mission_no += 1
        records.append({
            "cell": cell,
            "mission": mission_no,
            "cycle": cycle,
            "mission_start_time": float(group.loc[mask, schema.time].min()),
            "mission_end_time": float(group.loc[mask, schema.time].max()),
        })
    return pd.DataFrame(records)


def attach_capacity_soh_rul(missions: pd.DataFrame, audit: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if missions.empty:
        return missions
    tests = audit[audit["accepted"]].copy().sort_values("cycle")
    tests = tests.drop_duplicates("cycle", keep="last")
    x = tests["cycle"].to_numpy(float)
    y = tests["available_capacity_ah"].to_numpy(float)
    out = missions.copy()
    q = out["cycle"].to_numpy(float)
    capacity = np.full(len(out), np.nan)
    if len(x) >= 2:
        inside = (q >= x.min()) & (q <= x.max())
        capacity[inside] = np.interp(q[inside], x, y)
    elif len(x) == 1:
        capacity[q == x[0]] = y[0]
    out["capacity_ah"] = capacity
    initial = float(y[0]) if len(y) else np.nan
    out["initial_capacity_ah"] = initial
    out["soh"] = out["capacity_ah"] / initial
    if cfg["labels"].get("enforce_nonincreasing_soh", False):
        out["soh"] = out["soh"].cummin()
        out["capacity_ah"] = out["soh"] * initial
    for threshold in cfg["labels"]["thresholds"]:
        suffix = str(int(round(float(threshold) * 100)))
        crossing = out.loc[out["soh"] <= float(threshold), "mission"]
        crossing_mission = float(crossing.iloc[0]) if len(crossing) else np.nan
        out[f"crossing_mission_{suffix}"] = crossing_mission
        out[f"rul_{suffix}"] = np.maximum(crossing_mission - out["mission"], 0) if np.isfinite(crossing_mission) else np.nan
    horizon = int(cfg["labels"]["soh_horizon_missions"])
    future = out["soh"].shift(-horizon)
    out[f"soh_degradation_h{horizon}"] = (out["soh"] - future).clip(lower=0)
    out["has_interpolated_label"] = out["soh"].notna()
    return out


def build_windows(labels: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    max_length = int(cfg["labels"]["mission_sequence_length"])
    rows = []
    label_cols = [c for c in labels.columns if c == "soh" or c.startswith("rul_") or c.startswith("soh_degradation_")]
    for end in range(len(labels)):
        start = max(0, end - max_length + 1)
        window = labels.iloc[start : end + 1]
        row = {
            "cell": labels.iloc[end]["cell"],
            "window_start_mission": int(window.iloc[0]["mission"]),
            "window_end_mission": int(window.iloc[-1]["mission"]),
            "sequence_length": len(window),
            "maximum_sequence_length": max_length,
            "mission_sequence": json.dumps(window["mission"].astype(int).tolist()),
            "cycle_sequence": json.dumps(window["cycle"].tolist(), default=_jsonable),
        }
        row.update({col: labels.iloc[end][col] for col in label_cols})
        rows.append(row)
    return pd.DataFrame(rows)


def process_file(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df, schema, cell = load_cell_csv(path, cfg)
    audit = detect_capacity_tests(df, schema, cell, cfg)
    missions = build_mission_index(df, schema, cell, cfg)
    labels = attach_capacity_soh_rul(missions, audit, cfg)
    windows = build_windows(labels, cfg)
    info = {"path": str(path), "sha256": _sha256(path), "cell": cell, "rows": len(df), "schema": schema.__dict__}
    return audit, labels, windows, info


def run(input_dir: Path, output_dir: Path, config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = sorted(input_dir.glob(cfg["input_glob"]))
    paths = [p for p in paths if "impedance" not in p.name.lower()]
    if not paths:
        raise PipelineError(f"No inputs matching {cfg['input_glob']!r} in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audits, labels, windows, files, errors = [], [], [], [], []
    excluded = {x.upper() for x in cfg.get("excluded_cells", [])}
    for path in paths:
        try:
            audit, cell_labels, cell_windows, info = process_file(path, cfg)
            files.append(info)
            audits.append(audit)
            if info["cell"].upper() not in excluded:
                labels.append(cell_labels)
                windows.append(cell_windows)
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if not audits:
        raise PipelineError("No cell file was processed successfully")
    pd.concat(audits, ignore_index=True).to_csv(output_dir / "capacity_test_audit.csv", index=False)
    (pd.concat(labels, ignore_index=True) if labels else pd.DataFrame()).to_csv(
        output_dir / "mission_soh_rul_labels.csv", index=False
    )
    (pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()).to_csv(
        output_dir / "mission_windows.csv", index=False
    )
    manifest = {"config": cfg, "input_files": files, "excluded_cells": sorted(excluded), "errors": errors}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=_jsonable), encoding="utf-8")
    if errors:
        raise PipelineError(f"{len(errors)} file(s) failed; see run_manifest.json")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args(argv)
    run(args.input_dir, args.output_dir, args.config)


if __name__ == "__main__":
    main()
