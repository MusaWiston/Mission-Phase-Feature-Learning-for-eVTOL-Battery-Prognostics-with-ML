#!/usr/bin/env python3
"""Reconstruct leakage-safe mission features and health targets from CMU telemetry.

This implementation follows the checked manuscript protocol rather than the
legacy feature CSVs in the project repository.  It deliberately keeps a
superset of causal mission descriptors so that the exact 20-column manifest
can be audited against the published tree-control metrics before neural-model
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED = [
    "time_s",
    "Ecell_V",
    "I_mA",
    "QCharge_mA_h",
    "QDischarge_mA_h",
    "Temperature__C",
    "cycleNumber",
    "Ns",
]

EXCLUDED_HEALTH_CELLS = {"VAH06", "VAH07", "VAH09"}
CV_SETPOINT = {"VAH07": 4.0, "VAH23": 4.1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_mission(ns_values: Iterable[int]) -> bool:
    """Return True only when 4, 5, and 6 occur in chronological order."""

    expected = iter((4, 5, 6))
    wanted = next(expected, None)
    for value in ns_values:
        if value == wanted:
            wanted = next(expected, None)
            if wanted is None:
                return True
    return False


def _safe_percentile(values: pd.Series, q: float) -> float:
    values = values.dropna().to_numpy(float)
    return float(np.percentile(values, q)) if len(values) else np.nan


def _phase_table(frame: pd.DataFrame, q0_ah: float) -> pd.DataFrame:
    """Aggregate contiguous phase runs with causal, end-of-phase descriptors."""

    group = frame.groupby("phase_run_id", sort=False, observed=True)
    phase = group.agg(
        cycle_instance_id=("cycle_instance_id", "first"),
        raw_cycle_number=("cycleNumber", "first"),
        source_start=("source_row", "first"),
        source_end=("source_row", "last"),
        ns=("Ns", "first"),
        t_start_s=("time_s", "first"),
        t_end_s=("time_s", "last"),
        duration_s=("dt_s", "sum"),
        energy_Wh=("energy_increment_Wh", "sum"),
        n_rows=("time_s", "size"),
        mean_I_A=("I_A", "mean"),
        mean_abs_I_A=("abs_I_A", "mean"),
        max_abs_I_A=("abs_I_A", "max"),
        mean_abs_power_W=("abs_power_W", "mean"),
        start_voltage_V=("Ecell_V", "first"),
        end_voltage_V=("Ecell_V", "last"),
        min_voltage_V=("Ecell_V", "min"),
        mean_temp_C=("Temperature__C", "mean"),
        max_temp_C=("Temperature__C", "max"),
        gap_count=("large_gap", "sum"),
        sum_I=("I_A", "sum"),
        sum_V=("Ecell_V", "sum"),
        sum_IV=("I_times_V", "sum"),
        sum_I2=("I_squared", "sum"),
    ).reset_index()

    n = phase["n_rows"].to_numpy(float)
    sum_i = phase["sum_I"].to_numpy(float)
    sum_v = phase["sum_V"].to_numpy(float)
    numerator = phase["sum_IV"].to_numpy(float) - sum_i * sum_v / n
    denominator = phase["sum_I2"].to_numpy(float) - sum_i * sum_i / n
    variance = np.maximum(denominator / n, 0.0)
    valid = (n >= 12) & (np.sqrt(variance) >= 0.05) & (denominator > 1e-12)
    phase["vi_R_ohm"] = np.where(valid, -(numerator / denominator), np.nan)

    phase["mean_C_rate"] = phase["mean_abs_I_A"] / q0_ah
    phase["max_C_rate"] = phase["max_abs_I_A"] / q0_ah
    phase["voltage_drop_V"] = phase["start_voltage_V"] - phase["end_voltage_V"]
    phase["temp_rise_C"] = phase["max_temp_C"] - phase["mean_temp_C"]
    phase["temp_slope_C_per_s"] = phase["temp_rise_C"] / np.maximum(
        phase["duration_s"], 1.0
    )
    phase["thermal_gain_C_per_W_s"] = np.where(
        phase["mean_abs_power_W"] >= 1.0,
        phase["temp_slope_C_per_s"] / phase["mean_abs_power_W"],
        np.nan,
    )
    return phase


def _mission_row(cell_id: str, mission_id: int, phases: pd.DataFrame, q0_ah: float) -> dict:
    charge = phases.loc[phases["ns"].isin([0, 1])]
    cc = phases.loc[phases["ns"].eq(0)]
    cv = phases.loc[phases["ns"].eq(1)]
    flight = phases.loc[phases["ns"].isin([4, 5, 6])]

    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        good = values.notna() & weights.notna() & weights.gt(0)
        if not good.any():
            return np.nan
        return float(np.average(values.loc[good].to_numpy(float), weights=weights.loc[good]))

    chg_duration = float(charge["duration_s"].sum())
    cv_duration = float(cv["duration_s"].sum())
    cc_duration = float(cc["duration_s"].sum())
    chg_energy = float(charge["energy_Wh"].sum())
    cv_energy = float(cv["energy_Wh"].sum())
    cc_energy = float(cc["energy_Wh"].sum())
    dis_energy = float(abs(flight["energy_Wh"].sum()))

    return {
        "cell_id": cell_id,
        "mission_id": int(mission_id),
        "cycle_instance_id": int(phases["cycle_instance_id"].iloc[0]),
        "raw_cycle_number": int(phases["raw_cycle_number"].iloc[0]),
        "chg_total_dur_s": chg_duration,
        "chg_cc_dur_s": cc_duration,
        "chg_cv_dur_s": cv_duration,
        "chg_energy_Wh": chg_energy,
        "chg_cc_energy_Wh": cc_energy,
        "chg_cv_energy_Wh": cv_energy,
        "chg_cv_fraction": cv_duration / (cc_duration + cv_duration)
        if (cc_duration + cv_duration) > 0
        else np.nan,
        "cv_setpoint_V": CV_SETPOINT.get(cell_id, 4.2),
        "chg_mean_I_A": weighted_mean(charge["mean_abs_I_A"], charge["n_rows"]),
        "chg_mean_C_rate": weighted_mean(charge["mean_C_rate"], charge["n_rows"]),
        "chg_max_C_rate": float(charge["max_C_rate"].max()) if len(charge) else np.nan,
        "chg_cv_mean_I_A_proxy": (
            (cv_energy / CV_SETPOINT.get(cell_id, 4.2)) / (cv_duration / 3600.0)
            if cv_duration > 0
            else np.nan
        ),
        "dis_total_dur_s": float(flight["duration_s"].sum()),
        "dis_flight_Wh_abs": dis_energy,
        "dis_mean_C_rate": weighted_mean(flight["mean_C_rate"], flight["n_rows"]),
        "dis_mean_C_rate_med": float(flight["mean_C_rate"].median()),
        "dis_max_C_rate": float(flight["max_C_rate"].max()) if len(flight) else np.nan,
        "dis_max_C_rate_p90": _safe_percentile(flight["max_C_rate"], 90),
        "dis_vi_R_ohm_med": float(flight["vi_R_ohm"].median()),
        "dis_vi_R_ohm_p90": _safe_percentile(flight["vi_R_ohm"], 90),
        "temp_mean_C": weighted_mean(phases["mean_temp_C"], phases["n_rows"]),
        "temp_mean_C_med": float(phases["mean_temp_C"].median()),
        "temp_max_C_max": float(phases["max_temp_C"].max()),
        "temp_th_gain_med": float(phases["thermal_gain_C_per_W_s"].median()),
        "temp_th_gain_max": float(phases["thermal_gain_C_per_W_s"].max()),
        "temp_slope_med": float(phases["temp_slope_C_per_s"].median()),
        "gap_count": int(phases["gap_count"].sum()),
        "q0_ah_audit": float(q0_ah),
    }


def process_cell(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cell_id = path.stem.upper()
    raw = pd.read_csv(path, usecols=REQUIRED, low_memory=False)
    for column in REQUIRED:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw["source_row"] = np.arange(len(raw), dtype=np.int64)

    new_cycle = (
        raw["cycleNumber"].ne(raw["cycleNumber"].shift())
        | raw["time_s"].lt(raw["time_s"].shift())
    )
    raw["cycle_instance_id"] = new_cycle.cumsum().astype(np.int64)
    new_phase = new_cycle | raw["Ns"].ne(raw["Ns"].shift())
    raw["phase_run_id"] = new_phase.cumsum().astype(np.int64)

    dt_raw = raw["time_s"].diff()
    first_phase_row = raw["phase_run_id"].ne(raw["phase_run_id"].shift())
    raw["large_gap"] = (~first_phase_row) & dt_raw.gt(60.0)
    raw["dt_s"] = dt_raw.mask(first_phase_row | dt_raw.gt(60.0), 0.0)
    if raw["dt_s"].lt(0).any():
        raise ValueError(f"{cell_id}: negative within-phase time difference")
    raw["I_A"] = raw["I_mA"] / 1000.0
    raw["abs_I_A"] = raw["I_A"].abs()
    raw["power_W"] = raw["Ecell_V"] * raw["I_A"]
    raw["abs_power_W"] = raw["power_W"].abs()
    raw["energy_increment_Wh"] = raw["power_W"] * raw["dt_s"] / 3600.0
    raw["I_times_V"] = raw["I_A"] * raw["Ecell_V"]
    raw["I_squared"] = raw["I_A"] ** 2

    discharge = raw.loc[raw["I_A"].lt(-0.05)].copy()
    discharge["c5_dt_s"] = np.where(
        discharge["abs_I_A"].between(0.39, 0.81, inclusive="both"),
        discharge["dt_s"],
        0.0,
    )
    dg = discharge.groupby("cycle_instance_id", sort=False, observed=True)
    rpt = dg.agg(
        raw_cycle_number=("cycleNumber", "first"),
        source_start=("source_row", "min"),
        discharge_duration_s=("dt_s", "sum"),
        c5_duration_s=("c5_dt_s", "sum"),
        start_discharge_voltage_V=("Ecell_V", "first"),
        final_discharge_voltage_V=("Ecell_V", "last"),
        min_discharge_voltage_V=("Ecell_V", "min"),
        max_discharge_voltage_V=("Ecell_V", "max"),
        qdis_min_mAh=("QDischarge_mA_h", "min"),
        qdis_max_mAh=("QDischarge_mA_h", "max"),
    ).reset_index()
    rpt["c5_fraction"] = rpt["c5_duration_s"] / rpt["discharge_duration_s"].replace(0, np.nan)
    rpt["capacity_ah"] = (rpt["qdis_max_mAh"] - rpt["qdis_min_mAh"]) / 1000.0
    rpt["voltage_span_V"] = rpt["max_discharge_voltage_V"] - rpt["min_discharge_voltage_V"]
    rpt["current_ok"] = rpt["c5_fraction"].ge(0.70)
    rpt["voltage_ok"] = (
        rpt["start_discharge_voltage_V"].ge(4.10)
        & rpt["min_discharge_voltage_V"].le(2.55)
        & rpt["voltage_span_V"].ge(1.40)
    )
    rpt["capacity_ok"] = rpt["capacity_ah"].between(1.50, 3.75, inclusive="both")
    rpt["duration_ok"] = rpt["discharge_duration_s"].ge(7200.0)
    rpt["is_capacity_test"] = rpt[["current_ok", "voltage_ok", "capacity_ok", "duration_ok"]].all(axis=1)

    cycle_phase = (
        raw.loc[first_phase_row, ["cycle_instance_id", "phase_run_id", "Ns", "source_row"]]
        .sort_values("source_row")
        .groupby("cycle_instance_id", sort=False)["Ns"]
        .agg(lambda s: [int(v) for v in s])
    )
    valid_rpt = rpt.loc[rpt["is_capacity_test"]].copy().sort_values("source_start")
    if valid_rpt.empty:
        raise ValueError(f"{cell_id}: no valid capacity test")
    detected_capacity_ids = set(valid_rpt["cycle_instance_id"].astype(int))
    # The checked cell audit for VAH25 terminates the supported health trace at
    # the documented consecutive-RPT incident.  The two later candidates are
    # outside that audited trace and are not accepted as interpolation anchors.
    if cell_id == "VAH25":
        valid_rpt = valid_rpt.iloc[:10].copy()
    q0_ah = float(valid_rpt.iloc[0]["capacity_ah"])

    phases = _phase_table(raw, q0_ah)
    capacity_ids = detected_capacity_ids

    # A reconstructed cycle can contain an RPT followed by an ordinary
    # mission without a raw counter change.  Enumerate every ordered 4/5/6
    # block, classify the first block of a detected RPT cycle as diagnostic,
    # and retain later high-current blocks as missions.  This is the detail
    # that prevents complete missions appended to an RPT from being dropped.
    mission_blocks: list[tuple[int, pd.DataFrame]] = []
    for cycle_id, cycle_phases in phases.groupby("cycle_instance_id", sort=False):
        cycle_phases = cycle_phases.sort_values("source_start").reset_index(drop=True)
        ns = cycle_phases["ns"].astype(int).tolist()
        triplets: list[tuple[int, int, int]] = []
        state = 0
        picked: list[int] = []
        for index, value in enumerate(ns):
            wanted = (4, 5, 6)[state]
            if value == wanted:
                picked.append(index)
                state += 1
                if state == 3:
                    triplets.append(tuple(picked))
                    state = 0
                    picked = []
        if not triplets:
            continue

        previous_end = -1
        for triplet_number, (i4, i5, i6) in enumerate(triplets):
            flight = cycle_phases.iloc[[i4, i5, i6]]
            flight_mean_current = np.average(
                flight["mean_abs_I_A"].to_numpy(float),
                weights=np.maximum(flight["n_rows"].to_numpy(float), 1.0),
            )
            diagnostic = False
            if int(cycle_id) in capacity_ids:
                diagnostic = triplet_number == 0 or (
                    float(flight["duration_s"].sum()) >= 7200.0
                    or float(flight_mean_current) <= 1.0
                )

            start_candidates = [
                idx
                for idx in range(previous_end + 1, i4 + 1)
                if int(cycle_phases.iloc[idx]["ns"]) == 0
            ]
            block_start = start_candidates[0] if start_candidates else previous_end + 1
            next_zero = next(
                (
                    idx
                    for idx in range(i6 + 1, len(cycle_phases))
                    if int(cycle_phases.iloc[idx]["ns"]) == 0
                ),
                len(cycle_phases),
            )
            block_end = next_zero - 1
            if not diagnostic:
                block = cycle_phases.iloc[block_start : block_end + 1].copy()
                mission_blocks.append((int(block["source_start"].min()), block))
            previous_end = i6

    mission_blocks.sort(key=lambda item: item[0])
    mission_rows = [
        _mission_row(cell_id, mission_id, block, q0_ah)
        for mission_id, (_, block) in enumerate(mission_blocks, start=1)
    ]
    missions = pd.DataFrame(mission_rows)
    if missions.empty:
        raise ValueError(f"{cell_id}: no complete missions")
    mission_starts = np.array([source for source, _ in mission_blocks], dtype=np.int64)

    valid_rpt["cell_id"] = cell_id
    valid_rpt["mission_position"] = valid_rpt["source_start"].map(
        lambda source: int(np.sum(mission_starts < int(source)))
    )
    valid_rpt["SOH_pct"] = 100.0 * valid_rpt["capacity_ah"] / q0_ah
    valid_rpt["rpt_order"] = np.arange(1, len(valid_rpt) + 1)

    missions["chg_cum_Wh"] = missions["chg_energy_Wh"].cumsum()
    missions["dis_cum_Wh"] = missions["dis_flight_Wh_abs"].cumsum()

    # Duplicate RPTs at one mission position are combined by their median,
    # except that the first anchor retains the first chronological RPT so BOL
    # is exactly 100%.
    anchors = []
    for position, group in valid_rpt.groupby("mission_position", sort=True):
        if int(position) == int(valid_rpt.iloc[0]["mission_position"]):
            soh = float(group.sort_values("rpt_order").iloc[0]["SOH_pct"])
        else:
            soh = float(group["SOH_pct"].median())
        anchors.append((int(position), soh))
    anchor_x = np.array([x for x, _ in anchors], dtype=float)
    anchor_y = np.array([y for _, y in anchors], dtype=float)
    mission_x = missions["mission_id"].to_numpy(float)
    supported = (mission_x >= anchor_x.min()) & (mission_x <= anchor_x.max())
    missions["SOH_end_pct"] = np.nan
    missions.loc[supported, "SOH_end_pct"] = np.interp(mission_x[supported], anchor_x, anchor_y)
    # If a complete mission is appended inside the same reconstructed cycle as
    # the final RPT (the VAH22 terminal-record anomaly), it shares that final
    # cycle anchor rather than being treated as extrapolation beyond the test.
    final_rpt_cycle = int(valid_rpt.sort_values("source_start").iloc[-1]["cycle_instance_id"])
    same_final_cycle = missions["cycle_instance_id"].eq(final_rpt_cycle)
    if cell_id != "VAH25":
        missions.loc[same_final_cycle, "SOH_end_pct"] = float(anchor_y[-1])

    for threshold in (90, 85, 80):
        observed = missions.loc[missions["SOH_end_pct"].le(float(threshold)), "mission_id"]
        event = int(observed.iloc[0]) if len(observed) else None
        missions[f"RUL{threshold}_event_mission"] = event
        missions[f"RUL{threshold}_observed"] = event is not None
        missions[f"RUL{threshold}_missions"] = np.where(
            (event is not None) & missions["mission_id"].le(event if event is not None else -1),
            (event - missions["mission_id"]) if event is not None else np.nan,
            np.nan,
        )

    missions["SOH_m_plus_5_pct"] = missions["SOH_end_pct"].shift(-5)
    missions["SOH_drop_5_pct"] = missions["SOH_end_pct"] - missions["SOH_m_plus_5_pct"]
    info = {
        "cell_id": cell_id,
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(raw)),
        "cycle_instances": int(raw["cycle_instance_id"].nunique()),
        "missions": int(len(missions)),
        "valid_rpts": int(len(valid_rpt)),
        "q0_ah": q0_ah,
        "health_excluded": cell_id in EXCLUDED_HEALTH_CELLS,
    }
    return valid_rpt, missions, info


def run(raw_dir: Path, output_dir: Path) -> None:
    paths = sorted(raw_dir.glob("VAH*.csv"))
    if not paths:
        raise FileNotFoundError(f"No VAH*.csv files in {raw_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rpts, all_missions, files = [], [], []
    for path in paths:
        rpts, missions, info = process_cell(path)
        all_rpts.append(rpts)
        all_missions.append(missions)
        files.append(info)
        print(
            f"{info['cell_id']}: rows={info['rows']:,} missions={info['missions']:,} "
            f"RPTs={info['valid_rpts']}",
            flush=True,
        )

    capacity = pd.concat(all_rpts, ignore_index=True)
    missions = pd.concat(all_missions, ignore_index=True)
    health = missions.loc[~missions["cell_id"].isin(EXCLUDED_HEALTH_CELLS)].copy()
    capacity.to_csv(output_dir / "capacity_tests.csv", index=False)
    missions.to_csv(output_dir / "mission_features_targets_superset.csv", index=False)
    health.to_csv(output_dir / "health_mission_features_targets.csv", index=False)

    counts = {
        "raw_cells": int(len(files)),
        "missions_all_cells": int(len(missions)),
        "valid_rpts_all_cells": int(len(capacity)),
        "health_cells": int(health["cell_id"].nunique()),
        "health_supported_rows": int(health["SOH_end_pct"].notna().sum()),
        "soh_m_plus_5_rows": int(health["SOH_m_plus_5_pct"].notna().sum()),
        "rul90_rows": int(health["RUL90_missions"].notna().sum()),
        "rul85_rows": int(health["RUL85_missions"].notna().sum()),
        "rul80_rows": int(health["RUL80_missions"].notna().sum()),
        "rul90_cells": int(health.loc[health["RUL90_missions"].notna(), "cell_id"].nunique()),
        "rul85_cells": int(health.loc[health["RUL85_missions"].notna(), "cell_id"].nunique()),
        "rul80_cells": int(health.loc[health["RUL80_missions"].notna(), "cell_id"].nunique()),
    }
    manifest = {
        "protocol": {
            "cycle_instance": "raw cycle change or backward time",
            "phase_runs": "contiguous Ns runs",
            "max_valid_dt_s": 60,
            "rpt": {
                "c5_abs_current_A": [0.39, 0.81],
                "minimum_fraction": 0.70,
                "start_voltage_min_V": 4.10,
                "end_or_min_voltage_max_V": 2.55,
                "minimum_span_V": 1.40,
                "capacity_Ah": [1.50, 3.75],
                "minimum_duration_s": 7200,
            },
            "excluded_health_cells": sorted(EXCLUDED_HEALTH_CELLS),
            "SOH_target": "absolute unsmoothed SOH at mission m+5",
            "RUL": "observed crossings only; event row retained; post-event rows excluded",
        },
        "counts": counts,
        "files": files,
    }
    (output_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
