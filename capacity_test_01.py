import numpy as np
import pandas as pd

from evtol_capacity.pipeline import attach_capacity_soh_rul, audit_cycle, build_windows, resolve_schema


def config():
    return {
        "columns": {
            "cycle": ["cycle"], "time": ["time"], "step": ["step"], "phase": ["phase"],
            "current": ["current"], "voltage": ["voltage"],
            "charge_capacity": ["charge_capacity"], "discharge_capacity": ["discharge_capacity"],
        },
        "detection": {
            "charge_current_sign": 1, "current_deadband_a": 0.02,
            "full_charge_voltage_v": 4.15, "full_charge_min_fraction": 0.98,
            "full_charge_tail_points": 3, "min_charge_rows": 3, "min_discharge_rows": 3,
            "min_capacity_ah": 1.0, "max_capacity_ah": 4.0,
            "require_charge_then_discharge": True, "require_flight_label": True,
            "charge_phase_regex": "charge|cc|cv", "flight_phase_regex": "flight|takeoff|cruise|landing|discharge",
        },
        "labels": {
            "thresholds": [0.90, 0.85, 0.80], "soh_horizon_missions": 2,
            "mission_sequence_length": 3, "minimum_tests_per_cell": 2,
            "enforce_nonincreasing_soh": False,
        },
    }


def cycle(full=True):
    vtail = [4.10, 4.12, 4.15] if full else [3.90, 3.92, 3.95]
    return pd.DataFrame({
        "cycle": [1] * 6, "time": range(6), "step": [1] * 3 + [2] * 3,
        "phase": ["CC", "CV", "CV", "takeoff", "cruise", "landing"],
        "current": [1, .5, .1, -2, -1, -2], "voltage": vtail + [4.0, 3.8, 3.6],
        "charge_capacity": [1.0, 2.0, 2.5, 2.5, 2.5, 2.5],
        "discharge_capacity": [0, 0, 0, .5, 1.5, 2.4],
    })


def test_accepts_full_charge_followed_by_flight():
    df = cycle(True)
    result = audit_cycle(df, resolve_schema(df, config()), config())
    assert result["accepted"]
    assert result["available_capacity_ah"] == 2.5


def test_rejects_incomplete_charge():
    df = cycle(False)
    result = audit_cycle(df, resolve_schema(df, config()), config())
    assert not result["accepted"]
    assert "fully_charged" in result["rejection_reasons"]


def test_interpolation_thresholds_and_no_extrapolation():
    missions = pd.DataFrame({"cell": ["VAH01"] * 6, "mission": range(1, 7), "cycle": range(1, 7)})
    audit = pd.DataFrame({
        "accepted": [True, True, True], "cycle": [2, 4, 6],
        "available_capacity_ah": [2.5, 2.0, 1.75],
    })
    out = attach_capacity_soh_rul(missions, audit, config())
    assert np.isnan(out.loc[0, "soh"])
    assert out.loc[1, "soh"] == 1.0
    assert out.loc[5, "soh"] == .7
    assert out.loc[3, "crossing_mission_90"] == 3
    assert out.loc[1, "rul_90"] == 1


def test_windows_are_chronological_and_endpoint_labelled():
    labels = pd.DataFrame({
        "cell": ["VAH01"] * 5, "mission": range(1, 6), "cycle": range(11, 16),
        "soh": [1, .99, .98, .97, .96], "rul_90": [8, 7, 6, 5, 4],
    })
    windows = build_windows(labels, config())
    assert len(windows) == 5
    assert windows.iloc[0]["mission_sequence"] == "[1]"
    assert windows.iloc[2]["mission_sequence"] == "[1, 2, 3]"
    assert windows.iloc[2]["rul_90"] == 6
    assert windows.iloc[4]["mission_sequence"] == "[3, 4, 5]"
