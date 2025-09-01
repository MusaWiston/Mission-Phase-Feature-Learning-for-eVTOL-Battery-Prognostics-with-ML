#phase labeling Stage

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------- CONFIG ---------------------------
EOL_THRESHOLD_PCT = 85.0  # SOH% for EOL
REST_CURRENT_THRESH = 50.0  # mA, used for simple flags
REQUIRED_TS_COLS = {'time_s','Ecell_V','I_mA','Temperature__C','cycleNumber','Ns','SOC','power_W','cell_id'}
REQUIRED_SUM_COLS = {'cell_id','cycleNumber','SOH_pct','SOH_pct_smoothed','initial_capacity_mAh','is_capacity_test','RUL_cycles','RUL_censored'}

NS_TO_PHASE = {
    0: 'CC',
    1: 'CV',
    2: 'CV→Rest',
    3: 'Rest→TO',
    4: 'Take-off',
    5: 'Cruise',
    6: 'Landing',
    7: 'Landing→Rest',
    8: 'Rest1 (post-landing)',
    9: 'Rest2 (pre-charge)'
}
NS_FAMILY = {
    'CC':'charge', 'CV':'charge', 'CV→Rest':'transition', 'Rest→TO':'transition',
    'Take-off':'flight', 'Cruise':'flight', 'Landing':'flight',
    'Landing→Rest':'transition', 'Rest1 (post-landing)':'rest', 'Rest2 (pre-charge)':'rest'
}

# (Optional) test condition per cell
CELL_CONDITIONS = {
    'VAH01':'Baseline','VAH02':'Extended cruise (1000 s)','VAH05':'10% power reduction',
    'VAH06':'CC charge current C/2','VAH07':'CV 4.0V','VAH09':'20°C chamber',
    'VAH10':'30°C chamber','VAH11':'20% power reduction','VAH12':'Short cruise (400 s)',
    'VAH13':'Short cruise (600 s)','VAH15':'Extended cruise (1000 s)','VAH16':'CC 1.5C',
    'VAH17':'Baseline','VAH20':'Charge current 1.5C','VAH22':'Extended cruise (1000 s)',
    'VAH23':'CV 4.1V','VAH24':'CC C/2','VAH25':'20°C chamber','VAH26':'Short cruise (600 s)',
    'VAH27':'Baseline','VAH28':'10% power reduction','VAH30':'35°C chamber'
}

# --------------------------- LOGGING ---------------------------
def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------- HELPERS ---------------------------
def _ensure_cols(df: pd.DataFrame, cols: set, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.warning(f"{name}: missing columns: {missing}")

def _compute_dt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(['cell_id','cycleNumber','time_s']).copy()
    d['dt'] = d.groupby(['cell_id','cycleNumber'])['time_s'].diff().fillna(0).clip(lower=0)
    return d

def _rle_segments(series: pd.Series) -> pd.Series:
    """Return a run-length encoding id per contiguous equal value."""
    change = series.ne(series.shift(1, fill_value=series.iloc[0]))
    return change.cumsum()

def _detect_cycle_has_mission(ns_series: pd.Series) -> bool:
    vals = ns_series.dropna().astype(int).values
    has_to = 4 in vals
    has_ld = 6 in vals
    return bool(has_to and has_ld)

def _phase_family(name: str) -> str:
    return NS_FAMILY.get(name, 'other')

# --------------------------- CORE BUILD ---------------------------
def build_phase_instances_for_cell(ts_path: Path, sum_path: Path) -> pd.DataFrame:
    cell_id = ts_path.stem.split("_")[0]
    ts = pd.read_csv(ts_path)
    cs = pd.read_csv(sum_path)

    _ensure_cols(ts, REQUIRED_TS_COLS, f"{cell_id} timeseries")
    _ensure_cols(cs, REQUIRED_SUM_COLS, f"{cell_id} summary")

    # Keep minimal columns; compute dt
    cols_keep = list(REQUIRED_TS_COLS.union({'test_condition'} & set(ts.columns)))
    ts = ts[ [c for c in cols_keep if c in ts.columns] ].copy()
    ts = _compute_dt(ts)
    # Label exact phase names
    ts['phase_name'] = ts['Ns'].map(NS_TO_PHASE).fillna('unknown')

    # Add test condition
    ts['test_condition'] = CELL_CONDITIONS.get(cell_id, 'unknown')

    # Map cycle → SOH, init capacity
    cs_sorted = cs.sort_values('cycleNumber')
    soh_map = cs_sorted.set_index('cycleNumber')['SOH_pct_smoothed' if 'SOH_pct_smoothed' in cs_sorted.columns else 'SOH_pct'].to_dict()
    init_cap = float(cs_sorted['initial_capacity_mAh'].dropna().iloc[0]) if cs_sorted['initial_capacity_mAh'].notna().any() else np.nan

    # Determine mission_id per cycle (count only cycles that contain Take-off and Landing)
    cycles = sorted(ts['cycleNumber'].dropna().unique().tolist())
    mission_id_map = {}
    mission_counter = 0
    for cyc in cycles:
        ns_cyc = ts.loc[ts['cycleNumber']==cyc, 'Ns']
        if _detect_cycle_has_mission(ns_cyc):
            mission_counter += 1
            mission_id_map[cyc] = mission_counter
        else:
            mission_id_map[cyc] = np.nan

    # Determine first EOL mission (if any)
    # Use mission cycles only; EOL when SOH < threshold
    mission_cycles = [c for c in cycles if not np.isnan(mission_id_map.get(c, np.nan))]
    first_eol_mission = np.nan
    for cyc in mission_cycles:
        soh_val = soh_map.get(cyc, np.nan)
        if np.isfinite(soh_val) and soh_val < EOL_THRESHOLD_PCT:
            first_eol_mission = mission_id_map[cyc]
            break
    last_mission = max(mission_cycles, default=np.nan)
    last_mission_id = mission_id_map[last_mission] if mission_cycles else np.nan
    rul_censored = np.isnan(first_eol_mission)

    # Assign a contiguous-segment id per cycle by Ns
    ts['seg_id'] = ts.groupby(['cell_id','cycleNumber'])['Ns'].apply(_rle_segments).values

    # Aggregate per segment (phase instance)
    agg = ts.groupby(['cell_id','cycleNumber','seg_id','Ns','phase_name'], sort=True).agg(
        t_start_s=('time_s','min'),
        t_end_s=('time_s','max'),
        duration_s=('dt','sum'),
        energy_Wh=('power_W', lambda s: np.nan),   # placeholder; compute below with dt
        mean_power_W=('power_W','mean'),
        max_power_W=('power_W','max'),
        mean_abs_power_W=('power_W', lambda s: s.abs().mean()),
        mean_I_mA=('I_mA','mean'),
        max_I_mA=('I_mA','max'),
        min_I_mA=('I_mA','min'),
        mean_temp_C=('Temperature__C','mean'),
        max_temp_C=('Temperature__C','max'),
        start_SOC=('SOC','first'),
        end_SOC=('SOC','last'),
    ).reset_index()

    # Precise energy integration per segment: sum(power * dt)/3600
    # Build a quick lookup by (cycle, seg_id)
    ts['p_dt_Wh'] = ts['power_W'] * ts['dt'] / 3600.0
    energy_df = ts.groupby(['cell_id','cycleNumber','seg_id'])['p_dt_Wh'].sum().reset_index().rename(columns={'p_dt_Wh':'energy_Wh'})
    agg = agg.merge(energy_df, on=['cell_id','cycleNumber','seg_id'], how='left')
    agg.drop(columns=['energy_Wh_x'], inplace=True, errors='ignore')
    agg.rename(columns={'energy_Wh_y':'energy_Wh'}, inplace=True)

    # SOH per segment (approximately constant within cycle)
    # Use smoothed SOH if available; else raw SOH_pct
    agg['SOH_cycle_pct'] = agg['cycleNumber'].map(soh_map)

    # Mission mapping and per-mission targets
    agg['mission_id'] = agg['cycleNumber'].map(mission_id_map)
    agg['is_mission_cycle'] = agg['mission_id'].notna()

    # Phase order within mission (based on time ordering)
    agg['phase_order_in_mission'] = agg.groupby(['cell_id','mission_id'])['t_start_s'].rank(method='first').astype('Int64')

    # Phase family flags
    agg['phase_family'] = agg['phase_name'].map(_phase_family)
    agg['is_charge_phase'] = agg['phase_family'].eq('charge')
    agg['is_flight_phase'] = agg['phase_family'].eq('flight')
    agg['is_rest_phase']   = agg['phase_family'].eq('rest')
    agg['is_transition']   = agg['phase_family'].eq('transition')

    # Capacity (Ah) for C-rate features: SOH% * initial_capacity_mAh
    # Use per-cycle SOH; if missing, fall back to initial capacity
    cap_Ah = {}
    for cyc in cycles:
        soh = soh_map.get(cyc, np.nan)
        if np.isfinite(soh) and np.isfinite(init_cap):
            cap_Ah[cyc] = (soh/100.0) * (init_cap/1000.0)
        elif np.isfinite(init_cap):
            cap_Ah[cyc] = init_cap/1000.0
        else:
            cap_Ah[cyc] = np.nan
    agg['cap_Ah'] = agg['cycleNumber'].map(cap_Ah)

    # C-rate stats (mean of |I|/cap)
    agg['mean_C_rate'] = (agg['mean_I_mA'].abs()/1000.0) / agg['cap_Ah']
    agg['max_C_rate']  = (agg[['max_I_mA','min_I_mA']].abs().max(axis=1)/1000.0) / agg['cap_Ah']

    # Phase-level SOC/ SOH deltas
    agg['delta_SOC'] = agg['end_SOC'] - agg['start_SOC']
    agg['start_SOH_pct'] = agg['SOH_cycle_pct']
    agg['end_SOH_pct']   = agg['SOH_cycle_pct']  # within-cycle change is negligible; keep equal for targets
    agg['delta_SOH_pct'] = 0.0

    # Mission-level labels repeated on phases
    if not rul_censored and np.isfinite(first_eol_mission):
        agg['RUL_missions_after_phase'] = (first_eol_mission - agg['mission_id']).clip(lower=0)
        agg['RUL_missions_censored']    = False
    else:
        # censored: remaining missions until last observed mission
        last_mid = agg['mission_id'].dropna().max()
        agg['RUL_missions_after_phase'] = (last_mid - agg['mission_id']).clip(lower=0)
        agg['RUL_missions_censored']    = True

    # Also attach SOH at mission end (cycle-level SOH for mission cycles)
    agg['SOH_mission_end_pct'] = agg['SOH_cycle_pct']  # mission ends inside same cycle

    # Human-friendly extras
    agg['cell_condition'] = CELL_CONDITIONS.get(cell_id, 'unknown')

    # Drop segments not part of a mission if you only want mission-centric rows:
    # (keep charges/rests of the same cycle so models can use them)
    # If you want strictly flight-only rows, filter by is_flight_phase.
    return agg

# --------------------------- BATCH RUN ---------------------------
def build_all_phase_instances(processed_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_files = sorted(processed_dir.glob("VAH*_cleaned.csv"))
    sum_files = {p.stem.split("_")[0]: p for p in processed_dir.glob("VAH*_cycle_summary.csv")}

    all_frames = []
    for ts_path in ts_files:
        cell = ts_path.stem.split("_")[0]
        sum_path = sum_files.get(cell, None)
        if sum_path is None:
            logging.warning(f"Missing cycle summary for {cell}; skipping.")
            continue
        try:
            df_cell = build_phase_instances_for_cell(ts_path, sum_path)
            # save per cell
            cell_out = out_dir / f"{cell}_phase_instances.csv"
            df_cell.to_csv(cell_out, index=False)
            logging.info(f"Saved {cell_out.name} ({len(df_cell)} rows)")
            all_frames.append(df_cell)
        except Exception as e:
            logging.exception(f"Failed on {cell}: {e}")

    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        all_path = out_dir / "phase_instances.csv"
        all_df.to_csv(all_path, index=False)
        logging.info(f"Saved combined phase instances → {all_path} ({len(all_df)} rows)")
    else:
        logging.warning("No phase instances were created.")

# --------------------------- MANUAL ENTRY ---------------------------
if __name__ == "__main__":
    setup_logger()

    # ===== EDIT THESE TWO PATHS =====
    PROCESSED_DAY1 = Path(r"E:\BATTERY\processed_day1")        # where *_cleaned.csv & *_cycle_summary.csv live
    OUT_PHASE      = Path(r"E:\BATTERY\processed_day1\phase")  # output folder for phase instances
    # =================================

    logging.info(f"Building phase instances from {PROCESSED_DAY1}")
    build_all_phase_instances(PROCESSED_DAY1, OUT_PHASE)