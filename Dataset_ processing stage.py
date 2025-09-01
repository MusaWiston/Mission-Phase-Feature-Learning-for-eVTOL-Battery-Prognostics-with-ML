# Data processing stage
 
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# =================== CONFIG ===================
DEFAULT_EOL_THRESHOLD = 85.0      # SOH % at EOL
REST_CURRENT_THRESH   = 50.0      # mA (near-zero current)
VOLTAGE_LOWER, VOLTAGE_UPPER = 2.5, 4.3
HIGH_POWER_FRAC = 0.6
LOW_POWER_FRAC  = 0.2
MIN_SEG_SEC     = 30.0
REQUIRED_COLUMNS = {
    'time_s','Ecell_V','I_mA','QDischarge_mA_h','QCharge_mA_h',
    'Temperature__C','cycleNumber','Ns'
}

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

# =================== LOGGING ===================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

# =================== UTILITIES ===================
def vectorized_power(df: pd.DataFrame) -> pd.DataFrame:
    df['power_W'] = df['Ecell_V'] * df['I_mA'] / 1000.0
    return df

def safe_inspect(df: pd.DataFrame, drop_missing_pct=0.5) -> pd.DataFrame:
    """Drop constant and high-missing columns but preserve required ones."""
    missing_pct = df.isnull().mean()
    constant    = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_miss   = missing_pct[missing_pct > drop_missing_pct].index.tolist()
    to_drop     = [c for c in set(constant + high_miss) if c not in REQUIRED_COLUMNS]
    if to_drop:
        logging.info(f"Dropping columns (constant or >{int(drop_missing_pct*100)}% NaN): {to_drop}")
    return df.drop(columns=to_drop, errors='ignore')

def infer_discharge_sign(df: pd.DataFrame) -> int:
    """Return +1 if discharge is logged positive, -1 if negative."""
    if 'QDischarge_mA_h' not in df:
        return 1 if df['I_mA'].median() > 0 else -1
    dQ = df['QDischarge_mA_h'].diff().fillna(0)
    mask = dQ > 0
    if mask.sum() < 10:
        return 1 if df['I_mA'].median() > 0 else -1
    return 1 if df.loc[mask, 'I_mA'].mean() > 0 else -1

def detect_cv_setpoint(v_series: pd.Series) -> float:
    """Estimate CV setpoint; snap near 4.00/4.10/4.20 V."""
    v95 = float(np.nanpercentile(v_series, 95))
    for t in (4.00, 4.10, 4.20):
        if abs(v95 - t) <= 0.05:
            return t
    return round(v95, 2)

# =================== SUMMARIES ===================
def per_cycle_summary(df: pd.DataFrame, cell_id: str) -> pd.DataFrame:
    if 'power_W' not in df:
        df = vectorized_power(df)
    g = df.groupby('cycleNumber', sort=True)
    cs = g.agg(
        n_rows=('time_s','size'),
        duration_s=('time_s', lambda s: float(s.max() - s.min())),
        max_voltage=('Ecell_V','max'),
        min_voltage=('Ecell_V','min'),
        mean_voltage=('Ecell_V','mean'),
        max_temp=('Temperature__C','max'),
        mean_temp=('Temperature__C','mean'),
        max_current=('I_mA','max'),
        mean_current=('I_mA','mean'),
        std_current=('I_mA','std'),
        max_power=('power_W','max'),
        mean_power=('power_W','mean'),
        QDischarge_mAh=('QDischarge_mA_h','max'),
        QCharge_mAh=('QCharge_mA_h','max'),
        energy_discharge=('EnergyDischarge_W_h','max') if 'EnergyDischarge_W_h' in df.columns else ('time_s','size'),
        energy_charge=('EnergyCharge_W_h','max') if 'EnergyCharge_W_h' in df.columns else ('time_s','size'),
        Ns_mode=('Ns', lambda x: int(x.mode().iloc[0]) if not x.mode().empty else -1),
    ).reset_index()
    # if energies not present, set to NaN
    if isinstance(cs['energy_discharge'].dtype, pd.Series) or cs['energy_discharge'].dtype == 'O':
        pass
    cs['cell_id'] = cell_id
    return cs

# =================== CAPACITY TESTS (scored & robust) ===================
def mark_capacity_tests_strict(df: pd.DataFrame,
                               cycle_stats: pd.DataFrame,
                               sign_factor: int) -> pd.DataFrame:
    """
    Score-based capacity test detection:
      +1 S1: frac(Ns in {4,5,6}) < 2%      (no-flight)
      +1 S2: discharge_duration_s > max(1200, 1.8 * median_mission_discharge)
      +1 S3: QDischarge >= p90(file) or 0.9*max
      +1 S4: quasi-constant discharge current: std(Id)/|mean(Id)| <= 0.30
      +1 S5: deep-ish: min(V) <= 3.0 OR (maxV-minV) >= 0.8
    Mark as capacity test if score >= 3.
    Fallback: if none, pick top-2 no-flight by score then QDischarge.
    """
    d = df.sort_values(['cycleNumber','time_s']).copy()
    d['_Istd'] = d['I_mA'] * (-1 if sign_factor > 0 else 1)     # discharge negative
    d['_dt']   = d.groupby('cycleNumber')['time_s'].diff().fillna(0).clip(lower=0)

    flight_set = {4,5,6}
    feats = []
    for cyc, g in d.groupby('cycleNumber', sort=True):
        frac_flight = g['Ns'].isin(flight_set).mean() if 'Ns' in g else 0.0
        dis_mask = g['_Istd'] < -REST_CURRENT_THRESH
        dis_dur  = float(g.loc[dis_mask, '_dt'].sum())
        Id       = g.loc[dis_mask, '_Istd']
        quasi_cc = (Id.std() / (abs(Id.mean())+1e-6) <= 0.30) if len(Id) else False
        qd       = float(g['QDischarge_mA_h'].max() or 0.0)
        vmin     = float(g['Ecell_V'].min())
        vmax     = float(g['Ecell_V'].max())
        vdrop    = max(0.0, vmax - vmin)
        feats.append((cyc, frac_flight, dis_dur, quasi_cc, qd, vmin, vdrop))
    f = pd.DataFrame(feats, columns=['cycleNumber','frac_flight','dis_dur_s','quasi_cc','QDischarge_mAh','vmin','vdrop'])

    # references
    mission_dur_med = f.loc[f['frac_flight'] > 0.2, 'dis_dur_s'].median()
    mission_dur_med = float(mission_dur_med) if np.isfinite(mission_dur_med) else 800.0
    p90_Q = float(np.nanpercentile(f['QDischarge_mAh'].values, 90)) if len(f) else 0.0
    maxQ  = float(f['QDischarge_mAh'].max() or 0.0)

    S1 = (f['frac_flight'] < 0.02)
    S2 = (f['dis_dur_s'] > max(1200.0, 1.8 * mission_dur_med))
    S3 = (f['QDischarge_mAh'] >= max(p90_Q, 0.9 * maxQ))
    S4 = f['quasi_cc']
    S5 = (f['vmin'] <= 3.0) | (f['vdrop'] >= 0.8)

    f['score'] = S1.astype(int) + S2.astype(int) + S3.astype(int) + S4.astype(int) + S5.astype(int)
    f['is_capacity_test'] = f['score'] >= 3

    # fallback if none
    if not f['is_capacity_test'].any():
        candidates = f[S1].sort_values(['score','QDischarge_mAh'], ascending=[False, False]).head(2)
        f.loc[candidates.index, 'is_capacity_test'] = True

    out = cycle_stats.merge(f[['cycleNumber','is_capacity_test','score','dis_dur_s','frac_flight']],
                            on='cycleNumber', how='left')
    out['is_capacity_test'] = out['is_capacity_test'].fillna(False)
    return out

# =================== PHASE LABELING (exact Ns mapping) ===================
def label_phases(df: pd.DataFrame, sign_factor: int) -> pd.DataFrame:
    """
    Use Ns directly to label phases with the exact requested strings.
    """
    d = df.sort_values(['cycleNumber','time_s']).copy()
    if 'power_W' not in d:
        d = vectorized_power(d)
    d['phase'] = d['Ns'].map(NS_TO_PHASE).fillna('unknown')
    d['phase_source'] = 'Ns'
    return d

# =================== SOH & RUL ===================
def compute_soh(cycle_stats: pd.DataFrame) -> pd.DataFrame:
    cs = cycle_stats.sort_values('cycleNumber').reset_index(drop=True).copy()
    cap = cs[cs['is_capacity_test'] & cs['QDischarge_mAh'].notna()]
    if cap.empty:
        cs['initial_capacity_mAh'] = np.nan
        cs['SOH_pct'] = np.nan
        cs['SOH_pct_smoothed'] = np.nan
        return cs

    init_cap = float(np.nanmedian(cap.head(3)['QDischarge_mAh'].values))
    cs['initial_capacity_mAh'] = init_cap

    cs['SOH_pct'] = np.nan
    cs.loc[cap.index, 'SOH_pct'] = cs.loc[cap.index,'QDischarge_mAh'] / init_cap * 100.0
    cs['SOH_pct'] = cs['SOH_pct'].interpolate().ffill().bfill()
    cs['SOH_pct_smoothed'] = cs['SOH_pct'].rolling(5, min_periods=1).median()
    return cs

def compute_rul(cycle_stats: pd.DataFrame, eol_pct=DEFAULT_EOL_THRESHOLD) -> pd.DataFrame:
    cs = cycle_stats.sort_values('cycleNumber').reset_index(drop=True).copy()
    cs['RUL_cycles'] = np.nan
    if 'SOH_pct_smoothed' not in cs:
        return cs
    below = cs['SOH_pct_smoothed'] < eol_pct
    if not below.any():
        maxc = int(cs['cycleNumber'].max())
        cs['RUL_cycles']   = maxc - cs['cycleNumber']
        cs['RUL_censored'] = True
        return cs
    eol_cycle = int(cs.loc[below, 'cycleNumber'].iloc[0])
    cs['RUL_cycles']   = (eol_cycle - cs['cycleNumber']).clip(lower=0)
    cs['RUL_censored'] = False
    return cs

# =================== SOC (measured capacity) ===================
def compute_soc(df: pd.DataFrame, cycle_stats: pd.DataFrame, sign_factor: int) -> pd.DataFrame:
    """
    Coulomb counting per cycle using measured capacity:
      capacity_mAh(cycle) = SOH%/100 * initial_capacity_mAh
    Start-of-cycle SOC = 100%; discharge negative after standardization.
    """
    d = df.sort_values(['cycleNumber','time_s']).copy()
    if 'power_W' not in d:
        d = vectorized_power(d)

    if not {'SOH_pct','initial_capacity_mAh'}.issubset(cycle_stats.columns):
        d['SOC'] = np.nan
        return d
    if not cycle_stats['initial_capacity_mAh'].notna().any():
        d['SOC'] = np.nan
        return d

    cap0 = float(cycle_stats['initial_capacity_mAh'].dropna().iloc[0])
    soh_map = cycle_stats.set_index('cycleNumber')['SOH_pct'].to_dict()

    d['SOC'] = np.nan
    for cyc, g in d.groupby('cycleNumber', sort=True):
        soh = soh_map.get(int(cyc), np.nan)
        if not np.isfinite(soh) or cap0 <= 0:
            continue
        cap = (soh/100.0) * cap0
        I_std = g['I_mA'] * (-1 if sign_factor > 0 else 1)  # discharge negative
        t     = g['time_s'].values
        dt_h  = np.diff(t, prepend=t[0]) / 3600.0
        dQ    = (I_std.values * dt_h)  # mAh
        soc   = 100.0 + np.cumsum(dQ) / cap * 100.0
        d.loc[g.index, 'SOC'] = np.clip(soc, 0.0, 100.0)
    return d

# =================== ANOMALIES ===================
def detect_anomalies(df: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    out = cs.copy()
    out['anomaly_score'] = 0
    out.loc[(out['min_voltage'] < VOLTAGE_LOWER) | (out['max_voltage'] > VOLTAGE_UPPER), 'anomaly_score'] += 1
    dur_med = out['duration_s'].median() if out['duration_s'].notna().any() else 0
    out.loc[out['duration_s'] < 0.1*max(1.0,dur_med), 'anomaly_score'] += 1
    # Discontinuities (per-cycle max ΔV)
    try:
        vdiff = df.groupby('cycleNumber')['Ecell_V'].apply(lambda s: s.diff().abs().max())
        bad = vdiff[vdiff > 0.5].index.tolist()
        out.loc[out['cycleNumber'].isin(bad), 'anomaly_score'] += 1
    except Exception:
        pass
    out['any_anomaly'] = out['anomaly_score'] > 0
    return out

# =================== PER-FILE PIPELINE ===================
def process_file(path: Path, outdir: Path):
    cell_id = path.stem
    logging.info(f"Processing {cell_id}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logging.warning(f"{cell_id} missing columns: {missing}")

    df = safe_inspect(df)
    df = vectorized_power(df)
    df['cell_id'] = cell_id

    cs = per_cycle_summary(df, cell_id)

    # Current sign
    sign = infer_discharge_sign(df)  # +1 => discharge logged positive
    logging.info(f"  discharge sign: {'positive' if sign>0 else 'negative'}")

    # Capacity tests (scored)
    cs = mark_capacity_tests_strict(df, cs, sign_factor=sign)
    n_caps = int(cs['is_capacity_test'].sum())
    logging.info(f"  capacity tests detected: {n_caps}")
    if n_caps == 0 and 'score' in cs.columns:
        top = cs.sort_values('score', ascending=False).head(3)[['cycleNumber','score','dis_dur_s','frac_flight','QDischarge_mAh']]
        logging.info(f"  top candidates (score):\n{top.to_string(index=False)}")

    # Anomalies
    cs = detect_anomalies(df, cs)

    # Phases (Ns mapping EXACT)
    df = label_phases(df, sign_factor=sign)

    # SOH & RUL
    cs = compute_soh(cs)
    cs = compute_rul(cs, eol_pct=DEFAULT_EOL_THRESHOLD)

    # SOC (measured capacity)
    df = compute_soc(df, cs, sign_factor=sign)

    # Save
    outdir.mkdir(parents=True, exist_ok=True)
    ts_path = outdir / f"{cell_id}_cleaned.csv"
    cs_path = outdir / f"{cell_id}_cycle_summary.csv"
    df.to_csv(ts_path, index=False)
    cs.to_csv(cs_path, index=False)

    meta = {
        "cell_id": cell_id,
        "n_rows": int(len(df)),
        "n_cycles": int(cs.shape[0]),
        "n_capacity_tests": n_caps,
        "initial_capacity_mAh": float(cs['initial_capacity_mAh'].dropna().iloc[0]) if 'initial_capacity_mAh' in cs and cs['initial_capacity_mAh'].notna().any() else None,
        "rul_censored": bool(cs['RUL_censored'].iloc[0]) if 'RUL_censored' in cs else None
    }
    with open(outdir / f"{cell_id}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"Saved {ts_path.name}, {cs_path.name}")
    return cs, df, meta

# -------------------------
# MANUAL RUN (no arguments)
# -------------------------
if __name__ == "__main__":
    setup_logger()

    # ====== EDIT THESE TWO LINES ======
    DATA_DIR = Path(r"E:\BATTERY\eVTOL_battery_dataset")   # folder with VAH*.csv
    OUTDIR   = Path(r"E:\BATTERY\processed_day1")          # output folder
    # Optional: process a single file instead of the whole folder
    PROCESS_SINGLE_FILE = False
    SINGLE_FILE_PATH = Path(r"E:\BATTERY\eVTOL_battery_dataset\VAH27.csv")
    # ==================================

    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = [SINGLE_FILE_PATH] if PROCESS_SINGLE_FILE else sorted(DATA_DIR.glob("VAH*.csv"))
    logging.info(f"Found {len(files)} file(s)")

    all_cs = []
    for p in files:
        try:
            cs, df, meta = process_file(p, OUTDIR)
            cs['cell'] = meta['cell_id']
            all_cs.append(cs)
        except Exception as e:
            logging.exception(f"Failed on {p.name}: {e}")

    if all_cs:
        combined = pd.concat(all_cs, ignore_index=True)
        combined_path = OUTDIR / "per_cycle_summary_allcells.csv"
        combined.to_csv(combined_path, index=False)
        logging.info(f"Saved combined summary → {combined_path}")
    else:
        logging.warning("No summaries to combine.")
