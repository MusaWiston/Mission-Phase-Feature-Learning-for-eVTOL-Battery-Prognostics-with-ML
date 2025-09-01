# Feature engineering stage

import logging
from pathlib import Path
import time, shutil, tempfile, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ========================== CONFIG ==========================
# ---- EDIT THESE PATHS ----
PROCESSED_DAY1 = Path(r"E:\BATTERY\processed_day1")
OUTPUT_DAY2    = Path(r"E:\BATTERY\Feature_engineering")
# --------------------------
RNG_SEED = 42
SMALL_SEG_SEC = 2.0
POWER_MIN_FOR_GAIN = 1.0          # W; avoid divide-by-noise in rests
IC_BINS = 60                      # bins for incremental capacity histogram
MIN_POINTS_FOR_SLOPE = 12         # min samples to fit V~I slope
CV_VOLTAGE_ASSUMPTION = 4.19      # V proxy for CV(should be changed from 4.19 to 4.20 )
CLEANED_USECOLS = ['time_s','Ecell_V','I_mA','Temperature__C','cycleNumber','Ns','power_W','SOC']

# Retry policy for transient locks
MAX_RETRIES = 16
BASE_BACKOFF = 0.35  # seconds
JITTER = 0.15        # seconds
SKIP_LOCKED_CELLS = True  # if a cleaned CSV remains locked, skip IC/V-I for that cell

# Ns → Phase (fixed from Day 1)
NS_TO_PHASE = {
    0:'CC', 1:'CV', 2:'CV→Rest', 3:'Rest→TO', 4:'Take-off', 5:'Cruise',
    6:'Landing', 7:'Landing→Rest', 8:'Rest1 (post-landing)', 9:'Rest2 (pre-charge)'
}
PHASE_FAMILY = {
    'CC':'charge', 'CV':'charge', 'CV→Rest':'transition', 'Rest→TO':'transition',
    'Take-off':'flight', 'Cruise':'flight', 'Landing':'flight',
    'Landing→Rest':'transition', 'Rest1 (post-landing)':'rest', 'Rest2 (pre-charge)':'rest'
}

# ========================== LOGGING ==========================
def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ===================== ROBUST CSV READER =====================
def _size_stable(path: Path, wait_s=0.4) -> bool:
    try:
        s1 = path.stat().st_size
        time.sleep(wait_s)
        s2 = path.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False

def read_csv_retry(path, usecols=None, engine=None, allow_skip=False, **kwargs):
    """
    Read CSV resiliently on Windows. If allow_skip=True and locks persist,
    returns None instead of raising (caller can skip that cell).
    """
    path = Path(path)
    try:
        _ = _size_stable(path, wait_s=0.3)
        return pd.read_csv(path, usecols=usecols, engine=engine, **kwargs)
    except PermissionError as e:
        logging.warning(f"{path.name}: PermissionError → temp-copy workaround ({e})")

    last_err = None
    for k in range(MAX_RETRIES):
        try:
            if not _size_stable(path, wait_s=0.3 + random.uniform(0, JITTER)):
                time.sleep(BASE_BACKOFF * (k+1))
            tmp = Path(tempfile.gettempdir()) / f"tmp_{int(time.time()*1000)}_{path.name}"
            shutil.copyfile(path, tmp)
            df = pd.read_csv(tmp, usecols=usecols, engine=engine, **kwargs)
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            return df
        except (PermissionError, OSError) as err:
            last_err = err
            time.sleep(BASE_BACKOFF * (k+1) + random.uniform(0, JITTER))

    if allow_skip:
        logging.error(f"{path.name}: still locked after {MAX_RETRIES} retries → skipping.")
        return None
    raise PermissionError(f"Could not read {path} after {MAX_RETRIES} retries") from last_err

# ========================= LOAD DAY 1 ========================
def load_day1(proc_dir: Path):
    cs_path = proc_dir / "per_cycle_summary_allcells.csv"
    ph_path = proc_dir / "phase" / "phase_instances.csv"
    if not cs_path.exists():    raise FileNotFoundError(cs_path)
    if not ph_path.exists():    raise FileNotFoundError(ph_path)
    cs = read_csv_retry(cs_path)
    phases = read_csv_retry(ph_path)
    return cs, phases

# ================== TIMESERIES HELPERS (IC, VI) ==============
_cleaned_cache = {}
_skipped_cells = set()

def _load_cleaned_cell(cell_id: str):
    """
    Load and cache a cell's *_cleaned.csv once (robustly).
    Returns None if file remains locked (and SKIP_LOCKED_CELLS=True).
    """
    if cell_id in _cleaned_cache:
        return _cleaned_cache[cell_id]
    p = PROCESSED_DAY1 / f"{cell_id}_cleaned.csv"
    if not p.exists():
        logging.warning(f"Missing cleaned TS: {p.name}")
        _cleaned_cache[cell_id] = None
        return None

    # Header first to determine available columns (less OS touching)
    head = read_csv_retry(p, nrows=0, allow_skip=SKIP_LOCKED_CELLS)
    if head is None:
        _skipped_cells.add(cell_id); _cleaned_cache[cell_id] = None
        return None
    cols = [c for c in CLEANED_USECOLS if c in head.columns]

    df = read_csv_retry(p, usecols=cols, allow_skip=SKIP_LOCKED_CELLS)
    if df is None:
        _skipped_cells.add(cell_id); _cleaned_cache[cell_id] = None
        return None

    # Fill missing required columns with NaN (rare)
    for c in CLEANED_USECOLS:
        if c not in df.columns: df[c] = np.nan

    _cleaned_cache[cell_id] = df
    return df

def _extract_segment_ts(cleaned_df: pd.DataFrame, cycle: int, t0: float, t1: float) -> pd.DataFrame:
    if cleaned_df is None or cleaned_df.empty: return pd.DataFrame()
    g = cleaned_df[(cleaned_df['cycleNumber']==cycle) &
                   (cleaned_df['time_s'] >= t0) &
                   (cleaned_df['time_s'] <= t1)].copy()
    return g.sort_values('time_s')

def _ic_features(ts: pd.DataFrame) -> dict:
    if len(ts) < 8 or not {'time_s','I_mA','Ecell_V'}.issubset(ts.columns): return {}
    ts = ts.sort_values('time_s').copy()
    t = ts['time_s'].values
    I = ts['I_mA'].values
    V = ts['Ecell_V'].values
    if np.allclose(V.max(), V.min(), atol=1e-3): return {}

    dt_h = np.diff(t, prepend=t[0]) / 3600.0
    dQ = I * dt_h  # mAh (signed)

    vmin, vmax = np.percentile(V, 2), np.percentile(V, 98)
    if vmax - vmin < 0.05: return {}
    bins = np.linspace(vmin, vmax, IC_BINS+1)
    idx = np.digitize(V, bins) - 1

    hist = np.zeros(IC_BINS, dtype=float)
    for i, dq in zip(idx, dQ):
        if 0 <= i < IC_BINS: hist[i] += dq

    dV = (vmax - vmin) / IC_BINS
    ic = hist / max(dV, 1e-6)  # mAh/V
    centers = 0.5*(bins[:-1] + bins[1:])
    total = np.nansum(np.abs(ic))
    if total <= 0: return {}

    prob = np.abs(ic) / total
    centroid = float(np.nansum(centers * prob))
    spread   = float(np.sqrt(np.nansum(((centers - centroid)**2) * prob)))

    order = np.argsort(np.abs(ic))[::-1]
    p1 = order[0] if len(order)>0 else None
    p2 = order[1] if len(order)>1 else None
    return {
        'ic_centroid_V': centroid,
        'ic_spread_V': spread,
        'ic_peak1_V': float(centers[p1]) if p1 is not None else np.nan,
        'ic_peak1_height': float(ic[p1]) if p1 is not None else np.nan,
        'ic_peak2_V': float(centers[p2]) if p2 is not None else np.nan,
        'ic_peak2_height': float(ic[p2]) if p2 is not None else np.nan,
    }

def _vi_slope_R(ts: pd.DataFrame) -> dict:
    if len(ts) < MIN_POINTS_FOR_SLOPE or not {'I_mA','Ecell_V'}.issubset(ts.columns): return {}
    I = (ts['I_mA'].values / 1000.0).reshape(-1,1)  # A
    V = ts['Ecell_V'].values
    if np.nanstd(I) < 0.05:  # needs variability
        return {}
    try:
        lr = LinearRegression().fit(I, V)
        return {'vi_slope_R_ohm': float(-lr.coef_[0])}
    except Exception:
        return {}

# ============== BUILD PHASE-LEVEL FEATURES ===================
def add_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dur = np.maximum(d['duration_s'].values, 1.0)
    d['thermal_slope_K_per_s'] = (d['max_temp_C'] - d['mean_temp_C']) / dur
    denom = d['mean_abs_power_W'].replace(0, np.nan)
    d['thermal_gain_K_per_W']  = np.where(denom >= POWER_MIN_FOR_GAIN,
                                          d['thermal_slope_K_per_s'] / denom,
                                          np.nan)
    return d

def build_phase_features(phase_df: pd.DataFrame) -> pd.DataFrame:
    """
    Start from phase_instances.csv; add thermal, IC, and V–I slope per segment.
    If a cleaned CSV is locked, IC/V-I features are NaN for that cell (others proceed).
    """
    df = phase_df.copy()
    if 'duration_s' in df.columns:
        df = df[df['duration_s'] >= SMALL_SEG_SEC].copy()
    df = add_thermal_features(df)

    rows = []
    for (cell, cyc, seg), seg_rows in df.groupby(['cell_id','cycleNumber','seg_id']):
        cleaned = _load_cleaned_cell(cell)
        feats = {}
        if cleaned is not None and not cleaned.empty:
            t0, t1 = float(seg_rows['t_start_s'].iloc[0]), float(seg_rows['t_end_s'].iloc[0])
            ts = _extract_segment_ts(cleaned, int(cyc), t0, t1)
            if not ts.empty:
                try:
                    feats.update(_ic_features(ts))
                    feats.update(_vi_slope_R(ts))
                except Exception as e:
                    logging.warning(f"{cell} cyc{cyc} seg{seg}: IC/VI calc failed ({e})")
        base = seg_rows.iloc[0].to_dict()
        base.update(feats)
        rows.append(base)
    out = pd.DataFrame(rows).reset_index(drop=True)

    if _skipped_cells:
        (OUTPUT_DAY2 / "skipped_cells.txt").write_text("\n".join(sorted(_skipped_cells)), encoding="utf-8")
        logging.error(f"Skipped IC/V-I for locked cells: {sorted(_skipped_cells)}")
    return out

# ============== BUILD MISSION-LEVEL FEATURES =================
def build_mission_features(phase_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per mission using precomputed conditional columns
    (robust to pandas groupby quirks).
    """
    df = phase_feat.copy()
    df = df[df['mission_id'].notna()].reset_index(drop=True)

    # Family mapping
    df['phase_family'] = df['phase_name'].map(PHASE_FAMILY).fillna('other')

    # Ensure numeric cols exist
    for c in ['duration_s','energy_Wh','thermal_gain_K_per_W','vi_slope_R_ohm',
              'ic_centroid_V','ic_spread_V','ic_peak1_V','ic_peak1_height',
              'SOH_mission_end_pct']:
        if c not in df.columns:
            df[c] = np.nan

    # Conditional columns (durations & energies per family)
    for fam in ['flight','charge','rest','transition']:
        df[f'dur_{fam}_s']     = np.where(df['phase_family'].eq(fam), df['duration_s'].fillna(0.0), 0.0)
        df[f'energy_{fam}_Wh'] = np.where(df['phase_family'].eq(fam), df['energy_Wh'].fillna(0.0), 0.0)

    # CC/CV specific
    df['dur_cc_s']     = np.where(df['phase_name'].eq('CC'), df['duration_s'].fillna(0.0), 0.0)
    df['dur_cv_s']     = np.where(df['phase_name'].eq('CV'), df['duration_s'].fillna(0.0), 0.0)
    df['energy_cv_Wh'] = np.where(df['phase_name'].eq('CV'), df['energy_Wh'].fillna(0.0), 0.0)

    # Aggregate
    grp = (df.groupby(['cell_id','mission_id'], as_index=False)
             .agg(
                 n_segments=('seg_id','count'),
                 dur_flight_s=('dur_flight_s','sum'),
                 dur_charge_s=('dur_charge_s','sum'),
                 dur_rest_s=('dur_rest_s','sum'),
                 dur_transition_s=('dur_transition_s','sum'),
                 energy_flight_Wh=('energy_flight_Wh','sum'),
                 energy_charge_Wh=('energy_charge_Wh','sum'),
                 th_gain_med=('thermal_gain_K_per_W','median'),
                 th_gain_max=('thermal_gain_K_per_W','max'),
                 R_ohm_med=('vi_slope_R_ohm','median'),
                 R_ohm_p90=('vi_slope_R_ohm', lambda s: np.nanpercentile(s.dropna(), 90) if s.notna().sum() else np.nan),
                 ic_centroid_V=('ic_centroid_V','median'),
                 ic_spread_V=('ic_spread_V','median'),
                 ic_p1_V=('ic_peak1_V','median'),
                 ic_p1_h=('ic_peak1_height','median'),
                 SOH_end_pct=('SOH_mission_end_pct','max'),
                 cc_duration_s=('dur_cc_s','sum'),
                 cv_duration_s=('dur_cv_s','sum'),
                 cv_energy_Wh=('energy_cv_Wh','sum'),
             ))

    # Derived
    grp['flight_Wh_abs'] = grp['energy_flight_Wh'].abs()
    grp['eff_energy']    = grp['flight_Wh_abs'] / grp['energy_charge_Wh'].replace(0, np.nan)
    denom = (grp['cc_duration_s'].fillna(0) + grp['cv_duration_s'].fillna(0)).replace(0, np.nan)
    grp['cv_fraction']   = grp['cv_duration_s'] / denom
    grp['cc_fraction']   = grp['cc_duration_s'] / denom
    grp['cv_mean_I_A_proxy'] = (grp['cv_energy_Wh'] / CV_VOLTAGE_ASSUMPTION) / (grp['cv_duration_s'] / 3600.0).replace(0, np.nan)
    grp.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Targets: ΔSOH to next mission + last-mission flag
    grp['SOH_next_pct']  = grp.groupby('cell_id')['SOH_end_pct'].shift(-1)
    grp['dSOH_drop_pct'] = (grp['SOH_end_pct'] - grp['SOH_next_pct']).clip(lower=0)
    grp['is_last_mission'] = grp.groupby('cell_id')['mission_id'].transform(lambda s: s == s.max())
    return grp

# =================== MODELLING TABLES =======================
def _one_hot(df: pd.DataFrame, cols: list, drop=True) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns: continue
        dummies = pd.get_dummies(out[c].astype('category'), prefix=c, dummy_na=False)
        out = pd.concat([out, dummies], axis=1)
        if drop: out.drop(columns=[c], inplace=True)
    return out

def table_SOC_phase(phase_feat: pd.DataFrame):
    df = phase_feat.copy()
    df = df[df['mission_id'].notna()]
    base_cols = [
        'duration_s','energy_Wh','mean_abs_power_W','mean_temp_C','max_temp_C',
        'mean_C_rate','max_C_rate','thermal_slope_K_per_s','thermal_gain_K_per_W',
        'vi_slope_R_ohm','ic_centroid_V','ic_spread_V','ic_peak1_V','ic_peak1_height'
    ]
    X = df[[c for c in base_cols if c in df.columns]].copy()
    X = pd.concat([X, df[['phase_name','phase_family']]], axis=1)
    X = _one_hot(X, ['phase_name','phase_family'])
    y = df['delta_SOC']
    tbl = pd.concat([X, y.rename('target')], axis=1).dropna()
    return tbl, 'SOC_phase_delta'

def table_SOH_next(missions: pd.DataFrame):
    df = missions[~missions['is_last_mission']].copy()
    base_cols = [
        'flight_Wh_abs','dur_flight_s','dur_charge_s','dur_rest_s','dur_transition_s',
        'eff_energy','th_gain_med','th_gain_max','R_ohm_med','R_ohm_p90',
        'ic_centroid_V','ic_spread_V','ic_p1_V','ic_p1_h','cv_fraction','cc_fraction','cv_mean_I_A_proxy'
    ]
    X = df[[c for c in base_cols if c in df.columns]].copy()
    y = df['SOH_next_pct']
    tbl = pd.concat([X, y.rename('target')], axis=1).dropna()
    return tbl, 'SOH_next_mission'

def table_RUL_mission(phase_feat: pd.DataFrame):
    dfm = (phase_feat.dropna(subset=['mission_id'])
           .groupby(['cell_id','mission_id'])
           .agg(
               RUL=('RUL_missions_after_phase','max'),
               thermal_gain_med=('thermal_gain_K_per_W','median'),
               mean_C=('mean_C_rate','median'),
               max_C=('max_C_rate','median'),
               max_T=('max_temp_C','median'),
               vi_R=('vi_slope_R_ohm','median'),
               ic_centroid_V=('ic_centroid_V','median'),
               flight_Wh=('energy_Wh', lambda s: s[phase_feat.loc[s.index,'phase_family'].eq('flight')].sum()),
               charge_Wh=('energy_Wh', lambda s: s[phase_feat.loc[s.index,'phase_family'].eq('charge')].sum())
           ).reset_index())
    X = dfm[['thermal_gain_med','mean_C','max_C','max_T','vi_R','ic_centroid_V','flight_Wh','charge_Wh']]
    y = dfm['RUL']
    tbl = pd.concat([X, y.rename('target')], axis=1).dropna()
    return tbl, 'RUL_mission'

# ============== QUICK MODELS + IMPORTANCE ====================
def fit_and_importance(df_xy: pd.DataFrame, name: str, outdir: Path, n_estimators=300):
    outdir.mkdir(parents=True, exist_ok=True)
    if df_xy.empty or df_xy.shape[0] < 50:
        logging.warning(f"[{name}] Not enough rows for modeling ({df_xy.shape[0]}). Skipping.")
        return
    y = df_xy['target'].values
    X = df_xy.drop(columns=['target'])
    cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, shuffle=True
    )
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=RNG_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    r2 = rf.score(X_test, y_test)

    # Impurity-based
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    pd.DataFrame({'feature': imp.index, 'gini_importance': imp.values}).to_csv(outdir / f"{name}_gini_importance.csv", index=False)

    # Permutation
    pi = permutation_importance(rf, X_test, y_test, n_repeats=8, random_state=RNG_SEED, n_jobs=-1)
    pi_s = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    pi_std = pd.Series(pi.importances_std, index=cols).reindex(pi_s.index)
    pd.DataFrame({'feature': pi_s.index, 'perm_importance_mean': pi_s.values, 'perm_importance_std': pi_std.values}).to_csv(outdir / f"{name}_perm_importance.csv", index=False)

    # Plot
    plt.figure(figsize=(8, max(3, 0.4*len(cols))))
    plt.barh(pi_s.index[::-1], pi_s.values[::-1])
    plt.xlabel("Permutation importance")
    plt.title(f"{name} — RF (R²≈{r2:.2f})")
    plt.tight_layout()
    plt.savefig(outdir / f"{name}_perm_importance.png")
    plt.close()

    logging.info(f"[{name}] R²={r2:.3f} — saved importance CSVs and plot.")

# ============================ MAIN ===========================
def main():
    setup_logger()
    ensure_dir(OUTPUT_DAY2)
    ensure_dir(OUTPUT_DAY2 / "features")
    ensure_dir(OUTPUT_DAY2 / "importance")

    logging.info("Loading Day 1 artifacts…")
    cs, phases = load_day1(PROCESSED_DAY1)

    logging.info("Building phase-level features…")
    phase_feat = build_phase_features(phases)
    phase_out = OUTPUT_DAY2 / "features" / "phase_features.csv"
    phase_feat.to_csv(phase_out, index=False)
    logging.info(f"Saved {phase_out}")

    logging.info("Aggregating mission-level features…")
    mission_feat = build_mission_features(phase_feat)
    mission_out = OUTPUT_DAY2 / "features" / "mission_features.csv"
    mission_feat.to_csv(mission_out, index=False)
    logging.info(f"Saved {mission_out}")

    # Baseline importance screens
    soc_tbl, soc_name = table_SOC_phase(phase_feat)
    fit_and_importance(soc_tbl, soc_name, OUTPUT_DAY2 / "importance")

    soh_tbl, soh_name = table_SOH_next(mission_feat)
    fit_and_importance(soh_tbl, soh_name, OUTPUT_DAY2 / "importance")

    rul_tbl, rul_name = table_RUL_mission(phase_feat)
    fit_and_importance(rul_tbl, rul_name, OUTPUT_DAY2 / "importance")

    logging.info("Day 2 complete. Outputs → processed_day2/")

if __name__ == "__main__":
    main()
