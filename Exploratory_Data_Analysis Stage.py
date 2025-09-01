#eda Stage

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== CONFIG ===================
EOL_PCT = 85.0
SMALL_SEG_SEC = 2.0
CV_VOLTAGE_ASSUMPTION = 4.10  # V, proxy for CV voltage(4.19 to 4.20)
POWER_MIN_FOR_GAIN = 1.0      # W; below this, thermal_gain_K_per_W -> NaN (rest/transition stability)
PLOT_MAX_PHASES = 12          # safety cap for plotting loops

# =================== LOGGING ==================
def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# =================== LOAD =====================
def load_inputs(proc_dir: Path):
    cs_path     = proc_dir / "per_cycle_summary_allcells.csv"
    phases_path = proc_dir / "phase" / "phase_instances.csv"
    if not cs_path.exists():    raise FileNotFoundError(cs_path)
    if not phases_path.exists(): raise FileNotFoundError(phases_path)
    return pd.read_csv(cs_path), pd.read_csv(phases_path)

# ========== CAPACITY FADE / SOH PLOTS =========
def plot_capacity_fade(cs: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    soh_col = 'SOH_pct_smoothed' if 'SOH_pct_smoothed' in cs.columns else ('SOH_pct' if 'SOH_pct' in cs.columns else None)
    for cell, d in cs.groupby('cell_id'):
        plt.figure(figsize=(8,4))
        if 'is_capacity_test' in d.columns and d['is_capacity_test'].any() and 'QDischarge_mAh' in d.columns:
            dct = d[d['is_capacity_test']]
            plt.plot(dct['cycleNumber'], dct['QDischarge_mAh'], 'o', ms=3, label='Capacity tests (mAh)')
            plt.ylabel("Capacity (mAh)")
        if soh_col:
            ax2 = plt.gca().twinx()
            ax2.plot(d['cycleNumber'], d[soh_col], '-', lw=1.5, alpha=0.9, label='SOH (%)')
            ax2.axhline(EOL_PCT, ls='--', lw=1)
            ax2.set_ylabel("SOH (%)")
        plt.title(f"{cell} — Capacity fade / SOH")
        plt.xlabel("Cycle"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(outdir / f"{cell}_capacity_fade.png"); plt.close()

# ======= PHASE STRESS (C, Temp, Energy) =======
def summarize_phase_stress(phases: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    df = phases.copy()
    if 'duration_s' in df.columns:
        df = df[df['duration_s'] >= SMALL_SEG_SEC]

    stats = (df.groupby('phase_name')
               .agg(n=('seg_id','count'),
                    median_dur_s=('duration_s','median'),
                    median_energy_Wh=('energy_Wh','median'),
                    mean_mean_C=('mean_C_rate','mean'),
                    mean_max_T=('max_temp_C','mean'))
               .reset_index()
               .sort_values('n', ascending=False))
    stats.to_csv(outdir / "phase_stress_summary.csv", index=False)

    def barplot(x, y, fname, ylabel):
        plt.figure(figsize=(9,4))
        plt.bar(x, y)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel)
        plt.title(fname.replace('_',' '))
        plt.tight_layout()
        plt.savefig(outdir / f"{fname}.png")
        plt.close()

    # pass both filename and ylabel
    barplot(stats['phase_name'], stats['median_dur_s'],
            "median_duration_by_phase", "Median duration (s)")
    barplot(stats['phase_name'], stats['median_energy_Wh'],
            "median_energy_by_phase", "Median energy (Wh)")
    barplot(stats['phase_name'], stats['mean_mean_C'],
            "mean_C_rate_by_phase", "Mean C-rate")
    barplot(stats['phase_name'], stats['mean_max_T'],
            "mean_max_temp_by_phase", "Mean of max temp (°C)")


# ======= MISSION TABLE & ΔSOH CORR =============
def build_mission_table(phases: pd.DataFrame) -> pd.DataFrame:
    f = phases.copy()
    f = f[f['mission_id'].notna()]
    def sum_cond(s, mask): return s[mask.loc[s.index]].sum()
    fam = f[['phase_family']].copy()
    agg = (f.groupby(['cell_id','mission_id'])
            .agg(
                n_segments=('seg_id','count'),
                energy_flight_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('flight'))),
                energy_charge_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('charge'))),
                energy_rest_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('rest'))),
                energy_transition_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('transition'))),
                dur_takeoff_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('Take-off'))),
                dur_cruise_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('Cruise'))),
                dur_landing_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('Landing'))),
                dur_cc_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('CC'))),
                dur_cv_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('CV'))),
                dur_rest1_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('Rest1 (post-landing)'))),
                dur_rest2_s=('duration_s', lambda s: sum_cond(s, f.loc[s.index,'phase_name'].eq('Rest2 (pre-charge')) if 'Rest2 (pre-charge' in f['phase_name'].unique() else 0),
                mean_temp_C=('mean_temp_C','mean'),
                max_temp_C=('max_temp_C','max'),
                mean_C_rate=('mean_C_rate','mean'),
                max_C_rate=('max_C_rate','max'),
                SOH_end_pct=('SOH_mission_end_pct','max')
            ).reset_index())
    agg['SOH_next_pct']  = agg.groupby('cell_id')['SOH_end_pct'].shift(-1)
    agg['dSOH_drop_pct'] = (agg['SOH_end_pct'] - agg['SOH_next_pct']).clip(lower=0)
    agg['flight_Wh_abs'] = agg['energy_flight_Wh'].abs()
    return agg

def correlate_features_to_dSOH(missions: pd.DataFrame, outdir: Path):
    ensure_dir(outdir)
    feats = ['flight_Wh_abs','dur_cruise_s','max_temp_C','mean_C_rate','max_C_rate','dur_cc_s','dur_cv_s']
    df = missions.dropna(subset=['dSOH_drop_pct']).copy()
    rows = []
    for c in feats:
        if c not in df.columns: continue
        s = df[c]
        if s.notna().sum() < 5:
            rows.append({'feature': c, 'spearman_r': np.nan, 'n': int(s.notna().sum())}); continue
        r = pd.Series(s).rank().corr(pd.Series(df['dSOH_drop_pct']).rank())
        rows.append({'feature': c, 'spearman_r': float(r), 'n': int(s.notna().sum())})
        plt.figure(figsize=(4,3))
        plt.scatter(df[c], df['dSOH_drop_pct'], s=10, alpha=0.6)
        plt.xlabel(c); plt.ylabel("ΔSOH to next mission (%)"); plt.title(f"Spearman≈{r:.2f}")
        plt.tight_layout(); plt.savefig(outdir / f"scatter_{c}_vs_dSOH.png"); plt.close()
    pd.DataFrame(rows).to_csv(outdir / "dSOH_correlations.csv", index=False)

# ======= PHASE DEGRADATION FEATURES (ALL PHASES) =======
def compute_phase_degradation_features(phases: pd.DataFrame):
    df = phases.copy()

    # Thermal slope proxy & normalized thermal gain (all segments)
    # slope ≈ (max - mean) / duration; for stability, require duration>0
    dur = np.maximum(df['duration_s'].values, 1.0)
    df['thermal_slope_K_per_s'] = (df['max_temp_C'] - df['mean_temp_C']) / dur

    # Normalize by power when meaningful (>= POWER_MIN_FOR_GAIN)
    denom = df['mean_abs_power_W'].copy()
    df['thermal_gain_K_per_W'] = np.where(denom >= POWER_MIN_FOR_GAIN,
                                          df['thermal_slope_K_per_s'] / denom.replace(0, np.nan),
                                          np.nan)

    # Per-phase output
    phase_deg = df[['cell_id','mission_id','cycleNumber','seg_id','phase_name','phase_family',
                    'duration_s','energy_Wh','delta_SOC','mean_abs_power_W',
                    'mean_temp_C','max_temp_C','mean_C_rate','max_C_rate',
                    'thermal_slope_K_per_s','thermal_gain_K_per_W']].copy()

    # Mission-level charge acceptance & efficiency
    is_cc = df['phase_name'].eq('CC'); is_cv = df['phase_name'].eq('CV')
    cc = (df[is_cc].groupby(['cell_id','mission_id'], dropna=True)
              .agg(cc_duration_s=('duration_s','sum'),
                   cc_energy_Wh=('energy_Wh','sum'),
                   cc_mean_C=('mean_C_rate','mean')).reset_index())
    cv = (df[is_cv].groupby(['cell_id','mission_id'], dropna=True)
              .agg(cv_duration_s=('duration_s','sum'),
                   cv_energy_Wh=('energy_Wh','sum'),
                   cv_mean_C=('mean_C_rate','mean')).reset_index())
    cccv = cc.merge(cv, on=['cell_id','mission_id'], how='outer')
    cccv['cv_fraction'] = cccv['cv_duration_s'] / (cccv['cc_duration_s'] + cccv['cv_duration_s'])
    cccv['cc_fraction'] = cccv['cc_duration_s'] / (cccv['cc_duration_s'] + cccv['cv_duration_s'])
    cccv['cv_mean_I_A_proxy'] = (cccv['cv_energy_Wh'] / CV_VOLTAGE_ASSUMPTION) / (cccv['cv_duration_s']/3600.0)
    cccv.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Family energies & durations (flight/charge/rest/transition)
    fam = df[['phase_family']].copy()
    def sum_cond(s, mask): return s[mask.loc[s.index]].sum()
    family = (df.groupby(['cell_id','mission_id'])
                .agg(energy_flight_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('flight'))),
                     energy_charge_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('charge'))),
                     energy_rest_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('rest'))),
                     energy_transition_Wh=('energy_Wh', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('transition'))),
                     dur_flight_s=('duration_s', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('flight'))),
                     dur_charge_s=('duration_s', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('charge'))),
                     dur_rest_s=('duration_s', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('rest'))),
                     dur_transition_s=('duration_s', lambda s: sum_cond(s, fam.loc[s.index,'phase_family'].eq('transition'))))
                .reset_index())
    family['eff_energy'] = family['energy_flight_Wh'].abs() / family['energy_charge_Wh'].replace(0, np.nan)

    # Merge and attach mission labels if present
    mission_deg = cccv.merge(family, on=['cell_id','mission_id'], how='outer')
    for label in ['SOH_mission_end_pct', 'RUL_missions_after_phase', 'RUL_missions_censored']:
        if label in df.columns:
            lab = df.groupby(['cell_id','mission_id'])[label].max().reset_index()
            mission_deg = mission_deg.merge(lab, on=['cell_id','mission_id'], how='left')

    return phase_deg, mission_deg

# ======= PLOTS: ALL PHASES =====================
def _safe_unique(seq, maxn=PLOT_MAX_PHASES):
    xs = pd.Series(seq).dropna().unique().tolist()
    return xs[:maxn]

def plot_degradation_all_phases(phase_deg: pd.DataFrame, mission_deg: pd.DataFrame, outdir: Path):
    plots_dir = outdir / "plots_all"
    ensure_dir(plots_dir)

    # --- Global distributions by phase (duration, energy, ΔSOC, thermal)
    phases = _safe_unique(phase_deg['phase_name'])
    for feat, ylabel, fname in [
        ('duration_s','Duration (s)','duration'),
        ('energy_Wh','Energy (Wh)','energy'),
        ('delta_SOC','ΔSOC (pp)','delta_soc'),
        ('thermal_slope_K_per_s','Thermal slope (K/s)','thermal_slope'),
        ('thermal_gain_K_per_W','Thermal gain (K/W)','thermal_gain'),
    ]:
        plt.figure(figsize=(10,5))
        data = [phase_deg.loc[phase_deg['phase_name']==ph, feat].dropna().values for ph in phases]
        plt.boxplot(data, labels=phases, showfliers=False)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel)
        plt.title(f"{fname.replace('_',' ').title()} — by phase")
        plt.tight_layout(); plt.savefig(plots_dir / f"{fname}_box_by_phase.png"); plt.close()

    # --- Per-cell mission trends for each phase: median thermal gain & ΔSOC
    has_missions = phase_deg['mission_id'].notna().any()
    if has_missions:
        for cell, dcell in phase_deg.dropna(subset=['mission_id']).groupby('cell_id'):
            for ph in phases:
                sub = dcell[dcell['phase_name']==ph].sort_values('mission_id')
                if sub.empty: continue
                # median per mission for stability
                m = (sub.groupby('mission_id')
                        .agg(med_gain=('thermal_gain_K_per_W','median'),
                             med_dsoc=('delta_SOC','median'),
                             n=('seg_id','count'))
                        .reset_index())
                if m['med_gain'].notna().sum() > 0:
                    plt.figure(figsize=(8,3))
                    plt.plot(m['mission_id'], m['med_gain'], '-o', ms=3)
                    plt.xlabel("Mission"); plt.ylabel("Median thermal gain (K/W)")
                    plt.title(f"{cell} — {ph} — thermal gain over missions")
                    plt.tight_layout(); plt.savefig(plots_dir / f"{cell}_{ph}_trend_thermal_gain.png"); plt.close()
                if m['med_dsoc'].notna().sum() > 0:
                    plt.figure(figsize=(8,3))
                    plt.plot(m['mission_id'], m['med_dsoc'], '-o', ms=3)
                    plt.xlabel("Mission"); plt.ylabel("Median ΔSOC (pp)")
                    plt.title(f"{cell} — {ph} — ΔSOC per mission")
                    plt.tight_layout(); plt.savefig(plots_dir / f"{cell}_{ph}_trend_deltaSOC.png"); plt.close()

    # --- Charge acceptance plots (CV/CC) per cell
    if {'cv_fraction','cv_duration_s','cc_duration_s','cv_mean_I_A_proxy','mission_id'}.issubset(mission_deg.columns):
        for cell, d in mission_deg.dropna(subset=['mission_id']).groupby('cell_id'):
            d = d.sort_values('mission_id')
            if d['cv_fraction'].notna().sum():
                plt.figure(figsize=(8,3))
                plt.plot(d['mission_id'], d['cv_fraction'], '-o', ms=3)
                plt.xlabel("Mission"); plt.ylabel("CV fraction")
                plt.title(f"{cell} — CV fraction over missions")
                plt.tight_layout(); plt.savefig(plots_dir / f"{cell}_cv_fraction_trend.png"); plt.close()
            if d['cc_duration_s'].notna().sum() or d['cv_duration_s'].notna().sum():
                plt.figure(figsize=(8,3))
                if d['cc_duration_s'].notna().sum():
                    plt.plot(d['mission_id'], d['cc_duration_s'], '-o', ms=3, label='CC')
                if d['cv_duration_s'].notna().sum():
                    plt.plot(d['mission_id'], d['cv_duration_s'], '-o', ms=3, label='CV')
                plt.xlabel("Mission"); plt.ylabel("Duration (s)"); plt.legend()
                plt.title(f"{cell} — CC/CV durations")
                plt.tight_layout(); plt.savefig(plots_dir / f"{cell}_cc_cv_durations_trend.png"); plt.close()
            if d['cv_mean_I_A_proxy'].notna().sum():
                plt.figure(figsize=(8,3))
                plt.plot(d['mission_id'], d['cv_mean_I_A_proxy'], '-o', ms=3)
                plt.xlabel("Mission"); plt.ylabel("CV mean I (proxy, A)")
                plt.title(f"{cell} — CV current proxy over missions")
                plt.tight_layout(); plt.savefig(plots_dir / f"{cell}_cv_current_proxy_trend.png"); plt.close()

    # --- Family-level energy efficiency trend (includes charge/rest/transition)
    if {'eff_energy','mission_id'}.issubset(mission_deg.columns):
        for cell, d in mission_deg.dropna(subset=['mission_id']).groupby('cell_id'):
            d = d.sort_values('mission_id')
            if d['eff_energy'].notna().sum():
                plt.figure(figsize=(8,3))
                plt.plot(d['mission_id'], d['eff_energy'], '-o', ms=3)
                plt.xlabel("Mission"); plt.ylabel("|Flight Wh| / Charge Wh")
                plt.title(f"{cell} — Energy efficiency over missions")
                plt.tight_layout(); plt.savefig(plots_dir / f"{cell}_energy_efficiency_trend.png"); plt.close()

    # --- Global histograms for CC/CV/rest/transition durations and energies
    for fam_col, title_prefix in [('dur_charge_s','Charge'), ('dur_rest_s','Rest'), ('dur_transition_s','Transition')]:
        if fam_col in mission_deg.columns and mission_deg[fam_col].notna().any():
            plt.figure(figsize=(6,3))
            plt.hist(mission_deg[fam_col].dropna().values, bins=20)
            plt.xlabel("Seconds"); plt.ylabel("Missions")
            plt.title(f"{title_prefix} duration distribution")
            plt.tight_layout(); plt.savefig(plots_dir / f"{title_prefix.lower()}_duration_hist_all.png"); plt.close()

    for fam_energy, title_prefix in [('energy_charge_Wh','Charge'), ('energy_rest_Wh','Rest'), ('energy_transition_Wh','Transition')]:
        if fam_energy in mission_deg.columns and mission_deg[fam_energy].notna().any():
            plt.figure(figsize=(6,3))
            plt.hist(mission_deg[fam_energy].dropna().values, bins=20)
            plt.xlabel("Wh"); plt.ylabel("Missions")
            plt.title(f"{title_prefix} energy distribution")
            plt.tight_layout(); plt.savefig(plots_dir / f"{title_prefix.lower()}_energy_hist_all.png"); plt.close()

# =================== MAIN ======================
def main():
    setup_logger()

    # ===== EDIT THESE PATHS =====
    PROCESSED_DIR = Path(r"E:\BATTERY\processed_day1")
    EDA_DIR       = PROCESSED_DIR / "eda"
    # ============================

    ensure_dir(EDA_DIR)

    # 1) Load inputs
    cs, phases = load_inputs(PROCESSED_DIR)

    # 2) Capacity fade / SOH
    plot_capacity_fade(cs, EDA_DIR / "capacity")

    # 3) Phase stress summaries (all phases)
    summarize_phase_stress(phases, EDA_DIR / "phase_stress")

    # 4) Mission table & ΔSOH correlations
    missions = build_mission_table(phases)
    missions.to_csv(EDA_DIR / "missions_table.csv", index=False)
    correlate_features_to_dSOH(missions, EDA_DIR / "correlations")

    # 5) Phase degradation features (ALL phases) + mission-level features
    phase_deg, mission_deg = compute_phase_degradation_features(phases)
    deg_dir = EDA_DIR / "degradation"
    ensure_dir(deg_dir)
    phase_deg.to_csv(deg_dir / "phase_degradation_features_phase.csv", index=False)
    mission_deg.to_csv(deg_dir / "phase_degradation_features_mission.csv", index=False)

    # 6) Degradation plots for ALL phases (dists + per-cell trends)
    plot_degradation_all_phases(phase_deg, mission_deg, deg_dir)

    logging.info(f"EDA complete. Outputs in: {EDA_DIR}")

if __name__ == "__main__":
    main()