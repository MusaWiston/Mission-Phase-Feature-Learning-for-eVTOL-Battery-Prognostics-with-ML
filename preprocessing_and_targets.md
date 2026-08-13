# Supplementary preprocessing and target-construction procedure

This document is the procedural specification for
`scripts/build_reproducible_pipeline.py`. All numerical thresholds are stored
in `config/pipeline.json`; changing one creates a different preprocessing
protocol and requires a clean model rerun.

The CMU data descriptor states that a reference-performance test (RPT) was run
at the start of a campaign and after each set of 50 missions. An RPT measures
capacity by discharging from 4.2 V to 2.5 V at C/5. The dataset README also
warns that `cycleNumber` is the raw tester counter and is not globally
accurate after source files were concatenated. The implementation therefore
does not group all equal raw counter values together.

Primary sources:

- [CMU eVTOL Battery Dataset](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830)
- [Bills et al., Scientific Data 10, 344 (2023)](https://www.nature.com/articles/s41597-023-02180-5)

## 1. Input contract and chronological reconstruction

Required columns are `time_s`, `Ecell_V`, `I_mA`, `QDischarge_mA_h`,
`QCharge_mA_h`, `Temperature__C`, `cycleNumber`, and `Ns`. Source row order is
chronological. Positive current denotes charge and negative current denotes
discharge, following the dataset convention. Missing or non-finite required
values fail the cell instead of being silently imputed, and `cycleNumber` and
`Ns` must be integer-like. Optional source energy counters are not used in any
derived feature.

A new `cycle_instance_id` begins when either:

1. the raw `cycleNumber` differs from the preceding row; or
2. `time_s` moves backwards, indicating a recording-clock reset.

Thus, two non-contiguous runs with the same raw counter remain distinct.
Within each cycle instance, a new `phase_run_id` begins whenever `Ns` changes.
Elapsed time is calculated within each phase run. The first row of every phase
contributes zero seconds, so the interval crossing a phase boundary is not
assigned to the new phase. Negative differences are forbidden, and differences
greater than 60 s are flagged and contribute zero to duration, energy, and
coulomb integration.

```text
cycle_instance_id <- 0
for row r in source row order:
    if first row OR r.cycleNumber != previous.cycleNumber
                 OR r.time_s < previous.time_s:
        cycle_instance_id <- cycle_instance_id + 1

    dt_raw <- r.time_s - previous time in this phase run
    dt <- 0 if first row of phase OR dt_raw > 60 s else dt_raw
    record whether a large gap was excluded
```

The checked-in phase map is:

| `Ns` | Phase |
|---:|---|
| 0 | CC charge |
| 1 | CV charge |
| 2 | CV-to-rest transition |
| 3 | Rest-to-take-off transition |
| 4 | Take-off |
| 5 | Cruise |
| 6 | Landing |
| 7 | Landing-to-rest transition |
| 8 | Post-landing rest |
| 9 | Pre-charge rest |

A mission cycle must contain the ordered subsequence 4, 5, 6. The complete
observed `Ns` sequence for every reconstructed cycle is written to
`cycle_index.csv`, so this mapping can be audited against each raw file.

## 2. Exact capacity-test detector

The RPT detector is protocol-based. It has no score and no automatic
"best-candidate" fallback. A reconstructed cycle is a valid capacity test only
when all four conditions below are true.

| Criterion | Checked-in rule |
|---|---|
| Reference current | At least 70% of valid discharge time is within 0.39-0.81 A, i.e., 0.6 A (C/5 of 3.0 Ah) with +/-35% tolerance |
| Full voltage range | First discharge voltage is at least 4.10 V, final/minimum discharge voltage is at most 2.55 V, and voltage span is at least 1.40 V |
| Plausible capacity | Within-cycle `QDischarge_mA_h` delta is 1.50-3.75 Ah |
| Sufficient duration | Valid discharge duration is at least 7200 s |

The available capacity is

\[
Q_j = \frac{\max(Q_{\mathrm{dis},j})-\min(Q_{\mathrm{dis},j})}{1000}
\quad [\mathrm{Ah}],
\]

not the maximum charge-capacity accumulator. Direct current integration is
also written as `integrated_discharge_ah_audit`, but it does not replace the
published discharge-capacity measurement. Every individual criterion is
written to `cycle_index.csv`. Incomplete RPTs remain invalid.

```text
for reconstructed cycle c:
    D <- samples with current < -0.05 A
    fraction_C5 <- sum(dt where 0.39 <= abs(current) <= 0.81) / sum(dt in D)
    Q <- (max(QDischarge_mA_h) - min(QDischarge_mA_h)) / 1000

    current_ok  <- fraction_C5 >= 0.70
    voltage_ok  <- start_voltage >= 4.10 AND min_voltage <= 2.55
                   AND voltage_span >= 1.40
    capacity_ok <- 1.50 <= Q <= 3.75
    duration_ok <- sum(dt in D) >= 7200

    is_capacity_test <- current_ok AND voltage_ok AND capacity_ok AND duration_ok
```

The number of missions between consecutive RPTs is compared with the expected
50 +/- 5 and reported as an audit flag. Cadence is not used to force a cycle
to become an RPT, because the source documentation lists missing, extra, and
interrupted capacity tests.

## 3. Mission numbering and phase features

Capacity tests are excluded from missions. Remaining cycle instances that
contain the ordered take-off/cruise/landing subsequence receive consecutive
one-based `mission_id` values. Incomplete and non-mission cycles remain in the
cycle audit but are excluded from model tables.

Phase duration and energy are calculated only from valid within-phase sample
intervals:

\[
\Delta t_s=\sum_i dt_i, \qquad
E_s=\sum_i V_i I_i dt_i/3600.
\]

The code then computes current/C-rate summaries, temperature summaries,
thermal response, voltage-current slope, and voltage-binned incremental-
capacity descriptors. C-rate is normalised by the first valid RPT capacity,
which is known at beginning of life; interpolated future capacity is never used
as a predictor. Phase features are aggregated into one chronological mission
row using the formulas in `evt_battery/features.py`. Observed SOC change and
all health/RUL context are written only to `phase_targets.csv` or
`mission_targets.csv`; the canonical feature tables contain predictors only.

## 4. SOH labels

The first chronological valid RPT is the beginning-of-life reference:

\[
SOH_j = 100\,Q_j/Q_0.
\]

Each RPT is positioned by the number of complete missions observed before it.
SOH at mission end is linearly interpolated only between bracketing RPTs.
There is no forward fill, backward fill, or extrapolation. Missions after the
last RPT have `SOH_end_pct = NaN` and are not supervised health samples.

If two valid RPTs occur with no intervening mission, their median is used as
the interpolation anchor. At the first anchor, the first chronological RPT is
retained so beginning-of-life SOH is exactly 100%.

For the checked-in horizon `K = 5`, the SOH target at mission `m` is

\[
y^{SOH}_m = \max\{0, SOH_m-SOH_{m+5}\}.
\]

Both SOH values must be available; otherwise no sample is created.

## 5. Multi-threshold RUL labels and censoring

For each threshold `theta` in 90%, 85%, and 80%, the event mission is

\[
m_{EOL,\theta}=\min\{m:SOH_m\leq\theta\}.
\]

When this event is observed within the RPT-supported label range,

\[
RUL_\theta(m)=\max\{0,m_{EOL,\theta}-m\}.
\]

When the threshold is never observed, `RUL_theta_missions` is missing and
`RUL_theta_censored` is true. The last observed mission is never substituted
for an unobserved EOL event. This distinction prevents a right-censored cell
from being trained as though failure occurred at data collection end.

## 6. SOC labels

Each mission begins fully charged under the published protocol. Telemetry SOC
is coulomb-counted using the mission-specific interpolated capacity:

\[
SOC_i=\operatorname{clip}_{[0,100]}\left(100+
100\frac{\sum_{k\le i} I_kdt_k/3600}{Q_m}\right).
\]

For a phase, `dis_dSOC = SOC_end - SOC_start`. Capacity and SOH used for this
calculation are target-construction data and are excluded from the predictor
allow-list. Its mission aggregate, `dis_dSOC_sum`, is also labelled as target
context and cannot enter the SOH or RUL predictor matrix. The separate
`dis_dSOC_baseline` uses beginning-of-life capacity and remains an admissible
physics-based predictor.

## 7. Sliding-window samples

All windows use stride 1, left padding, a binary validity mask, and at least
one observed time step. No window can cross a cell boundary; SOC windows also
cannot cross a mission boundary.

| Task | Input ending at prediction time | Target | Maximum length |
|---|---|---|---:|
| SOC | Retained flight-phase features through phase `t` within one mission | Observed `dis_dSOC` at retained flight phase `t+1` | 16 phases |
| SOH | Mission features through mission `m` | Non-negative SOH drop from `m` to `m+5` | 20 missions |
| RUL | Mission features through mission `m` | Event-observed RUL at `m`, separately for 90%, 85%, and 80% | 20 missions |

For SOC, a group of length `n` creates at most `n-1` samples:

```text
for input_end in 0 .. n-2:
    start <- max(0, input_end - 16 + 1)
    X <- phase_features[start : input_end + 1]
    y <- dis_dSOC[input_end + 1]
```

For SOH and RUL, a mission at position `m` creates one sample when its target
is finite and event-observed. The window is the latest 20 mission rows ending
at `m`. Target columns are rejected if supplied in the predictor list.

## 8. Leakage-free LOCO preprocessing

The outer test group is one complete cell. Remaining cells are assigned to
three deterministic, sample-count-balanced inner folds; inner fold 0 is the
validation group and the other folds are training groups. Median imputation,
mean, and standard deviation are fitted only on valid training sequence rows.
The fitted values are then applied unchanged to validation and test cells.
Padded positions are set to zero after transformation.

## 9. Reproduction command and outputs

```bash
python scripts/build_reproducible_pipeline.py \
  --data-dir /path/to/cmu_evtol_v2 \
  --output-dir data/reproducible \
  --config config/pipeline.json
```

The output directory must be empty. This fail-fast rule prevents files from a
previous configuration or partial run from being mixed into a new manifest.

The command writes:

- `cycle_index.csv`: every reconstructed cycle and every RPT criterion;
- `capacity_tests.csv`: valid RPTs, positions, capacities, and SOH;
- `phase_features.csv` and `phase_targets.csv`;
- `mission_features.csv` and `mission_targets.csv`;
- `feature_dictionary.csv`: the exact predictor allow-list used for the run;
- `samples/*.npz`: padded values, masks, lengths, targets, IDs, and feature names;
- `loco_folds.csv`: outer test, inner validation, and training cell roles; and
- `manifest.json`: resolved configuration, input/output SHA-256 hashes, row
  counts, sample counts, processed cells, and any explicitly continued errors.

Before reporting new metrics, inspect `cycle_index.csv` and
`capacity_tests.csv` for every cell, confirm the phase map against raw `Ns`
sequences, and rerun every downstream model. The legacy archived features and
metrics were produced by a different preprocessing path and are not evidence
for the corrected pipeline.
