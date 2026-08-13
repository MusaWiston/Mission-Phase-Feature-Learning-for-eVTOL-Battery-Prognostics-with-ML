# Supplementary pseudocode: capacity-test and target construction

```text
INPUT: raw telemetry files for each cell; detector configuration
OUTPUT: audited capacity tests, mission SOH/RUL labels, chronological windows

FOR each cell file:
    resolve required column names
    coerce cycle, time, current, voltage, capacity to numeric
    sort rows by cycle, timestamp, and original row order

    FOR each raw cycle:
        charge_rows ← phase matches {charge, CC, CV} AND signed current > deadband
        flight_rows ← phase matches flight phases AND signed current < −deadband
        IF no phase labels match, classify using signed current

        full_charge ← median(last N charge voltages)
                      ≥ voltage_setpoint × minimum_fraction
        capacity ← maximum cumulative charge capacity among charge_rows
        chronological ← last(charge time) ≤ first(flight time)

        accept cycle iff:
            enough charge rows AND enough flight rows AND full_charge
            AND capacity within physical bounds AND chronological
        write all checks and rejection reasons to audit table

    tests ← accepted cycles ordered by cycle
    C0 ← capacity at first accepted test
    FOR each complete flight mission between first and last accepted test:
        Cm ← linear interpolation of test capacities at mission cycle
        SOHm ← Cm / C0

    FOR threshold θ in {0.90, 0.85, 0.80}:
        crossingθ ← first mission m where SOHm ≤ θ
        RULθ(m) ← max(crossingθ − m, 0), if crossing exists; otherwise missing

    degradationK(m) ← max(SOHm − SOH(m + K), 0), K = 5
    create rolling mission windows ending at m with maximum length 20

EXCLUDE VAH06, VAH07, VAH09 from model labels/windows, but retain their audits
SAVE configuration, resolved schemas, input hashes, and errors in run manifest
```

