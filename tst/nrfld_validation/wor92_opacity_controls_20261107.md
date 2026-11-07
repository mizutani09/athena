# WOR-92 opacity and EoS controls after the three-cycle pilot

2026-11-07 JST. This follow-up tests whether short local convergence
predicts later behavior and separates Planck coupling opacity from
Rosseland transport opacity. The purpose is numerical diagnosis; changing
opacity here does not select the physical opacity model for collapse.

## Reproducibility and controls

All runs used the isolated WOR-105 + WOR-92 trial source described in
`wor92_wor105_continuation_20261107.md`: base
`ebf7634a8e877edfd58724de1e3810638a76ca7e`, uncommitted trial diff
SHA-256 `6e507ec3ad350a0fb6a3d1e3f06a73ebaf4998f55c214df3593fb95f766d388e`.
The table and ideal binary SHA-256 values were
`cf1e1520ca23334a08615439d8c616b6e88551801ac64101cd6c454f63d3d27a`
and `ff9887c40d80c1a2a64a1a77bb104992298b917dd28921ca84c37f1db89877c0`.
No source patch was committed.

The 32³ root, 8³ blocks, mixed-level AMR, two MPI ranks, restart 40,
two MG V-cycles, 300 max FGMRES iterations, flux-form matvec, mixed frame,
exact local gas trial, and strict true-residual target `2e-8` were fixed.
Precision-floor acceptance was disabled. Only the indicated EOS/opacity,
CFL and bounded cycle count varied. The successful ideal gas run is a
sanity control; the table EoS runs isolate opacity settings.

At the initial core state `T=14 K`, `rho=5.251585416e-18 g/cm³`,
log-bilinear interpolation of the saved Semenov table gives
`κP=0.21524823808128721` and `κR=0.064784772977394392 cm²/g`.
`κ=1` is therefore a substantially different transport strength, and
must not be called a matched constant-opacity control.

| EOS / opacity | CFL | Bound | Last cycle | Linear solves | Max true residual | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Table / Semenov | .05 | 10 | 5 | 11 | 1.800e-5 | fails first solve of cycle 5 |
| Ideal / κP=1, κR=1 | .05 | 10 | 10 | 30 | 1.895e-8 | strict pass |
| Table / κP=1, κR=1 | .05 | 10 | 10 | 20 | 1.861e-8 | strict pass |
| Table / κP=.2152, κR=.06478 | .05 | 10 | 5 | 12 | 2.484e-3 | fails second solve of cycle 5 |
| Table / κP=1, κR=.06478 | .05 | 10 | 5 | 12 | 5.907e-4 | fails second solve of cycle 5 |
| Table / κP=.2152, κR=1 | .05 | 10 | 10 | 20 | 1.949e-8 | strict pass |
| Table / Semenov | .3 | 3 | 0 | 2 | 3.880e-8 | strict fail; floor would accept |
| Table / κP=1, κR=1 | .3 | 3 | 3 | 6 | 1.886e-8 | strict pass |
| Table / κP=1, κR=.06478 | .3 | 3 | 3 | 7 | 1.955e-8 | strict pass |
| Table / κP=.2152, κR=.06478 | .3 | 3 | 3 | 7 | 1.975e-8 | strict pass |
| Table / exact initial Semenov κP, κR fixed | .3 | 3 | 3 | 7 | 1.984e-8 | strict pass |

All completed runs had zero precision-floor acceptances and met the
configured Newton criterion. The failed runs stopped at 300 FGMRES
iterations; no automatic restart/retry was used. The `.05` table/Semenov
case completed five Newton solves before its sixth-cycle attempt
(`cycle=5`) failed. Its local-stencil backward-error proxy then rose to
`3.8243e-6`, unlike the near-roundoff proxy of the original `.3`
three-cycle failure. These are different numerical failure regimes.

The `.05` matrix shows that low **Rosseland** opacity alone is sufficient
to reproduce a later FGMRES failure in this setup; lowering Planck
opacity alone is not. The two constant-low-κR cases fail in the second
linear solve of cycle 5, while Semenov fails in the first, so their states
and failure trajectories are not identical. At `.3`, even the
17-digit initial-opacity constant control passes strictly for three
cycles while the Semenov table path does not. A spatially/time-varying
opacity field or the table update path matters for that precision-limited
early failure, but this experiment does not distinguish those mechanisms.
The small differences near the `2e-8` target may also be sensitive to
state changes; a full Jacobian/condition analysis remains necessary.

### Direct coefficient ranges at the later failure

A further isolated diagnostic patch recorded the first active-cell opacity
evaluation for every block in cycles 4 and 5, writing directly to one file
per rank. The source-patch SHA-256 is
`20082492ddd246ceaf60d7739d4cdad22d81eeaad9f745b15eec15af455bfb23`;
the table diagnostic binary SHA-256 is
`5b726ae497c67d97f4615cb459b1877b629a4d0ed125df43bb829eefe63a352f`.
The diagnostic repeat had the *same* cycle-5 failure residual
`1.8003263813e-5` as the original, so logging did not change this result.

| Cycle | Active-cell temperature (K) | κP range (cm²/g) | κR range (cm²/g) |
| ---: | ---: | ---: | ---: |
| 4 | 13.9524–13.9981 | 0.213882–0.215192 | 0.0643564–0.0647672 |
| 5 | 13.9584–13.9979 | 0.214054–0.215188 | 0.0644103–0.0647659 |

All 120 leaf blocks were represented in both cycles. These values show
that the failing state remains near 14 K, with κR variation under 0.6%
across active cells. The small variation does not prove it is harmless:
the `.3` three-cycle result remains sensitive to table versus fixed
opacity, while the `.05` longer failure also occurs with fixed low κR.
The [diagnostic JSON](wor92_opacity_controls_results_20261107/opacity_range_diagnostics.json)
and [rank logs/patch](wor92_opacity_controls_results_20261107/opacity_range_evidence.tar.gz)
contain the complete measurements and hashes.

These tests do not compare Semenov extinction to Optab absorption, and
do not make a physical-model selection from numerical convergence.

## Artifacts and next step

Run
`python3 tst/nrfld_validation/wor92_opacity_controls_20261107.py`
to regenerate [JSON](wor92_opacity_controls_results_20261107/summary.json),
[CSV](wor92_opacity_controls_results_20261107/runs.csv), and
[figure](wor92_opacity_controls_results_20261107/opacity_controls.png).
The [archive](wor92_opacity_controls_results_20261107/logs_inputs_tables.tar.gz)
contains the exact inputs, raw logs, exit codes, trial source diff and
table files. The JSON records every input/log/table hash. Example run from
the bundle directory:

```bash
cd /home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  timeout 300s mpirun -np 2 /path/to/isolated/table/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_low_rosseland_cfl005_strict_10cycles.used \
  -d /path/to/fresh/output
```

The next targeted experiment should record the actual κP/κR and diffusion
coefficient ranges at the failing Newton state, then freeze either Planck
or Rosseland table updates while retaining the other table. Compare the
resulting true residual, level/edge localization and assembled AMR-aware
Jacobian/backward error. Only a supported numerical cause should lead to
a minimal source change and regression suite. WOR-92 remains **In
Progress**; strict Semenov CFL=.3 and long collapse are unverified. No
XD2000 login or job submission occurred.
