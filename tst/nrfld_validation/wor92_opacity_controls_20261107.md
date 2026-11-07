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

### Separate the two table components at CFL 0.3

An additional trial-only switch replaced **one** table-returned opacity
with its 17-digit initial constant, while still executing both table
lookups. The identical diagnostic binary and input were used for all four
controls; only the environment switch and MPI rank count varied. The
source-patch SHA-256 is
`ce9d7dbe7b7b6f54a17ea62a980930a0a9c3f8fe2f75e20f343827ee556155d3`,
and the binary SHA-256 is
`5b259bf3c0a4b768a6a35aa9f2d9cebfd3b9151f78b80ed5583069a772ffb691`.

| Table-derived component(s) | Fixed component | Ranks | Cycles / 3 | Max true residual |
| --- | --- | ---: | ---: | ---: |
| κP and κR | none | 2 | fails cycle 0 | 3.87966e-8 |
| κR | κP=0.21524823808128721 | 2 | fails cycle 0 | 4.78875e-8 |
| κP | κR=0.064784772977394392 | 2 | 3 strict | 1.99855e-8 |
| κP | same fixed κR | 4 | 3 strict | 1.95989e-8 |

The unfrozen diagnostic binary reproduced the earlier failure residual
exactly. Holding κR fixed was sufficient for this three-cycle strict
pass in both rank layouts, while holding κP fixed left the failure.
Thus the Rosseland **values** returned by the table are implicated in the
early `2e-8` threshold crossing under this controlled state. The two-rank
pass is very close to threshold and does not establish a robust
production setting or identify a defective opacity lookup. It remains
necessary to inspect the Schur matrix, coefficient scaling and
preconditioner with the actual table-derived κR field.

The [component JSON](wor92_opacity_controls_results_20261107/component_freeze.json),
[CSV](wor92_opacity_controls_results_20261107/component_freeze.csv),
[figure](wor92_opacity_controls_results_20261107/component_freeze.png),
and [logs/input/source patch](wor92_opacity_controls_results_20261107/component_freeze_evidence.tar.gz)
record this A/B test. The run command below adds either
`ATHENA_WOR92_FIXED_PLANCK=1` or `ATHENA_WOR92_FIXED_ROSSELAND=1` and uses
`input_table_component_freeze_cfl030_strict_3cycles.used`.

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

The next targeted experiment should compare Schur diagonal/off-diagonal
scaling, low-frequency error modes and MG correction quality for the
fixed-R versus table-R states. An assembled AMR-aware Jacobian/backward
error remains needed. Only a supported numerical cause should lead to
a minimal source change and regression suite. WOR-92 remains **In
Progress**; strict Semenov CFL=.3 and long collapse are unverified. No
XD2000 login or job submission occurred.
