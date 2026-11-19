# WOR-174 solar top-face opacity reconstruction

Date: 2026-11-20 JST

## Scope

This change records the previously uncommitted WOR-91 upper-boundary
experiment as a solar-application option.  It does not change the general FLD
operator and is disabled by default:

```text
problem/reconstruct_top_face_opacity = false
```

At a physical global upper boundary, the option assumes `log(sigma_R)` is
linear between the two outermost cell centres, integrates that extrapolation
over the outer half cell, and supplies a positive ghost value whose arithmetic
average with the outer cell is that integral average whenever the implied
ghost is positive.  For an outward opacity decrease steeper than a positive
arithmetic ghost can represent, the implementation uses `TINY_NUMBER`; the
result then remains finite and positive but no longer exactly equals the
log-linear integral.  The arithmetic branch is the active branch of the
capped FLD face-opacity rule in the optically thin solar upper atmosphere.
Internal z-MeshBlock faces are not modified.  The mature A/B analysis must
therefore also check whether the selected solar state enters this fallback
regime before interpreting a flux change as exact half-cell reconstruction.

The Planck coefficient is deliberately unchanged: it is a cell-local
matter-radiation coupling coefficient rather than a face diffusion
coefficient.

## Provenance and isolation

- branch: `exp/wor174-solar-top-face-optin-20261120`
- base: `1193d9394bc002acb4410181e7e5b370de3f3441`
- worktree: `/home/kosuke/simulation/athena-wor174-solar-top-face-optin-20261120`
- results: `/home/kosuke/simulation/test/rad_nr/wor174_solar_top_face_optin_20261120`
- local validation binary SHA-256:
  `ca7c733f47a83448677497c92bb79090f8b272c43de0b6653725e2921723e865`

The existing dirty WOR-91 worktree was read for provenance only and was not
modified.

Input data SHA-256 values:

- NATA EOS: `981b6dad13d1ceb55352088dd15797bfa73d6a12408e3503c5c538a395c938fe`
- rho-T opacity: `cc5912dd3896c39c1ea3fde0d315dc16c136303a8c363ab58fa56b705b909aaf`
- initial profile: `22c4c592d8dc9d48c4208b62a87f28f43fc58dfcafaa9a837a3f952bba3400f2`

## Validation

### Pure reconstruction test

Command:

```sh
g++ -std=c++11 -Wall -Wextra -pedantic -O2 \
  tst/nrfld_validation/test_solar_top_face_opacity.cpp \
  -o /tmp/wor174_top_face_test
/tmp/wor174_top_face_test
```

PASS.  The test covers the constant-opacity limit, the analytical exponential
integral, a near-constant cancellation case, a steep-gradient floor, and
negative/non-finite inputs.

### Full application build

An isolated source copy was configured with
`simple_convection_lhllc_fld`, Cartesian coordinates, NATA table EOS, LHLLC,
MPI, HDF5, and NR-FLD.  `make -j4` completed without error.

### Real-solver A/B smoke test

The strongest local smoke case used 2 MPI ranks, a `32 x 32 x 16` mesh, and
`8 x 8 x 8` MeshBlocks.  It therefore contains two vertical MeshBlock layers
and exercises the distinction between internal and physical upper faces.
Each case ran two hydrodynamic timesteps.

| option | Newton | linear solves / FGMRES iterations | final top flux | wall | max RSS |
|---|---:|---:|---:|---:|---:|
| omitted (default) | 2/2 converged, 6 iterations | 6 / 70 | `6.27508e10` | 1.31 s | 85,448 KiB |
| explicit `false` | 2/2 converged, 6 iterations | 6 / 70 | `6.27508e10` | 1.24 s | 85,564 KiB |
| `true` | 2/2 converged, 6 iterations | 6 / 70 | `6.27977e10` | 1.26 s | 85,276 KiB |

The default and explicit-false history files are byte-for-byte identical.
The enabled case changes the intended top radiative flux by about 0.075%
without changing Newton or FGMRES iteration counts.  A separate 1-rank smoke
test also passed, with 2/2 Newton steps converged and the same default-off
bitwise check.

No non-finite opacity, table-domain violation, Newton failure, or FGMRES
breakdown was reported in any accepted smoke case.

### Mature-seed representability audit

The exact t=65000 restart selected for the long XD A/B has paired full-volume
primitive and user-output snapshots.  Re-evaluating the Rosseland table on
the outermost two active planes gives 4,096 top-face samples.  Of these, 331
(8.081%) imply a non-positive arithmetic ghost and therefore take the
positive-floor fallback.  The zero-ghost representability threshold is a log
opacity slope of approximately `-3.18725`; the nearest sampled slope is still
`0.00106` away from that threshold.  The median slope is `-2.39195`; the
range is `-3.67655` to `-1.76872`.  The fallback makes the represented face
opacity at most 1.09303 times the analytical log-linear target, while the
remaining 91.919% are exact to floating-point precision.

In contrast, the local t=0 smoke snapshots and the XD t=5.5 short A/B have
zero fallback columns.  The limitation is therefore exposed by the evolved
mature atmosphere and would not have been detected by repeating only the
startup regression.

This does not make the option non-finite or unstable, but it narrows the
claim: at the mature solar surface it is a positive approximation to the
half-cell integral in a minority of columns, not an exact reconstruction in
every column.  The long-run analyzer now repeats this count on the final
snapshot of both A/B branches so a flux change can be interpreted together
with any evolution of the fallback fraction.  Seed primitive/user snapshot
SHA-256 values are `558eab12672816caf8a3e5d4481407786b9158b4f27ce08a5f2fc9120e6f529c`
and `a88b970a33be161140f947d11c1e5bff915696ceb325231b80efaa8bd589baf8`.

### XD clean-source build and same-restart smoke

The option was also built in an isolated XD worktree at clean source commit
`f3bfffa4f31d9589ebb2660bed30ba2da02304a8`.  The executable SHA-256 is
`473b9b6ef5eaf5d30783ca9e3a7ea731a69eb4fb02dcb30f89ee781d27a87c8f`.
Build job `343649` completed `0:0` in 1:19.  The generated configuration was
checked against the validated solar build; the validated Intel-MPI Makefile
was used to avoid the current configure script's inappropriate
`-qopenmp-simd` annotation of the table-EOS exception path.

Job `343651` then ran enabled and disabled modes from the exact same restart
on 32 ranks, from code time 5.0 to 5.5.  Both paths completed `0:0`; all nine
Newton records per path converged (including the zero-dt initialization), all
16 active linear solves used exactly the same FGMRES iteration sequence, 142
iterations total and at most 11 per solve.  There was no fatal message,
opacity anomaly, or changed timestep history.

At the six-digit history precision, the enabled/disabled mean top-radiation
difference over this very early transient is only `-1.58e-6` relative and the
final values round to equality.  Float32 final HDF5 relative L2 differences
remain small but nonzero: `1.33e-10` in density, `7.00e-11` in pressure,
`1.68e-9` in radiation energy, and at most `1.30e-6` among velocity
components.  This is a stability/default-isolation smoke, not the physical
selection test.

The first mature-state submission, XD job `343667`, is excluded from physical
results.  Both concurrent steps stopped in `MPI_Init` with
`PMI2_Job_GetId returned 14`; Slurm reported exit `1:0` after 5 s and step
MaxRSS values of 13.1 and 13.2 MiB.  The launcher exported Intel MPI's PMI2
library but, unlike the passing single-step smoke, did not explicitly select
Slurm's `pmi2` plugin.  No timestep was attempted.

Commit `f6f65eab1` adds `srun --mpi=pmi2` and a launcher-only concurrent-step
gate.  M-test job `343701` ran two simultaneous 32-rank one-node Athena
invocations.  Athena initializes MPI before processing `-h`, so this tests the
failed path without advancing a physical simulation.  Both steps exited zero,
the job completed `0:0` in 4 s, and step MaxRSS was 52.3/54.7 MiB.  The long
A/B was then resubmitted as job `343705`.

The accepted mature-state same-restart A/B, job `343705`, starts at code time
65,000 from the selected high-entropy HD2 state and evolves enabled/disabled
paths concurrently to 66,000.  Slurm completed `0:0` in 30:17.  The disabled
and enabled steps took 20:24 and 30:15, respectively; their maximum per-rank
RSS was 105,180 and 95,020 KiB.  Both final restarts independently validate at
time 66,000.

The versioned 36-column clean-history analyzer rejects the two
advective-radiation columns that existed only in the historical dirty checkout,
and reports 100/500/1000-time windows, energy storage/closure, mass drift,
timestep, velocity RMS, Newton, and FGMRES work.  Its HTML, CSV, PNG, JSON, raw
logs, final full-volume snapshots, restart metadata, accounting, and file
hashes are under
`test/rad_nr/wor174_solar_top_face_optin_20261120/xd_long_343705`.

### Mature-state A/B result

| final window | disabled top radiation | enabled top radiation | disabled/enabled mass drift | enabled dt change | enabled Vz RMS change |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.12367 Fsun | 1.16746 Fsun | 0.3424% / 0.3593% | -16.24% | +177.36% |
| 500 | 0.12685 Fsun | 1.16483 Fsun | 0.3424% / 0.3593% | -16.55% | +174.37% |
| 1000 | 0.17652 Fsun | 1.16489 Fsun | 0.3424% / 0.3593% | -15.61% | +136.31% |

The enabled top flux remains close to 1.165 Fsun for the full 1,000-time
comparison, while the disabled atmosphere falls to approximately 0.124 Fsun
in the final 100-time window.  The modification therefore repairs the intended
upper-boundary radiative loss in this mature state; it is not merely a startup
or diagnostic difference.  The flow responds dynamically: the 1,000-time
mean RMS vertical velocity rises from 0.02525 to 0.05966 in code units, and the
bottom hydro flux changes with it.

All 13,571 disabled and 15,760 enabled Newton solves converged.  The maximum
reported final nonlinear L2 values were `6.50e-8` and `2.17e-8`.  There were
no FGMRES breakdowns, inexact accepts, opacity table-domain violations,
clamps, non-finite results, or fatal messages.  The enabled mode is more
expensive: Newton iterations rise from 30,508 to 48,983 (+60.6%), total
FGMRES iterations from 38,587 to 165,045 (4.28 times), and wall time by about
48.3%.  Individual FGMRES solves remain small (maximum 4 versus 5 iterations).

The final representability audit is also part of the acceptance boundary.  The
enabled atmosphere has 527/4,096 fallback columns (12.866%), up from 331/4,096
(8.081%) in the common seed.  The positive fallback represents a face opacity
up to 1.16656 times the analytical log-linear target.  The other 87.134% of
columns remain exact to floating-point precision.  Thus the selected option is
a stable and physically useful positive closure for this solar application,
but it is not an exact half-cell reconstruction in every mature surface
column.

The clean history's available energy ledger gives 1,000-time closure errors of
-0.2021 Fsun (disabled) and -0.2932 Fsun (enabled).  Because the two historical
advective-radiation boundary terms are not present in the clean 36-column
source, this comparison must not be promoted to a complete boundary-energy
conservation proof.  The like-for-like difference is -0.0911 Fsun and remains
a limitation for a future expanded diagnostic; it does not invalidate the
solver, restart, opacity, or mass checks above.

## Decision

PASS as an opt-in solar boundary closure, PASS for default-off regression, and
PASS for the mature same-restart production-selection gate.  The option is
recommended for the tested `simple_convection_lhllc_fld` solar setup because it
prevents the mature top flux from collapsing far below Fsun.  It must remain
application-scoped and default-off: the fallback is approximate in a minority
of mature columns, the flow state and cost change substantially, and this test
does not establish a complete clean-source boundary-energy ledger.  No general
FLD default, collapse source, or publication candidate is changed by this
decision.
