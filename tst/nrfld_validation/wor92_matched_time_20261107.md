# WOR-92: same-state matched-time CFL sensitivity

The clean Tomida-family table EoS + Semenov opacity pilot can advance at CFL=0.3, but a same-checkpoint comparison shows material time-step sensitivity. These short local tests start from the identical saved cycle-100 restart (`t=2266.625011212` code units), use the same 2-rank 120-block AMR topology and isolated binary, and end at exactly `t=2436.7392236433016`. The input has the flux-form/mixed-frame solver settings, general-EoS exact local gas root, accepted-state opacity-domain guard, and opt-in mean-preserving radiation ghost method. The only intended runtime change across the three trajectories is `time/cfl_number`.

## Results

| CFL | End cycle | Newton converged | End ρmax (g/cm³) | End Tgas,max (K) | End gas internal energy (erg) | End radiation energy (erg) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 120 | 20/20 | 4.73882e-14 | 20.0534 | 2.08131e42 | 2.52441e42 |
| 0.15 | 140 | 40/40 | 4.75471e-14 | 17.8022 | 1.89224e42 | 1.86239e42 |
| 0.075 | 176 | 76/76 | 4.76434e-14 | 17.0738 | 1.85049e42 | 1.88670e42 |

Relative to CFL=0.3, CFL=0.15 differs by **0.335%** in maximum density, **11.23%** in maximum gas temperature, **9.08%** in total gas internal energy, and **26.22%** in total radiation energy. CFL=0.075 differs by 0.539%, 14.86%, 11.09%, and 25.26% respectively. The 0.15→0.075 result is closer for gas temperature and internal energy than the 0.3→0.15 increment, but radiation energy is nonmonotone; these three points do not establish an asymptotic order.

All three runs converge every Newton stage in at most three iterations. Their FGMRES true relative residuals stay below the strict `2e-8` target (maxima 1.970e-8, 1.996e-8, and 1.991e-8). Opacity diagnostics report zero out-of-table/clamped trial lookups; accepted active T stays at least 14.252 K. There is no AMR regrid and no ghost limiter activation. Thus neither a failed inner solve, a 5 K opacity table clamp, nor a changed AMR topology explains this particular matched-time difference.

A further CFL=0.3 replay from the same checkpoint disables both `ATHENA_WOR92_INEXACT_TRUE_CAP` and `ATHENA_WOR92_POSITIVE_NR_GHOST`. It reaches cycle120 with **byte-identical history** and identical cycle, Newton, local-gas, and FGMRES numerical traces (after excluding the intentionally changed gate-status fields). All 60 FGMRES solves meet the strict target. This directly excludes those two opt-in mechanisms as the source of the late matched-time difference.

To check whether the maximum-temperature location alone caused the difference, the CFL=0.3 final restart was reopened with `time/tlim` equal to its saved time, producing a final-state HDF5 dump without advancing a step. Its full fields and the two CFL controls have identical block levels/logical locations. Across all 120 blocks and 8³ cells each, relative L2 differences against CFL=0.3 are:

| Field | CFL=0.15 | CFL=0.075 | 0.15 versus 0.075 |
|---|---:|---:|---:|
| density | 0.353% | 0.572% | 0.220% |
| pressure | 10.93% | 14.40% | 3.899% |
| radiation energy | 26.22% | 25.27% | 1.307% |

These full-field differences show that the result is spatially distributed. The HDF5 primitive fields are stored in float32, while the history values have limited printed precision; the differences are much larger than either limit. The angular-momentum history is also sensitive and changes sign at CFL=0.075, so a simple relative percentage for that nearly canceling integral is not useful.

An immediate no-step dump of the starting cycle-100 checkpoint reproduces its source history values exactly (`Tgas,max=20.5938 K`, gas internal energy `2.08564e42 erg`, radiation energy `2.78408e42 erg`). The first completed step already differs strongly: CFL=0.3 at `t=2275.43` gives `Tgas,max=20.6337 K`, `Erad=2.77235e42 erg`; CFL=0.15 at `t=2271.65` gives `19.4249 K`, `2.47475e42 erg`; CFL=0.075 at `t=2269.14` gives `18.9552 K`, `2.29397e42 erg`. These first-step states are at different times, so they are a diagnostic of an early response and not a matched-time convergence estimate. They motivate checking whether a per-step source, boundary update, or splitting term is scaled consistently with dt.

**Decision:** CFL=0.3 is a workable short solver pilot, but this comparison blocks a claim that its table collapse trajectory is temporally converged. The next focused test is a shorter matched-time sequence from an earlier checkpoint, plus an audit of explicit transport/mixed-frame energy terms and the time integration splitting. First-core collapse and post-regrid opacity-domain audit remain open; WOR-92 stays **In Progress**.

## Reproduction and evidence

Run `python3 tst/nrfld_validation/wor92_matched_time_20261107.py`. It checks the exact matched end time, Newton convergence, strict FGMRES residuals, opacity diagnostics, identical mesh topology, the starting state, and the strict/no-ghost control, then writes [JSON](wor92_matched_time_results_20261107/summary.json), [case CSV](wor92_matched_time_results_20261107/cases.csv), [field CSV](wor92_matched_time_results_20261107/fields.csv), [trajectory CSV](wor92_matched_time_results_20261107/trajectories.csv), [plot](wor92_matched_time_results_20261107/matched_time.png), and a [compressed archive](wor92_matched_time_results_20261107/evidence.tar.gz) with full rank-combined logs/history and initial plus three final primitive HDF5 states. SHA-256 values for the source checkpoint, binary, input, logs, histories, restarts, and state dumps are in JSON/CSV. The binary hash is `0e6af159f3c545d765667ab45a3b7feaad7a3ddbb0c05e8897633822c86f5c1b`; the source is branch `codex/wor92-domain-guard-trial-20261107`, commits `6da81d20`, `ff632195`, `14633c31`.

From `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, use a fresh output directory for each CFL value:

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7 \
  ATHENA_WOR92_POSITIVE_NR_GHOST=2 \
  timeout 300s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena \
  -r /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/table_domain_mean_limiter_restart74_to100_2rank/cfl030_flux_precision16_local.final.rst \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_domain_guard_cfl030_restart.used \
  -d /path/to/fresh/output time/cfl_number=0.15 \
  time/nlim=1000 time/tlim=2436.7392236433016
```

Repeat at `0.075`; the CFL=0.3 continuation uses `time/nlim=120` as in the [long continuation report](wor92_long_continuation_20261107.md). For the CFL=0.3 HDF5 state dump, restart its cycle-120 `final.rst` with the same input/flags and `time/nlim=1000 time/tlim=2436.7392236433016` in a fresh directory. No XD2000 access or merge occurred.

For the strict/no-ghost control, repeat the CFL=0.3 `time/nlim=120` command from the cycle-100 restart in a fresh directory, omitting `ATHENA_WOR92_INEXACT_TRUE_CAP` and `ATHENA_WOR92_POSITIVE_NR_GHOST`. The archived strict-control log and history are compared by the script.
