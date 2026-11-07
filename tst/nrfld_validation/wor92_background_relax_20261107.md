# WOR-92: exploratory dt-consistent background relaxation

The [reset ledger](wor92_sink_removal_ledger_20261107.md) measured roughly `7.81e41 erg` removed from background radiation **per step**, nearly independent of dt. An isolated source patch replaces the exact per-step overwrite, when explicitly requested, by exponential relaxation toward the 14 K background state:

`new = old + [1 − exp(−dt/τ)] × (target − old)`.

`problem/background_relaxation_time_code=0` retains the original overwrite. The pilot applies the same factor to gas internal energy and radiation energy, then recomputes table-EoS pressure. Its τ values are **exploratory**, not calibrated reservoir physics. With the input's units, one code-time unit is 14.99 years, so τ=8 and 32 are about 120 and 480 years. The source change remains an uncommitted trial in a separate worktree; the [patch](wor92_background_relax_results_20261107/relaxation_trial.patch.gz) is archived here for review. No production configuration is adopted.

## Same-checkpoint controls

All runs use the same cycle-100 checkpoint, Tomida-family table EoS, Semenov opacity, 2 ranks, 120 AMR blocks, mixed-frame local gas root, true-residual inexact gate, and opacity accepted-state guard. They stop at exact matched times. The τ=0 short control has a **byte-identical history** to the original clean source at CFL=0.3. Every τ>0 run's Newton stages converge in at most three iterations; its FGMRES true residual remains below the strict `2e-8` target and opacity diagnostics show no table-outside/clamped lookup. No regrid or radiation ghost limiter activation occurs in these intervals.

| Interval | Treatment | CFL=0.3 Erad (10^42 erg) | CFL=0.075 Erad | Relative difference | Largest background-radiation T error |
|---|---|---:|---:|---:|---:|
| t=2266.625→2275.431 | Original exact overwrite | 2.77235 | 1.91849 | −30.80% | 13.32% (at CFL=0.3) |
| same short interval | τ=8 | 3.01259 | 2.81396 | −6.59% | 15.70% |
| same short interval | τ=32 | 3.32278 | 3.29027 | −0.98% | 18.56% |
| t=2266.625→2436.739 | Original exact overwrite | 2.52441 | 1.88670 | −25.26% | 10.50% (at CFL=0.3) |
| same long interval | τ=8 | 2.99884 | 2.60662 | −13.08% | 15.56% |

The τ=8 long control also reaches CFL=0.15: Erad=`2.73237e42 erg`, between the 0.3 and 0.075 values. At the long matched time, gas internal energy is `2.14140/2.01713/1.96761e42 erg` and maximum gas temperature `20.3605/18.4938/17.8086 K` for CFL=0.3/0.15/0.075. Thus the CFL=0.3→0.075 gas internal-energy difference is still **8.12%** and maximum-temperature difference **12.53%**. The τ=8 update reduces but does not resolve the temporal sensitivity. The larger τ=32 nearly removes the short radiation-energy difference while allowing roughly 19% background-temperature error, similar to leaving the reservoir weakly constrained.

The exact overwrite behaves like a strong background reservoir whose energy removal is set by the number of steps. Exponential relaxation gives a finite removal rate as dt decreases, which explains the improved short-interval radiation comparison. Its timescale, tracer geometry, gas/radiation coupling, and boundary energy flux need physical justification. The remaining gas-temperature and long-interval energy differences also require a transport/splitting and conservation audit before any first-core claim. WOR-92 remains **In Progress**.

## Reproduction and evidence

The report branch is `codex/wor92-sink-ledger-trial-20261107`, base source commit `7fd53c00bdba49095ff5ad13659b2b4c09e1451c`. Apply the archived patch to that source in a fresh worktree, copy `configure.py`, `Makefile.in`, and `src` into an isolated build directory, then configure and build as in the [ledger report](wor92_sink_removal_ledger_20261107.md). Trial binary SHA-256: `9834c90766919e895a5743f77658600bfc76d473d370a0af98e781e7b6a4d32c`. The patch is opt-in; do not apply it to main or a production branch without choosing τ physically and validating the trajectory.

The [input copy](wor92_background_relax_results_20261107/input_table_background_relax_trial.used) adds the parameter in the `<problem>` block; Athena++ requires that before it can be overridden on the command line. From `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, use the [matched-time command](wor92_matched_time_20261107.md) with this input, a fresh output directory, `problem/background_relaxation_time_code=8` (or `0`/`32`), and `time/tlim=2275.4309789960685` or `2436.7392236433016`. Keep the same source checkpoint and seven `ATHENA_NRFLD_*` flags, `ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7`, and `ATHENA_WOR92_POSITIVE_NR_GHOST=2`. Each run had a 300-second cap for the long controls; no failed calculation was retried without changing the setup.

Regenerate [JSON](wor92_background_relax_results_20261107/summary.json), [case CSV](wor92_background_relax_results_20261107/cases.csv), [comparison CSV](wor92_background_relax_results_20261107/comparisons.csv), [plot](wor92_background_relax_results_20261107/relaxation_controls.png), and [full log/history archive](wor92_background_relax_results_20261107/evidence.tar.gz) with:

```bash
python3 tst/nrfld_validation/wor92_background_relax_20261107.py
```

All input, checkpoint, binary, log, history, restart, and patch hashes are recorded in JSON/CSV. No XD2000 access or merge occurred.
