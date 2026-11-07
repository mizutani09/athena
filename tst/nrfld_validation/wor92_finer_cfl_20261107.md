# WOR-92: one more CFL halving at the background-reset stop

From the same cycle-100 checkpoint, three background treatments were advanced to the same short end time `t=2275.4309789960685` with CFL=0.3, 0.075, and a new 0.0375 control. The last value takes eight steps. All nine local 2-rank runs converge every Newton stage; all FGMRES true residuals meet `2e-8`, the opacity table reports zero outside/clamped lookup, and no AMR blocks are created or destroyed. The comparison is a short local sensitivity test, not a first-core or spatial-convergence claim.

| Treatment | CFL | Steps | Tgas,max (K) | Gas internal energy (10^42 erg) | Radiation energy (10^42 erg) | Background radiation-T error |
|---|---:|---:|---:|---:|---:|---:|
| Exact per-step reset | 0.3 | 1 | 20.6337 | 2.08936 | 2.77235 | 13.32% |
| Exact per-step reset | 0.075 | 4 | 17.9008 | 1.86636 | 1.91849 | 3.35% |
| Exact per-step reset | 0.0375 | 8 | 17.6627 | 1.83688 | 1.73783 | 0.83% |
| Reset off | 0.3 | 1 | 20.9965 | 2.17506 | 3.49846 | 20.10% |
| Reset off | 0.075 | 4 | 19.0989 | 2.09239 | 3.51758 | 20.27% |
| Reset off | 0.0375 | 8 | 19.0107 | 2.08962 | 3.52431 | 20.32% |
| Exploratory τ=8 relaxation | 0.3 | 1 | 20.7554 | 2.11915 | 3.01259 | 15.70% |
| Exploratory τ=8 relaxation | 0.075 | 4 | 18.6105 | 2.00379 | 2.81396 | 13.74% |
| Exploratory τ=8 relaxation | 0.0375 | 8 | 18.4885 | 1.99571 | 2.78164 | 13.41% |

From CFL=0.075→0.0375, maximum gas temperature changes by 1.33% with exact reset, 0.46% with reset off, and 0.66% with τ=8. Thus a substantial portion of the residual gas-temperature difference is ordinary time-step sensitivity, though its asymptotic order has not been established. Radiation energy changes over the same halving by **9.42%** with exact per-step reset, **0.19%** with reset off, and **1.15%** with τ=8. The direct [reset ledger](wor92_sink_removal_ledger_20261107.md) explains why radiation remains strongly dependent on the number of exact resets. The τ=8 trial improves numerical behavior but its approximately 13–16% background-temperature error and arbitrary timescale prevent adoption. Turning the reset off leaves approximately 20% background error.

The source for the τ=8 run remains an uncommitted isolated trial, archived in the [relaxation report](wor92_background_relax_20261107.md). No new production source change is made here. WOR-92 remains **In Progress** pending a physically specified background reservoir/boundary treatment, matched-time CFL and mesh validation over a longer trajectory, post-regrid opacity-domain checks, and first-core formation.

Run `python3 tst/nrfld_validation/wor92_finer_cfl_20261107.py` to regenerate the [JSON](wor92_finer_cfl_results_20261107/summary.json), [CSV](wor92_finer_cfl_results_20261107/cases.csv), [plot](wor92_finer_cfl_results_20261107/finer_cfl.png), and [full logs/history archive](wor92_finer_cfl_results_20261107/evidence.tar.gz). Binary, input, checkpoint, run log, history, and restart SHA-256 values are recorded. Use the [background reset](wor92_background_sink_20261107.md) and [relaxation](wor92_background_relax_20261107.md) commands with fresh output directories, `time/cfl_number=0.0375`, and `time/tlim=2275.4309789960685`. No XD2000 access or merge occurred.
