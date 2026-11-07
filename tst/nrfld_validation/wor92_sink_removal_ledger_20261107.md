# WOR-92: measured per-step background energy reset

The [same-checkpoint CFL controls](wor92_background_sink_20261107.md) implicated the problem generator's per-step background radiation reset in the large temporal sensitivity. This isolated diagnostic branch measures the energy actually removed at the first reset. The opt-in instrumentation in `src/pgen/protostellar_core_fld.cpp` accumulates `(target - pre-reset state) × active-cell volume` for gas internal and radiation energy, converts to erg with the run's code units, and prints one line per MeshBlock at cycle100. It does not change the update.

All three runs use the same cycle-100 checkpoint, Tomida-family table EoS and Semenov opacity, 2 ranks, 120 blocks, and the current `fix_background_temperature=true`, `background_fix_radiation=true`, `background_sink_after_implicit=false` input. Each advances exactly one step with a fresh output directory.

| CFL | Actual first dt | Background cells reset | Gas energy removed (erg) | Radiation energy removed (erg) |
|---:|---:|---:|---:|---:|
| 0.3 | 8.80597 | 23,136 | 3.08007e40 | 7.81004e41 |
| 0.15 | 5.02957 | 23,184 | 3.08124e40 | 7.81448e41 |
| 0.075 | 2.51478 | 23,184 | 3.07194e40 | 7.80662e41 |

The radiation reset removes almost the **same ~7.81e41 erg per step** while dt changes by a factor of 3.5. This is about 28.1% of the checkpoint's `2.78408e42 erg` total radiation energy. The effective removal per unit code time therefore rises from `8.87e40` to `3.10e41 erg` as CFL falls from 0.3 to 0.075. The exact overwrite in `ApplyBackgroundHeatSink` has no dt factor, matching the measured behavior. Repeating it more often directly creates step-count dependence. The matched-time 1/2/4-step controls show a 30.8% radiation-energy difference under the current reset, versus 0.55% with the reset off; this is a supported numerical mechanism, not merely an association.

For each CFL, the diagnostic run's first-step physical history columns match the corresponding uninstrumented control exactly. CFL=0.15 has a different *next-step dt* history column because the uninstrumented control is stopped at an earlier `time/tlim`; the physical state columns are identical. Thus the ledger has no detected effect on these trajectories. Every run reaches the cycle101 limit with Newton convergence and no opacity-domain event. The block ledger prints are included in the archive.

This diagnosis does **not** authorize turning the reset off for production. Without it, the background tracer's temperature error rises to about 20% in the short control; resetting after the implicit solve pins the background to 14 K but retains significant CFL sensitivity. A physical reservoir/boundary model or a relaxation timescale must be chosen and validated with CFL/mesh comparisons. The gas maximum-temperature sensitivity remains partly unexplained. WOR-92 remains **In Progress**.

## Reproduction

The independent worktree is `codex/wor92-sink-ledger-trial-20261107` from base `873a1b4c3bda584b52e1cae2b41e333ddb2f793a`. Copy its `configure.py`, `Makefile.in`, and `src` to a new build directory, then run:

```bash
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -b -hdf5 -mpi --cflag=-O3
make -j4
```

Isolated diagnostic binary SHA-256: `e23da02cd35ba8cb93d0d5e4061f732b1b27fa550f6746e74ae7d2b0ce8140d0`. From `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, use a fresh `-d` directory and the common [WOR-92 run flags](wor92_background_sink_20261107.md), adding `ATHENA_WOR92_BACKGROUND_SINK_LEDGER=1`, `time/cfl_number=0.3` (then 0.15/0.075), `time/nlim=101 time/tlim=10000.0`, and the saved cycle-100 checkpoint as `-r`. No XD2000 access or merge occurred.

Regenerate [JSON](wor92_sink_removal_ledger_results_20261107/summary.json), [CSV](wor92_sink_removal_ledger_results_20261107/cases.csv), [figure](wor92_sink_removal_ledger_results_20261107/removal_vs_dt.png), and [full log/history archive](wor92_sink_removal_ledger_results_20261107/evidence.tar.gz) with:

```bash
python3 tst/nrfld_validation/wor92_sink_removal_ledger_20261107.py
```

The JSON/CSV give binary, input, checkpoint, log, history, and restart SHA-256 values and assert 120 unique block reports per run plus identical physical histories to the clean controls.
