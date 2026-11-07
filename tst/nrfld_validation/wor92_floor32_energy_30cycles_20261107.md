# WOR-92: bounded 30-cycle local-root / precision-floor extension

The Tomida table EoS + Semenov opacity trial at CFL=0.3 was extended from ten to 30 early collapse cycles, with the exact table local gas root and precision-floor factor 32. This is a bounded exploratory run, not a strict solver acceptance or first-core calculation. It used the same isolated baseline binary (SHA-256 `5b259bf3c0a4b768a6a35aa9f2d9cebfd3b9151f78b80ed5583069a772ffb691`), root `32^3`, 120 AMR leaf blocks, two MPI ranks, flux-form matvec, FGMRES restart 40 / MG2 / maximum 300, and mixed-frame transport. The input changed only `nlim=10` to `nlim=30` from the prior ledger run; input SHA-256 is `faa9139a2d1f85f8c0e14def7cfb60c5af6e937b67a9b888b00b7b9b0d4dda3d`. Source report branch started at Git `5e831ca09074ec461d25534c8aed19a4967a2e5e`, with the previously documented uncommitted trial solver source. No production source was changed for this extension.

| Measure | 30-cycle result |
| --- | ---: |
| Completed cycles / Newton convergences | 30 / 30 |
| FGMRES solves / precision-floor accepts | 62 / 21 |
| Largest FGMRES true relative residual | `6.21057134e-8` (strict target `2e-8`) |
| Largest Newton iteration count | 3 |
| Largest converged Newton final L2 | `5.30391857e-7` |
| Maximum area-weighted coarse-fine face mismatch, all cycles | `3.10830249e-7` |
| Maximum rank signed energy-balance fraction, all cycles | `1.30288670e-8` |
| Maximum rank absolute-term energy-balance fraction, all cycles | `2.18188535e-7` |
| Maximum weighted coarse-fine mismatch, cycles 20–29 | `1.06237860e-7` |
| Maximum signed energy-balance fraction, cycles 20–29 | `7.43738589e-9` |
| Final code time / maximum density / maximum gas temperature | `851.42` / `7.97031e-18 g/cm³` / `13.987 K` |

The first ten numerical FGMRES, Newton, and cycle lines are byte-identical to the previous ten-cycle run. The maximum coarse-fine mismatch and energy balance defect over all 30 cycles occurred in that first ten-cycle segment; the last ten showed no growth in these particular metrics. All 120 leaf blocks and 24 coarse-fine face groups were accounted for in each stage. The history file contains a duplicate final row, removed only for analysis. The maximum density is many orders of magnitude below the first-core regime; the result checks early-time behavior only. The 21 floor accepts are strict linear-convergence failures, so WOR-92 remains **In Progress**.

Artifacts: [per-stage CSV](wor92_floor32_energy_30cycles_results_20261107/stages.csv), [all paired faces CSV](wor92_floor32_energy_30cycles_results_20261107/faces.csv), [summary JSON](wor92_floor32_energy_30cycles_results_20261107/summary.json), [figure](wor92_floor32_energy_30cycles_results_20261107/floor32_energy.png), and [input, full log, both rank ledgers, history](wor92_floor32_energy_30cycles_results_20261107/evidence.tar.gz). Regenerate the audit from this worktree with:

```bash
WOR92_AUDIT_CYCLES=30 python3 tst/nrfld_validation/wor92_floor32_energy_audit_20261107.py
```

Exact run command from the table bundle directory, with a new output directory and ledger prefix:

```bash
cd /home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_NRFLD_ENERGY_LEDGER=1 \
  ATHENA_NRFLD_INTERFACE_LEDGER=1 \
  ATHENA_NRFLD_RANK_LEDGER_PREFIX=/path/to/new/output/ledger \
  timeout 300s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_precision/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_local_jacobian_floor32_cfl030_30cycles.used \
  -d /path/to/new/output
```

The next decisive checks remain exact assembled AMR componentwise backward error, conservative coarse-fine flux repair, a more effective Schur preconditioner/scaling, and a longer physical collapse comparison against a strictly converged reference. The [solver-control report](wor92_solver_controls_20261107.md) documents strict failures and rejected tuning attempts. XD2000 was not used.
