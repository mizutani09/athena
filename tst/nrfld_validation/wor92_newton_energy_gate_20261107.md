# WOR-92: cycle-92 energy defect is controlled by the Newton stopping gate

The mean-preserving positive ghost pilot at CFL=0.3 has a rank signed implicit energy-balance fraction of about `2.7e-6` near cycle 92. To isolate its cause, a restart was saved **immediately before cycle 92**. Its cycle, Newton and FGMRES traces through cycle 91 match the earlier CFL=0.3 run. Two short runs from this identical state differed only in the configured Newton maximum-residual gate; both used the same opt-in local gas root, bounded true-residual inexact FGMRES gate, opacity clamp with accepted-state guard, and mean-preserving ghost prolongation.

| Cycle 92 from identical checkpoint | Maximum gate `1e-5` | Maximum gate `3e-6` |
| --- | ---: | ---: |
| Newton iterations | 3 | **4** |
| Final Newton L2 | `4.82082e-7` | `1.20012e-7` |
| Final Newton maximum | `9.67505e-6` | `4.89065e-7` |
| Maximum rank signed implicit energy-balance fraction | `2.69560e-6` | **`3.68475e-11`** |
| Maximum rank absolute-term energy-balance fraction | `2.70168e-6` | `3.74225e-8` |

This is direct evidence that the cycle-92 signed energy defect is driven largely by **stopping Newton at the allowed localized maximum residual**, rather than by the ghost limiter at cycles 87–88. One more Newton iteration reduces the signed balance by about 73,000-fold from the same initial state. A longer run from cycle 74 with the `3e-6` gate also reaches cycle 94; its cycle-92 signed balance is `5.41e-12`.

A global change to the stricter gate is **not yet viable**. Starting the same clean pilot from cycle 0 with `3e-6` stops at the first step after eight Newton iterations and exhausted backtracking: `final_l2=5.27336e-7`, `final_max=5.01502e-6`. The existing `1e-5` gate allows that early step and all 30 previously audited early cycles. The follow-up should use an accuracy-aware or adaptive nonlinear stopping policy, or investigate why the early localized maximum stalls, while requiring explicit energy/flux budgets. A fixed relaxed gate is also insufficient for the later energy accuracy. This result does not establish a physically converged first core.

The [per-cycle CSV](wor92_newton_energy_gate_results_20261107/cycles.csv), [JSON](wor92_newton_energy_gate_results_20261107/summary.json), [figure](wor92_newton_energy_gate_results_20261107/energy_gate.png), [all control logs](wor92_newton_energy_gate_results_20261107/evidence.tar.gz), and [audit script](wor92_newton_energy_gate_20261107.py) contain the evidence and SHA-256 hashes. Regenerate with:

```bash
python3 tst/nrfld_validation/wor92_newton_energy_gate_20261107.py
```

The pre-cycle-92 checkpoint SHA-256 is recorded in JSON; it is at `/home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/table_mean_limiter_checkpoint92_baseline_2rank/cfl030_flux_precision16_local.final.rst`. The clean isolated binary SHA-256 is `0e6af159f3c545d765667ab45a3b7feaad7a3ddbb0c05e8897633822c86f5c1b`, from dedicated source branch commit `14633c31` (with prior commits `6da81d20`, `ff632195`).

Representative tight control, from the data bundle directory, using a fresh output directory:

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
  ATHENA_NRFLD_ENERGY_LEDGER=1 \
  timeout 60s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena \
  -r /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/table_mean_limiter_checkpoint92_baseline_2rank/cfl030_flux_precision16_local.final.rst \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_domain_guard_cfl030_restart.used \
  -d /path/to/new/output time/nlim=95 time/tlim=10000.0 \
  nrfld/nr_threshold=3e-6 nrfld/nr_accept_max_threshold=3e-6
```

For the baseline control, omit the two `nrfld/*` overrides. For the early-step control, replace `-r` with the initial input only and use `time/nlim=3`; its first step fails under `3e-6`. The opacity input contains `fld/opacity_table_domain_policy=clamp` and `fld/opacity_accepted_state_guard=true`. WOR-92 remains **In Progress**. No XD2000 access or merge occurred.
