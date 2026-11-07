# WOR-92: exact table local gas root with an inexact-Newton linear gate

## Decision supported by the local tests

The proposed acceptance principle is reasonable for a **bounded numerical pilot**: an FGMRES iterate with a slightly missed strict target can be passed to Newton if its independently recomputed true relative residual is finite and bounded, and the complete nonlinear solve actually converges. With the exact Tomida-table local gas root enabled, a copied-build trial accepted FGMRES true residuals up to `1e-7` (five times the requested `2e-8`) and disabled the existing local-stencil precision-floor rule. At CFL=0.3 on the 32³-root, two-rank mixed-AMR test, this gate reproduced the prior factor-32 floor run for **all 30 early cycles**: every FGMRES `(iterations, true residual)` pair, every Newton convergence line, and every cycle line matched exactly. The 30-cycle history file was byte-identical. At ten cycles the complete `prim` and FLD HDF5 arrays, levels, and logical locations were bitwise identical. Of 62 linear solves, 21 exceeded `2e-8`; their maximum true residual was `6.21057134e-8`. Newton converged in all 30 cycles under the configured `final_l2 <= 1e-5` and `final_max <= 1e-5` criteria.

This evidence supports using the local root and a **guarded inexact-Newton policy** instead of requiring every inner solve to reach `2e-8`. It does not establish that the current floor or `1e-7` cap is suitable for the full collapse. The gate is a diagnostic trial in an isolated copied build, not a committed production change. The general-EOS local root itself remains a trial source change because its table-EOS problem-generator dependency has not been cleanly integrated and validated.

## Limits exposed by the controls

| Check | Result | Interpretation |
| --- | --- | --- |
| CFL 0.3, floor32 versus true-residual cap `1e-7`, 30 cycles | 30/30 Newton convergences; identical linear and nonlinear numerical traces | Both gates accept the same early trajectory. |
| CFL 0.3, Newton tolerance tightened from `1e-5` to `1e-6` | Cycle 0 backtracking exhausted after 8 iterations; `final_l2=5.273e-7`, `final_max=5.015e-6`; gas L2 `3.600e-13` | Local gas solve is accurate, but the coupled nonlinear residual has a floor above the tighter maximum criterion. |
| CFL 0.15, floor32 | Cycle 0 second solve `2.594e-8`; local-stencil proxy `1.536e-14` exceeds `32ε`, so FGMRES aborts | The existing floor is not a robust acceptance rule across CFL. |
| CFL 0.15, true-residual cap `1e-7` | Cycle 0 Newton converges in 2 iterations; cycle 1 FGMRES stops at `1.461e-6` after 300 iterations | A larger failure remains, so the cap must reject it. No matched-time CFL convergence comparison is available. |
| Restart from cycle 30 with floor32 | 48 more Newton-converged cycles through cycle 77; only 3 of 107 new linear solves used floor; at cycle 66 Newton `final_max=8.646e-6` | The nonlinear tolerance margin becomes narrow later. |
| Same restart, energy and AMR-face ledger | Maximum rank signed energy-balance fraction `4.013e-6` at cycle 66; maximum weighted coarse-fine mismatch `4.587e-6` at cycle 34 | Conservation defects remain below the `1e-5` Newton criterion but are not negligible or proven acceptable for science output. |

The restart stops at the **start of cycle 78** because a Planck opacity lookup requests approximately `3.813 K`, below the Semenov table's `5 K` lower bound. This is a domain guard, not an FGMRES or Newton nonconvergence. A separate short restart from the saved time-2000 checkpoint using the table's `clamp` policy completes cycle 80 with 368 clamped lookups and no nonfinite or negative opacity results. Per-cycle accepted-state FLD snapshots at cycles 75–80 have active-cell gas-temperature minima from `14.028 K` down to `11.494 K`; no saved active cell is below `5 K`. This is consistent with the out-of-domain lookup arising during an intermediate trial evaluation, though ghost-cell lookups are not captured by these snapshots. Clamping the trial value is an exploratory way to let Newton backtrack; production handling should keep an explicit accepted-state domain check. It must not silently reinterpret the opacity model outside its table.

At the last accepted pre-error state, `rho_max=3.13153e-15 g/cm³` and `Tgas,max=13.734 K` at code time `2056.79`. This is still far below the intended first-core density near `1e-10 g/cm³`. The user-proposed inexact-Newton criterion can replace the **inner strict gate for a pilot**, while WOR-92's full physical and numerical completion remains open. Semenov-versus-Optab opacity definitions are a separate physics question.

## Reproducible evidence

The [summary JSON](wor92_inexact_newton_results_20261107/summary.json), [run CSV](wor92_inexact_newton_results_20261107/runs.csv), [per-solve CSV](wor92_inexact_newton_results_20261107/solves.csv), [accepted-temperature CSV](wor92_inexact_newton_results_20261107/accepted_temperature.csv), [residual figure](wor92_inexact_newton_results_20261107/inexact_residuals.png), [gate patch](wor92_inexact_newton_results_20261107/inexact_gate.patch), [opacity-count diagnostic patch](wor92_inexact_newton_results_20261107/opacity_count.patch), and [logs, exact inputs, history and ten-cycle HDF5 evidence](wor92_inexact_newton_results_20261107/evidence.tar.gz) provide hashes and controls. The [restart-stage CSV](wor92_floor32_restart30_results_20261107/stages.csv), [face CSV](wor92_floor32_restart30_results_20261107/faces.csv), [JSON](wor92_floor32_restart30_results_20261107/summary.json), [figure](wor92_floor32_restart30_results_20261107/floor32_energy.png), and [rank logs](wor92_floor32_restart30_results_20261107/evidence.tar.gz) cover cycles 30–77. Per-cycle opacity snapshots remain in the isolated local output directory; their individual SHA-256 values and temperature minima are in the accepted-temperature CSV.

Starting report-branch Git commit: `44611e8ece4c7da695733a39a0892646703d7607`. Baseline binary SHA-256: `5b259bf3c0a4b768a6a35aa9f2d9cebfd3b9151f78b80ed5583069a772ffb691`; inexact trial binary: `a1600941beb720be472bf8f4d789a9b70ecfdf308d3cd606f1d3278bebc4eded`; opacity-count diagnostic binary: `902b8e2cbf8f0fbd6ba2e75a945216a02b7838fe8a06b78586fc9e6fbc59f20f`. All source modifications were confined to copied build trees. Regenerate the result tables, figures, and archives from the report worktree:

```bash
WOR92_AUDIT_MODE=restart30 python3 tst/nrfld_validation/wor92_floor32_energy_audit_20261107.py
python3 tst/nrfld_validation/wor92_inexact_newton_audit_20261107.py
```

Representative inexact trial command from `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle` (use a fresh `-d` output path):

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7 \
  timeout 300s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_inexact_trial/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_inexact_cfl030_30cycles.used \
  -d /path/to/new/output
```

For the diagnostic clamp control, restart with `-r` pointing to the archived `cfl030_flux_precision16_local.00002.rst` from the time-2000 output, pass `fld/opacity_table_domain_policy=clamp`, and use `output3/dt=1.0` to save each accepted state. Do not set `ATHENA_WOR92_FIXED_ROSSELAND`, including to zero, in table-opacity runs.

The exact input files for the tighter Newton and CFL 0.15 controls are in the evidence archive. For either control, replace the representative command's input with `input_table_local_jacobian_floor32_nr1e6_cfl030_10cycles.used` or `input_table_inexact_cfl015_matched_time.used`; the latter also uses `output2/dt=290.74615879854679 output3/dt=290.74615879854679`. The domain-control command, with the same seven `ATHENA_NRFLD_*` settings as above and a fresh output path, was:

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  timeout 180s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_opacity_count_trial/bin/athena \
  -r /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/table_floor32_restart30_to80_2rank/cfl030_flux_precision16_local.00002.rst \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_local_jacobian_floor32_cfl030_30cycles.used \
  -d /path/to/new/output time/nlim=80 time/tlim=2100.0 \
  fld/opacity_table_domain_policy=clamp output3/dt=1.0
```

## Candidate production criterion and remaining checks

1. Integrate the general-EOS local gas root and its table-EOS problem-generator support on a clean isolated branch; retain the mixed-frame and bracket stress checks already documented in [general-EOS local solve](wor92_local_general_eos_20261107.md) and [bracket trial](wor92_local_bracket_trial_20261107.md).
2. Use a bounded **true** linear residual, no breakdown/nonfinite result, and enforce actual Newton decrease and configured final residual on every step. On failure, reject the step or reduce `dt`; a broad unconditional tolerance relaxation is unsupported by the CFL 0.15 control. An adaptive inexact-Newton forcing criterion may be preferable to the fixed `1e-7` pilot cap.
3. Record total-energy and coarse-fine flux errors against a science accuracy budget, and establish a timestep/mesh convergence reference. The existing CFL 0.15 control cannot provide one because its cycle-1 linear solve stalls well above the trial cap.
4. Handle opacity table bounds during Newton trial evaluations without letting an invalid trial abort the entire calculation, while enforcing table validity for accepted states. Continue the physical collapse to first-core density before marking WOR-92 Done.

WOR-92 remains **In Progress**. No XD2000 access or merge occurred.
