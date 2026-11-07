# WOR-92: CFL=0.3 solver controls and ten-cycle ledger (2026-11-07)

## Conclusion

The Tomida table EoS + Semenov opacity run stops in the second FGMRES solve of cycle 0, before a Newton update can be accepted. At actual `dt=31.114819909`, the first solve meets the requested true relative residual `2e-8` at iteration 156 (`1.218907647e-8`); the second reaches iteration 300 at `3.879657772e-8`. Independent long-double residual recomputation agrees with this true residual. The current evidence supports a precision/conditioning conflict with the strict **normwise** stopping target. The frozen radiation Schur operator is diffusion dominated (maximum diagonal `3.93e10`, minimum reaction fraction `2.62e-11`), and convergence depends strongly on the Rosseland opacity field. This remains a supported diagnosis, not a complete proof of the unique cause.

The exact **table EoS** local gas-radiation root is a promising improvement: it reduces the first Newton gas residual from approximately `0.0584` (local root off) to `4e-13` (on). It does not resolve the strict second FGMRES solve. A separate precision-floor acceptance experiment completed ten early cycles, with 22 FGMRES solves, ten Newton convergences, and four linear solves accepted by the floor. This is exploratory; no production validation or full collapse has been established. WOR-92 remains **In Progress**.

## New controls

All strict runs used root `32^3`, small AMR, two MPI ranks, table opacity, flux-form matvec, exact table local gas root, FGMRES maximum 300, restart 40, and no precision floor. Each bounded run used an isolated output directory. The table gives the second FGMRES true relative residual; each target was `2e-8`.

| Trial | Second residual | Outcome |
| --- | ---: | --- |
| Baseline MG2 | `3.87965777e-8` | fails |
| Long-double Hessenberg/Givens | `4.49781536e-8` | fails, worse |
| MG V-cycles 4 | `2.90045368e-8` | fails |
| MG V-cycles 5 | `9.67807047e-8` | fails, worse |
| Smoother omega 0.7 | `2.78309984e-8` | fails |
| Smoother omega 0.8 | `3.29999334e-8` | fails |
| Smoother omega 0.9 | `2.58962975e-8` | fails |
| Disable basis reuse | `3.87965777e-8` | identical to baseline |
| Disable compact basis | `3.87965777e-8` | identical to baseline |

Four pre/post smoothing sweeps exceeded the local 180-second bound before completing the first solve; its log is preserved and it was not retried. Increasing MG effort or changing smoother omega is nonmonotonic. The long-double projected Krylov calculation, basis reuse and compact storage do not explain the strict failure. These diagnostic source edits remain in copied build trees; no production source change was staged.

The AMR backward-error diagnostic used `|r_i|/(|b_i|+|(Ax)_i|)`, an upper bound on the exact assembled componentwise backward error because `|(Ax)_i| <= sum_j |A_ij x_j|`. It gave `1.0` on the first solve and `0.12946` on the failed second solve, with all 61,440 cells above machine epsilon. This bound is too loose to certify the actual AMR error; it does **not** show that the iterate has a large exact backward error. A uniform-grid case without coarse-fine faces already showed exact assembled componentwise backward error `8.88913e-17` even while failing the same normwise target. The AMR row expansion, including ghost interpolation weights, remains necessary before replacing the acceptance test.

## Ten-cycle precision-floor audit

The floor-32 run with the same table and local gas root completed ten early cycles at CFL=0.3. Enabling rank energy and interface ledgers produced an identical numerical FGMRES/Newton/cycle trace to the prior run (SHA-256 `4397c44c3e6d1633c55d28a9d633a8df77edfb0ffd1048132d8dacd727d60272`). Across ten stages, all 120 leaf blocks and 24 coarse-fine face groups per stage were paired. Maximum area-weighted coarse-fine face mismatch was `3.10830249e-7`; maximum per-rank signed energy-balance fraction was `1.30288670e-8`, and maximum absolute-term-normalized fraction was `2.18188535e-7`. The signed interior diffusion imbalance ranged to approximately 319 code energy units. These errors are small on the early-run energy scale, but flux conservation is not exact, and ten cycles do not cover the collapse regime.

## Artifacts and reproduction

Base source Git commit: `521c2b845419b0ee8404b2cd3888e7ab9bc46668` on isolated branch `codex/wor92-wor105-retest-20261107`, plus the six previously documented uncommitted table/local-root trial source files. Baseline binary SHA-256: `5b259bf3c0a4b768a6a35aa9f2d9cebfd3b9151f78b80ed5583069a772ffb691`. Long-double trial binary SHA-256: `517c104d0bd949b2b0b65fdccd4c0fb83fe12efebebb9cce03c89858d87f27fb`. AMR upper-bound diagnostic binary SHA-256: `d89ac18eeb9f732246f9e40e1efc5c082a76dd34b11e8c6013bbc0a11aa0f9db`.

The [solver CSV](wor92_solver_controls_results_20261107/solves.csv), [JSON](wor92_solver_controls_results_20261107/summary.json), [figure](wor92_solver_controls_results_20261107/second_solve.png), and [logs, inputs, and diagnostic source](wor92_solver_controls_results_20261107/evidence.tar.gz) contain the strict controls. The [energy CSV](wor92_floor32_energy_audit_results_20261107/stages.csv), [face CSV](wor92_floor32_energy_audit_results_20261107/faces.csv), [JSON](wor92_floor32_energy_audit_results_20261107/summary.json), [figure](wor92_floor32_energy_audit_results_20261107/floor32_energy.png), and [rank logs/input](wor92_floor32_energy_audit_results_20261107/evidence.tar.gz) contain the floor audit. Regenerate both sets from this worktree with:

```bash
python3 tst/nrfld_validation/wor92_solver_controls_20261107.py
python3 tst/nrfld_validation/wor92_floor32_energy_audit_20261107.py
```

Representative strict command (use the archived input for each control and a fresh isolated output directory):

```bash
cd /home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  timeout 120s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_precision/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_component_freeze_cfl030_strict_3cycles.used \
  -d /path/to/new/isolated/output
```

For the AMR upper-bound diagnostic, use the `build_table_be_upper_trial` binary and `ATHENA_WOR92_BE_UPPER_DIAG=1`. For the energy audit, use archived `input_table_local_jacobian_floor32_cfl030_10cycles.used`, baseline binary, and set `ATHENA_NRFLD_ENERGY_LEDGER=1`, `ATHENA_NRFLD_INTERFACE_LEDGER=1`, `ATHENA_NRFLD_RANK_LEDGER_PREFIX=/path/to/new/ledger`. Omit `ATHENA_WOR92_FIXED_ROSSELAND` entirely for table opacity; merely setting it to zero selects fixed opacity in this trial source.

## Remaining validation

1. Expand coarse-fine ghost interpolation into the global matrix rows and compute the exact assembled componentwise backward error at all failed solves. Validate any precision-based acceptance on mixed AMR and uniform grids before changing production stopping criteria.
2. Reduce or recondition the radiation Schur operator, then compare true residual, Newton convergence, conservation, and physical fields over more cycles against a converged reference. The short floor run is only a candidate route.
3. Audit local gas root and Jacobian derivatives across table discontinuities and temperatures reached during collapse; compare against the ideal-gas WOR-93 control with its actual compile-time EoS setting. Keep Semenov-versus-Optab definitions as a separate physics issue.
4. Repeat after a demonstrated conservative AMR flux fix. Existing weighted-state transfer changes improve one Jacobian mismatch but are not a validated solver repair.
