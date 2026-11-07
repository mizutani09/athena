# WOR-92: coupled Schur global Jv check on AMR and uniform grids

Date: 2026-11-07. Base source: `ebf7634a8e877edfd58724de1e3810638a76ca7e` (WOR-105 source plus the isolated, uncommitted WOR-92 solver trials described in the preceding reports). This diagnostic was built only in a copied build tree. No production source change from this experiment is committed.

## Question and method

The existing global Jacobian diagnostic covers periodic, uncoupled radiation Test C. The existing local gas/radiation finite-difference diagnostic checks cell interiors, but cannot check coarse-fine ghost transfer. This experiment probes the **coupled Schur radiation direction** at the first Newton linearization: perturb the global radiation field by a smooth direction, exchange Newton state boundaries, recompute the residual with frozen face coefficients, and compare its finite difference to the assembled Schur matvec. The gas state and local gas-elimination ratio are held at the base state. Three perturbation sizes (`10^-4`, `10^-5`, `10^-6`) and four cell groups (interior, same-level face, coarse-fine face, all) are logged.

The `tangent_l2_rel` column substitutes the finite-difference *state ghost tangent* for the correction ghost in the same assembled stencil. Thus it isolates the discrepancy between state-boundary and correction-boundary transfers. The diagnostic restores the saved arrays after each check. A same-binary, diagnostic-on/off run with tangential reconstruction disabled produced identical four FGMRES iteration counts and true residuals, including the final failed solve (see [runs.csv](wor92_coupled_global_jv_results_20261107/runs.csv)). An earlier version that mixed pre-sync baseline residual with post-sync ghosts was discarded; the archived patch is the corrected version.

## Results

At cycle 0 and perturbation `10^-5`:

| Case | Interior Jv error | Same-level face | Coarse-fine face | Ghost transfer difference | Outcome |
|---|---:|---:|---:|---:|---|
| Table Rosseland, tangential on | `9.17e-10` | `6.20e-10` | `2.99e-5` | `1.57e-3` | second FGMRES: `300`, true relative residual `3.88e-8` > `2e-8` |
| Table Rosseland, tangential off | `9.17e-10` | `6.20e-10` | `4.37e-2` | `2.69e-3` | cycle 1 second FGMRES: `300`, `6.33e-8` |
| Fixed Rosseland `0.0652`, tangential on | `9.17e-10` | `6.20e-10` | `3.05e-5` | `1.56e-3` | 3 cycles complete; 7/7 FGMRES solves meet target |
| Uniform level 1, table Rosseland | `2.61e-9` | `2.04e-9` | no coarse-fine cells | `1.30e-11` at same-level faces | second FGMRES: `300`, `9.12e-8` |

In the tangential-on table case the coarse-fine error is `2.9866106e-5`, `2.9866112e-5`, `2.9866133e-5` as perturbation decreases, whereas the state-ghost tangent control is `1.26e-14`, `2.88e-14`, `2.58e-13`. This is a stable discrepancy between the assembled correction transfer and the derivative of the Newton state transfer at these coarse-fine faces. Tangential reconstruction reduces it by about three orders of magnitude. The fixed-R run shows a similar discrepancy while converging; the uniform table run fails without any coarse-fine faces. The interface discrepancy is therefore a separate consistency problem and is neither necessary nor sufficient for this particular FGMRES stop.

The frozen-coefficient radiation-direction check does not test the full two-variable Jacobian, derivatives of the opacity/limiter, physical boundary conditions, or actual FGMRES Krylov directions. It does not establish a production fix. The fixed-R and uniform results strengthen the earlier finding that variable Rosseland diffusion and solver conditioning need further investigation, including componentwise backward error on the actual full system.

## Reproduction and evidence

The isolated binary is `/home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_coupled_jv/bin/athena`, SHA-256 `636209373ab24c5a9f5fee803b6ac86031c23b9e1a4a62a12d691fb145de0025`. Its diagnostic-only source diff is [diagnostic.patch](wor92_coupled_global_jv_results_20261107/diagnostic.patch). Exact logs, exit codes, and inputs are in [evidence.tar.gz](wor92_coupled_global_jv_results_20261107/evidence.tar.gz); hashes are in [summary.json](wor92_coupled_global_jv_results_20261107/summary.json). The [Jv CSV](wor92_coupled_global_jv_results_20261107/jv.csv) contains all cycles and perturbation sizes, and the [figure](wor92_coupled_global_jv_results_20261107/jv_regions.png) compares cycle 0 regions.

Representative command, run from the bundle directory so table paths resolve:

```bash
cd /home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_COUPLED_GLOBAL_JV=1 \
  timeout 150s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_coupled_jv/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_component_freeze_cfl030_strict_3cycles.used \
  -d /path/to/fresh/output
```

For the fixed-R case add `ATHENA_WOR92_FIXED_ROSSELAND=1` and use `input_table_fixed_rosseland_high_cfl030_strict_3cycles.used`. For uniform level 1 use `input_uniform_l1_restart40_mg2_strict.used`. For tangential-off set `ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=0`; for the diagnostic-off control omit `ATHENA_WOR92_COUPLED_GLOBAL_JV`. The exact archived inputs and hashes avoid ambiguity. Regenerate CSV, JSON, figure and archive with:

```bash
cd /home/kosuke/simulation/athena_wor92_wor105_retest
python3 tst/nrfld_validation/wor92_coupled_global_jv_20261107.py
```

## Next experiments

1. Compare the state ghost derivative and linear correction ghost on a minimal single coarse-fine interface, and identify which interpolation or limiter term accounts for the `2.99e-5` residual. Keep this separate from the table-opacity stopping investigation.
2. For table and fixed-R controls, instrument the actual FGMRES direction at the second solve: full precision `b-Ax`, a componentwise backward error that accounts for AMR ghost interpolation, and row-level contributions by local gas coupling and diffusion coefficient contrast.
3. Test a controlled variable-R homotopy or diffusion coefficient scaling at a fixed initial state. If it changes convergence consistently, evaluate a targeted preconditioner or scale correction. Do not accept a relaxed precision floor as strict convergence.

WOR-92 remains **In Progress**. No production change is ready to commit or merge.
