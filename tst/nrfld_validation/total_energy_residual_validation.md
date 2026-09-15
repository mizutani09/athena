# Coupled total-energy residual validation (2026-09-15)

Base commit: `a09dd40d`. Tests used the production patch in an isolated worktree,
without the earlier Jacobian/boundary diagnostic instrumentation. The parent
repository's build files, original inputs/results, and restart files were not
modified. No 256-cubed or remote restart validation was performed in this step.

The coupled convergence basis is gas plus total energy. Both combined L2 and
combined max must satisfy the original threshold. Exact equation roots are
unchanged, but finite-tolerance weighting is now based on total internal energy,
not the small radiation energy alone. Radiation-only and fixed-radiation modes
retain their original equations. See the README for diagnostic field semantics.

## Regression and analytic comparisons

- `nrmgfld/coupled_residual_norms`: passed. Includes stiff coupled exchange on
  two MPI ranks and radiation-only/fixed-radiation cases on one rank; checks
  active/inactive channels, norm aggregation, and both stopping criteria.
- `run_suite.py --max-cores 4 --no-plots`: both cases passed.
  Diffusion relative L1 error: `0.00425236` (limit `0.01`).
  Coupling maximum log gas-energy error: `0.00042907082136` (limit `0.005`).

## Solar smoke tests

Input: downloaded `test_long2/athinput.simple_convection_lhllc_fld_general_hpc_256_mb32_t10`.
Only resolution, cycle/time limits, output suppression, external-file paths,
and iteration-summary verbosity were overridden. The saved input was not edited.
EOS/opacity/profile settings, CFL 0.3, domain `[0, 3.072]^3`, noise amplitude
`1e-3`, and 32-cubed MeshBlocks were retained. All outputs including restart
were disabled with `output1/dt=-1` through `output6/dt=-1`.
Build: `--prob=simple_convection_lhllc_fld --eos=general/eos_table --flux=lhllc
-nrmgfld -mpi -hdf5` (double precision).

| Grid | MPI | Cycles | Threshold | Newton iterations | Final/max combined residual |
|---|---:|---:|---:|---:|---:|
| 32 cubed | 1 | 1 | 1e-6 | 29 | 5.7807743e-7 |
| 64 cubed | 8 | 10 | 1e-6 | 6-30 per cycle | all cycles <= 9.0044756e-7 |
| 128 cubed | 8 | 1 | 1e-6 | 28 | 9.3906533e-7 |
| 64 cubed | 8 | 1 | 1e-8 | 38 | 6.8971912e-9 |

All requested cycles converged. Opacity out-of-domain, clamp, negative, zero,
and nonfinite result counters were zero. These are short numerical-stability
tests, not evidence of thermal/statistical stationarity or production flux accuracy.

External files under `simulation/test/rad_nr/test_20260811`, SHA-256:

```text
AthenaNataEosTable_solar_wide.tab  127c5219f9e6a10085b2791043d0c72f482c2b5ce5c6e3dbe53dd08e7db71017
opacity_table_rhoT_3p0_6p3.h5     cc5912dd3896c39c1ea3fde0d315dc16c136303a8c363ab58fa56b705b909aaf
solar_initial_profile.dat       2d3e8902e291dfa1cb6d237e11d6dea57889c8d121abf6cae08a5f50b36e0186
```

Before a long production job, rebuild with this patch and run a 256-cubed
1-10-cycle smoke test. For a remote restart, separately verify restart/binary
compatibility with a short continuation before submitting a full-duration job.
