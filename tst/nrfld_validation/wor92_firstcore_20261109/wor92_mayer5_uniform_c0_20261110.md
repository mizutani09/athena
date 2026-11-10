# Single-binary Mayer §5 ideal-EOS collapse, 2026-11-10

## Initial-condition audit

The reference is [Mayer et al. (2025), §5.1](https://arxiv.org/html/2510.12620#S5). This run starts at cycle 0 with the same repaired ideal-EOS binary through every continuation. It uses the validated source at `582acd5999c1a5681ab1bc1d1af292676642ed40` and binary SHA-256 `56d379d96cea55cb50497ccc4a7ea9a10897784f82f8f48a6b9817671b62993d`. The binary's `NRFLD.cpp` and `protostellar_core_fld.cpp` hashes match the isolated worktree. The earlier path used the old binary through c1090 and is not the trajectory of this run.

| Quantity | Mayer §5.1 | This input |
| --- | --- | --- |
| Core | Uniform, nonrotating, unmagnetized sphere; 1 M☉, 3000 au | Same; `transition_width=0` (previous run used 187.5 au smoothing) |
| Initial core density | Approximately `5.25e-18 g cm^-3` | `5.2515854e-18 g cm^-3`; c0 3D cells inside the sphere contain `1.00086 M☉` (0.086% Cartesian discretization excess) |
| Thermodynamics | 14 K, fixed μ=2.381, γ=5/3 | Same ideal gas and initial gas/radiation temperature |
| Opacity | Fixed κP=κR=1 cm² g^-1 | Same |
| Box and boundaries | 12000 au cube; periodic hydro, isolated gravity | Same nominal geometry; Athena++ multipole gravity boundary |
| Background | Passive-tracer region, raised pressure, fixed initial temperature | Passive tracer, density `0.01 ρcore`, pressure `0.01 Pcore`, fixed 14 K gas/radiation. The paper gives no numerical background density or pressure boost; this remains an unmatched choice. |
| Refinement | AREPO Lagrangian target mass `3.33e-7 M☉`, split/merge outside factor 2 | Cartesian Jeans AMR with `N_J=32`. The input records the same target mass but it is inactive in Jeans-only mode; the spatial methods are not equivalent. |
| Time stepping | Global and local variants in paper | Global RK2, CFL 0.3; compare with its global-time-step result |

The original first-core comparison reached the threshold after a repair at c1090, but accepted c1095–1097 states contained eight pressure-floor cells. This run gates **every history step** on `P_min>1e-18 dyn cm^-2` and Jeans resolution ≥16 cells. At any violation, the long job records the first bad history row, writes its restart, and exits nonzero. Thus its exit status will not silently report a clean first-core path after the known artifact.

## XD execution

Isolated XD directory: `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_uniform_c0_20261110/`.

1. [Input](wor92_mayer5_uniform_c0_20261110.in) and [M-test-a preflight](wor92_mayer5_uniform_mtest_c0_c10.sbatch): 1 node, 64 ranks, c0→10; exact binary hash, restart, no pressure floor, Jeans gate.
2. [M-large-a continuation](wor92_mayer5_uniform_mlarge_c10_firstcore.sbatch): 5 nodes, 320 ranks, after the successful preflight; ten-cycle restart segments to the first saved ρmax≥1e-10 g cm^-3, with all-step pressure/Jeans checks. It also checks solver exit and finite positive history at each segment.

All jobs record source, input and binary hashes, module list, solver logs, per-step 15-digit HST, runtime and Slurm memory. The actual job IDs and status are appended after submission and completion.

## Submitted jobs and preflight result

The input SHA-256 is `2d7198a19316a462e3ca4d44b41f0eec79795239f084e5184d50962c8224474f`. The two scripts assert both input and binary hashes before running. On XD, their input and shell syntax checks passed. The source files `NRFLD.cpp` and `protostellar_core_fld.cpp` matched the local isolated worktree hashes.

M-test-a job **339293**, 1 node / 64 MPI ranks, completed `0:0` in **3:33**. It wrote `PASS_C0_TO_C10 cycle=10 meshblocks=960`. All ten nonzero-step Newton solves converged (five additional zero-dt setup statuses); 20 FGMRES solves used 1865 iterations and had maximum final independent true relative residual `1.99698e-8`. The largest final Newton maximum residual was `5.803e-6`, below the `1e-5` acceptance threshold. Across all accepted c0→10 history rows, minimum pressure was `1.91215e-11 dyn cm^-2` and minimum Jeans resolution was `60.57056` cells. Gas mass changed from `2.274089536825101e33` to `2.274089536825090e33 g`, relative `−4.8163e-15`. c10 maximum density was `5.30851670e-18 g cm^-3`; Slurm step MaxRSS was `17,078,612 K`. The c0 3D HDF5 density/volume integration gives the core mass quoted above. Preflight logs, provenance, full-precision HST and c0 3D data are copied to the local external evidence directory `firstcore_integration_evidence/ideal_mayer5_uniform_c0_20261110/job_339293/`.

After the preflight passed, submitted **339296** to M-large-a, 5 nodes / 320 MPI ranks, with `afterok:339293`. The job uses the exact same ideal binary and input. At submission it was `PENDING`, reason `QOSMaxJobsPerUserLimit`, because another ten-node M-large-a job (338743) was running under the same account; the dependency was already satisfied. The scheduler gave no start-time estimate. Do not cancel the unrelated job. Job 339296 will start automatically when the account limit permits. Its status and first-core/step-gate outcome have not yet been validated.
