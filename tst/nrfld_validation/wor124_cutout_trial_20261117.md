# WOR-124 central cutout diagnostic (2026-11-17)

## Purpose and status

Test whether a first-core checkpoint can be continued on a smaller central
domain without remapping the constrained-transport magnetic field. This is a
diagnostic, not a validated second-core setup. The resulting cube still has
periodic fluid and FLD boundaries, while its gravity solve uses only the
mass inside the crop.

## Reproducible construction

- Isolated branch: `exp/wor124-central-cutout-20261117`, based on
  `4f886c5d1ff99c91794f5bdcf52198e0a25d0cad`. Existing worktrees and
  jobs were not changed.
- Input: WOR-124 combined weak-field B32 restart at cycle 1780,
  `combined_B32uG_floor_amr11_3node.00182.rst`, 4096 MeshBlocks.
- [`wor124_crop_restart.py`](wor124_crop_restart.py) selects the complete
  central cube `[-375,375]^3` au at existing MeshBlock boundaries, rebases the
  octree, and copies each MeshBlock's restart body without interpolation.
  Hydro, FLD, passive scalar, and all three face-centered magnetic arrays
  therefore retain their saved bytes. An octree completeness check rejects
  holes and overlapping blocks. The crop has 2528 MeshBlocks, 16 root cells
  per axis, and the same finest physical cell width as its parent.
- The generated restart SHA-256 is
  `4362523f0f4a097bcfd705f33f3ad04d6c9c82676bcdcfc98ee53081693f9459`.
- Existing XD trial binary SHA-256:
  `2a8b93e1dbfbeb82f03660407f883cd472b8e6db1bded58fefe5af575ea8955c`.
  EOS table SHA-256:
  `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`.
  Opacity table SHA-256:
  `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`.
  The XD `source_commit.txt` records the original source base
  `7dccd1afed951217168d54155f92dcca93d51bb0`; its floor-trial
  `general_mhd.cpp` matches this branch at SHA-256
  `fc526cb4545b11deccd40923d0cf1afbbaea7f8a30a4bb09b8e29865dceaa968`.
- XD test script: [`wor124_cutout_gate.sbatch`](wor124_cutout_gate.sbatch),
  M-test-a, one node, 64 MPI ranks, the same binary and solver options for
  full-domain and cropped runs. A/B starts from the same cycle 1780 state and
  stops at cycle 1785.

## Five-cycle A/B result

| Run | XD job | Blocks at end | Wall time | Slurm step MaxRSS | Result |
|---|---:|---:|---:|---:|---|
| Central crop | 341264 | 2535 | 70 s | 25.7 GB | PASS, numerical gate |
| Full domain | 341272 | 4124 | 126 s | 37.7 GB | PASS, reference |

All Newton diagnostics reported `converged`; the maximum FGMRES true relative
residual was below `8.6e-9` in both runs. Opacity out-of-domain, clamping,
negative, zero, and nonfinite counters remained zero. The cropped run's
`divB_max` was about `1.73e-10` in code units at cycle 1785, comparable to
the parent history's `1.68e-10`.

[`wor124_compare_cutout.py`](wor124_compare_cutout.py) compared 2532 of 2535
cropped MeshBlocks with parent blocks at exactly the same coordinates and
refinement level (1,296,384 cells). Three unmatched blocks are around 52 au;
they reflect different AMR decisions after the restart. Relative L2 errors:

| Radius | Density | Gas temperature | Radiation energy | Radiation temperature | B1 |
|---|---:|---:|---:|---:|---:|
| 0–10 au | 6.72e-7 | 6.88e-7 | 1.36e-6 | 6.88e-7 | 7.79e-7 |
| 10–50 au | 3.69e-7 | 2.33e-4 | 1.47e-4 | 5.35e-4 | 1.91e-7 |
| 50–150 au | 6.31e-6 | 3.80e-3 | 7.65e-2 | 2.07e-2 | 1.19e-5 |
| 150–300 au | 2.83e-8 | 5.41e-3 | 1.42e-1 | 3.37e-2 | 2.77e-8 |
| 300–650 au | 1.43e-3 | 7.19e-3 | 1.48e-1 | 3.51e-2 | 1.11e-3 |

On exactly matched cells, the volume-integrated density differs by -0.0140%,
while volume-integrated radiation energy differs by +13.86%. Central maximum
density and temperature history differences at cycle 1785 are -3.26e-7 and
-1.49e-7, respectively.

## Interpretation and remaining work

This is a technically valid no-remap restart and a useful short-time core
diagnostic. It is **not** a physically validated long-time central zoom.
The fluid and implicit FLD solve still use periodic boundaries at ±375 au;
the current conservative FLD prototype rejects nonperiodic boundaries. The
large radiation discrepancy outside about 50 au appears within five steps.
Because changing the domain also changes the gravity solve and MG hierarchy,
the test alone does not assign that discrepancy uniquely to the periodic
boundary. It does show that continuing this crop as a physical second-core
calculation would be unjustified.

M-test job 341274 extended the crop from cycle 1780 to 1830 and completed
successfully in 602 s. All 52 Newton diagnostics were `converged`; 151 FGMRES
solves required 7–11 iterations and their largest true relative residual was
`1.98e-8`, below the `2e-8` target. Opacity out-of-domain remained zero.
Against the existing full-domain history, central maximum density differs by
-6.19e-5, maximum gas temperature by -2.78e-5, and maximum radiation energy
by -1.11e-4 at cycle 1830. These are small over this approximately five-year
physical interval, but the error is growing. The periodic crop remains
unvalidated for longer continuation. A planned ±750 au short comparison
(job 341275) was canceled before it started when the user requested a pause.
No second-core production job was launched from the periodic crop.

Local HTML report and raw comparisons:
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/cutout_trial_20261117/report/index.html`.
