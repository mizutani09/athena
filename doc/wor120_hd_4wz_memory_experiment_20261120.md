# WOR-120: Berta 4wz memory experiment (2026-11-20)

## Motivation

The first 168-rank surface-L3 sustained 4wz attempt (`343499`) exhausted the
three-node `3 x 110 GiB` Slurm cgroup after startup.  It reached about 2.06 GB average
RSS per rank, while the accepted one-cycle test did not run long enough to expose the
steady scratch footprint.

This branch (`wor-120-hd-4wz-memory`) changes storage only.  It does not alter a
reconstruction formula, mask, time integrator, or evolved variable.

## Reuse and removed allocations

For a Berta WENO-Z MeshBlock:

- `wl3d_` and `wr3d_` are legacy fourth-order face-average states and are never read.
- the point-recovery Laplacian is dead before mask dilation, so its storage also holds
  the temporary dilated mask;
- after Hydro and scalar primitive recovery, `u_cc` is dead and can hold the
  face-flux Laplacian during `CalculateFluxes`.

The Berta path therefore omits `wl3d_`, `wr3d_`, and `scr2_nkji_`.  The legacy
fourth-order path allocates all three exactly as before.  For a 32-cubed block with
four ghost cells, the omitted logical allocation is
`3 x 5 x 40^3 x 8 = 7,680,000` bytes per block, or 17.28 GiB over 2,416 blocks.

## Local validation

The exact MPI SCF configuration builds.  On the four-rank, 400-block surface-level-0
SCF one-cycle test:

| build | launcher/child peak RSS | main-loop CPU time |
|---|---:|---:|
| production storage | 1,551,276 KiB | 4.019 s |
| reused storage | 1,389,120 KiB | 3.330 s |

Peak RSS falls by 162,156 KiB (10.45%).  The complete history files are byte-for-byte
identical, including both scalar masses and all WENO fallback counts.  A separate
uniform-grid legacy `xorder=4` SCF smoke also completes, confirming that its
conditionally retained arrays are allocated.

The decisive gate is the committed 100-cycle surface-L3 script on three 110 GiB M
nodes.  This branch remains separate from production until that gate completes and
the measured rank/node RSS confirms safe headroom.
