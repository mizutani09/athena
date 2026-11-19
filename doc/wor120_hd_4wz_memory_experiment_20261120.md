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

## XD memory1 result

The first reduced-storage gate (`343539`) used the exact 2,416-block surface-L3
restart on 168 ranks and three 110 GiB M nodes.  It still exhausted the cgroup after
134 seconds, before cycle 1, but its average rank RSS fell from about 2.16 GB in the
original attempt (`343499`) to about 2.06 GB.  This agrees with the 17.28 GiB logical
allocation reduction and confirms that the failure is capacity-only, not numerical.

## Memory2 plus scalar PLM

The follow-on branch `wor-120-hd-4wz-memory2` removes the remaining Berta Hydro
scratch array.  It stores the conserved Laplacian in `w_cc` before that array becomes
the primitive output, and dilates the integer troubled-cell mask in place using 1 for
original trouble and 2 for newly marked neighbours.  The legacy fourth-order path is
unchanged.  Scalar point recovery similarly reuses `r_cc` and then `s_cc`; its legacy
one-cycle history is byte-for-byte identical.

For Berta 4wz, passive scalars use the already validated conservative PLM path.  The
code now also omits all storage used only by the inactive legacy fourth-order scalar
path: `s_cc`, `r_cc`, four block scratch/state arrays, and three point-mass-flux
arrays.  Hydro remains Berta WENO-Z plus RK4; only tracer transport is second order.
This is the robustness contract tested by scalar-PLM job `343506`, which kept the B16
topology fixed and reduced its energy excursion by a factor of about 34.

Relative to memory1, memory2 removes another 10.304 MB per B32 block: one NHYDRO
array, six NSCALARS arrays, and the three directional point-mass-flux arrays.  This is
23.19 GiB over 2,416 blocks.  The cumulative logical reduction from the original
implementation is 40.47 GiB.  In the local four-rank, 400-block SCF smoke, peak RSS
changes from
1,551,276 KiB (original) to 1,389,120 KiB (memory1) and 1,170,804 KiB (memory2 plus
scalar PLM), a cumulative 24.5% reduction.  The resulting history is byte-for-byte
identical to the independently validated scalar-PLM branch.

A separate `-fsanitize=address` debug build also completes a three-dimensional SCF
4wz cycle with scalar PLM and no invalid access.  This specifically exercises the
Berta path after the legacy fourth-order scalar arrays have been left unallocated.

## XD memory2 result

The exact surface-L3 gate (`343554`) completed all 100 cycles on the same 168 ranks
and three 110 GiB nodes.  The MPI step took 220.08 s, reported 1.28 GiB average RSS
per rank and 4.25 GiB for the largest rank, and left stderr empty.  For comparison,
Slurm reported 1.92 GiB average for memory1 and 2.01 GiB for the original storage.
The new layout therefore passes the node cgroup with substantial sustained headroom.

The mesh remains fixed at 2,416 blocks.  At 100 cycles the stellar-mass drift is
`2.30e-12`; the maximum mechanical-energy drift is `1.27e-4`, radial Mach RMS
`2.49e-3`, and absolute virial residual `2.51e-3`.  No NaN, multigrid warning, or
runtime failure occurs.  Total and stellar-mass-weighted Hydro fallback fractions are
22.0% and 18.8%.

The 100-cycle main-loop measurement is 180.16 s.  Scaling to the approximately 4,752
cycles needed for `0.1 t_dyn` gives 2.38 h; allowing restart I/O, final output, and
performance variation, the sustained script requests four hours.  Job `343567` uses
the identical accepted restart and binary and keeps a final restart active for later
continuation.  Its dependent A/B post-processing job `343568` compares the physical
histories and also compiles `compare_scf_cost.cpp`.  The latter records both methods'
cycle counts, Athena CPU and wall times, throughput, and Slurm per-rank RSS in a
machine-readable `cost_comparison.json`; this avoids estimating the final 4wz/x2 cost
ratio from the short memory gate.
