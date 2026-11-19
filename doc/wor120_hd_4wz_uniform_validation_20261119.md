# WOR-120: HD 4wz uniform-grid validation (2026-11-19)

## Scope and isolation

- Worktree: `/home/kosuke/simulation/athena-wor120-hd4wz`
- Branch: `wor-120-hd-4wz`
- Base: WOR-162 commit `1f3647935e4e`
- Reference: `athena_st/4th-order` commit `38b4ea1bf579`
- Current scope: Newtonian HD on Cartesian uniform and locally order-reduced
  SMR/AMR grids; no FLD, MHD, or relativity

The existing WOR-120 and WOR-162 worktrees and the BSCF repository were not modified.
`time/xorder=4wz` is rejected for FLD, MHD, and relativity until their separate
numerical contracts are implemented and validated. On SMR/AMR, it uses WENO-Z in
safe same-level regions and a conservative PLM fallback around coarse-fine
transitions. Characteristic `4wzc` remains rejected on multilevel meshes.

## Implemented numerical path

The HD subset follows Berta et al. (2024), JCP 499, 112701:

1. Recover cell-center point conserved values from evolved volume averages with
   the fourth-order Laplacian correction (Eq. 16).
2. Convert those point conserved values to point primitive values through the EOS.
3. Reconstruct primitive point states on faces with the five-point WENO-Z formula
   of Eq. 24--26.
4. Perform one HLLC Riemann solve for pointwise face fluxes.
5. Convert pointwise fluxes to face-average fluxes with the transverse Laplacian
   correction of Eq. 30.
6. Advance with the existing fourth-order RK integrator.

The legacy `xorder=4/4c` path is unchanged. It reconstructs face-average PPM states,
de-averages those states, and performs a second Riemann solve. The 4wz path needs only
one Riemann solve per face.

The uniform-grid path now includes the Berta/Jameson density-and-pressure curvature
sensor, one-cell dilation, and a local van-Leer PLM fallback. Troubled faces also skip
the fourth-order transverse flux correction. A point-state quality gate rejects
non-finite or large density/pressure excursions before they reach reconstruction.

On SMR/AMR meshes, the complete radius-three footprint around every coarse-fine
transition is forced onto the same local fallback. `time/correct_ic=true` is rejected
on multilevel meshes because Athena++'s midpoint initial-condition correction is not
coarse-fine aware; production SCF data use `time/correct_ic=false`.

## Validation

Build:

```text
python3 configure.py --prob=linear_wave --coord=cartesian --flux=hllc \
  --nghost=4 -mpi --cxx=g++ --mpiccmd=/usr/bin/mpicxx
make -j4
```

All runs used HLLC, `integrator=rk4`, `xorder=4wz`, `correct_ic=true`, and
`correct_err=true`.

### Two-dimensional left-going sound wave

| Nx1 x Nx2 | RMS-L1 error | measured order |
|---:|---:|---:|
| 16 x 8 | 3.475397e-8 | - |
| 32 x 16 | 1.354085e-9 | 4.682 |
| 64 x 32 | 6.760068e-11 | 4.324 |
| 128 x 64 | 6.236959e-12 | 3.438 |
| 256 x 128 | 5.510431e-12 | roundoff limited |

The entropy-wave errors at the same resolutions were `2.461055e-8`, `9.471842e-10`,
`4.649607e-11`, `2.793100e-12`, and `1.508892e-12`.

### Three-dimensional left-going sound wave

| Nx1 x Nx2 x Nx3 | RMS-L1 error | measured order |
|---:|---:|---:|
| 16 x 8 x 8 | 4.671578e-8 | - |
| 32 x 16 x 16 | 2.192300e-9 | 4.413 |
| 64 x 32 x 32 | 1.238193e-10 | 4.146 |

The entropy-wave errors were `3.231063e-8`, `1.534710e-9`, and `8.710554e-11`,
with finest-pair order 4.139.

The complete `hydro4/hydro_linwave_3d` regression, including the pre-existing
`RK4+4c` and `SSPRK5_4+4` controls, passed. A 64 x 32 x 32 run split into 16-cubed
MeshBlocks produced byte-identical error files with one and two MPI ranks.

The smooth-wave convergence values above are unchanged after adding the troubled-cell
path. In particular, the 64 x 32 x 32 RMS-L1 error remains `1.238193e-10`. The
one-rank and two-rank, 16-cubed MeshBlock error files have the same SHA-256 digest:
`110a44d09a148cde0305bfed844e50d8488e63937aab29cfc669cac02db57395`.

### General-EOS discontinuities

The six existing hydrogen general-EOS Riemann conditions were run with HLLC,
`integrator=rk4`, `xorder=4wz`, and 256 cells. Every run completed with finite,
positive density and pressure. Errors below are measured against the regression
suite's exact hydrogen Riemann solver and all satisfy its existing tolerances.

| case | density L1 | pressure L1 | velocity L1 |
|---:|---:|---:|---:|
| 1 | 3.390395e-10 | 4.471681e-11 | 2.685362e-3 |
| 2 | 5.751343e-9 | 9.660648e-10 | 4.969766e-3 |
| 3 | 6.483969e-8 | 1.253814e-8 | 5.253033e-3 |
| 4 | 4.013655e-8 | 1.423257e-8 | 1.150833e-2 |
| 5 | 4.162444e-7 | 4.274404e-8 | 6.744184e-3 |
| 6 | 3.411669e-7 | 3.145498e-8 | 4.890738e-3 |

### Static refinement

A 64 x 32 x 32 sound wave was evolved for one period through the five level-1 static
refinement regions in `athinput.linear_wave3d`, using 16-cubed MeshBlocks and two MPI
ranks. Both methods used midpoint initialization and the same 143 cycles.

| method | RMS-L1 error | largest max/L1 |
|---|---:|---:|
| VL2 + `xorder=2` | 3.612619e-8 | 5.3560 |
| RK4 + `xorder=4wz` | 2.439165e-8 | 6.3540 |

The 4wz composite-grid error is about 1.48 times smaller. Total mass and total energy
in the periodic history output remain unchanged at the printed precision from
`6.75000` and `6.07500`; the existing conservative restriction and reflux paths are
unchanged.

### Cost comparison

For the 64 x 32 x 32 sound-wave case in the serial regression runner:

| method | RMS-L1 error | CPU time |
|---|---:|---:|
| RK4 + legacy `4c` | 3.59475e-10 | 36.54 s |
| RK4 + `4wz` | 1.23819e-10 | 13.88 s |

This local smooth-wave result makes 4wz about 2.6 times faster and 2.9 times more
accurate than the legacy fourth-order path. It is not yet a stellar-surface or AMR
performance estimate.

## SCF surface-AMR smoke tests

The production-facing tests use the adiabatic SCF RSG problem, HLLC, two XD nodes
and 112 MPI ranks. They do not yet include FLD. BSCF and its repository remain
untouched.

An existing WOR-162 surface-L3 restart could not be reused because it was written by
an `NGHOST=2` binary, while 4wz requires `NGHOST=4`; the per-MeshBlock restart record
sizes are intentionally incompatible. A fresh `NGHOST=4` state is therefore required.

The first fresh surface-L2 A/B smoke (`343421`) completed both one-cycle cases with
2,472 MeshBlocks and 10,125,312 active cells. It exposed repeated gravity multigrid
warnings caused by requesting a `1e-12` absolute defect after the FMG solution had
already converged as far as this AMR hierarchy permits. The warning is not an FLD
FGMRES problem: FGMRES exists in the NRFLD linear solver, not the self-gravity driver.

The input now uses the WOR-162-validated efficient equivalent,
`gravity/mgmode=fmg`, `threshold=-1`, `niteration=0`. This retains the accepted FMG
solution and omits only the divergent polishing V-cycle. The repeated surface-L2 job
`343434` produced zero multigrid warnings and completed normally:

| method | wall time | max mass drift | max J drift | max mechanical-energy drift | max radial Mach RMS |
|---|---:|---:|---:|---:|---:|
| RK2 + x2 | 7.96 s | 0 | 2.22e-9 | 5.44e-10 | 1.699e-4 |
| RK4 + 4wz | 8.23 s | 4.27e-12 | 1.46e-9 | 1.35e-7 | 1.707e-4 |

The one-cycle runtime difference is only 3.4% and is too short for a production
throughput conclusion. Cause-separated job `343467` shows that the 4wz fallback mask
covers 5,940,960 of 10,125,312 active cells (58.67%): 46.80% comes from the Berta
density/pressure curvature sensor and another disjoint 11.87% is added by the
coarse-fine safety band. The counts sum exactly. Weighting by the stellar tracer gives
a smaller fallback fraction of 37.76%, so 62.24% of the stellar mass remains on 4wz.

The reference threshold is the intentionally conservative `1e-2`; it has not been
relaxed for these acceptance runs. The radius-3 coarse-fine band occupies 18.75% of a
16-cubed block width on every touching face. A 32-cubed comparison (`343471`) reduces
that fraction to 9.375% and reduces ghost-inclusive cell-centered storage per active
cell from `(24/16)^3=3.375` to `(40/32)^3=1.953`. Its coarser AMR decision granularity
refines a wider physical shell, so the active-cell count grows rather than remaining
fixed:

| MeshBlock | blocks | active cells | x2 wall | 4wz wall | estimated x2/4wz aggregate RSS | sensor fallback | AMR-added fallback | total fallback | stellar-mass fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16^3 | 2,472 | 10.13M | 8.08 s | 8.40 s | 48.9 / 64.1 GiB | 46.80% | 11.87% | 58.67% | 37.76% |
| 32^3 | 680 | 22.28M | 7.86 s | 9.51 s | 53.3 / 74.7 GiB | 23.03% | 7.99% | 31.03% | 13.80% |

Although 32-cubed blocks evolve 2.20 times as many active cells, one-cycle 4wz wall
time rises by only 13% and aggregate memory by about 16%. They leave 86.2% of the
stellar mass on the high-order path, versus 62.2% for 16-cubed blocks. The result is
promising but not yet a production choice because initialization dominates these
short timings; both layouts need the 100-cycle gate.

The L3 workflow creates the expensive `NGHOST=4` state once with `nlim=0`, writes one
forced final restart, and branches RK2+x2 and RK4+4wz from that byte-identical state.
Delaying the normal restart `next_time` avoids a redundant multi-gigabyte initial
restart. A C++ comparison tool writes final A/B differences and the WENO-Z fallback
fractions to JSON.

## Next gates

1. Run 100-cycle surface-L2 A/B gates for both 16-cubed and 32-cubed blocks, including
   several dynamic-AMR updates and steady-state throughput measurements.
2. Rehearse the `nlim=0` seed/restart/A-B branch workflow at surface L2.
3. Generate the fresh central-L6 plus surface-L3 `NGHOST=4` state with the selected
   block size and run the
   same-restart HD A/B comparison.
4. If the L3 one-cycle gate passes, extend the more promising configuration far enough
   to assess secular stability and production throughput.
