# WOR-120: HD 4wz uniform-grid validation (2026-11-19)

## Scope and isolation

- Worktree: `/home/kosuke/simulation/athena-wor120-hd4wz`
- Branch: `wor-120-hd-4wz`
- Base: WOR-162 commit `1f3647935e4e`
- Reference: `athena_st/4th-order` commit `38b4ea1bf579`
- Current scope: Newtonian HD, Cartesian uniform grid, no FLD, no MHD

The existing WOR-120 and WOR-162 worktrees and the BSCF repository were not modified.
`time/xorder=4wz` is rejected for FLD, MHD, relativity, and SMR/AMR until their
separate numerical contracts are implemented and validated.

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

The coarse-fine footprint mask remains deferred. Consequently `xorder=4wz` is still
rejected for SMR/AMR and is not yet ready for the central-L7 plus surface-L3 star.

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

### Cost comparison

For the 64 x 32 x 32 sound-wave case in the serial regression runner:

| method | RMS-L1 error | CPU time |
|---|---:|---:|
| RK4 + legacy `4c` | 3.59475e-10 | 36.54 s |
| RK4 + `4wz` | 1.23819e-10 | 13.88 s |

This local smooth-wave result makes 4wz about 2.6 times faster and 2.9 times more
accurate than the legacy fourth-order path. It is not yet a stellar-surface or AMR
performance estimate.

## Next gates

1. Mark the full three-cell footprint around coarse-fine interfaces as troubled so
   existing conservative restriction and reflux remain authoritative.
2. Validate static-refinement wave crossing and conservation.
3. Build with the SCF table EOS and run positivity/domain smoke tests.
4. Only then enable the central-L7 plus surface-L3 SCF input for a short HD A/B run.
