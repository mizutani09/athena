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
stellar mass on the high-order path, versus 62.2% for 16-cubed blocks. The sustained
gates below make 32-cubed blocks the production choice.

The B16 100-cycle gate (`343482`) completed both evolutions. Its batch status is
`FAILED` only because the original postprocessor required identical final times to
`1e-12`, while two fixed-cycle CFL runs ended at `0.99352325` and `0.99350330` code
time (a relative difference of `2.01e-5`). Reprocessing with the corrected comparator
passes and records both times explicitly.

| B16, 100 cycles | x2 | 4wz |
|---|---:|---:|
| wall time | 34.71 s | 88.28 s |
| main-loop time | 22.39 s | 79.98 s |
| zone-cycles/s | 4.522e7 | 1.414e7 |
| approximate aggregate RSS | 64.3 GiB | 76.8 GiB |
| initial/final MeshBlocks | 2,472 / 2,472 | 2,472 / 2,808 |
| maximum mechanical-energy drift | 4.87e-4 | 1.69e-2 |
| maximum radial Mach RMS | 2.48e-3 | 2.49e-3 |

The x2 case passes all nine HD gates. The 4wz case passes mass, angular momentum,
Mach, perturbation-energy, radius, and virial gates, but fails the `1e-2` mechanical-
energy gate. The energy changes are discrete rather than runaway: 4wz surface AMR
grows through 2,640, 2,752, and 2,808 blocks in the first 40 cycles, while x2 remains
at 2,472. The largest energy excursion coincides with the first topology change and
the result settles near `-1.015e-2` after the last change. Final 4wz fallback is 48.10%
of cells but only 9.92% of stellar mass, improved from 58.67% and 37.76% initially.
The B16 layout is therefore not the production choice unless a continuation proves
that the regrid offset is bounded; the B32 100-cycle result is the deciding gate.

The B32 100-cycle gate (`343484`) passes all nine HD acceptance gates for both
methods. In contrast to B16, neither branch changes topology: both remain at 680
MeshBlocks for all 100 cycles. The 4wz mechanical-energy error is smaller than the
B16 value by a factor of about 1,007, so the B16 failure is a topology/granularity
effect rather than secular instability of 4wz.

| B32, 100 cycles | x2 | 4wz |
|---|---:|---:|
| wall time | 29.02 s | 99.17 s |
| main-loop time | 20.92 s | 90.23 s |
| zone-cycles/s | 1.065e8 | 2.469e7 |
| approximate aggregate RSS | 53.8 GiB | 75.0 GiB |
| initial/final/maximum MeshBlocks | 680 / 680 / 680 | 680 / 680 / 680 |
| maximum mass drift | 0 | 4.62e-10 |
| maximum mechanical-energy drift | 7.63e-8 | 1.68e-5 |
| maximum radial Mach RMS | 2.47e-3 | 2.49e-3 |
| final total/stellar-mass fallback | 0 / 0 | 31.25% / 13.84% |

At fixed cycles, 4wz costs 3.42 times the wall time and 4.31 times the measured
main-loop time of x2 in this gravity-dominated short run. Its aggregate RSS is about
1.39 times larger. The two methods nevertheless agree to `4.62e-10` in stellar mass,
`2.19e-8` in angular momentum, and `1.67e-5` in mechanical energy at the final time.
The B32 layout combines the lower fallback coverage, lower block/ghost overhead, and
stable AMR topology needed for the surface-L3 production gate.

The seed/restart rehearsal (`343483`) also passes. Fresh initialization and the one
forced restart took 63.38 s; the x2 and 4wz one-cycle branches took 11.26 and 11.72 s.
Exactly one 1.913 GB restart was written, with SHA-256
`2ab9a3dcd18de7bf211c57e9ffb0002d506afb51a6fa5276da37723032ba887a`.
Both branches reproduce the earlier one-cycle diagnostics without multigrid warnings,
NaNs, or runtime errors.

The L3 workflow creates the expensive `NGHOST=4` state once with `nlim=0`, writes one
forced final restart, and branches RK2+x2 and RK4+4wz from that byte-identical state.
Delaying the normal restart `next_time` avoids a redundant multi-gigabyte initial
restart. A C++ comparison tool writes final A/B differences and the WENO-Z fallback
fractions to JSON.

The B32 surface-L3 seed and one-cycle A/B job (`343497`) passes. Initialization creates
2,416 MeshBlocks (79,167,488 active cells), takes 144.04 s on three XD nodes, and
writes one 8.659 GB restart. Both methods retain exactly the same topology:
264 blocks at L2, 1,976 at L3, 56 each at L4 and L5, and 64 at L6.

| B32 surface L3, one cycle | x2 | 4wz |
|---|---:|---:|
| wall time including restart read | 47.49 s | 40.71 s |
| main-loop time | 1.029 s | 3.714 s |
| zone-cycles/s | 7.691e7 | 2.132e7 |
| approximate aggregate RSS | 128 GiB | 173 GiB |
| maximum mass drift | 0 | 1.64e-12 |
| maximum mechanical-energy drift | 8.28e-9 | 1.29e-7 |
| maximum radial Mach RMS | 1.699e-4 | 1.707e-4 |
| total/stellar-mass fallback | 0 / 0 | 21.69% / 18.75% |

The wall values are dominated by the shared restart read and must not be used as a
solver-speed ratio. Main-loop throughput makes 4wz about 3.61 times more expensive per
cycle. At the initial `dt=0.01008`, a conservative constant-throughput estimate for
`0.1 t_dyn = 47.006` is about 1.3 h for x2 and 4.8 h for 4wz, both within the 12 h
allocation. The dependent sustained jobs use the same seed and three nodes.

The first sustained 4wz attempt (`343499`) was stopped by the Slurm memory cgroup at
cycle zero; it was not a numerical failure.  Its 168 ranks averaged about 2.01 GiB RSS
and one rank reached about 3.90 GiB, exhausting the 110 GiB-per-node limit on three M
nodes.  The x2 job `343498` completed normally at `t=47.006`, cycle 4,752, with fixed
2,416-block topology and passes all nine HD gates.  Its batch state is `FAILED` only
because that version disabled final restart output and then required the file in its
postprocessor; the evolution step itself is `COMPLETED`, stderr is empty, and the
history is intact.  The tentative P-node replacement `343525` remains only as a
fallback until the reduced-storage M-node run starts.

The selected memory2 implementation is documented separately in
`wor120_hd_4wz_memory_experiment_20261120.md`.  It reduces the original Berta-path
logical storage by 40.47 GiB over this 2,416-block mesh, keeps Hydro on RK4+WENO-Z,
and transports passive scalars with conservative PLM.  The exact 100-cycle M-node
gate `343554` completes with 1.28 GiB average RSS per rank and fixed topology.  The
sustained replacement `343579` therefore uses the original three-node M allocation,
the same accepted restart, and the same target time; dependent job `343580` performs
the physical and measured-cost A/B comparison.  Its executable SHA-256 is
`042fcd1701f4215c038a38ec80746ca7671163ba06472eb8c9aa8219b7e9811e`.

An isolated B16 experiment (`343506`) transports the stellar/radiative tracers with
PLM while leaving Hydro 4wz unchanged. It eliminates the B16 topology growth: the 4wz
branch remains at 2,472 blocks, its maximum energy drift falls from `1.69e-2` to
`4.92e-4`, and wall time falls from 88.28 to 68.62 s. This confirms that the B16
failure is driven primarily by the unrelated legacy fourth-order scalar path feeding
the tracer AMR condition.  Memory2 adopts this conservative PLM scalar path: it is an
explicitly second-order tracer contract, while the HD flux update remains fourth
order in smooth regions, and it also permits omission of scalar-only legacy
fourth-order block storage.

## Next gates

1. Complete memory2 job `343579` and its dependent comparison `343580`; compare
   stability, accuracy, topology, restart integrity, memory, throughput, and allocated
   node-hours against completed x2 job `343498`.
2. If the sustained result passes, retain three 110 GiB M nodes as the minimum tested
   4wz production layout and cancel the redundant pending P-node fallback `343525`.
3. Record explicitly in production inputs that HD uses RK4+4wz while passive scalar
   transport uses PLM; fourth-order tracer transport would require its own Berta mask
   and convergence contract and is outside this HD acceptance gate.
