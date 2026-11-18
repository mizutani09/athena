# WOR-162 HD-relaxation scale and cost budget

## Decision

The first undamped reference should remain the existing root64 + central-L5
mesh and run for three dynamical times.  It is cheap enough to expose the
dominant mode and gives the required baseline before changing the mesh.

The first surface-convergence mesh should add block-local surface refinement
through level 3, selected by the stellar tracer and a pressure-scale-height
criterion.  This gives about 5.2 cells per pressure scale height at
`rho=1e-9 g cm^-3`.  A level-4 surface run is a convergence check, not the
default production mesh.  Surface level 5 is not justified: the minimum
one-block-thick shell alone makes it a roughly 200-million-cell calculation.

These estimates are intentionally separated from the forthcoming XD MPI
benchmark.  The cell and physical-step estimates are geometric; the memory
and CPU-hour estimates are calibrated from a local serial run and must be
updated with measured parallel efficiency before submitting the long run.

## Physical scales

The target rotating model has `M_gas=15.30 Msun`, `M_core=5.70 Msun`,
`R_eq=960.12 Rsun`, and `R_p=877.17 Rsun`.  Its inner layer interface is at
about `68.25 Rsun`, and the compact spline core support radius is
`h=5.091 Rsun`.

| quantity | value | interpretation |
|---|---:|---|
| mean ellipsoidal density | `3.661e-8 g cm^-3` | total gas + core mass |
| code time | `2.2000e4 s = 0.25463 d` | Athena input time unit |
| `sqrt(R_eq^3/GM)` | `1.0341e7 s = 119.69 d` | adopted dynamical time, `470.06` code time |
| uniform-sphere free-fall time | `1.1486e7 s = 132.94 d` | `pi/sqrt(8) t_dyn` |
| equatorial acoustic crossing | `2.2530e7 s = 260.76 d` | center-to-surface integral of `dr/c_s` |
| rotation period | `1.5708e8 s = 4.98 yr` | `15.19 t_dyn` |
| `Omega/Omega_K` | `0.4137` | target is rotating but not near shedding |
| equatorial rotation speed | `26.72 km s^-1` | initial rigid rotation |
| surface escape speed | `91.35 km s^-1` | at `R_eq` |
| maximum sound speed | `510.8 km s^-1` | central region; controls the CFL step |

The one-way acoustic crossing is `2.18 t_dyn`, so a three-`t_dyn` run sees
one global acoustic crossing and several dynamical response times, but only
0.20 rotation periods.  It is sufficient for mapping-induced radial drift;
it is not sufficient to measure a slow rotational instability.

The root32 benchmark settled to `dt=0.03336` code time after 20 steps.
Scaling the central-L5 spacing by two gives `dt=0.01668` for root64, or about
`2.82e4` steps per dynamical time and `8.45e4` steps for the reference run.

## Surface scales and resolution

The pressure scale height is computed directly from the SCF equatorial ray as
`H_P=|d ln P/dr|^-1`.  It agrees with the independent WOR-161 estimate.

| density | radius | `H_P` | root64 cells per `H_P` | level-3 cells per `H_P` |
|---:|---:|---:|---:|---:|
| `1e-8 g cm^-3` | `734.8 Rsun` | `68.36 Rsun` | 1.99 | 15.9 |
| `1e-9 g cm^-3` | `898.8 Rsun` | `22.46 Rsun` | 0.65 | 5.23 |
| `1e-10 g cm^-3` | `945.8 Rsun` | `5.56 Rsun` | 0.16 | 1.29 |
| `1e-11 g cm^-3` | `956.9 Rsun` | `1.23 Rsun` | 0.036 | 0.29 |

The existing surface spacing is `34.375 Rsun`; the stellar diameter is only
about 56 cells and the equator-pole radius difference is 2.4 cells.  Central
refinement cannot fix this.  Conversely, the spline support diameter has 9.48
finest cells and the inner layer interface lies in the level-4 region, so the
central length scales are already much better resolved than the surface.

Density alone does not identify the eventual FLD photosphere.  The level-3
recommendation therefore targets the HD mapping error at `rho=1e-9`, not a
claim that the radiation decoupling layer is resolved.  WOR-166 must combine
this with optical-depth and opacity information.

## Cell, memory, and CPU budget

The shell estimates use the surface area of the oblate spheroid and the
minimum shell thickness of one 16-cell MeshBlock at each refinement level.
They include all coarser transition shells.  Memory uses the measured local
root32 resident set (`1.831748 GB` for `1,179,648` active cells), and CPU cost
uses `103.25 CPU s` for 20 steps.  The benchmark wrote the expensive gravity
history every five cycles, so its per-zone CPU cost is conservative for the
production history cadence; MPI communication is not represented.

| surface level | surface `dx` | cells per `H_P` at `1e-9` | estimated active cells | memory | CPU h / `t_dyn` |
|---:|---:|---:|---:|---:|---:|
| 0 | `34.375 Rsun` | 0.65 | `1.41e6` | 2.1 GiB | 48 |
| 1 | `17.188 Rsun` | 1.31 | `2.00e6` | 3.0 GiB | 69 |
| 2 | `8.594 Rsun` | 2.61 | `4.37e6` | 6.5 GiB | 150 |
| 3 | `4.297 Rsun` | 5.23 | `1.38e7` | 20.5 GiB | 474 |
| 4 | `2.148 Rsun` | 10.45 | `5.17e7` | 76.6 GiB | 1771 |
| 5 | `1.074 Rsun` | 20.90 | `2.03e8` | 301 GiB | 6959 |

The level-3 value is about 10 times the baseline cost but does not reduce the
timestep, because the existing central L5 region already controls it.  On 112
physical cores, the ideal level-3 cost is 4.2 wall hours per `t_dyn`; a
realistic request must use the measured XD efficiency rather than this ideal
number.  Level 4 should initially run for only `0.25--0.5 t_dyn` as a
convergence discriminator.

At ten stored double-precision fields, an uncompressed full snapshot is about
`1.1 GB` for level 3 and `4.1 GB` for level 4.  Full 3-D output should therefore
be no more frequent than `0.25 t_dyn`; equatorial slices and reduced radial
profiles can be written much more often.

## Initial mapped-state audit

WOR-162 adds C++ history diagnostics for radial and nonradial perturbation
energy, radial Mach number, radius moments, gravitational and pressure virial
terms, mass beyond the initial oblate surface, and surface-tracer extent.

At root64 before evolution:

- mass-weighted radial Mach number is `2.54e-5`;
- the maximum radial Mach number in cells with stellar fraction above 0.5 is
  `7.34e-2`, localized to the low-sound-speed surface;
- the mapped virial residual is `8.22e-3`;
- the nonradial perturbation energy is `5.42e-9` of the gravitational virial
  scale.

The BSCF equilibrium virial error is `2.59e-5`.  The much larger mapped value
is therefore an Athena discretization/mapping signal, consistent with the
surface and SMR force residual measured in WOR-161.  Root32 gives a still
larger mapped virial residual of about `2.6e-2`.

## Execution sequence

1. Benchmark root64 on XD at several MPI task counts, with history output
   disabled except at the beginning and end.
2. Run the undamped root64 baseline for `3 t_dyn` and evaluate mass, angular
   momentum, mechanical energy, radial Mach, perturbation energy, radius, and
   virial drift.
3. Implement tracer/`H_P`-based block refinement and run surface level 3 for
   at least `1 t_dyn`; extend to `3 t_dyn` if it materially reduces drift.
4. Use a short level-4 run only to establish convergence of the level-3
   result.
5. Generate and test the nonrotating and `Omega=2e-8 s^-1` SCF profiles in the
   same order.  No damping is part of the reference result.

The long-run input is `inputs/rsg/athinput.scf_rsg_relaxation`.  It uses a
100-cycle history cadence, `0.01 t_dyn` slices, `0.1 t_dyn` restarts, and
`0.25 t_dyn` full primitive snapshots.

## Reproduction

```sh
g++ -O2 -std=c++11 -Wall -Wextra -pedantic \
  analysis/wor162/estimate_relaxation_scales.cpp \
  -o /tmp/estimate_relaxation_scales
/tmp/estimate_relaxation_scales data/ic/rsg21_scf_omega4e-8.csv \
  > analysis/wor162/scale_estimates.json

g++ -O2 -std=c++11 -Wall -Wextra -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o /tmp/analyze_relaxation_history
/tmp/analyze_relaxation_history RUN/scf_rsg_relax.hst \
  RUN/relaxation_diagnostics.csv RUN/relaxation_summary.json
```

Machine-readable scale estimates and the root64 initial audit are stored in
`analysis/wor162/`.
