# WOR-162 HD-relaxation scale and cost budget

## Decision

The original root64 + central-L5 run exposed a non-conservative external-core
energy update, not a resolution requirement.  Replacing the cell-centred
velocity kick work with face-mass-flux work reduced the `0.0213 t_dyn`
mechanical-energy error from 26.6% to `1.29e-6` and the radial Mach RMS from
0.0770 to 0.0138 at the same resolution.

Central L5 (`dx=1.074 Rsun`, 4.74 cells per core-support radius) is the minimum
accepted mesh.  Central L6 (`dx=0.537 Rsun`, 9.48 cells per support radius) is
the quality-production recommendation: it reduces radial kinetic energy by a
factor 32 relative to L5 while costing 2.47 times as much.  L7 provides little
additional improvement in RMS Mach and is reserved for convergence checks.

The first surface-convergence mesh should add block-local surface refinement
through level 3, selected by the stellar tracer and a pressure-scale-height
criterion.  This gives about 5.2 cells per pressure scale height at
`rho=1e-9 g cm^-3`.  A level-4 surface run is a convergence check, not the
default production mesh.  Surface level 5 is not justified: the minimum
one-block-thick shell alone makes it a roughly 200-million-cell calculation.

At `0.1 t_dyn`, central L5 conserves mechanical energy to `6.54e-6`, but a
compact-core nonradial signal and `1.52e-3` angular-momentum drift remain.
An equatorial slice places 99.9999% of nonradial perturbation energy inside
`20 Rsun`.  Central L6 removes this failure: at the same time its nonradial
energy is `2.44e-5 |W|` and angular-momentum drift is `8.32e-5`.  The remaining
Mach signal is confined to the low-mass surface, so FLD remains a no-go until
surface refinement.  The cell and physical-step estimates are geometric.
Memory and CPU-hour columns below are calibrated from a local serial run; XD
MPI measurements supply the production wall-clock estimates.

## Physical scales

The target rotating model has `M_gas=15.30 Msun`, `M_core=5.70 Msun`,
`R_eq=960.15 Rsun`, and `R_p=877.21 Rsun`.  Its inner layer interface is at
about `68.56 Rsun`, and the compact spline core support radius is
`h=5.091 Rsun`.

| quantity | value | interpretation |
|---|---:|---|
| mean ellipsoidal density | `3.661e-8 g cm^-3` | total gas + core mass |
| code time | `2.2000e4 s = 0.25463 d` | Athena input time unit |
| `sqrt(R_eq^3/GM)` | `1.0341e7 s = 119.69 d` | adopted dynamical time, `470.06` code time |
| uniform-sphere free-fall time | `1.1486e7 s = 132.94 d` | `pi/sqrt(8) t_dyn` |
| equatorial acoustic crossing | `2.2536e7 s = 260.83 d` | center-to-surface integral of `dr/c_s` |
| rotation period | `1.5708e8 s = 4.98 yr` | `15.19 t_dyn` |
| `Omega/Omega_K` | `0.4137` | target is rotating but not near shedding |
| equatorial rotation speed | `26.72 km s^-1` | initial rigid rotation |
| surface escape speed | `91.35 km s^-1` | at `R_eq` |
| maximum sound speed | `510.8 km s^-1` | central region; controls the CFL step |
| core-support sound crossing | `6.93e3 s = 0.080 d` | rapid central adjustment scale |

The one-way acoustic crossing is `2.18 t_dyn`, so a three-`t_dyn` run sees
one global acoustic crossing and several dynamical response times, but only
0.20 rotation periods.  It is sufficient for mapping-induced radial drift;
it is not sufficient to measure a slow rotational instability.

At `rho=1e-9 g cm^-3`, the local `H_P/c_s` time is 14.2 days or
`0.119 t_dyn`.  Motion that grows much earlier than this but after several
0.080-day core crossings points first to the compact-core handoff rather than
to a global surface response.

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
refinement cannot fix this.  At central L5 the spline support diameter has
9.48 finest cells; L6 raises this to 18.96 cells.  The inner layer interface
lies in the level-4 region.  The short central convergence study confirms
that L6 is sufficient, whereas the surface remains grossly underresolved.

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
timestep, because the existing central L5 region already controls it.  Level
4 should initially run for only `0.25--0.5 t_dyn` as a convergence
discriminator.

### XD MPI calibration

The XD2000 benchmark used one shared node, Intel oneAPI 2025.1, 1,409,024
active cells, and 50 steps.  History was written only at the beginning and
end.

| MPI ranks | wall time | main-loop time | zone-cycles/s |
|---:|---:|---:|---:|
| 8 | 45.95 s | 43.10 s | `1.63e6` |
| 16 | 26.43 s | 23.58 s | `2.99e6` |
| 32 | 15.82 s | 13.75 s | `5.12e6` |
| 56 | 12.02 s | 9.28 s | `7.59e6` |

At 56 ranks the original steady benchmark cost was `0.186 s/step`.  The
conservative core-work L5 run, including normal history and slice output,
measured `1.27 h/t_dyn` over the first `0.0213 t_dyn` and `1.36 h/t_dyn` over
`0.1 t_dyn`.  A three-`t_dyn` central-L5 run is therefore a 5-hour-class
one-node allocation.  Central L6 measured `3.14 h/t_dyn`, and L7 measured
`7.28 h/t_dyn`.

Scaling the measured rate by cell count gives about `12--14 h/t_dyn` for
surface level 3 with central L5 on 56 ranks; two nodes should be budgeted at
roughly `7--10 h/t_dyn` until that larger mesh is benchmarked.  Combining the
quality-recommended central L6 with surface level 3 approximately doubles the
step count, giving `25--30 h/t_dyn` on one node or a prudent
`15--20 h/t_dyn` two-node allocation.  A surface-level-4 `0.25 t_dyn`
discriminator is already an approximately 8-hour, two-node calculation with
central L5 and about twice that with central L6.

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

The production BSCF equilibrium virial error is `9.02e-6`.  The much larger mapped value
is therefore an Athena discretization/mapping signal, consistent with the
surface and SMR force residual measured in WOR-161.  Root32 gives a still
larger mapped virial residual of about `2.6e-2`.

## Quantitative acceptance gates

The undamped three-`t_dyn` result is acceptable as an HD handoff only if all
of the following hold after the first `0.25 t_dyn` transient:

| metric | gate | reason |
|---|---:|---|
| stellar mass drift | `<1e-4` | exclude boundary/floor-driven envelope loss |
| target-model angular momentum drift | `<1e-3` | keep numerical torque below the intended rotation signal |
| total HD + self + core mechanical-energy drift | `<1e-2` | limit known non-conservative gravity-source accumulation |
| mass-weighted radial Mach RMS | `<0.03` | artificial radial flow remains below expected RSG convective speeds |
| radial kinetic energy / `|W_self+W_core|` | `<5e-4` | independent energetic bound on coherent pulsation |
| nonradial perturbation KE / `|W_self+W_core|` | `<5e-4` | bound artificial shear and angular-momentum redistribution |
| mean and RMS stellar-radius drift | `<1%` | smaller than a root surface cell but resolvable in mass moments |
| absolute virial residual | `<0.02` and non-growing | allow the measured `8.2e-3` initial mapping residual without secular runaway |

The local maximum Mach number is reported but is not a standalone rejection
gate because one underresolved, low-sound-speed surface cell dominates it.
It must decrease with surface refinement.  Surface level 3 is considered
resolved for the HD handoff only if a level-4 discriminator changes the RMS
Mach, radius drift, and radial-energy peak by less than 10%.  Passing the
global root64 gates without this surface convergence is a conditional HD pass,
not yet an FLD go decision.

## Early-time central-resolution discriminator

Undamped runs were compared at the same `t=10` code time
(`0.02127 t_dyn`, 2.546 days), varying only the innermost static-refinement
level.  The compact-core support radius is `5.091 Rsun`.  The tabulated wall
cost includes the common history and equatorial-slice output and was measured
on one 56-core XD node.

| core-work scheme | central level | minimum `dx` | cells / support radius | steps | wall h / `t_dyn` | max radial Mach RMS | final radial KE / `|W|` | max virial residual | max mechanical-energy drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cell-centred kick | 5 | `1.074 Rsun` | 4.74 | 700 | 1.71 | `7.70e-2` | `1.281e-2` | `3.90e-2` | 26.6% |
| cell-centred kick | 6 | `0.537 Rsun` | 9.48 | 1,116 | 3.43 | `3.80e-2` | `4.446e-3` | `1.19e-2` | 17.4% |
| cell-centred kick | 7 | `0.269 Rsun` | 18.96 | 1,998 | 7.16 | `1.56e-2` | `4.527e-4` | `3.22e-3` | 11.1% |
| face-flux work | 5 | `1.074 Rsun` | 4.74 | 507 | 1.27 | `1.377e-2` | `2.431e-4` | `8.22e-3` | `1.29e-6` |
| face-flux work | 6 | `0.537 Rsun` | 9.48 | 1,008 | 3.14 | `1.265e-2` | `7.502e-6` | `3.05e-3` | `3.21e-6` |
| face-flux work | 7 | `0.269 Rsun` | 18.96 | 2,011 | 7.28 | `1.261e-2` | `8.066e-7` | `1.73e-3` | `1.04e-5` |

The old apparent `dx^0.6` energy convergence was a source-discretization
artifact.  `CoreGravitySource` now evaluates external-potential work with the
two face mass fluxes and their potential differences, matching Athena++'s
self-gravity/point-mass pattern.  For a static core potential this conserves
`E_gas + integral(rho Phi_core dV)` up to boundary and AMR reflux errors.  It
also removes the artificial heating that reduced the CFL step in the old
level-5 run.

With face-flux work, all central levels conserve mass to printed precision,
have angular-momentum drift below `2.3e-5`, and pass the short-time Mach,
radial-energy, radius, virial, and mechanical-energy gates.  L6 and L7 agree
in Mach RMS to 0.3%; L6 radial energy is 32 times below L5 and already only
`7.5e-6` of the binding scale.  L7 costs another factor 2.3 for a quantity
that is already negligible.  This sets L5 as the accepted floor, L6 as the
quality recommendation, and L7 as a convergence-only mesh.

The L5 extension to `0.1 t_dyn` keeps mechanical-energy drift at `6.54e-6`,
radial energy at `2.32e-4 |W|`, mean-radius drift at `1.13e-4`, and mass drift
at zero.  However, radial Mach RMS reaches 0.204, nonradial perturbation energy
reaches `2.38e-2 |W|`, and angular-momentum drift reaches `1.52e-3`.  The
small radial energy and radius drift rule out renewed expansion.  The final
equatorial slice instead places 99.9999% of nonradial energy and 99.88% of
radial energy inside `20 Rsun`: central L5 is an acceptable short discriminator
but not a multi-dynamical-time production resolution.

The corresponding L6 extension passes the energetic and conservation gates:
mechanical-energy drift is `1.52e-5`, angular-momentum drift `8.32e-5`, radial
energy `7.14e-6 |W|`, nonradial energy `2.44e-5 |W|`, mean-radius drift
`7.82e-5`, and mass drift zero.  The radial Mach RMS still reaches 0.204, but
the equatorial profile identifies its cause: the `950--1000 Rsun` bins have
Mach 23--40 while containing only `5.5e-6` of the slice radial energy.  Inside
`25 Rsun`, where 99.78% of radial and 99.97% of nonradial slice energy reside,
the Mach RMS is only 0.0101.  This separates the now-resolved compact-core
energetics from the remaining surface-resolution error.

The resolution prescription is therefore:

- central L5 is sufficient for economical discriminators; central L6 is the
  validated quality HD reference and L7 remains a short convergence check;
- add surface level 3 (`dx=4.30 Rsun`, 5.2 cells per
  `H_P` at `rho=1e-9 g cm^-3`) before a multi-dynamical-time reference;
- use a surface-level-4 short run to require less than 10% change in RMS Mach,
  angular-momentum drift, radius drift, and perturbation-energy peaks.

## Execution sequence

1. Keep the validated face-mass-flux core-work update and central L6.
2. Implement tracer/`H_P`-based block refinement and run surface level 3 for
   at least `1 t_dyn`; extend to `3 t_dyn` if it materially reduces drift.
3. Use a short surface-level-4 run only to establish convergence of the
   level-3 result.
4. Test the generated nonrotating and `Omega=2e-8 s^-1` SCF profiles in the
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

g++ -O2 -std=c++11 -Wall -Wextra -pedantic \
  analysis/wor162/analyze_relaxation_slice.cpp \
  -o /tmp/analyze_relaxation_slice
/tmp/analyze_relaxation_slice RUN 00001 \
  RUN/slice_00001.csv RUN/slice_00001.json
```

Machine-readable scale estimates and the root64 initial audit are stored in
`analysis/wor162/`.  The level-5/6/7 comparison is in
`analysis/wor162/core_resolution_comparison.csv`; each `root64_*` directory
contains its history-derived diagnostics and equatorial-slice radial-energy
profiles.
