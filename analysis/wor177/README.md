# WOR-177: MESA-fit SCF to Athena++ HD mapping

## Outcome of the first gate

The accepted WOR-175 zero-rotation model can be mapped into the existing
Athena++ `scf_rsg` problem, but its compact core requires a better source SCF
resolution and three more central refinement levels than the old fixed-index
model.  The accepted source is `nr=16384`, `nmu=16`, `lmax=12`, with a
`1.2 R_photo` radial domain.  The Athena mesh is root64 over +/-1100 Rsun with
central L8, giving `dx_min=9.34167e9 cm = 0.13428 Rsun` and 5.49 cells per
softened-core support radius.

The old `nr=4096` WOR-175 analysis profile remains a valid SCF convergence
result, but is rejected for Cartesian mapping.  The SCF code integrates
sampled `4 pi r^2 rho` and `4 pi r^2 P` with the trapezoid rule, while the
mapper integrates the piecewise-linear fields between samples.  Their exact
spherical differences expose the unresolved core:

| SCF radial points | cells/core support | mass interpolation error | pressure-volume error | SCF virial |
|---:|---:|---:|---:|---:|
| 4,096 | 2.81 | 1.0109% | 18.5357% | 0.6171% |
| 16,384 | 11.25 | 0.06225% | 1.2015% | 0.04149% |

The roughly factor-16 error reduction from a factor-four resolution increase
is the expected second-order behavior.  The `nr=16384` source is therefore the
minimum accepted HD-map profile; `nr=32768` remains the convergence option.

## Conservative map gate

The standalone root64/L8 audit and the actual NGHOST=4 Athena++ initialization
agree:

| diagnostic | standalone map | Athena++ initialization | gate |
|---|---:|---:|---:|
| stellar mass error | 0.06313% | 0.06313% | < 0.1% |
| ideal-gamma internal-energy error | 1.2193% | 1.2193% | < 2% |
| force L2 | 0.06118 | 0.05522 | < 0.1 |
| multigrid potential L2 | -- | 0.003839 | < 0.01 |
| mapped virial residual | -- | 0.03302 | < 0.05 |

The energy gate refers to the Athena ideal-gamma representation
`integral(P dV)/(gamma-1)`, not the cold-barotrope internal energy used for the
WOR-175 binding-energy comparison.  This distinction must remain explicit in
later HD diagnostics.

## One-cycle x2 / 4WZ smoke

Both methods complete on the same 512-block central-L8 mesh without NaNs.
The physical duration is only about `2.3e-6 t_dyn`, so this is an immediate
update gate, not a stability result.

| method | energy drift | radial Mach RMS | nonradial KE/abs(W) | virial | local CPU |
|---|---:|---:|---:|---:|---:|
| RK2 + x2 | `+3.53e-6` | `2.94e-3` | `4.92e-5` | 0.0250 | 6.84 s |
| RK4 + 4WZ | `-6.31e-6` | `3.11e-3` | `5.85e-5` | 0.0246 | 25.93 s |

At this mesh topology 4WZ falls back to PLM in 43.2% of available cells and
those cells contain 50.0% of stellar mass, primarily because every nested
central level introduces coarse-fine faces.  The one-cycle state is therefore
not yet a clean fourth-order stellar evolution comparison.

## Remaining resolution constraint

The MESA-fit photospheric layer is substantially sharper than the old
fixed-index model.  At `rho=1e-9 g cm^-3`, `H_P=4.45 Rsun`; root64 has only
0.130 cells per scale height and surface-L3 has 1.04.  Surface-L6 would provide
8.29 cells per scale height but is not affordable as the old whole-surface
block shell.  The next implementation step is a narrower gradient/enthalpy
selected surface hierarchy or a controlled atmosphere match.  A longer HD
job should not be submitted before that choice is tested, because central L8
already implies roughly 451,000 cycles per dynamical time.

The first adaptive audit exposed an additional implementation constraint.
The legacy tracer criterion selects every block whose stellar mass fraction
lies anywhere between `1e-6` and `0.999999`.  With a `1e-14 g cm^-3`
atmosphere, that is a thick outer volume rather than a narrow surface.  Its
serial L3 initialization consumed about 31 GiB of RAM plus 26 GiB of swap.
WOR-177 therefore adds `surface_refinement_use_tracer=false`, which follows
only the requested density isosurface while preserving the old behavior by
default.  Even this narrow central-L8 plus surface-L3 mesh approached 29 GiB
during serial initialization, so all further adaptive audits run as
distributed-memory XD jobs; the local workstation is limited to compilation
and small unit tests.

The first two-node, 112-rank order-3 L3 gate (XD job 344217) reached the
30-minute `M-test-a` limit during adaptive initialization.  It did not report
a numerical or allocation failure.  Slurm measured at most 1.266 GiB RSS per
rank (about 142 GiB summed over ranks), which fits the requested two-node
220-GiB allocation, but no history row was produced before timeout.  The next
gate therefore uses one cell-center sample only to determine the AMR topology
and resource cost.  This is not an accepted map-quality run; order 3 remains
required for the physical initialization once the mesh cost is known.

The one-point topology gate (XD job 344235) also timed out after 30 minutes
at the same initialization stage and the same 1.267-GiB maximum RSS per rank.
This rules out tensor-Gauss mapping as the dominant cost.  Before extending
the queue limit, a one-node scaling gate measures central-L8-only and
surface-L1/L2 initialization separately.  This identifies whether the cost
jump is tied to a particular surface level or to the combined AMR/MG tree.

XD job 344243 completed that scaling gate in 13 seconds.  Central L8 alone
uses 512 blocks; surface L1 uses 848 blocks; surface L2 uses 2,472 blocks.
Their launcher wall times are 3.47, 3.57, and 4.37 seconds respectively, and
surface L2 uses at most 756,620 KiB RSS per rank (about 40 GiB summed over 56
ranks).  The pathological transition is therefore specifically at surface
L3.  A central-L6/surface-L3 control next distinguishes a surface-only cost
from an interaction with the two additional compact-core levels.

The central-L6/surface-L3 control (XD job 344246) completes in 11.94 seconds
with 8,688 blocks: L1=128, L2=2,312, L3=6,072, L4=56, L5=56, and L6=64.
This confirms that surface L3 itself is healthy and isolates the slowdown to
the deeper compact-core hierarchy.  Central L7 is the next compromise: its
0.2686-Rsun cell width gives 2.75 cells per 0.737-Rsun core-support radius,
versus only 1.37 cells at L6 and 5.49 cells at L8.

The root64 central-L7/surface-L3 control entered the same high-cost path as
central L8 and was stopped after 5 minutes once its RSS and progress matched
the two earlier 30-minute timeouts.  A shallower equivalent mesh is therefore
tested next: root128, surface L2, and central L6.  It retains the same
4.296875-Rsun surface spacing and 0.268555-Rsun central spacing as the
root64/L3/L7 case while reducing the maximum relative AMR level from 7 to 6.

Both follow-up controls entered the same high-cost path: root64/L3/L7 was
stopped at 5:16 and root128/L2/L6 at 1:48 after their per-rank RSS and progress
matched the prior timeouts.  Because the latter has the same physical surface
and central spacings with a shallower relative tree, the bottleneck follows
the simultaneous physical resolution range, not the relative AMR level or
SCF quadrature.  The practical mesh-only baseline is therefore root64
surface-L3 plus central-L6 (8,688 blocks, 11.94-second initialization).  Its
1.37 cells per current core-support radius are insufficient for production.
A separate BSCF worktree therefore scanned larger core-softening supports
before any Athena evolution was resumed.  Although a 3-Rsun support gives a
very small SCF virial residual and 5.59 central-L6 cells per support, it
retains only 24.7% of the accepted model's binding energy and increases the
replacement particle mass from 4.861 to 5.880 Msun.  Even the 1-Rsun case
loses 24.9% of the binding energy while remaining below two L6 cells.
Enlarging the core support is rejected as a physical model change rather than
a resolution fix.

## HD surface match and selected mesh

The accepted core must consequently retain central L8.  The affordable way
to avoid the simultaneous central-L8/surface-L3 cost path is to regularize
only the low-mass outer layer and use surface L2.  A density scan of the
source profile gives:

| match density (g cm^-3) | Hp (Rsun) | L2 cells/Hp | exterior gas mass |
|---:|---:|---:|---:|
| `4e-9` | 40.05 | 4.66 | 1.67% |
| `3e-9` | 23.14 | 2.69 | 0.734% |
| `2e-9` | 4.92 | 0.573 | 0.105% |
| `1e-9` | 4.45 | 0.518 | 0.0480% |

The sharp scale-height collapse occurs below `3e-9 g cm^-3`, so that density
is the selected connection point.  `scf_surface_taper.cpp` preserves the SCF
interior and all source potentials, then replaces the exterior with a finite
`gamma=5/3` polytrope satisfying `h + Phi_eff = constant`.  Density and
pressure are continuous at the connection.  The taper reaches zero at
`6.48295e13 cm = 931.9 Rsun`, and changes the piecewise-linear gas mass by
only `+1.297e-4` and the pressure-volume integral by `+3.606e-6`.

The root32/L9 conservative-map audit (the same minimum spacing as root64/L8)
has a `1.27e-5` mass error and `0.0611843` force L2 residual.  The latter is
unchanged at the shown precision from the unmodified profile.  The selected
HD candidate is therefore root64, original central L8, tapered surface L2,
order-3 cell mapping, and RK4+4WZ evolution.  Surface L2 plus central L8 was
already measured at 2,472 blocks and about 4.4 seconds for a one-point
topology gate; the full quadrature/MG initialization remains the next XD gate.

That gate completed as XD job 344387.  Full order-3 mapping, gravity MG, and
one RK4+4WZ cycle took 7.28 seconds on 56 ranks.  The mesh has 2,136 blocks
and the largest per-rank RSS was 977,700 KiB, corresponding to a conservative
52.2-GiB summed upper bound on the 110-GiB node.  The mapped stellar-mass
error is `9.02e-6`, ideal-gamma energy error is `-0.592%`, MG potential L2 is
`0.00650`, mapped force L2 is `0.0726`, and maximum virial residual is
`0.0308`; all accepted map gates remain satisfied.  The one-cycle 4WZ PLM
fallback covers 42.8% of stellar mass, down from 50.0% on the central-only
mesh, but a sustained gate is still needed to judge stability and fallback
evolution.

The 100-cycle sustained gate (XD job 344389) completed in 139.04 seconds with
the same 2,136 blocks and 979,524-KiB maximum RSS per rank.  Mass drift stays
below `8.23e-10`; the largest mechanical-energy drift is `5.59e-4`.  Radial
Mach RMS peaks at `0.0214` by cycle 20 and declines to `0.0181`; radial
kinetic energy similarly falls from `1.56e-3` to `9.57e-4` of `abs(W)`.
The local maximum Mach rises to `0.308`, comparable to the `0.293` surface
maximum in the earlier WOR-120 surface-L2 gate, but the old history columns
cannot identify its mass or position.  The 1,000-cycle gate therefore adds
interior/taper and density-binned maxima, the mass above Mach 0.1 and 0.3,
the mass-weighted radius and geometric-mean density of the Mach>0.1 cells,
and a cycle-1,000 restart.  This reaches approximately `0.00215 t_dyn` and is
the next decision point before a dynamical-time production run.

The 1,000-cycle gate (XD job 344410) completed in 1,339.59 seconds.  Mass and
mechanical-energy drifts are `-5.78e-9` and `-5.31e-4`; radial Mach RMS falls
from `0.0181` at cycle 100 to `0.0145`, radial kinetic energy falls to
`7.94e-4 abs(W)`, and the virial residual remains bounded at `-0.00953`.
The dense/interior maximum Mach peaks at `0.428` and falls to `0.210`.  Cells
above Mach 0.1 contain 0.0866 Msun at the end and describe an inner pulse with
mass-weighted radius 4.27 Rsun and geometric-mean density 0.157 g cm^-3.

The formal maximum Mach of 4.32 is instead in the sub-`3e-9 g cm^-3` taper.
Only `6.76e-8 Msun` is above Mach 0.3 at the end, so this is a negligible-mass
surface cell rather than a global instability.  The 2,136-block topology and
43.0% 4WZ-fallback stellar-mass fraction stay nearly fixed.  This accepts the
mesh/model as a bounded short-HD baseline.  A continuation from the saved
restart is still required to follow the outward-moving inner pulse before a
full-dynamical-time production run; the measured one-node cost is about
7.21 wall-days per dynamical time.

The three-node continuation to `0.13 t_dyn` (XD job 344423) completed normally
at the requested time, but decisively rejects this configuration as a
long-time HD solution.  There were no NaNs, runtime failures, or multigrid
iterative warnings; the failure is physical/numerical rather than a scheduler
or solver crash.  Nonradial perturbation energy first exceeds `1e-3 |W|` at
`0.00534 t_dyn`, reaches `0.113 |W|` by the cycle-5,000 checkpoint
(`0.01011 t_dyn`), and the interior Mach diagnostic becomes catastrophic at
`0.02090 t_dyn`.  Mass drift exceeds 1% at `0.02194 t_dyn` and unity at
`0.02335 t_dyn`.  The final mass and mechanical-energy drifts are therefore
not meaningful physical values (`3.48e3` and `8.35e3`, respectively).

No continuation checkpoint is accepted: even the first saved checkpoint at
cycle 5,000 already fails the nonradial-energy and mechanical-energy gates.
The cycle-1,000 restart from job 344410 remains the last accepted seed.  The
failed 31-GiB continuation directory is retained for forensic analysis and
must not be used for production.  Relative to the stable WOR-120 reference,
the conservative face-flux core-work update is present, so a missing source
term is ruled out.  The leading resolution hypothesis is instead the compact
core: its 0.737-Rsun support radius spans only 5.49 finest cells at central L8,
whereas the stable WOR-120 central-L6 model spans about 9.48 cells.  Central L9
would provide 10.97 cells.  Spatial localization from the pre-catastrophe
checkpoints and a short same-profile fresh-map L8/L9 comparison are required
before another long continuation.

The compact-core localization and L8/L9 discriminator (XD job 344576) confirms
that diagnosis.  Equatorial slices at the rejected L8 cycle-5,000 and
cycle-10,000 checkpoints place more than `99.9999998%` of nonradial kinetic
energy and more than `99.9993%` of radial kinetic energy inside `20 Rsun`.
The surface-L2 taper is therefore not the source of this failure.

A fresh-map comparison evolved both core meshes to `0.012 t_dyn`.  L8
reproduces the failure: nonradial energy reaches `0.213 |W|`, radial energy
reaches `1.81e-3 |W|`, and mechanical-energy drift reaches `1.83e-2`.
L9, with 10.97 cells across the core support radius, passes all nine short
numeric gates.  Its corresponding maxima are `1.56e-4 |W|`, `4.38e-5 |W|`,
and `1.58e-4`; radial Mach RMS remains below `0.00654`.  L9 costs 1.85 times
L8 in wall time (89.98 versus 48.52 minutes on three nodes) while using only
56 additional MeshBlocks.

This is a successful short discriminator, not yet a production acceptance.
For `t>=0.004 t_dyn`, the L9 nonradial energy follows an exponential fit with
`R^2=0.992` and e-folding time `0.00474 t_dyn`.  Extrapolation predicts the
`5e-4 |W|` gate crossing near `0.0174 t_dyn`.  The next gate must therefore
compare fresh L9 and L10 through at least `0.025 t_dyn`; L10 provides about
21.9 cells across the compact-core support radius.  Job 344576 deliberately
disabled restart output, so that comparison starts from fresh maps and must
retain final restarts.

## Diffuse core excision supersedes the L10 path

The L10 comparison above is no longer the preferred next calculation.  A
separate BSCF worktree, branch `wor-177-core-excision` at commit `44eaba0`,
moves the MESA cut from `5.70324` to `6.27977 Msun`, immediately inside the
`6.29664 Msun` radiative--convective boundary.  The excised burning shell and
radiative core are represented by a `6.2112 Msun` softened particle.  Only
the inward continuation beyond the last retained MESA state is replaced by a
matched `gamma=5/3` law; the retained convective-envelope table is unchanged.

This choice lowers the central gas density from `21.98` to
`4.81e-3 g cm^-3` and expands the compact-spline support from `0.7369` to
`4.0548 Rsun`.  The support is resolved by 7.55 L6 cells or 15.10 L7 cells,
instead of 10.97 L9 cells in the old model.  The `nr=16384` source SCF has a
`1.04e-4` scalar-virial residual and a photospheric-radius error of 0.0082%.
The surface taper changes mass by `1.44e-4` and pressure volume by `4.85e-6`.
This is an HD core/surface regularization, not an FLD photosphere.

The standalone conservative mapper gives the following fresh-map results:

| mesh | cells/support | mass error | ideal-gas energy error | force L2 |
|---|---:|---:|---:|---:|
| root64 + central L6 | 7.55 | -6.15e-7 | -1.46e-7 | 1.997e-2 |
| root64 + central L7 | 15.10 | -6.73e-7 | 3.78e-7 | 5.232e-3 |

Both maps are acceptable for an HD discriminator.  L6 is the production-cost
target and L7 is the resolution control.  Relative to the old L9 hierarchy,
their minimum cell widths are respectively 8 and 4 times larger.  Since the
Athena++ timestep is global, a first conservative cost expectation is about
one eighth and one quarter of the old L9 cycle count, with a small additional
per-step saving from removing the innermost refinement shells.  The actual
CFL and wall-time ratios must be measured by the same `0.025 t_dyn` L6/L7
fresh-map run; no long binary calculation should be priced from geometry
alone.

That discriminator completed as XD job 344840.  Both resolutions reached
`0.025 t_dyn` without NaNs, memory failures, or iterative-MG warnings and pass
all nine numeric gates.  L6 used 2,024 blocks and 1,571 cycles in 698.96 s on
three nodes; L7 used 2,080 blocks and 3,152 cycles in 1,477.71 s.  The
measured costs are therefore 23.30 and 49.26 node-hours per dynamical time,
and L7 costs 2.114 times L6 in wall time.  L6 is about 16.1 times faster than
the old compact-core L9 measurement, while L7 is about 7.61 times faster.

The L6 maxima are `6.81e-4` mechanical-energy drift, `3.14e-3` radial Mach
RMS, `5.64e-5 |W|` radial kinetic energy, and `4.15e-5 |W|` nonradial kinetic
energy.  The corresponding L7 maxima are `7.64e-4`, `4.16e-3`, `2.52e-5`,
and `1.42e-5`.  At the endpoint the nonradial energies are `1.60e-5 |W|` and
`3.57e-6 |W|`.  More than 99.4% of the nonradial energy remains inside
`20 Rsun`; the high formal surface Mach belongs to negligible taper mass.
L7 shows clear overall decay.  L6 also decays strongly from its early peak,
but has a small late increase between about `0.020` and `0.025 t_dyn`.

L6 is consequently the provisional production candidate, with L7 retained as
the resolution control.  Before binary work, the accepted L6 restart is being
continued to `1.0 t_dyn` to distinguish a harmless core transient from a
slowly growing mode.  The measured estimate for the additional evolution is
7.57 wall-hours on three nodes, within a 12-hour allocation.  A one-dynamical-
time pass, bounded virial and energy histories, and no sustained nonradial
growth are required for final HD acceptance.

This taper begins below the MESA photosphere and is deliberately an HD
regularization.  It is not accepted as an FLD photospheric model.  Radiation
work must instead restore a thermodynamically and optically controlled outer
layer and repeat its own resolution audit.

## Reproduction

```sh
python3 configure.py --prob=scf_rsg --coord=cartesian --eos=adiabatic \
  --flux=hllc --nscalars=2 --grav=mg --nghost=4 --cxx=g++
make -j4
python3 tests/scf_mapping/test_mesa_mapping.py --athena bin/athena
g++ -O2 -std=c++11 -Wall -Wextra -Wpedantic -Werror \
  tools/rsg/scf_surface_taper.cpp -o /tmp/scf_surface_taper
/tmp/scf_surface_taper data/ic/rsg21_mesa_scf_omega0_nr16384.csv \
  data/ic/rsg21_mesa_scf_omega0_nr16384_taper_rho3e-9_gamma5_3.csv \
  analysis/wor177/surface_taper_audit.csv 3e-9 1.6666666666666667
```

The original BSCF-rsg repository is unchanged.  Core-softening rejection was
performed on the separate `wor-177-core-softening` worktree; source profile
generation parameters and checksums remain recorded in `source_manifest.json`.
