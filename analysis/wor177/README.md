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
