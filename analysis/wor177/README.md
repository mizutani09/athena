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

## Reproduction

```sh
python3 configure.py --prob=scf_rsg --coord=cartesian --eos=adiabatic \
  --flux=hllc --nscalars=2 --grav=mg --nghost=4 --cxx=g++
make -j4
python3 tests/scf_mapping/test_mesa_mapping.py --athena bin/athena
```

The upstream BSCF-rsg repository is unchanged; profile generation parameters
and checksums are recorded in `source_manifest.json`.
