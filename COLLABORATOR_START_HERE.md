# Protostellar first-core calculation: collaborator source v1

This branch is a **fixed, scoped research version** for 3D Cartesian AMR
protostellar collapse with a Tomida H/He material EOS, Semenov opacity, and
Newton–FGMRES–MG FLD. It is suitable for starting a related calculation
before the FLD paper source is finalized. It is not a claim that every
problem generator, boundary condition, or EOS is qualified.

## Obtain and identify the code

```bash
git clone --branch collab/protostellar-collapse-first-core-v1 --single-branch \
  https://github.com/mizutani09/athena.git athena-firstcore-v1
cd athena-firstcore-v1
git rev-parse HEAD:src
# Expected: c4ab200383c7317a2feccc50aa9578d5cf71478d
sha256sum inputs/radiation/athinput.protostellar_firstcore_tabular_v1
# Expected: aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6
```

Use this named branch at a pinned commit or the `protostellar-firstcore-v1`
tag. The branch and tag identify the source; the data archive described below
is supplied separately. The solver `src/` tree is byte-identical to the
validated first-core run source `ff60f8f07842acb00e7753932bcfe30867e3a66e`.
The [qualification record](doc/fld_collapse_source_delta_20261116.md)
describes the diffusion, Jeans AMR, coupling, conservation, streaming,
restart, and mixed-frame checks on that source.

### Branch names

| Reference | Purpose |
| --- | --- |
| `collab/protostellar-collapse-first-core-v1` | **Use this** for the collaborator calculation and report its pinned commit/tag. |
| `codex/wor114-collapse-candidate-20261116` | Integration audit and wider paper-source candidate from which this branch was made; retain for traceability. |
| `codex/wor95-table-source-regression-20261116` | Isolated diffusion/Jeans test evidence on the same solver source. |
| `wor-120-radiation-high-order` and WOR-122 worktrees | Separate, still-changing RSG/high-order research; do not merge into this version. |
| `master` | Upstream-tracking branch of the fork, not this FLD source. |

Existing branches are intentionally retained because their issues, reports,
and Draft PR refer to them. The new `collab/` name and immutable tag make
the student-facing choice unambiguous.

## Obtain the two tables

Ask the project owner for `protostellar_firstcore_v1_assets.tar.gz`; the
archive is kept outside public Git. Extract it anywhere and verify it:

```bash
sha256sum protostellar_firstcore_v1_assets.tar.gz
# Expected: 86b9d4b1c5f533eed704b2521b49db9693cd6d67f3500f5e596c8450254779b1
tar -xzf protostellar_firstcore_v1_assets.tar.gz
cd protostellar_firstcore_v1_assets
sha256sum -c SHA256SUMS
```

The verified table SHA-256 values are:

| Asset | SHA-256 |
| --- | --- |
| `data/eos/protostellar_tomida2015_material.tab` | `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43` |
| `data/opacity/protostellar_semenov2003_nrm_c_a.h5` | `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94` |

The input uses table paths relative to the **process working directory**.
Run from the extracted asset directory, or use the smoke script, which stages
the tables into an isolated build. The table files are research assets;
their redistribution terms must be checked before giving the archive outside
the collaboration.

## Run the short acceptance check

The script checks the frozen source tree, exact input, both table hashes,
builds an isolated MPI/HDF5 executable, and performs one table-EOS AMR step
with the recorded numerical environment. It rejects a fatal message even if
an MPI launcher returns zero.

```bash
cd /path/to/athena-firstcore-v1
python3 tst/nrfld_validation/collapse_firstcore_v1_smoke.py \
  --assets-root /path/to/protostellar_firstcore_v1_assets \
  --output-root /path/to/new/smoke_result --mpi-ranks 2 --build-jobs 2
```

The result directory contains `summary.json`, full logs, the binary,
effective input, HST/HDF5 output, and resource use. Build requirements are
an MPI C++ compiler, parallel HDF5, Python 3 with NumPy and h5py, and
`/usr/bin/time`. Compiler and binary hashes differ by machine; the solver
source, input, table hashes, and physical acceptance checks are portable.
The [recorded local check](examples/protostellar_firstcore_v1/verification.json)
passed with both 1 and 2 MPI ranks on the exact input and archive. Both
runs converged in one Newton solve with two FGMRES calls. Their maximum
true relative residuals were `5.659e-8` and `5.735e-8`, respectively,
accepted under the stated `1e-7` inexact cap; gas-mass relative changes
were `−3.00e-15` and `−3.22e-15`. Peak density matched exactly, and peak
temperature differed by `6.55e-13` relative. This checks startup and MPI
reproducibility; the long XD result above establishes first-core arrival.

For a manual build, the verified source configuration is:

```bash
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table -b --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -hdf5 -mpi --cflag=-O3
make -j2
source examples/protostellar_firstcore_v1/env.sh
mkdir -p /path/to/new/output
cd /path/to/protostellar_firstcore_v1_assets
mpirun -np 2 /path/to/athena-firstcore-v1/bin/athena \
  -i /path/to/athena-firstcore-v1/inputs/radiation/athinput.protostellar_firstcore_tabular_v1 \
  -d /path/to/new/output time/nlim=1 time/ncycle_out=1
```

The input is byte-identical to the XD cycle-0 input. It starts with
`mesh/numlevel=2` and `time/nlim=30`; the `time/nlim=1` override above is a
smoke test. The full first-core calculation increased the maximum AMR level
at selected checkpoints and used segmented restarts. Its historical XD
scripts under `tst/nrfld_validation/wor92_firstcore_20261109/` record those
decisions but contain site-specific paths and are not a portable launcher.
Plan memory and a Jeans-resolution stop rule for a new long run.

## What has been demonstrated

The exact solver source, input, and tables were used for XD jobs
338469/338485/338486 from cycle 0 through 1080 at CFL 0.3. Peak density
reached `1.04414476e-10 g cm^-3` at 29.11046 kyr with a first-core shock
candidate at 7.507–7.874 au. Gas-mass relative variation was at most
`5.28e-15`; the minimum resolved Jeans length was 31.59 cells. The XD
binary SHA-256 was
`908d5def56faf5e0993526f1c709b2593eec814a2a26ffd83d4a0133b6ac0810`.
The same `src/` tree passes local short regressions; this handoff does not
claim that the full XD trajectory was rerun on another compiler.

The tested path explicitly enables the conservative AMR prototype,
shared coefficients, tangential reconstruction, variable limiter, coupled
operator, regrid, assembled Jv, centered face, positive Newton ghosts, and
bounded-inexact FGMRES acceptance through `env.sh`. These are part of the
numerical configuration, not optional performance settings. The accepted
linear true-residual cap is `1e-7`; a claim that every linear solve reached
the nominal `2e-8` target would be incorrect. Newton convergence remains
its separate acceptance criterion.

## Scope for a related project

The verified application uses a nonrotating, nonmagnetic 1-solar-mass sphere,
periodic gas/radiation boundaries, multipole gravity boundaries, cubic
Cartesian cells, and a fixed background treatment. The Semenov Planck
**extinction** table is used as an absorption proxy. The background fixing
acts as a source/sink, so total energy of the full collapse is not expected
to be constant. This version demonstrates first-core formation, not H2
dissociation or second-core formation.

The multilevel conservative operator is guarded for specific problem
generators and geometries. A new problem generator, different boundaries,
noncubic cells, different EOS/opacity, or altered background/reset ordering
requires a short validation at the changed condition before a long run.
Do not silently fall back to `ATHENA_NRFLD_LEGACY_INTERFACE=1`; that selects
the nonconservative AMR operator for diagnostics. RSG high-order and surface
AMR work (WOR-120/122) remains on separate branches. The optional
post-implicit ambient reset change from WOR-115 is not part of the validated
table run because `background_sink_after_implicit=false`.

For the publication-wide source decision, see [WOR-114](https://linear.app/work-on-fld/issue/WOR-114).
For this application's measured physical limits, see [WOR-115](https://linear.app/work-on-fld/issue/WOR-115).
