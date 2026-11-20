# Protostellar-collapse solver v2

This document describes the frozen first-core solver candidate intended for
collaborator calculations.  Use branch
`release/protostellar-collapse-solver-v2-20261120` and its annotated release
tag; do not substitute `main` or an older AMR-FLD branch.

## Build

```sh
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table -b --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -hdf5 -mpi --cflag=-O3
make -j4
```

The validated assets are:

- `data/eos/protostellar_tomida2015_material.tab`, SHA-256
  `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`;
- `data/opacity/protostellar_semenov2003_nrm_c_a.h5`, SHA-256
  `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`.

## Run

```sh
source inputs/radiation/protostellar_firstcore_v2.env
mpirun -np 2 bin/athena \
  -i inputs/radiation/athinput.protostellar_firstcore_tabular_v1 \
  -d output
```

The supplied input selects FGMRES restart 40, maximum 300 iterations, and two
MG V-cycles.  Do not increase the restart width as an initial response to a
difficult step: the release measurements found no convergence or timing
benefit at restart 160, while the basis memory was about four times larger.
MG=1 is also not the shared default because it reached the linear cap on a
real collapse checkpoint.

The normal calculation uses the accurate conservative tangential AMR face
form.  If that complete nonlinear solve fails, the opt-in recovery in the
environment file reloads the uncommitted step and retries it consistently with
the robust two-point face form.  It does not accept a failed Newton state and
does not respond to unrelated runtime errors.  Thus the robustness cost is
paid only on a failing step.

## Quick validation

```sh
python3 tst/nrfld_validation/collapse_solver_v2_smoke.py \
  --assets-root /path/containing/data \
  --output-root /new/empty/result/directory \
  --mpi-ranks 2 --build-jobs 4
```

The smoke script refuses a changed solver source tree, physical input, or
table.  A complete record of the parameter selection, local regressions, and
XD cycle-964--970 gate is in
`tst/nrfld_validation/protostellar_collapse_solver_v2_20261120.md`.

Changing initial conditions, rotation, magnetic field, refinement criteria,
EOS, or opacity is supported as application work, but the new setup must be
checked for table-domain coverage, positive finite states, conservation, and
Newton convergence.  This freeze validates first-core calculations; it does
not by itself extend the tables or resolution to second-core formation.
