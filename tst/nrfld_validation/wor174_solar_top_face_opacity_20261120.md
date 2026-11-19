# WOR-174 solar top-face opacity reconstruction

Date: 2026-11-20 JST

## Scope

This change records the previously uncommitted WOR-91 upper-boundary
experiment as a solar-application option.  It does not change the general FLD
operator and is disabled by default:

```text
problem/reconstruct_top_face_opacity = false
```

At a physical global upper boundary, the option assumes `log(sigma_R)` is
linear between the two outermost cell centres, integrates that extrapolation
over the outer half cell, and supplies a positive ghost value whose arithmetic
average with the outer cell is that integral average.  The arithmetic branch
is the active branch of the capped FLD face-opacity rule in the optically thin
solar upper atmosphere.  Internal z-MeshBlock faces are not modified.

The Planck coefficient is deliberately unchanged: it is a cell-local
matter-radiation coupling coefficient rather than a face diffusion
coefficient.

## Provenance and isolation

- branch: `exp/wor174-solar-top-face-optin-20261120`
- base: `1193d9394bc002acb4410181e7e5b370de3f3441`
- worktree: `/home/kosuke/simulation/athena-wor174-solar-top-face-optin-20261120`
- results: `/home/kosuke/simulation/test/rad_nr/wor174_solar_top_face_optin_20261120`
- local validation binary SHA-256:
  `ca7c733f47a83448677497c92bb79090f8b272c43de0b6653725e2921723e865`

The existing dirty WOR-91 worktree was read for provenance only and was not
modified.

Input data SHA-256 values:

- NATA EOS: `981b6dad13d1ceb55352088dd15797bfa73d6a12408e3503c5c538a395c938fe`
- rho-T opacity: `cc5912dd3896c39c1ea3fde0d315dc16c136303a8c363ab58fa56b705b909aaf`
- initial profile: `22c4c592d8dc9d48c4208b62a87f28f43fc58dfcafaa9a837a3f952bba3400f2`

## Validation

### Pure reconstruction test

Command:

```sh
g++ -std=c++11 -Wall -Wextra -pedantic -O2 \
  tst/nrfld_validation/test_solar_top_face_opacity.cpp \
  -o /tmp/wor174_top_face_test
/tmp/wor174_top_face_test
```

PASS.  The test covers the constant-opacity limit, the analytical exponential
integral, a near-constant cancellation case, a steep-gradient floor, and
negative/non-finite inputs.

### Full application build

An isolated source copy was configured with
`simple_convection_lhllc_fld`, Cartesian coordinates, NATA table EOS, LHLLC,
MPI, HDF5, and NR-FLD.  `make -j4` completed without error.

### Real-solver A/B smoke test

The strongest local smoke case used 2 MPI ranks, a `32 x 32 x 16` mesh, and
`8 x 8 x 8` MeshBlocks.  It therefore contains two vertical MeshBlock layers
and exercises the distinction between internal and physical upper faces.
Each case ran two hydrodynamic timesteps.

| option | Newton | linear solves / FGMRES iterations | final top flux | wall | max RSS |
|---|---:|---:|---:|---:|---:|
| omitted (default) | 2/2 converged, 6 iterations | 6 / 70 | `6.27508e10` | 1.31 s | 85,448 KiB |
| explicit `false` | 2/2 converged, 6 iterations | 6 / 70 | `6.27508e10` | 1.24 s | 85,564 KiB |
| `true` | 2/2 converged, 6 iterations | 6 / 70 | `6.27977e10` | 1.26 s | 85,276 KiB |

The default and explicit-false history files are byte-for-byte identical.
The enabled case changes the intended top radiative flux by about 0.075%
without changing Newton or FGMRES iteration counts.  A separate 1-rank smoke
test also passed, with 2/2 Newton steps converged and the same default-off
bitwise check.

No non-finite opacity, table-domain violation, Newton failure, or FGMRES
breakdown was reported in any accepted smoke case.

## Decision

PASS as an opt-in solar boundary closure and PASS for default-off regression.
This is not evidence to enable the option globally or by default.  A long,
same-restart solar A/B run remains necessary before selecting it for a
production solar setup; the option is now reproducible and testable for that
experiment.
