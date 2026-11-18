# WOR-161 Athena++ gravity alignment artifacts

The production result uses the WOR-160 root 64^3 mesh with five nested static
refinement levels.  Athena++ solves gas self-gravity with isolated l<=4
multipole boundaries.  The compact spline core is an external field and is not
inserted into the Poisson source.

## Recommended initial-state result

| diagnostic | result |
|---|---:|
| stellar mass error | -2.17765e-5 |
| angular momentum error | -5.03364e-6 |
| HD energy error | 2.98076e-3 |
| gas-potential mass-weighted L2 error | 5.82891e-4 |
| self/core gravitational-energy error | 1.54949e-4 / 2.83916e-4 |
| force mass-weighted L2 residual | 6.20517e-2 |
| force residual median / p90 / p99 on the equatorial slice | 1.05e-3 / 1.05e-2 / 1.03e-1 |
| atmosphere mass | 2.55217e27 g |

Changing the atmosphere reference density from 1e-15 to 1e-13 g cm^-3
changes the force L2 residual by less than 6e-9 relative.  The maximum is not
a robust convergence measure because it selects individual cells at the
stellar/atmosphere classification edge.

The force map shows two localized limitations: the unresolved stellar surface
and weaker square imprints at static-refinement interfaces.  At the root-grid
surface spacing of 34.375 Rsun, the pressure scale height is only 0.66 cell at
rho=1e-9 g cm^-3 and 0.16 cell at rho=1e-10 g cm^-3.  The central spline
support radius is independently resolved by 4.74 finest cells.

`gravity_slice_root64_l5.csv` contains the potential, force, work, and torque
fields.  Regenerate the JSON and figures from an Athena++ UOV tab directory:

```sh
wor161_tabs=$(mktemp -d /tmp/wor161-tabs.XXXXXX)
bin/athena -i inputs/rsg/athinput.scf_rsg_map -d "$wor161_tabs" \
  time/nlim=0 output2/dt=-1 output3/dt=1
python3 analysis/wor161/analyze_gravity.py --tab-dir "$wor161_tabs"
```

Run the C++ unit, mapping audit, and gravity-enabled Athena++ smoke test:

```sh
python3 tests/scf_mapping/test_mapping.py --athena bin/athena
```
