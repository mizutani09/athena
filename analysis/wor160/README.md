# WOR-160 conservative SCF-to-Athena++ mapping artifacts

The C++ mapper reads the 2048 x 32 axisymmetric SCF table and volume-averages
the conserved HD variables onto Cartesian cells with tensor Gauss-Legendre
quadrature.  The recommended audit uses a 64^3 root mesh, five nested static
refinement levels, and third-order quadrature in each coordinate.

## Recommended result

| diagnostic | relative error |
|---|---:|
| stellar mass | -2.17765e-5 |
| Athena discrete Jz | -5.03364e-6 |
| ideal-HD total energy | 2.98079e-3 |
| recovered pressure L1 | 4.33667e-6 |
| cell-center vs cell-average density L1 | 5.09024e-4 |
| cell-center vs mass-weighted temperature L1 | 1.77119e-3 |
| radiative/convective interface radius | -8.88858e-4 |
| mapped force residual L2 | 6.90504e-2 |

The atmosphere contributes `2.44371e28 g`, which is tracked independently
from the `3.04156e34 g` mapped stellar mass.  The minimum cell width is
`7.47334e10 cm = 0.211 h`.

The force residual is measured rather than accepted as an equilibrium result.
WOR-161 will add and reconcile the Athena++ discrete gravitational source,
softened core, atmosphere and boundary treatment; WOR-160 does not claim a
stable time evolution without that work.

Regenerate the summary and figure:

```sh
python3 analysis/wor160/analyze_mapping.py
```

Run the C++ loader, remap audit and compiled Athena++ smoke test:

```sh
python3 tests/scf_mapping/test_mapping.py --athena bin/athena
```

The small CSV files record uniform-grid failure and SMR convergence.  The
recommended composite slice is `smr_slice_root64_l5.csv`, and the actual
Athena++ history output is `athena_root64_l5.hst`.
