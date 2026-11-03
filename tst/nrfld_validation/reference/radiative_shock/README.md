# Steady radiative-shock reference snapshots

These CSV files are copied from the user's July 2026 runs so the validation
runner works without those output directories:

| Case | Historical source | SHA-256 |
|---|---|---|
| Mach 2 | `/mnt/d/output/rad/test_20260703/test_mach2/mach2_semianalytic_bvp.csv` | `cac52345494e4cc60e339036d24e4c6100550f182d453a671d822d9d350f65c3` |
| Mach 5 | `/mnt/d/output/rad/test_20260703/test_mach5/mach5_semianalytic.csv` | `59df4baa78b2317d3858c05669b2a52a78190af88843b4e0502b485af7fb5665` |

They represent steady semi-analytic profiles, with a branch discontinuity at
the embedded shock. The runner checks and records their hashes and copies the
exact CSV bytes into each result directory. It does not regenerate a new
reference for every simulation. The historical solution scripts are preserved
here for inspection. The Mach 5 BVP script is an alternative formulation; it
was not established as the exact producer of the saved Mach 5 CSV.

As a cross-check, the July Mach 5 CSV was compared by interpolation at 1000
locations with the later August reference used in
`/home/kosuke/simulation/test/rad_nr/runs/mach5_uniform_latest_20260915/`.
The two agree in relative L1 to about `2.4e-7` for density, `1.6e-7` for gas
temperature, and `3.1e-8` for radiation temperature. This checks consistency
between saved references, not their physical correctness.
