# RSG material-EOS production-candidate validation (2026-10-30)

## Scope

Uniform-grid `rsg_star_in_box` with the material-only Tomida-2013 EOS,
temperature-extended gray opacity, table-consistent convective HSE model,
nonlinear FLD limiter, matter-radiation coupling, radiation force, and a
six-face Marshak-vacuum boundary.

## Physics assets

- EOS: `data/eos/rsg_tomida2013_material_eos.tab`, SHA-256
  `7051d1c5222f3ca7dddbbba1c48d473cb39bc5e76badb869b082625473a71576`.
- Opacity: `data/opacity/rsg_gray_opacity_t7.h5`, SHA-256
  `6f5d39922be2436daf45c049c081adcf65e2b71ff15ab05ef4c066413ae65a14`.
- HSE profile: `data/ic/st35gm03n07_tomida_gray_hse.dat`, SHA-256
  `d24ab36d205cc1561267b0d7e0b2c39a71d592fac52a0da3ea88e5361c394227`.
- HSE mass inside 830 Rsun: 2.9992386261 Msun.
- Resolved-interior HSE residual: max 0.0084845, RMS 0.0007773.
- Entropy forward/inverse round trip on 257 profile points: maximum errors
  0.0001296 dex in energy, 0.0009395 dex in temperature, and 0.0012976 dex
  in pressure.

The opacity high/low-temperature tails are an audited engineering
continuation of the supplied OPTAB snapshot.  Source composition and license
are not embedded in that snapshot, so this remains a production *candidate*,
not a publication-qualified opacity dataset.

## Build

```text
BUILD_JOBS=16 scripts/configure_rsg.sh
```

Configuration: Cartesian, general table EOS, HLLC-FLD, NR-MG-FLD, MPI,
OpenMP, and HDF5.  Production pgen: `rsg_star_in_box`.

## 32^3 hydro-CFL test

Input: `inputs/rsg/athinput.st35gm03n07_fld`, with command-line overrides
`time/nlim=100`, `mesh/nx1,2,3=32`, and `meshblock/nx1,2,3=32`.

- Fluid CFL number: 0.3.
- Result: all 100 steps converged; no backtracking exhaustion.
- Newton iterations: 3--5 per step.
- FGMRES iterations: typically 7--16 per Newton solve, relative target 1e-4.
- Nonlinear target: 1e-8.  The observed final maximum residual remained below
  the target; representative worst early value was 8.89e-9.
- Simulated code time: 448.7981 (approximately 114.3 days with the configured
  21999.97 s code-time unit).
- Wall time: 125.99 s; maximum RSS: 125180 KiB; one MPI rank, one thread.
- Opacity diagnostics: out-of-domain=0, clamped=0, negative=0, zero=0,
  nonfinite=0.

The hydrodynamic CFL step decreased from 6.64 to a minimum near 0.38 as the
initial model developed fluid motion.  This was normal CFL adaptation, not an
FLD retry or solver-imposed radiation time step.

Run directory:
`work_runs/freytag_rsg/production_pgen_32_cfl03_100cycle_20261030/`.

## Resolution smoke tests and floor-plateau fix

The first 64^3 initialization exposed 32 cells whose density was one ULP below
`hydro/dfloor`.  The 1-D profile is exactly equal to the physical density floor
there; the discrepancy came only from `exp(interp(log(rho)))`.  The pgen now
accepts at most 64 machine epsilons of floor-roundoff, snaps those cells to the
exact floor while preserving specific internal energy, and retains a fatal
error with cell diagnostics for any material floor violation.  The same scan
finds 280 such one-ULP cells at 128^3.

- 64^3, 8 blocks of 32^3, CFL=0.3: 3/3 steps converged, Newton 4--5,
  FGMRES 9--12, wall 41.09 s, max RSS 702168 KiB, opacity violations zero.
- 128^3, 64 blocks of 32^3, CFL=0.3: 1/1 step converged, Newton 4,
  FGMRES 13--15, final max residual 1.98e-11, wall 106.93 s, max RSS
  5335736 KiB, opacity violations zero.

Run directories:
`work_runs/freytag_rsg/production_pgen_64_cfl03_3cycle_floorfix_20261030/`
and
`work_runs/freytag_rsg/production_pgen_128_cfl03_1cycle_floorfix_20261030/`.

## Regression coverage

The following NR-FLD regression modules pass after the production changes:

- `general_eos_exchange_jacobian`: analytic material/radiation exchange
  Jacobian against finite differences.
- `coupled_residual_norms`: coupled residual normalization and convergence.
- `coupling_mode`: HLLC and LHLLC in legacy Riemann-pressure mode, the new
  gas-only Riemann-star/source-force mode, and radiation-force-off mode.
- `parameter_contract`: normal, radiation-only, fixed-opacity,
  coupling-disabled, opacity-table-disabled, and rejected-parameter cases.
- `opacity_domain_policy`: fatal and clamped out-of-domain policies.
- `adiabatic_gradient`: general-EOS thermodynamic derivative behavior.

The coupling test now distinguishes where radiation pressure enters the
Riemann star state from the explicit FLD force.  The parameter-contract test's
synthetic opacity domain and single-block table-disabled case were widened so
that it tests the parameter contract rather than failing in an unrelated
polytropic ghost cell outside its deliberately tiny table.

## Final production-build smoke test

After restoring the production build, a fresh 32^3, one-block, CFL=0.3 run
converged in five Newton iterations.  Its FGMRES solves required 7--8
iterations, the final maximum nonlinear residual was 1.67e-12, and all
opacity diagnostics were zero.  Wall time was 2.11 s and maximum RSS was
124972 KiB.  Run directory:
`work_runs/freytag_rsg/final_32_cfl03_1cycle_20261030/`.

## Revalidation

```text
python3 tools/rsg/validate_rsg_production.py
```

The validator checks the input physics switches, asset checksums, table
domains, absence of EOS sentinel holes, HSE mass/residual, and entropy
round-trip accuracy.  It passed all 26 checks for the recorded assets. Runtime
solver validation still requires the Athena++ command above.
