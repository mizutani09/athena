# WOR-30: FLD coefficient and MG matrix update audit (2026-10-23)

## Intended and observed call graph

One implicit `Solve_general(stage, dt)` performs:

1. `linearMGDriver::BeginTimeStep()` invalidates the cached hierarchy.
2. `NRFLD::LoadVariables()` resets
   `fixed_linear_coefficients_initialized_`.
3. After NR state boundary exchange,
   `NRFLD::CalculateCoefficientsOnce()` evaluates face opacity/flux-limiter
   closure and mixed-frame face drift from the initial implicit-step state.
4. `NRFLD::CalculateCoefficients()` builds the initial residual, source and
   Schur-reduced fine-grid coefficients, then marks the fixed coefficients
   initialized.
5. Each new Newton direction calls `CalculateCoefficients()` again.  It
   reevaluates the nonlinear thermal residual, gas-EOS derivative, reduced RHS
   and local Schur terms, but it does not reevaluate the face limiter/opacity
   closure.
6. `linearMGDriver::Solve()` loads the current RHS and coefficient data,
   restricts the required coefficient hierarchy, assembles every level's
   matrix with that level's `dt/dx^2`, and runs MG.

Line-search trials reuse one direction and therefore do not rebuild the
Jacobian unless an expert retry explicitly requests a new linear solve.

## Cache paths

The first linear solve after `BeginTimeStep`, FMG mode, an AMR update, or
`cache_coefficient_hierarchy=false` loads/restricts every coefficient and calls
`CalculateMatrixAll()`.

On subsequent MGI solves with caching enabled:

- without implicit mixed-frame drift, only `DCCS` changes.  The code loads and
  restricts `DCCS` through MeshBlock, octet and root levels, then rebuilds all
  matrices;
- with implicit mixed-frame coupling, the `*S` drift coefficients also contain
  the current local Schur factor.  The code deliberately reloads/restricts all
  coefficients and rebuilds all matrices.

The scalar cached `DCCS` path would be wrong for a nonzero drift row, because
full coefficient restriction separates and rescales its center/face drift
parts.  The `mixed_implicit` full-rebuild condition prevents that combination
in the current code.

## Numerical cache control

An opt-in coefficient/matrix fingerprint was added to the existing coefficient
diagnostic.  It reports per linear solve and MG level the L1/L2 norms of all
coefficient slots and assembled matrix slots; it does not change the solve.

For variable-limiter Test C q=5e5, both uniform fine and aligned AMR were run
with cache enabled and disabled from the same executable:

- uniform: both converge in 3 Newton iterations, zero rejected V-cycles;
- aligned AMR: both converge in 4 Newton iterations, zero rejected V-cycles;
- coefficient and matrix fingerprints agree at every recorded solve and level;
- every numerical final HDF dataset, including `Er` and `sigma_R`, is
  byte-identical between cache modes.

Raw results:

- `tst/outputs/wor30_cache_true_20261023`
- `tst/outputs/wor30_cache_false_20261023`

## Conclusion and remaining risk

No stale/new coefficient mixture is found in the single-rank, uncoupled Test C
cache path.  In particular, the q=7e5 uniform failure should not be described
as limiter coefficients changing during Newton; they are lagged for the
implicit step and the failure is already a linear-MG behavior.

Still unverified are a coupled two-variable/Schur case where `DCCS` changes,
implicit mixed-frame drift where the full hierarchy must change, MPI>1, an
actual AMR topology update, general-EOS derivatives, and the collapse failure
state.  These checks are required before closing WOR-30.
