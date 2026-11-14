# WOR-30: FLD coefficient and MG matrix update audit (2026-10-23)

## Completion audit, 2026-11-04

The previous single-rank, radiation-only Test C control is extended below.
No inconsistent cached MG operator was found. The diagnostic now includes
64-bit, cell-addressed coefficient and matrix fingerprints on MeshBlock, root,
and active AMR octet cells. It also checks `A*1` against the corresponding
coefficient row sums. This code runs only when
`nrfld/coefficient_diagnostics=true`.

| Quantity | Evaluated/frozen scope | MG handling |
| --- | --- | --- |
| `sigma_P`, `sigma_R` opacity | Set before an implicit stage, read at stage entry; not changed by the Newton iterate | Face values enter the fixed `*F` coefficients |
| Face limiter `lambda`, diffusion `D`, Marshak face derivative | `CalculateCoefficientsOnce` at stage entry; fixed for all Newton directions in that stage | `DCCF` and six `*F` faces copied/restricted on first solve; cached on later scalar-only updates |
| Face mixed-frame `q*v` drift | Same stage-entry face state and closure | Local Schur scaling changes each Newton direction; all six `*S` faces and `DCCS` are rebuilt/restricted when mixed drift is active |
| Gas temperature derivative, thermal exchange and `Fg` | Recomputed from each Newton gas/radiation iterate | Changes the Schur RHS and `DCCS` (and `*S` when mixed drift is active) |
| Schur RHS `src`, local diagonal `DCCS` | Every new Newton direction | Source reloaded every solve; `DCCS` reloaded/restricted on cached scalar path |
| MG matrix and optional Galerkin coarse matrix | Every linear solve from the current restricted coefficients | `CalculateMatrixAll` assembles the seven-point matrix at each level using its own `dt/dx²`; `BuildGalerkinHierarchy` then builds Galerkin levels when selected |
| Smoother/FGMRES coefficients | Never reevaluated inside a linear solve | Both consume the assembled matrix; FGMRES reports a recomputed true residual |

Call sequence: `Solve_general` → `BeginTimeStep` (invalidate hierarchy) →
`LoadVariables` (reset fixed-face flag) → stage-entry NR boundary exchange →
`CalculateCoefficientsOnce` → initial `CalculateCoefficients` → defect norm →
Newton direction task (`CalculateCoefficients`) → `linearMGDriver::Solve` →
full `SetupMultigrid(false)` or cached `SetupMultigrid(true)` plus
`SetupCoefficient(DCCS)` → `CalculateMatrixAll` → optional Galerkin build →
MG preconditioner/FGMRES → Newton update and defect. A line-search trial
reuses its direction and does not change this matrix. `amr_updated`, FMG,
cache disabled, or implicit mixed drift force full hierarchy setup.

### Paired numerical controls

Each pair uses one executable/input except `cache_coefficient_hierarchy=true`
versus `false`, and is checked with
`tst/nrfld_validation/compare_coefficient_cache.py`. The checker requires
clean termination, exact per-solve coefficient **and matrix** fingerprints,
identical Newton/FGMRES summaries and every numerical HDF dataset, and a
constant-vector operator row error below `1e-12`.

| Case | MPI | MG fingerprint rows | AMR octet rows | HDF snapshots | Max `A*1` row error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stiff coupled sound wave, static level-1 AMR, fixed limiter, FGMRES | 2 | 36 | 6 | 8 | `2.94e-16` |
| Coupled moving closed box, implicit mixed drift, `v=1e8 cm/s`, FGMRES | 2 | 42 | 0 | 4 | `4.86e-16` |
| RSG material-table general EOS, variable limiter, Galerkin/FGMRES, CFL 0.3 | 1 | 35 | 0 | 2 | `6.14e-15` |
| Variable-coefficient radiation Test C, adaptive AMR topology change, FGMRES | 2 | 30 | 3 | 8 | `4.11e-16` |

All four comparisons passed. In the coupled sound wave, `DCCS` changed between
Newton directions while face diffusion extrema were exactly fixed within each
implicit stage; octet coefficient and matrix hashes changed together. In the
mixed-frame case, the maximum drift fraction reached `4.01e-6`, so the full
nonsymmetric-rebuild branch was active. The adaptive case had levels `{0,1}`
through cycle 2 and only level `{0}` from cycle 3; the cached and uncached
operators still matched at every recorded solve. The RSG case exercises the
production general-EOS derivative, variable closure and Galerkin path.

Raw paired runs are under `/mnt/d/output/rad/wor30_sound_octet_20261104`,
`wor30_mixed_strong_cache_20261104`, `wor30_rsg_cache_20261104`, and
`wor30_dynamic_amr_octet_20261104`. Each contains exact input and command;
`comparison.json` records the checker result. The earlier single-rank Test C
raw runs remain listed below. No code-path correction was indicated by these
controls. A full late-collapse checkpoint comparison was not run locally:
the earlier 256³/512-rank failure state is remote and belongs to WOR-21.

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

The coupled, mixed-drift, MPI, AMR topology, general-EOS and Galerkin controls
were subsequently completed in the 2026-11-04 audit above. Direct testing at
the remote late-collapse failure state remains outside this local audit.
