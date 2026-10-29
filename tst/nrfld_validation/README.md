# NRFLD validation runner

This runner automates the diffusion and matter-radiation coupling tests and
organizes radiative-shock results.  It records reproducibility metadata and
keeps each test's plotting implementation in a separate file.

For execution-time breakdowns and HD/MHD/FLD/self-gravity scaling experiments,
see [performance measurements (WOR-20)](PERFORMANCE.md). The separate
`performance.py` runner creates a timestamped directory in the launch directory.

Run both initial cases with at most four build jobs or MPI ranks:

```bash
python tst/nrfld_validation/run_suite.py --max-cores 4
```

Run one case:

```bash
python tst/nrfld_validation/run_suite.py --case diffusion --max-cores 4
python tst/nrfld_validation/run_suite.py --case couple --max-cores 4
```

## Diffusion convergence (space and time)

```bash
python tst/nrfld_validation/diffusion_convergence.py --max-cores 2
```

This additional validation reuses `nrfld_diff`, the isolated-build helpers and
manifest machinery. It writes `summary.json`, `manifest.json`, per-run inputs,
logs/profiles and `convergence.png` under `validation/diff_convergence_<timestamp>/`.
Use `--binary /path/to/nrfld_diff/athena` to reuse an already built MPI executable;
its configuration and SHA256 are checked/recorded. `--no-plots` skips the figure.
The parent build is never reconfigured. Runs are sequential, with at most
`--max-cores` MPI ranks/build jobs and one OpenMP thread per rank.

The problem is a Gaussian with constant diffusion coefficient: the existing
`fixed_flux_limiter=true` sets lambda=1/3. Coupling, radiation force and gravity
are absent, and the gas is uniform and static. The 1-D solution is embedded in
an N x 8 x 8 transverse-periodic box with cubic cells and 8-cubed MeshBlocks;
the transverse lengths shrink with dx. This is not an AMR/collapse validation.
All profiles are compared at the same code time (default 0.02) using the same
point-sampled analytic solution as the pgen. The norm is
`sum(abs(E-Eexact))/sum(abs(Eexact))`, avoiding division by small Gaussian tails.

Three measurements distinguish the errors `C_x dx^2 + C_t dt`:

1. **Fixed diffusion number** `chi*dt/dx^2`: N=16,32,64 with 40,160,640 steps.
   Doubling N halves CFL and quarters dt (verified from the actual timestep).
   Expected error is second order in dx and first order in dt along this path.
   Because both errors scale together, this alone does NOT prove temporal order.
2. **Spatial order**: for each N repeat at half dt, then compare
   `2 E(dt/2) - E(dt)` with the analytic solution. This cancels the leading
   backward-Euler error. The extrapolated field is analysis-only; the evolution
   scheme is not changed. Expected slope versus dx is 2.
3. **Temporal order on a fixed grid**: N=32 at 40,80,160,320 steps. Ratios of
   successive profile differences cancel the common spatial truncation error.
   Expected slope versus dt is 1.

The runner checks finite positive energy, all Newton solves, actual dt and final
time. Acceptance ranges are 1.7–2.3 for spatial/fixed-number dx slopes and
0.8–1.2 for fixed-grid temporal slopes. Default single-precision HDF5 timestamps
are cross-checked against the full-precision termination log. All overrides are
materialized in each `input.used` because Athena's command-line parser cannot
add a missing parameter.

The collapse investigation and the exact experimental rollback scope are in
[protostellar convergence review](protostellar_convergence_review_20261016.md).
The measured diffusion/coupling regression gate and its remaining collapse
A/B test are summarized in [WOR-27 baseline regression](wor27_regression_20261024.md).
The remaining same-restart solver gate can be submitted with
`job_mayer5_smoother_ab_pdebug.sh`; compare the three array results with
`compare_collapse_smoother_ab.py ARRAY_JOB_ID`.  The array compares the
unchanged RB baseline, RB with a one-cycle non-monotone window, and the existing
double-Jacobi smoother.  A minimal `-i` overlay safely adds the new patience
parameter when it is absent from the older restart header and requests a common
one-cycle xline HDF5 output for field comparison.
The bounded non-monotone Test C control and its limitations are recorded in
[WOR-25 non-monotone window](wor25_nonmonotone_window_20261024.md).

## Test C: material interface versus static AMR

`nrfld_diff_amr_interface` is a separate 1D-like pgen for distinguishing a
variable diffusion coefficient from an AMR-transfer problem.  The fields vary
only in x1, while the mesh remains three-dimensional because the multigrid
implementation requires cubic 3-D cells.  Build and run the five controlled
cases with:

```bash
python configure.py --prob=nrfld_diff_amr_interface -nrmgfld --nghost=2 --cxx=g++
make -j2
tst/nrfld_validation/run_test_c.sh
# Include the concise restriction/prolongation diagnostic:
DIAGNOSTIC_VERBOSITY=1 tst/nrfld_validation/run_test_c.sh
```

The cases are C0 (uniform coefficient/uniform mesh), C1 (uniform coefficient/
SMR), C2 (material interface/uniform mesh), C3a (material and SMR interfaces
aligned), and C3b (material interface shifted inside the fine region).  Set
`nrfld/diagnostic_verbosity=1` prints one `[NRFLD MG transfer]` record for the
baseline-scale first V-cycle of each Newton direction.  Each record contains
the finest-level residual at these stages:

```text
before_pre
after_pre_before_restrict
after_prolong_before_post
after_post
```

The same line reports `prolong_factor=after_prolong_before_post/
after_pre_before_restrict` and `full_factor=after_post/before_pre`.
Coarse-correction retry solves are deliberately omitted so a failed Newton
step cannot emit one record per retry.  Verbosity 0 retains the original
unsplit V-cycle task sequence.  At verbosity 1 or 2 the finest task list is
split at the two diagnostic points; the small Test C comparison gives the
same final Newton residual, but production comparisons should retain a
verbosity-0 control run.

Results are written under `~/simulation/test/rad_nr/runs/<timestamp>/` by
default.  Use `--output-root` or `--run-name` to change this location.  Every
case is built in its own `build_source/` snapshot, so `configure.py`,
`Makefile`, `src/defs.hpp`, `obj/`, and `bin/` in the parent repository are not
modified. The snapshot includes uncommitted source changes. Each case contains
its pgen-specific executable, `input.used`, build/config/run logs,
`manifest.json`, raw output, and an optional figure. `summary.json` collects the
final metrics. Plot implementations are kept separately in
`plots/plot_diff.py`, `plots/plot_couple.py`, and
`plots/plot_radiative_shock.py`.

`--max-cores` is a hard upper bound for both `make -j` and MPI ranks.  The
actual MPI count is also limited by the number of MeshBlocks.  Cases run
sequentially because they use different problem generators.

Useful development options:

```bash
python tst/nrfld_validation/run_suite.py --help
python tst/nrfld_validation/run_suite.py --case diffusion --no-plots
python tst/nrfld_validation/run_suite.py --case diffusion --no-build \
  --binary /path/to/athena
```

`--no-build` checks the selected executable with `athena -c` and rejects a pgen
mismatch before running it. If `--binary` is omitted, `bin/athena` is used as
the explicit legacy default. Use one calculation case per `--no-build`
invocation. The manifest records input, binary and external-file SHA-256
digests, effective settings, compiler/MPI versions, a limited environment
allowlist, timeout/exit records, and separate analysis/calculation provenance.
Missing references and timeouts still leave a final manifest.

## Newton convergence residuals

Ordinary coupled NR-FLD uses the equivalent equation basis `{Fg, Fg + Fr}`:
the gas equation and the total-energy equation. The total residual is evaluated
directly as `delta_egas + delta_erad - dt * diffusion`, without subtracting stiff
equal-and-opposite exchange sources. Its scale is the larger of the absolute
transient/transport terms and the old/current total internal energy, with a
`1e-30` floor. Exchange is excluded from this scale. This matches the stable
Schur elimination and avoids amplifying exchange roundoff by `egas / erad`.

`only_rad=true` retains the radiation residual and `fixed_u_rad=true` retains
the gas residual. The log fields `gas_*`, `radiation_*`, and `total_energy_*`
report active convergence equations; inactive fields are zero (in coupled mode,
`radiation_*=0` is not a measurement of the original radiation residual).
The combined L2 is the quadrature sum of active L2 norms, and the combined max
is their maximum. Both must satisfy `nrfld/nr_threshold`; the stopping rule,
Jacobian, physical boundaries, and initial profiles are unchanged.

The exact equation roots are unchanged, but finite-tolerance weighting differs
from testing `Fr / erad` alone. Check energy conservation and radiation/flux
accuracy with a tolerance study before adopting a long production run.
The regression covers stiff coupled exchange and both single-equation modes:

```bash
cd tst/regression
python3 run_tests.py nrmgfld/coupled_residual_norms --hide_make
```

See [total-energy residual validation](total_energy_residual_validation.md)
for the solar smoke tests and analytic-solution comparisons.

## Radiative shocks downloaded from another machine

Radiative shocks are plot-only by default. The suite does not configure,
compile, or launch Athena++ unless `--run-radiative-shock` is explicitly given.
Point it at a downloaded run directory containing `output/*uov_x*.athdf`:

```bash
python tst/nrfld_validation/run_suite.py \
  --case radiative_shock_mach2 \
  --radiative-shock-mach2-dir ~/simulation/test/rad_nr/test_20260811/radiative_shock/mach2 \
  --radiative-shock-mach2-analytic ~/simulation/test/rad_nr/test_20260723/test_radiative_shock_mach2/mach2_semianalytic_bvp.csv \
  --max-cores 4
```

Mach 5 uses the corresponding `--radiative-shock-mach5-dir` and
`--radiative-shock-mach5-analytic` options. The large HDF5 results remain in
their downloaded location. Small reproducibility files (`athena`, `athinput*`,
and `problem_parameters.txt`) are copied into the validation case directory.

To deliberately run the expensive calculation locally, add the explicit flag:

```bash
python tst/nrfld_validation/run_suite.py \
  --case radiative_shock_mach2 --run-radiative-shock --max-cores 4
```

For a short smoke test, also pass `--radiative-shock-nlim 2`. Without that
option, the time and cycle limits in the selected input file are retained.

The shock plotter can also be used independently:

```bash
python tst/nrfld_validation/plots/plot_radiative_shock.py \
  --case mach2 --run-dir ~/simulation/test/rad_nr/test_20260811/radiative_shock/mach2 \
  --analytic ~/simulation/test/rad_nr/test_20260723/test_radiative_shock_mach2/mach2_semianalytic_bvp.csv
```

Mach 2 and Mach 5 use shared loading and AMR-profile merging logic. The Mach 5
plot enables a temperature inset by default. Use `--no-inset` or `--inset` to
override this behavior.

## Controlled Test C / WOR-22

`test_c_controlled.py` reuses the Test C pgen in an isolated build and compares
uniform coarse/fine grids with two equal-size static-AMR layouts. Hydro is
fixed and all layouts use the same dt, material profile and radiation profile.
No solver/smoother changes are applied. Resource bounds, source/binary hashes,
effective inputs, actual mesh geometry, NR results, history energy change, and
field comparisons are saved in a manifest. This is a radiation-only solver
stress test, not a physical free-streaming or collapse benchmark.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_controlled.py \
  --output tst/outputs/wor22_reproduce \
  --diffusion-numbers 1e8 --contrasts 1 --nr-threshold 1e-7 \
  --pulse-center .5 --pulse-width .04 --verbosity 2
```

The shifted-AMR case currently fails with backtracking exhausted; exit status
1 is therefore expected for this reproducer. Successful cases must reach the
requested cycle/time and strictly converge; raw abort status is not ignored.
See [the measured results and limitations](wor22_reproduction_20261019.md).
`--binary` reuses a diagnostic binary (configuration checked); `--reanalyze`
refreshes saved metrics without running Athena. Output directories must be new
unless `--reanalyze` is used. Existing production binaries are not overwritten.

## Test C operator audits / WOR-23 and WOR-24

`test_c_operator_audit.py` runs the interior-AMR controls with diagnostics OFF
and ON, checks identical NR results, history bytes and HDF datasets, and measures
both initial and actual correction directions. The 27 simulations use one MPI
rank, a 30-second/4-GiB per-case limit and two build jobs. An isolated build is
used unless `--binary` is supplied; the main executable is not overwritten.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_operator_audit.py \
  --output tst/outputs/test_c_operator_audit

OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_composite_reference.py \
  --audit tst/outputs/test_c_operator_audit \
  --output tst/outputs/test_c_operator_audit/composite_reference.json

OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_coarse_operator_reference.py \
  --audit tst/outputs/test_c_operator_audit \
  --output tst/outputs/test_c_operator_audit/coarse_operator_reference.json

OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_galerkin_reference.py \
  --output tst/outputs/test_c_operator_audit/galerkin_reference.json
```

The audit exit code checks execution, diagnostic presence and instrumentation
neutrality, **not convergence of all physical solves**. Some NR failures are
intentional reproducers; inspect the per-case status in each manifest.
The independent 96-cell/128-cell direct reference uses no MG iteration and
checks its first correction against the measured Athena correction.  It also
separates direct minmod-on-correction from the base-state tangent
`G'(u) delta`; these are diagnostic controls and are not interchangeable.
The independent uniform-grid MG reference reconstructs the variable-limiter
coefficient hierarchy, red-black smoother, restriction and trilinear
prolongation.  It checks the first V-cycle stage by stage and reports
`A_H v` versus `R A_h P v` for several modes, including the actual first
restricted residual.  This separates a uniform smoother/coarse-correction
failure from AMR-boundary effects; it does not model the AMR hierarchy.  Its
additional minmod-correction control tests (and does not recommend) replacing
the linear MG transfer globally by a nonlinear limiter.
The optional SciPy Galerkin reference retains the same fine operator, smoother,
restriction and prolongation, but replaces rediscretized coarse matrices with
`R A_h P`.  It is an offline causal control, not a production option.
It now records the residual and spectral summary at every recursive level.
The controlled harness also accepts `--smoother jacobi-rb|jacobi-double|jacobi`
and `--omega VALUE` for existing-smoother controls; neither changes the
production default.  See `wor31_multilevel_controls_20261024.md`.
The controlled harness also accepts `--symmetric-rb-sweeps true|false` for the
WOR-29 order comparison and `--cache-coefficient-hierarchy true|false` plus
`--coefficient-diagnostic-limit N` for the WOR-30 cache audit.  See
`wor29_rb_order_20261023.md` and `wor30_coefficient_update_audit_20261023.md`.
It also accepts `--linear-solver`, `--coarse-operator`, `--prolongation` and
the `--fgmres-*` controls for WOR-54.  AMR FGMRES results must pass the
coarse/fine matvec-linearity and conservation checks described in
`wor54_fgmres_amr_plan_20261029.md`; convergence alone is not that validation.
For NR-FLD correction ghosts, `--amr-correction-prolongation tangent` freezes
the minmod branch at the Newton base state and applies its linearized mapping;
`minmod` and `constant` remain comparison controls.
The FGMRES performance controls are independently testable with
`--[no-]fgmres-compact-basis`, `--[no-]fgmres-reuse-basis`, and
`--[no-]fgmres-batched-reductions`; use `--fgmres-restart 5` and `10` for the
memory/convergence comparison.  Runtime logs report basis/workspace bytes,
allocation rebuilds, and collective-reduction calls for each linear solve.

New runtime diagnostics (disabled by default):

- `nrfld/global_jacobian_check=true`: boundary-inclusive finite-difference Jv;
  limited to periodic, uncoupled `only_rad` and one MPI rank. In addition to a
  smooth direction, check the actual direction selected by
  `global_jacobian_iteration=0` (zero-based **linear-solve call index** within
  each implicit solve, not necessarily the NR iteration if retries occur).
  `global_jacobian_epsilon=1e-4` starts a four-value epsilon sweep.
- `nrfld/handoff_diagnostics=true`: common residual after MG, retrieval,
  boundary exchange, and the existing fine sweeps; one MPI rank only.
  `handoff_diagnostic_limit=4` limits the number of recorded linear solves
  over the process lifetime. No extra smoothing is performed.

`closure=frozen` tests the actual lagged-per-step operator. `refreshed` is a
hypothetical recalculated closure, not the current production Newton operator.
Residual norms are volume-weighted; Jv errors use max(norm(FD),norm(Av)), while
handoff residuals use norm(rhs). They must not be confused with the normalized
nonlinear residual in `[NR]`. `full_step_min` tests the **raw** correction before
positivity floors, damping or update caps. See
[the measured operator audit](wor23_wor24_operator_audit_20261023.md) for the
results, restrictions, and what remains unproven for collapse.
