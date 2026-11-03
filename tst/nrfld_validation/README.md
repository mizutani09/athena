# NRFLD validation runner

This runner automates the diffusion and matter-radiation coupling tests and
organizes radiative-shock results.  It records reproducibility metadata and
keeps each test's plotting implementation in a separate file.

For execution-time breakdowns and HD/MHD/FLD/self-gravity scaling experiments,
see [performance measurements (WOR-20)](PERFORMANCE.md). The separate
`performance.py` runner creates a timestamped directory in the launch directory.

Run the initial diffusion case and both coupling branches with at most four
build jobs or MPI ranks:

```bash
python tst/nrfld_validation/run_suite.py --max-cores 4
```

Run one case:

```bash
python tst/nrfld_validation/run_suite.py --case diffusion --max-cores 4
python tst/nrfld_validation/run_suite.py \
  --case couple_low --case couple_high --max-cores 4
```

`couple_low` starts from `e_gas=1e2 erg cm^-3`; `couple_high` starts from
`1e10 erg cm^-3`. Both use `E_rad=1e12 erg cm^-3` as a fixed radiation bath
and the current FGMRES solver by default (`--couple-linear-solver mg` selects
the historical MG control). The high case resolves its much shorter initial
cooling time using `dt0=1e-17 s`, growth factor `1.01`, and 1800 steps;
`--couple-high-dt0`, `--couple-high-dt-growth`, and `--couple-high-nlim` expose
these settings. The two cases share one isolated build in a combined run.
The legacy `--case couple` alias remains the old low-energy MG setup.

Coupling acceptance requires every requested Newton solve to converge, a
maximum absolute log gas-energy difference below `5e-3` against the fixed-bath
analytic ODE, and maximum relative fixed-radiation drift below `1e-10`.
`gas + radiation` is not conserved here because the radiation state is held
fixed by design. Each run directory contains `summary.json`, `index.html`,
per-case plots, logs, resource use, and manifests. HTML images and artifact
links are relative to the run directory; copy the directory as a unit.

### Integrated coupling validation

The integrated runner performs the simple low/high growing-timestep test and
then the fixed-step sweep. The latter reproduces the low/high plots and
first-step error comparison from the earlier `mgfld/couple` calculation, then
measures the error at every later step. It uses the current source and FGMRES
by default:

```bash
python3 tst/nrfld_validation/coupling_sweep.py --max-cores 4
```

This builds one isolated `nrfld_couple` executable and runs the simple test
first: low starts at `dt0=1e-11 s`, grows by 1.05 for 100 steps; high starts
at `dt0=1e-17 s`, grows by 1.01 for 1800 steps. Both require measured
`max(dt)/t_couple < 1`, maximum analytic log error below `5e-3`, and every
Newton/linear solve to converge. The sweep then runs 10 steps for each
combination of low/high initial gas energy and constant physical timestep
`1e-7`, `1e-6`, `1e-5`, or `1e-4` s. The domain is expanded to ±5000 cm in
each dimension, with `cfl_number=1`, so that the hydro CFL limit does not
prevent the requested timestep. The runner checks the *measured* timestep in
the history file. It also checks every Newton and FGMRES solve, fixed-radiation
drift, and the last three steps against the analytic fixed-bath ODE. The first
step is reported separately: an implicit update can have a large initial
error while subsequent steps converge to equilibrium. The reported
`t_couple` is the linearized relaxation time at equilibrium,
`e_eq/(4 c sigma_P E_rad)`; each manifest also gives the initial source time.

Results go to `/mnt/d/output/rad/validation_couple_<timestamp>_JST/` by
default. Open `index.html` at that directory's root. It shows the simple
test before the detailed sweep, with larger text and relative image/file
links. Sweep plots are also available as PDF. `summary.json` and
`first_step_errors.csv` give machine-readable values. Each simple case has
its input, raw output, log, resource use, plot, and manifest under
`simple/<low|high>/`; sweep cases use `cases/<low|high>/dt_<value>/`.
Use `--output-root` and `--run-name` to select
another location/name; `--dt-values`, `--steps`, and `--linear-solver mg`
support controlled variations.

## Diffusion convergence (space and time)

The current FGMRES validation combines the 3×3 convergence matrix, spatial
profiles, time traces, and an H.264 MP4 in one run:

```bash
python3 tst/nrfld_validation/diffusion_validation.py --max-cores 8
```

By default it follows the earlier `mgfld/diff` experiment at code time
`0.0535561`: `N=128,256,512`, each at CFL multipliers `1,1/2,1/4` relative
to a base CFL of about `0.299`. Along the diagonal `N=128/CFL×1`,
`N=256/CFL×1/2`, `N=512/CFL×1/4`, dt scales as dx². The three fixed-N
absolute/relative L1 values and their raw change rates are recorded, with
the original history mean pointwise relative L1 as an additional comparison.
Spatial order is measured both along the diagonal and after time error
cancellation with `2 E(dt/2)-E(dt)`. Temporal order uses successive solution
differences at the middle resolution to remove the fixed spatial error.

The runner checks the measured dt and common final time, every Newton and
FGMRES solve, positive finite radiation energy, and spatial/temporal orders.
It writes a directory under `/mnt/d/output/rad/` with `index.html` at its
root. Open that file for the error table, convergence figure, final spatial
profiles and error distributions, fixed-position time traces, snapshots, L1
history, and an embedded 21-frame H.264 video. The HTML's image and video
links are relative to the run directory. Each case contains its materialized
input, raw HDF5/history output, profile CSV, log, resource use and manifest.
Use `--resolutions 16 32 64 --base-steps 40 --final-time 0.02 --max-cores 4`
for the smaller exploratory matrix.

## Marshak wave against the analytic Robin-boundary solution

Run the four-resolution FGMRES comparison with:

```bash
python3 tst/nrfld_validation/marshak_validation.py --max-cores 8
```

The runner builds `nrfld_marshak` with `general/marshak` EOS in an isolated
source snapshot, then runs N=32, 64, 128 and 256 through 1000 fixed steps. Each problem
uses cubic 32³ MeshBlocks and only 32 cells across each passive transverse
direction; the transverse domain shrinks with the x1 cell size. The default
template's former 0.1 K initial gas state causes a first-step Newton line-search
failure in both MG and FGMRES with the current solver. The template and
validation use 1 K, which has normalized initial gas energy v≈5.67×10⁻¹²;
the analytic
reference assumes zero initial energy. The effective input is saved for each
case, and a failed test remains visible as a failure in the HTML report.

The report is at `/mnt/d/output/rad/validation_marshak_<timestamp>_JST/index.html`.
It compares normalized radiation `u` and gas `v` at cycle 34 (τ≈0.0102) and
cycle 1000 (τ≈0.3), and lists mean absolute L1, relative L1 and maximum
absolute error. It also includes a shared log-axis `u`/`v` comparison with
the analytic curves and selected x-axis ticks from the supplied plotting
script, and a relative-L1-versus-resolution plot with adjacent measured rates.
The time step stays fixed, so these rates include time error and are not a
standalone measurement of spatial order. Other artifacts include profile
and error plots, history plots, an
embedded H.264 evolution movie, raw HDF5/history files, CSV profiles, inputs,
logs and manifests. The analytic reference uses the Su–Olson-type
semi-infinite Robin-boundary inverse Laplace solution already used by the old
Marshak plotting script. The finite domain has a cold outer boundary, so the
reported error includes that approximation. `--resolutions 32 64` provides a
smaller exploratory run. `--reuse-build /path/to/previous/run/build` copies
an existing validated build into a new run directory, while
`--resume-from /path/to/previous/run` copies its case results and build, checks
the saved profiles, and calculates only missing resolutions.

The profiles use each HDF5 cell's native `x_dimless`; no translation or fitted
registration is applied. The HTML reports a small-shift fit only as a diagnostic,
separate from every error value.

To test the spatial order at τ=0.3, extend a passed four-resolution report with
half-time-step calculations at N=32,64,128:

```bash
python3 tst/nrfld_validation/marshak_spatial.py \
  --report-root /mnt/d/output/rad/validation_marshak_<timestamp>_JST \
  --max-cores 8
```

The problem generator accepts `problem/delta_tau` (default 3e-4). The spatial
study first repeats the N=32 default-step case with its new binary to verify
equivalence with the saved baseline. It then runs Δτ=1.5e-4 for 2000 steps at
N=32,64,128 and forms `2 U(Δt/2)-U(Δt)` at matching cell centers. This cancels
the leading first-order time error in the analysis; the actual evolved fields
are not modified. The root HTML gains a spatial-convergence figure, measured
L1 rates, and links to new logs, inputs, profiles and manifests. The fixed-step
curve alone still contains time error and is not a spatial-order measurement.

## Optically thin streaming front

Run the four-condition FGMRES study with:

```bash
python3 tst/nrfld_validation/streaming_validation.py --max-cores 4
```

The runner builds `nrfld_streaming_front` in an isolated source snapshot and
calculates `chi_R=1e-4,1e-7 cm^-1` with both `dt=dx/(2c)` and `dx/(20c)`.
It uses the source problem's 128×32×32 grid, advances to 2 ns, and writes a
directory under `/mnt/d/output/rad/` with `index.html` at its root.
The four cases use `nr_threshold=1e-8`. An initial `1e-10` trial stalled on
the first step of both `chi_R=1e-7` runs with final maximum residuals around
`1e-10`; a `1e-9` trial later encountered a `1.21e-9` maximum residual.
This threshold is stored in every effective input. The very thin, small-step
case also reached the default FGMRES limit of 200 iterations at step 96. A
300-step control passed using restart 40, at most 500 iterations, and two MG
preconditioner V-cycles (maximum observed 24 iterations); these stronger
settings are used for all four cases. `--reuse-build /path/to/build` copies a
verified streaming build into a new run when only the input settings change.
The kinematic reference is `x_front=x_front0+c*t`. At each snapshot, the
analysis interpolates the unique 90%, 50%, and 10% crossings of radiation
energy between the two initial levels. It reports 50% position error in cm and
cell widths, 10% leading-edge error, 10–90% width, fitted front speed, energy
ahead of the expected front, and mean error relative to an ideal sharp step.
These describe front
capture; the sharp step is not an analytic FLD profile. Numerical profiles are
never shifted to match the reference. PASS checks completed, converged
Newton/FGMRES steps, time stepping and defined front crossings; accuracy
figures are reported separately.

The HTML links to the two-panel profile figure based on the older streaming
plot, front zoom, position/width evolution, H.264 movie, raw HDF5, CSV, input,
logs and manifests. For analysis-only regeneration from saved output, use
`--postprocess /mnt/d/output/rad/validation_streaming_<timestamp>_JST`.

For the separate WOR-59 time/space convergence and control sweep, run:

```bash
python3 tst/nrfld_validation/streaming_front_convergence.py --max-cores 4
```

It keeps the same cubic cell spacing but uses an 8×8 transverse slab. A
128-cell slab was numerically identical to the original 128×32×32 run. The
runner varies `dt` at fixed `N`, `N` at fixed physical `dt`, and `N` with
`dt=dx/(20c)`, plus controls for the initial step's cell placement, right
boundary distance, opacity, and solver tolerance. Its output root contains
`index.html`, per-case inputs/logs/HDF5, `summary.json`, position/width plots,
and explicit convergence-trend checks. Calculation PASS still means the solver
completed; the separate trend checks quantify front capture. The reference
curve is never shifted. See `wor59_streaming_front_20261104.md` for the
measured results and interpretation.

## Shock tube against 1D Rad-EOS hydro

Run the full comparison with:

```bash
python3 tst/nrfld_validation/shock_tube_validation.py --max-cores 8
```

The runner builds NR-FLD from an isolated snapshot of this checkout and
calculates `N=64,128` through code time 0.1. It builds a separate, strictly
1D `general/rad` EOS and HLLC hydro reference from
`/home/kosuke/simulation/mhd/athena_st` at `N=2048,4096`. The two original
shock-tube inputs have different thermodynamic states. The runner therefore
derives the reference left/right total pressure from the FLD input density and
temperature and checks the initial density, temperature, pressure and velocity.
The Rad-EOS reference source needs three small compatibility changes for this
comparison: use the FLD unit constant and gas constant, and initialize total
energy through `EgasFromRhoP`. The adapter applies them only to the isolated
build snapshot; neither checkout is edited.

The 1D reference is cached under
`/mnt/d/output/rad/reference_cache/shock_tube_rad_eos/`. Its key includes the
reference source, input template, adapter and physical state; stored inputs
and final HDF5 hashes are checked before reuse. A normal repeat therefore
reruns the FLD cases while reusing a matching reference. The run directory
under `/mnt/d/output/rad/` remains self-contained because it also includes a
copy of the reference data and source snapshot. Open its root `index.html` for
final profiles, per-field errors, error histories, an embedded H.264 movie,
logs, CSV, inputs and manifests. The 4096-cell reference is conservatively
averaged onto each native FLD cell for error calculation, without spatial
alignment. The 2048-versus-4096 difference estimates reference resolution
error. Final-time comparisons use the same exact time; intermediate snapshots
can differ slightly because Athena writes on the step crossing each output
mark, and the report records the maximum offset.

The default shock-tube run uses `nrfld/backward_error_convergence=true` and
disables relaxed Newton acceptance. This convergence norm estimates the
relative gas-energy state correction needed to satisfy a stiff gas row. The
raw residual and Jacobian used for the Newton update remain unchanged. The
N64/N128 run at `/mnt/d/output/rad/shock_tube_strict_monitored_20261103/index.html`
reached `1e-10` with zero relaxed steps; maximum final L2 values were
`1.24e-11` and `8.78e-12`. The previous normalized gas residual stalled at
about `2e-7` on the first step, even as the state correction reached floating
point resolution. The old behavior can be reproduced with `--legacy-residual`;
its report must retain `completed_with_solver_caveat`. The strict and legacy
final profiles agree at the saved HDF precision and have the same Rad EOS
reference L1 metrics.
For analysis-only regeneration, use `--postprocess /path/to/run`. The
`--import-reference-root` and `--import-fld-root` options are for importing
verified earlier runs into this layout.

The older lightweight convergence script remains available:

```bash
python3 tst/nrfld_validation/diffusion_convergence.py --max-cores 2
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

The same line reports `correction_factor=after_prolong_before_post/
after_pre_before_restrict` and `full_factor=after_post/before_pre`.
With `linear_solver=fgmres`, the MG preconditioner calls `SolveVCycle()`
directly.  The same verbosity prints stage records for up to the first two
V-cycles of the first preconditioner application in each linear solve; later
Krylov iterations are not traced to keep the output bounded.
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

## Radiative shocks: Mach 2 and Mach 5

The dedicated runner executes both shock problems sequentially on a periodic
slab. Each MeshBlock and each physical cell is cubic; the default local run is
256×4×4 cells for each problem. It uses the current FGMRES solver, records
Newton/linear convergence, and compares with the bundled steady semi-analytic
profiles. The result directory has `index.html` at its root, with raw and
shock-aligned profiles, width-weighted L1 error and its time history, peak
gas-temperature capture, movies, inputs, logs, CSV, and hashes:

```bash
python3 tst/nrfld_validation/radiative_shock_validation.py --mpi-ranks 4
```

The local default reaches 0.05 s for Mach 2 and 0.04 s for Mach 5. These are
modest-resolution diagnostics; the narrow Mach 5 gas-temperature spike may
still be missed even if the global L1 is small. The HTML therefore reports
the sampled temperature peak and its error separately. The raw-coordinate
plot shows the numerical shock displacement. The aligned comparison translates
the numerical density jump midpoint to the semi-analytic jump midpoint; its
shift is listed, and only aligned profiles enter L1. The last two independently
aligned snapshots provide a measure of profile change. The physical gate
below also checks the gas-temperature spike. A completed coarse calculation
with an unresolved spike is `accuracy_not_met`, not `passed`.

The reference CSV and generator scripts are versioned under
`reference/radiative_shock/` and copied into each run, so a large computer does
not need the old `/mnt/d/output/rad/test_20260703` directories. Each run also
contains a snapshot of the build source and binary. An isolated build is made
by default; `--reuse-build /path/to/build_source` verifies a matching source
snapshot before importing an existing build. For a large computer with a
scratch filesystem, use for example:

```bash
python3 tst/nrfld_validation/radiative_shock_validation.py \
  --preset hpc --output-root "$SCRATCH/rad" --mpi-ranks 16 --build-jobs 16
```

The HPC preset uses 1024×4×4 cells for Mach 2 and 4096×4×4 for Mach 5.
Runtime can be many hours, especially for Mach 5; use `--case mach2` or
`--case mach5` to run separately. `--nx1-mach2`, `--nx1-mach5`, and `--slab`
change the geometry while preserving cubic cells and blocks. A short
`--nlim 2` run is marked exploratory and cannot be mistaken for a completed
analytic comparison.

`--layout smr` prepares a three-level static-refinement slab with x1
refinement around the shock. A two-step Mach 2 test verified that this path
starts and converges, but the transverse directions also refine: on the
current computer it used more memory and time than the uniform 256×4×4
case. It is therefore experimental and is not the local default. The local
SMR root grid is 64×4×4; `--preset hpc --layout smr` uses root grids 256×4×4
and 1024×4×4, each with up to four times finer x1 spacing in the inner
region. A full SMR shock run and its comparison are still required before
claiming an SMR validation result.

For a single adaptive level, use `--layout amr`. The local root grids are
128×4×4, with finest x1 spacing equal to the uniform local 256-cell runs.
The problem generator's density/gas-temperature curvature tagger must create
level-one blocks; the runner checks that the final HDF5 contains both levels.
Refined levels take additional substeps, so the reported number of Newton
solves can exceed the number of global mesh cycles. Every solve is checked for
strict convergence and its FGMRES residual target.

The AMR calculation produces its own complete report. Once both Mach cases
finish, add that report to the uniform report's root HTML with:

```bash
python3 tst/nrfld_validation/radiative_shock_validation.py \
  --layout amr --run-name validation_radiative_shock_amr1 \
  --mpi-ranks 4
python3 tst/nrfld_validation/radiative_shock_validation.py \
  --postprocess /mnt/d/output/rad/validation_radiative_shock_<uniform-run> \
  --attach-amr-run /mnt/d/output/rad/validation_radiative_shock_amr1
```

The second command copies the AMR run into `amr/` and adds its metrics and
figures to the uniform `index.html`; both remain portable as one directory.
The HPC AMR preset uses root grids 512×4×4 and 2048×4×4, with one finer
level. AMR can cost substantially more time than the uniform slab because
its mesh changes dynamically, even when memory use is modest.

The shared validation HTML baseline is 21 px (19 px on narrow screens).
The suite applies the same size to copied older reports at import time.

## Advecting radiation pulse

`advection_pulse_validation.py` runs the static and moving Zhang et al. Section
6.7 profiles on the same 512×4×4 periodic slab. The moving background speed is
10^6 cm/s, and the input's final physical time is 4.80056×10^-5 s, close to
the nominal 4.8×10^-5 s benchmark time. The analysis shifts
the moving solution back by the prescribed `v0*t` with periodic wrapping and
subtracts `v0` from its velocity. It reports relative L1 and maximum pointwise
differences for density, gas/radiation temperature and radiation energy, plus
pulse-contrast errors, velocity difference, profiles, time history, CSV, and an
H.264 movie. The supplied plotting script is saved in
`reference/advect_pulse_plot_user.py` and informs the three-panel figure.
Intermediate snapshots are paired by output index; their time difference
must be no greater than the largest logged timestep of either run (plus HDF
rounding). This also works for the coarser `fast` grid. The final comparison
still requires the same exact target time.
The physical setup follows [Zhang et al., CASTRO II, Sections 6.7–6.8](https://ccse.lbl.gov/Publications/wqzhang/castro2.pdf).

```bash
python3 tst/nrfld_validation/advection_pulse_validation.py --mpi-ranks 4
```

The input uses CFL 0.03. A short current-code test found that the template's
four backtracking attempts stopped at the first Newton solve; twelve attempts
allowed strict Newton convergence without changing CFL, initial state, or
acceptance tolerance. The runner records these effective settings. For a
completed manually launched static/moving pair, `--postprocess RUN_DIR
--reuse-build BUILD_SOURCE` builds the same report without recalculating.

## Static equilibrium

`static_equilibrium_validation.py` runs the Section 6.8 2D radial pressure
balance on a 64×64×4 slab. Every cell is compared with its own initial state;
the report includes gas/radiation temperature, density and total-pressure
drift, velocity, total mass/energy drift, 2D maps, radial profiles, time
history, CSV, and a movie. No spatial translation is applied.

```bash
python3 tst/nrfld_validation/static_equilibrium_validation.py --mpi-ranks 4
```

Both scripts leave a root `index.html` and preserve the effective inputs,
binary/source snapshot, logs, and machine-readable summary/manifest. Strict
Newton and FGMRES convergence plus reaching the target time are prerequisites
for the physical accuracy gates below.

## Full sequential validation

`run_all.py` runs couple, diffusion, Marshak wave (including time-extrapolated
spatial convergence), streaming, Rad-EoS shock tube, radiative shocks Mach 2/5
(uniform and one-level AMR), radiation pulse advection, static equilibrium,
3-D thermal wave, periodic closed-box conservation, AMR FGMRES restart, and
mixed-frame implicit-placement comparison in that order. It launches one heavy calculation at a time, gives each child
its own report directory, and places the parent `index.html`, `suite.json`,
manifest, command logs, and process-memory records at the suite root.
The parent HTML shows only the verdict, a short numerical result, a comment,
and a link to each child report. Three physical accuracy gates now distinguish
`passed` from `accuracy_not_met`: radiative shocks, pulse advection, and static
equilibrium. A completed coarse calculation can therefore appear as
`accuracy_not_met` without being misreported as a solver failure.
New runs use this directory layout:

```text
/mnt/d/output/rad/validation_suite_<date>_JST/
  index.html
  suite.json
  couple/index.html
  diffusion/index.html
  ...
  mixed_frame/index.html
  logs/
  resources/
```

The existing `validation_fast_20261103` and
`validation_suite_complete_20261103` reports retain their older `cases/`
subdirectory; `--resume` and their parent HTML still support it.

```bash
python3 tst/nrfld_validation/run_all.py --tier full --max-cores 4
python3 tst/nrfld_validation/run_all.py --tier fast --max-cores 4
python3 tst/nrfld_validation/run_all.py --tier hpc --max-cores 32 --plan
```

The `fast` twelve-problem run completed on 2026-11-03 at
`/mnt/d/output/rad/validation_fast_20261103/index.html`. All child reports
passed their execution/solver checks; the shock-tube child was refreshed with
the strict backward-error run. After adding physical accuracy gates, the
aggregate is `complete_with_accuracy_gap`: 11 PASS and the Mach 5 radiative
shock accuracy gap described below. The
largest sampled simultaneous process
tree RSS was 1.83 GiB (thermal wave); no cgroup OOM event occurred. The
parent runner was externally interrupted once after the shock-tube child had
already completed. `--resume` verified and recovered that child report, then
continued. Two coarse-grid analysis checks (pulse intermediate output time
offset and HDF final-time rounding in mixed-frame) were corrected and their
existing simulations were reprocessed without rerunning the physics.

`fast` uses smaller supported grids and omits the optional Marshak spatial
extension and radiative-shock AMR attachment. `full` uses the established
local accuracy matrices. `hpc` selects larger diffusion, Marshak, shock,
advection, and equilibrium presets; inspect its plan and machine allocation
before running. All tiers execute sequentially. The suite monitor samples
the child process tree every 0.5 s and saves the simultaneous Athena-rank
RSS sum, whole process-tree RSS sum (including compilation), each
Athena MPI rank's VmHWM, and the cgroup OOM counter delta under
`resources/NAME.json`. The process-tree sum counts shared pages in each
process, so it is a conservative usage indicator; MPI rank VmHWM is the
per-process peak. A cgroup OOM count or SIGKILL is reported separately from
a normal nonzero solver exit.
On this WSL2 machine, 30 process-tree polls took a median 0.80 ms each,
about 0.16% of one core at the default 0.5 s interval; the measurement is
saved in `validation_memory_smoke_20261103/monitor_overhead.json`.

```bash
python3 tst/nrfld_validation/run_all.py --tier fast \
  --only fgmres_amr_restart --max-cores 4 --run-name memory_check
jq '{mpi_ranks, peak_athena_rss_sum_kib, peak_process_tree_rss_kib, cgroup_memory_event_delta, oom_classification}' \
  /mnt/d/output/rad/memory_check/resources/fgmres_amr_restart.json
```

If a suite is interrupted externally, its root HTML remains available and
`--resume` archives the interrupted case before retrying. A failed case is
classified as timeout, solver failure, observed cgroup OOM kill, possible
SIGKILL, or report/filesystem failure; the raw command log and resource JSON
remain the evidence for that classification.

### Physical accuracy gates for the final three tests

All three still require target time, finite positive fields, and strict Newton
and FGMRES convergence before accuracy is assessed. Thresholds apply to the
saved physical metrics and are written with measured values to each child
`summary.json` and HTML.

| Problem | Required accuracy |
| --- | --- |
| Mach 2/5 radiative shock, including attached AMR | Each aligned density, gas-temperature and radiation-temperature L1 error ≤2%; raw shock-position error ≤5% of the domain width; final aligned change ≤1%; gas-temperature peak error ≤10%; gas-spike amplitude recovery ≥50% of the semi-analytic spike. The shift used for shape L1 is reported separately and does not relax the position check. |
| Advected radiation pulse | After the prescribed periodic shift by `v0*t`, background-subtracted density, gas-temperature and radiation-energy pulse-contrast L1 errors each ≤10% for `N<128`, or ≤1% for `N≥128`. Maximum velocity mismatch is ≤5% or ≤1% of the imposed 10 km/s, respectively. |
| Static equilibrium | Maximum pointwise relative drift of density, gas/radiation temperature, radiation energy and total pressure ≤1e-4; total-pressure spatial range/mean ≤1e-4; mass and total-energy drift ≤1e-5; spurious speed ≤1e-5 cm/s. Scalar HDF5 rounding limits sensitivity below about 1e-7 relative. |

The saved local Mach 5 shocks do not yet recover half the semi-analytic gas
temperature spike: the 64-cell uniform run recovers 2.5%, and the 256-cell
uniform and one-level AMR runs about 10.4%. Their bulk profiles and solver
converge, but their accuracy verdict is `accuracy_not_met`. This is an explicit
resolution limitation, not a failure to finish the calculation. The fast and
full pulse and equilibrium reports satisfy their respective gates.

To apply the current gates to older completed reports without rerunning the
physics or regenerating plots/videos:

```bash
python3 tst/nrfld_validation/run_all.py \
  --reassess-suite /mnt/d/output/rad/validation_fast_20261103
```

The three standalone runners also accept `--reassess REPORT_DIR`.

The four added reports can also run independently:

```bash
python3 tst/nrfld_validation/run_thermal_wave_matrix.py
python3 tst/nrfld_validation/closed_box_validation.py
python3 tst/nrfld_validation/fgmres_amr_restart_validation.py
python3 tst/nrfld_validation/run_mixed_frame_matrix.py --preset full
```

The thermal-wave matrix evolves an analytic 3-D profile for 101 steps with
`fld/only_rad=false`, then compares all leaf cells to the similarity solution.
The old radiation-only convergence setting can stagnate; its failure is
preserved in the exploratory logs rather than treated as a pass. The
mixed-frame report records both raw shock position/error and density-jump
aligned shape error, plus the pulse after a physical `v0*t` shift. Its
explicit placement control requires a fixed flux limiter. The matrix also
runs a one-cycle local finite-difference Jacobian check for each placement
(`jacobian_implicit` and `jacobian_explicit`). With
`fld/mixed_frame_ledger_limit=2`, the first two source updates per block log
the signed and absolute lambda/q exchanges, the gas/radiation pair balance,
and any radiation floor correction. The report checks that q appears only in
its selected path, that the pair balance is within 1e-12 of the exchanged
energy, and that the local Jacobian-vector error shrinks with epsilon.
The optional ledger is disabled by default. Legacy imported shock runs may
have command-line grid/CFL overrides missing from `input.used`; their report
labels that provenance and checks the actual HDF grid and first-step dt.

Interrupted suites can be continued with `--resume /path/to/suite`; failed
cases require `--retry-failed` and their previous directories are archived
under `failed_attempts/`. `--only NAME` selects one or more problems. Use
`--import NAME=/path/to/completed/report` to make a portable aggregate from
completed child reports without rerunning them; imported reports retain their
own provenance. `--resume SUITE --import NAME=DIR --refresh-imported` replaces
one completed imported case after archiving its earlier report; a source
`suite_resource.json` is copied into the aggregate's memory table when
available. `--plan` prints the order before starting any calculation.

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
