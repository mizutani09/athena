# WOR-182 rotating FLD star with a material tabulated EOS

## Scope

This branch adds a fresh-map path for the accepted q=0.25 RLOF-synchronous,
rotating-atmosphere, 4wz, full mixed-frame FLD star using Athena++'s general
EOS interface.  It is intentionally separate from `wor-182-rotating-full-fld`;
an ideal-EOS restart must not be reinterpreted with a table EOS.

## EOS choice

Use `data/eos/rsg_tomida2013_material_eos.tab` (SHA-256
`7051d1c5222f3ca7dddbbba1c48d473cb39bc5e76badb869b082625473a71576`).
Its metadata declares `material_only=true` and excludes photon pressure,
energy, and entropy.  Do not combine `rsg_nata_eos.tab` with FLD: that table
contains photon terms and would double count radiation.

The Tomida table assumes X=0.70, Y=0.28 and omits metals.  It is therefore a
numerical integration candidate, not yet an exact MESA-composition EOS.

## Implementation

Branch: `wor-182-tabulated-eos`

- `scf_rsg.cpp` solves
  `P_material(rho,e)+a*T(rho,e)^4/3=P_SCF` for every stellar cell average.
- Conserved gas energy and the FLD gas cache use the table internal energy.
- Opacity temperature and output temperature use `TempFromRhoEg`.
- Mach diagnostics use `AsqFromRhoP`.
- The accepted atmosphere-matched profile, rho=1.7e-11 g cm^-3, and a
  corotating numerical atmosphere are materialized in the dedicated input.
- The atmosphere keeps the accepted endpoint pressure 22.24714707 dyn cm^-2.
  The material table maps this to about 1.08e4 K, preserving mechanical rather
  than exact 1e4-K matching for the first A/B gate.

Configure:

```text
--prob=scf_rsg --coord=cartesian --eos=general/eos_table --flux=hllc \
--nscalars=2 --grav=mg --nghost=4 -nrmgfld -hdf5 -mpi
```

## XD gates

The fresh-map launcher is
`analysis/wor182/xd_q025_corot_tabulated_eos_gate.sh`; continuations use
`analysis/wor182/xd_q025_corot_tabulated_eos_restart_gate.sh`.

### Fresh map, job 347859

- 3 M-test-a nodes, 168 MPI ranks
- `nlim=0`
- completed normally in 69.31 s
- all LTE pressure splits succeeded; no NaN, table-domain, or fatal error
- final restart size: 1.680 GB

### One step, job 347860

- 3 M-test-a nodes, 168 MPI ranks
- fresh map, CFL=0.1, dt=1e-5
- completed normally in 37.66 s
- Newton: 3 iterations, final L2 2.48e-10, final max 7.64e-9
- each FGMRES solve: 2 iterations; relative residual <=1.46e-7
- mass drift after the step is approximately -2.3e-13
- interior radial Mach remains 3.60e-6; surface-taper maximum changes only
  from 1.24035e-3 to 1.24040e-3
- full-force relative maximum decreases from 0.442 to 0.139 during the first
  adjustment step

### Ten steps, job 347862

- fresh map, CFL=0.1, 10 steps to t=1e-4
- all Newton solves converged: 3 iterations on step 1, then 2 iterations
- final Newton max residual stayed below 9.84e-11 after step 1
- mass drift remained approximately -2.3e-13
- interior/surface radial Mach at t=1e-4: 3.60e-6 / 1.24085e-3
- full-force relative maximum: 0.1393

### One hundred steps, job 347863

- fresh map, CFL=0.1, 100 steps to t=1e-3
- completed normally in 209.87 s
- all Newton solves converged; final max residual 4.06e-11
- relative drift: mass -4.68e-13, angular momentum +1.76e-10,
  gas+radiation energy +1.55e-9
- interior/surface radial Mach: 4.70e-6 / 1.245e-3
- full-force relative maximum: 0.144

Commit `dfcb9836c` moves the table-temperature-floor fatal report outside the
OpenMP SIMD row, allowing the production Intel SIMD build without weakening
the failure contract.  Commit `83ea88750` limits the 2019 legacy general-EOS
high-order warning to the old PPM/4 paths; 4wz has EOS-aware point conversion
and an explicit density/pressure quality fallback.  Intel SIMD build job
347865 completed, and its 10-step numerical gate 347868 reproduced the
non-SIMD residual history in 79.18 s versus 101 s for job 347862.

### Startup release, job 347871

- Intel SIMD build, fresh map, CFL=0.1
- completed 110 steps to t=1e-2 in 209.81 s
- after the t=1e-3 cap, dt increased through the normal Athena factor-of-two
  growth limiter; every solve converged in 3--5 Newton iterations
- final Newton max residual: 5.68e-9
- relative drift from t=0: mass -1.40e-12, angular momentum +1.76e-9,
  gas+radiation energy +4.09e-9
- interior/surface radial Mach: 4.67e-5 / 1.300e-3
- full-force relative maximum: 0.174

### CFL=0.2 nonlinear-floor diagnosis and gate

The first continuations from t=0.01 (jobs 347886 and 347887) reached t=0.1,
but the final `dt=0.04332` solve was accepted only at the relaxed tolerance:
L2=3.96e-7 and max=1.72e-4.  Tightening FGMRES from 1e-4 to 1e-5 did not
change that residual, ruling out linear-solve accuracy as the cause.

The verbosity-2 reproduction, job 347889, localized the plateau to a
total-energy row next to a radiation cell already clipped to `TINY_NUMBER`.
The linear solve repeatedly requested a negative correction in that floor
cell, while positivity clipping applied zero; neighbouring rows therefore
used a correction that could never be realized.  Commit `9559f3ccb` treats
this lower bound as a Newton active-set constraint (`delta_Er=0`) and uses a
projected residual for the constrained radiation equation.

Job 347891 repeated exactly the CFL=0.2 path after this change:

- completed normally from t=0.01 to 0.1 in 64.35 s
- all three timesteps converged normally; there was no relaxed acceptance
- the previously stalled final timestep converged in 7 Newton iterations
- final L2/max residual: 2.53e-12 / 1.48e-9
- FGMRES retained its production 1e-4 relative threshold

### CFL=0.3 and 4wz AMR ghost admissibility

The first full CFL=0.3 timestep, `dt=0.1216300565`, converged in 7 Newton
iterations.  Before the following step, jobs 347892 and 347895 exposed a
separate explicit-stage issue: multidimensional coarse/fine interpolation
made finite negative radiation-energy ghost values near -9.3e-10, although
the owning active cells remained positive.  Commit `dc0f01759` bounds direct
FLD prolongation, and commit `667fb5a2c` enforces ghost-stencil admissibility
at the 4wz radiation reconstruction consumer while retaining fatal checks for
all negative active cells and non-finite ghosts.

Job 347897 then completed from t=0.1 to 0.3:

- 3 M-test-a nodes, 168 MPI ranks, 47.89 s wall time
- `dt=0.1216300565` and `dt=0.0783699435` both converged in 7 Newton iterations
- final max residuals: 2.62e-9 and 1.27e-9
- no negative-radiation, relaxed-Newton, or table-domain warning
- the production CFL=0.3 gate is therefore passed through t=0.3

### CFL=0.3 medium gate and lower-bound active set

Job 347898 completed the continuation from t=0.3 to t=3.0:

- 3 M-test-a nodes, 168 MPI ranks, 24 timesteps, 129.70 s wall time
- 23 solves converged normally; one solve at cycle 121 was accepted at the
  relaxed nonlinear tolerance
- relative drift over the recorded continuation was mass `-2.94e-9`, angular
  momentum `+4.59e-7`, and gas+radiation energy `-1.19e-5`
- at t=3 the interior and rho-at/above-surface radial Mach maxima were
  `3.17e-3` and `2.94e-3`
- the low-density surface/taper maximum was 660, but only `8.97e28 g` of
  stellar material had radial Mach above 0.1; the table-margin gate below
  subsequently identified this as a thermodynamic-domain artifact

The verbosity-2 reproduction showed why the single solve stalled.  Once a
trial clipped a radiation cell to its lower bound, later unconstrained linear
directions could still point below the bound even with a positive local
reduced right-hand side because of strong neighbour coupling.  Commit
`48ca4a67f` therefore retains cells that were actually clipped as active
constraints for the remainder of that timestep, rolls the mask back with a
rejected line-search trial, and releases it at the next timestep.

The first reproduction with that change, job 347902, exposed finite negative
values only in the first AMR ghost layer of the following Newton stencil.
These were interpolation undershoots, not negative owning active cells.
Commit `949b5a5cf` bounds those values only when they are copied into the
private Newton stencil; non-finite ghosts and every negative active value
remain fatal.

Job 347912 repeated the exact two-step interval containing the old relaxed
solve:

- completed normally in 42.83 s
- both solves converged normally in 7 Newton iterations
- final max residuals were `9.44e-9` and `4.14e-9`
- no relaxed acceptance, fatal error, or warning occurred

Commit `1a17ccea2` appends four history diagnostics giving the minimum distance
in dex to the lower and upper density and gas-energy coordinates of the EOS
table.  A negative value unambiguously signals table extrapolation.

### EOS-domain and outer-Mach diagnosis

Job 347913 advanced the old t=3 restart by one short `dt=0.01` step with the
new history fields.  It showed density lower/upper margins of `+6.00/+2.19`
dex and a gas-energy upper margin of `+1.083` dex, but the lower gas-energy
margin was `-0.265` dex.  The explicit hydro conversion restored its configured
`tfloor=1000 K` before this step, reducing the low-density surface Mach maximum
from 660 to 0.078.  Together these results identify the old huge Mach as an
EOS-extrapolated sound-speed artifact after the implicit update, not a
resolved mass ejection.

Commit `aed3eed02` makes the implicit FLD gas lower bound honor the same
general-EOS temperature floor as the explicit hydro conversion, including
when density equals the configured density floor.  It also bounds the private
Newton starting iterate before any table temperature or derivative is
evaluated.

Job 347915 repeated the same t=3 to 3.01 step with this fix:

- completed normally in 29.72 s; Newton converged in 6 iterations with final
  max residual `1.04e-9`
- EOS margins (lower/upper) were `+6.00/+2.19` dex in density and
  `+0.181/+1.083` dex in gas energy, so every active cell was in-table
- interior, resolved-surface-density, and low-density-surface radial Mach
  maxima were `0.00318`, `0.00295`, and `0.0533`
- no stellar mass had radial Mach above 0.1, and there was no relaxed solve,
  fatal error, or warning

Job 347916 continued the corrected restart from t=3.01 to t=10 at CFL=0.3:

- 3 M-test-a nodes, 168 MPI ranks, 60 normal Newton solves, 381.71 s wall
  time; there was no relaxed solve, fatal error, or warning
- maximum Newton iteration count was 7 and maximum final nonlinear residual
  was `6.31e-9`; 411 FGMRES solves used at most 22 iterations
- relative drift was mass `-9.13e-10`, angular momentum `+1.35e-6`, and
  gas+radiation energy `-6.26e-5`
- the minimum EOS margins remained positive: density lower/upper
  `+6.00/+2.190` dex and gas-energy lower/upper `+0.181/+1.083` dex
- final interior and rho-at/above-surface Mach maxima were `0.00696` and
  `0.00645`
- the low-density taper maximum increased monotonically to `0.209`; the
  stellar-tracer mass above Mach 0.1 was `1.86e29 g` (`9.4e-5 Msun`), while
  no stellar material exceeded Mach 0.3

Job 347920 read the t=10 restart without advancing it and localized the maximum
to symmetric L3 cells at ellipsoidal radius 1.077, just outside the initial
surface.  The maximum cell had `rho=1.89e-10 g cm^-3`, almost entirely stellar
tracer, `vr=-0.918 km s^-1`, `cs=4.40 km s^-1`, and Mach 0.209.  The negative
radial velocity and multi-quadrant symmetry identify a subsonic contraction of
the low-density taper rather than asymmetric ejection.  Job 347923 therefore
continues to t=30; persistent growth will trigger an A/B that extends L4
surface refinement into this density band.

### t=30 mixed-drift failure and monotone implicit stencil

The original centered mixed-frame continuation, job 347923, reached
`t=28.001433492` and failed at cycle 346 with
`backtracking_exhausted`.  Its 149 earlier solves converged normally.  The
resolved interior remained quiet (Mach `0.00803`), but the low-density taper
had reached Mach `0.891`.

Job 347939 reproduced the exact failing step from a newly materialized restart.
Tightening FGMRES from `1e-4` to `1e-6` (job 347940) did not change the failed
Newton direction, and permitting line-search scales down to `1/512` (job
347943) only delayed the failure by one iteration.  This rules out the linear
tolerance and the default backtracking range as the cause.

The failure cell was on an L3/L4 interface.  It had
`Egas=1.0611e-3`, `Er=1e-20`, and one radiation neighbour at
`Er=1.6955e-4`.  Its timestep-integrated thermal exchange was only
`8.73e-25`, whereas the centered `q v.grad(Er)` term was `4.03e-2`, about
38 times the cell gas energy.  Thus the centered advection-like source asked
an already floor-bound radiation cell for an inadmissible update.  Turning off
both mixed-frame source and transport passed the one-step control, while
retaining transport without its matched source produced negative radiation
states and is not an admissible production mode.

The implicit mixed drift now uses a first-order upwind derivative of the
cell-centred average face speed.  Its Schur contribution uses the identical
positive diagonal and non-positive off-diagonals.  This is intentionally
low-order: the implicit FLD source needs monotonicity rather than the 4wz
hydrodynamic reconstruction order.

- Job 347956 passed the exact formerly failing step in seven full Newton
  iterations, with no backtracking.  Final L2/max residuals were
  `2.23e-12/1.70e-9`; FGMRES used at most 18 iterations.
- Job 347958 repeated that step with a finite-difference Jacobian check.  At
  perturbation `1e-4`, gas and radiation relative L2 errors were
  `1.05e-12` and `4.87e-11`, respectively.
- Job 347957 continued from `t=28.0014` to 30.  All 17 solves converged in
  6--7 Newton iterations with no backtracking; 103 FGMRES solves used 4--18
  iterations.  Segment drifts were mass `-4.08e-10`, angular momentum
  `+3.18e-7`, and gas+radiation energy `-8.18e-6`.  Every EOS margin remained
  positive.

The dedicated Zhang et al. moving-radiation-pulse regression used the separate
ideal-EOS build and `analysis/wor182/xd_upwind_pulse_regression.sh`:

- At the reference CFL=0.03, static job 347964 completed 4482 steps and moving
  job 347965 completed 4571 steps to code time 100.  Every nonlinear solve
  converged normally with no backtracking.
- The expected Galilean displacement was `v0*t=5.2942344`; the measured
  radiation-excess centroid displacement was `5.2938828`, a relative speed
  error of `6.64e-5`.
- After translating the static final profile and fitting only its amplitude,
  the moving/static relative shape errors were L1 `4.40e-4` and L2 `3.62e-4`.
  The excess-profile widths were `33.44264` (static) and `33.44331` (moving).

The monotone stencil therefore retains the established Galilean pulse
transport while removing the non-positive centered update at a sharp front.

This resolves the nonlinear failure but not the outer-layer stability gate.
At `t=30`, the interior and rho-at/above-surface Mach maxima were only
`0.00785` and `0.00760`, while the low-density taper maximum had grown to
`0.942`; `1.94e30 g` exceeded radial Mach 0.1 and `2.61e29 g` exceeded 0.3.
The next physical gate must therefore address the outer taper/refinement rather
than extend this configuration blindly.

### Surface-refinement A/B: retain relative L2

Jobs 347973 and 347981 restarted the same accepted `t=10` state and advanced
to `t=11.5` with the surface density band held at relative levels L2 and L3,
respectively.  The successful L3 layout used 324 ranks on 9 nodes (36 ranks
per node).  An earlier 504-rank layout exceeded a safe per-node memory
packing, and a 168-rank/9-node layout supplied too little compute per node.

- L2 used 1688 MeshBlocks on 168 ranks/3 nodes and 99.24 s wall time.  Its
  final interior, taper, and `rho >= rho_surface` radial Mach maxima were
  `0.00735`, `0.2421`, and `0.00679`.  No stellar mass exceeded Mach 0.3.
- L3 used 7904 MeshBlocks on 324 ranks/9 nodes and 301.53 s wall time.  Its
  corresponding maxima were `0.00955`, `0.3115`, and `0.00955`; about
  `1.78e27 g` exceeded Mach 0.3.
- Both runs retained positive EOS margins and had no Newton or FGMRES failure.

Jobs 347985 (L2) and 347984 (L3) then continued their fixed AMR topologies
from `t=11.5` to 13.  L2 completed in 97.29 s with 14 Newton solves (at most
7 iterations) and 81 FGMRES solves (at most 19 iterations).  L3 completed in
322.87 s with 14 Newton solves (at most 8 iterations) and 91 FGMRES solves
(at most 29 iterations).  At `t=13`, the taper Mach maximum was `0.2723` for
L2 versus `0.3449` for L3, and the interior maximum was `0.00763` versus
`0.01251`.

Relative L3 therefore costs 4.68 times as many MeshBlocks and about 6.4 times
as many rank-seconds over the controlled continuation without improving the
stability metric.  Relative L2 is the production choice.  The worsening with
resolution points to thermal/radiative adjustment of the mechanically
matched low-density taper, rather than an under-resolved hydrodynamic mode.
Longer table-EOS runs should first make that layer radiatively consistent or
demonstrate that its negligible-mass motion remains dynamically decoupled.

### Taper radiation-balance localization

Experimental branch `exp/wor182-rad-taper-diagnostic-20261126` reconstructs
the cell-centred diffusion term from the same six lagged NR-FLD face
coefficients and records both signed and absolute diffusion and thermal
exchange powers.  The thermal term uses the conserved gas energy, the same
tabulated-EOS temperature inversion as NR-FLD, and the solver's exact
floating-point LTE resolvability guard.  This last detail is essential:
without it, multiplication by `c*sigma_P` turns an adjacent-ULP LTE difference
into a spurious subsecond exchange time.

After the first normal `dt=1e-5` startup step (XD job 348047), the taper
contained `3.012e32 g` (`0.1515 Msun`).  Its signed/absolute diffusion powers
were `+1.793e39/+1.305e40 erg s^-1`, and its signed/absolute material-to-
radiation exchange powers were `+4.432e38/+5.403e38 erg s^-1`.  The shortest
local total-energy times were 259 s for diffusion and 12,617 s for exchange.
The positive integrated signs mean that the taper radiation field initially
gains energy, but substantial cell-to-cell cancellation is present.

The initial surface-Mach maximum was instead in a cell at ellipsoidal radius
1.08485 and density `4.40e-11 g cm^-3`.  That cell was locally *losing*
radiation by diffusion at `4.45e-2 erg cm^-3 s^-1`, with a 3,681 s local
energy time; its material-radiation exchange was at the LTE roundoff floor.
Its pressure, gravity, and radiation acceleration magnitudes were 0.891,
0.590, and 0.0433 cm s^-2, and the full discrete force residual was
0.445 cm s^-2 (relative residual 0.273).  Thus the fresh outermost layer has
both an hour-scale radiative divergence and a sizeable mechanical mismatch.

Job 348034 repeated one production-CFL step from the accepted time-10 restart.
At time 10.1217, the signed/absolute taper diffusion powers were
`-1.012e38/+1.258e38 erg s^-1`, while the signed/absolute exchange powers were
`-5.062e38/+7.469e38 erg s^-1`.  The shortest local diffusion and exchange
times were 400 and 1,382 s, respectively; the hydrodynamic step was 2,677 s
in physical units, so these stiff terms are appropriately handled by the
implicit solve.

The time-10 surface-Mach maximum (job 348043) had moved inward to ellipsoidal
radius 1.07690 and density `1.89e-10 g cm^-3`.  It was contracting at Mach
0.211, but its local diffusion term was weak heating with a `4.34e7 s` time,
and its thermal exchange again vanished at the LTE roundoff threshold.  Its
pressure, gravity, and radiation accelerations were 0.0932, 0.587, and
0.00287 cm s^-2; the full force residual was 0.442 cm s^-2 (relative 0.556)
and pointed predominantly inward once centrifugal support is included.

The spatial correlation therefore distinguishes cause from response.  Rapid
radiative and mechanical adjustment begins in the outermost matched cells,
while the later Mach maximum is a deeper, mechanically under-supported layer
reached by the resulting contraction.  Direct local radiation cooling at the
time-10 Mach maximum is ruled out.  Surface L3 cannot repair this physical
initial-data mismatch, consistent with its worse and much more expensive A/B
result.

### Controlled corotation-relative relaxation

Experimental branch `exp/wor182-radiative-relaxation-20261126` adds an
initialization-only velocity relaxation for the radiative star.  It damps the
stellar-tracer-weighted velocity perturbation relative to rigid corotation,
not the inertial velocity or the atmosphere.  The exponential update changes
the conserved total energy by exactly the kinetic-energy difference, so gas
internal energy and the implicit FLD state are not artificially cooled.  A
linear time taper removes the damping completely before the subsequent
acceptance interval.  All three input parameters default to negative values,
which leaves the production baseline bitwise on the old path.

The history output separately records inertial-frame work and the positive
corotating perturbation energy removed in each step, along with their powers.
The helper algebra and taper endpoints have a standalone C++ check.  Both the
local GNU/MPI/HDF5 table-EOS build and the XD Intel/MPI/IPO build passed.  A
one-step XD gate (job 348049) converged in three Newton iterations; every
FGMRES solve took two iterations and the final nonlinear maximum residual was
`7.64e-9`.

Jobs 348051 (relaxation time 5, taper from time 20 to 50) and 348052 (disabled)
then formed a controlled fresh-map A/B to time 1 at CFL 0.1.  Both completed
136 normal implicit solves, retained positive EOS-table margins, used at most
18 FGMRES iterations, and had no stellar mass above radial Mach 0.1.  At time
1 the relaxed/control results were:

- low-density taper Mach maximum `0.01820/0.01919`
- interior Mach maximum `0.002164/0.002352`
- stellar perturbation kinetic energy `9.51e39/1.17e40 erg`

Thus the operator already reduces the mapped transient without changing
solver robustness.  The relaxed run applied `1.29e42 erg` of inertial-frame
work while removing `1.85e39 erg` of corotating perturbation energy.  The
large distinction is expected: correcting a small velocity error on top of
the bulk corotation changes inertial energy by approximately `Omega dJ`.
Audit of this gate exposed a units-only error in the new corotating-energy
history column; commit `e4195901d` corrects its scaling before the longer
relaxation run.  The evolved conserved state was unaffected.

Job 348055 then continued the relaxed time-1 state to time 20 at CFL 0.3,
with full damping strength throughout.  It completed in 877 s on three XD
nodes.  All 158 Newton solves converged normally in at most seven iterations,
FGMRES used at most 21 iterations, and there was no backtracking, relaxed
acceptance, warning, or fatal error.  The minimum gas-energy EOS margin was
`+0.181 dex`, and no stellar material reached radial Mach 0.3.

At time 9.96 the taper Mach maximum was 0.102 and only `1.85e28 g` exceeded
Mach 0.1.  The unrelaxed time-10 reference had corresponding values 0.209 and
`1.86e29 g`, so damping reduced the maximum by about a factor two and the
affected mass by about a factor ten.  Over the full segment the taper maximum
peaked at 0.205 and ended at 0.186, while the interior maximum stayed below
0.00355.  The artificial operator supplied `5.45e42 erg` of inertial work and
removed `3.48e41 erg` of corotating perturbation energy.  Segment mass drift
was `+1.32e-9`; angular momentum changed by `+1.68e-4`, as expected for the
controlled return toward rigid corotation.

The outer maximum had begun to decline by time 20 but was not yet quiescent.
Reducing the damping there would therefore be premature.  The next continuous
schedule keeps full strength through time 50 and tapers linearly to zero from
50 to 100; job 348057 tests the first interval.  Because both schedules have
unit strength at time 20, changing the future taper introduces no force jump.

Job 348057 continued with full strength from time 20 to 50 at CFL 0.3.  It
completed in 1,192 s on three XD nodes.  All 248 Newton solves converged
normally in at most seven iterations, FGMRES used at most 19 iterations, the
minimum gas-energy EOS margin remained `+0.181 dex`, and there was no warning,
relaxed acceptance, backtracking failure, or fatal error.

The taper Mach maximum rose to 0.259 and then declined to 0.227; the interior
maximum was only 0.00301, and no stellar material reached Mach 0.3.  At the
end, `9.28e29 g` (`4.67e-4 Msun`) exceeded Mach 0.1, about 0.31 percent of the
artificial taper mass.  The segment changed the mean and RMS stellar radii by
only `1.20e-5` and `5.10e-6`, respectively.  It removed `6.46e41 erg` of
corotating perturbation energy.  However, its final removal power was still
`1.06e36 erg s^-1`, almost the segment maximum `1.07e36 erg s^-1`; the damping
was still actively balancing the outer mismatch rather than becoming
irrelevant.

The zero-force acceptance criterion therefore overrides the earlier nominal
taper date.  Job 348059 keeps full strength continuously to time 100 and moves
the proposed linear taper to 100--150.  Again, both the old and new schedules
have unit strength at the restart time, so this introduces no discontinuity.

Job 348059 completed the full-strength interval from time 50 to 100 in 2,139 s
on three XD nodes.  All 413 Newton solves converged normally in at most eight
iterations, FGMRES used at most 18 iterations, and there was no solver anomaly.
The minimum gas-energy EOS margin was again `+0.181 dex`.  Relative to the
time-50 restart, the final mass, angular momentum, mechanical-energy, and
mechanical-plus-radiation-energy changes were `+2.82e-9`, `+1.51e-5`,
`+1.19e-5`, and `+1.54e-4`; the mean and RMS stellar-radius changes were only
`+1.75e-5` and `+8.55e-6`.

The mechanically relevant acceptance gate did not pass.  The taper Mach
maximum ended at 0.235 after reaching 0.236, while the interior maximum stayed
below 0.00287 and no stellar material reached Mach 0.3.  The mass above Mach
0.1 increased to `1.83e30 g` (`9.18e-4 Msun`).  More decisively, the artificial
operator removed `1.42e42 erg` of corotating perturbation energy and its final
removal power was `1.50e36 erg s^-1`, close to the segment maximum
`1.54e36 erg s^-1` and larger than at time 50.  The calculation is numerically
robust, but it has reached a damped, forced outer flow rather than a star that
is ready to remain stable without relaxation.  The scheduled time-100 taper
must therefore not be started.

The time-100 one-step diagnostic job 348119 completed normally and reported
the 32 largest surface-Mach cells.  The global maximum had radial Mach 0.235
at ellipsoidal radius 1.063 and density `2.95e-10 g cm^-3`.  In these cells the
gravity acceleration was commonly much larger than the pressure acceleration;
the full-force residual was `0.57--0.73 cm s^-2` for representative maxima.
Their matter-radiation exchange was numerically negligible (thermal-exchange
times near `1e21 s`), so the maximum-Mach flow is a mechanical outer-profile
mismatch, not direct radiative heating or cooling.  Velocity damping masks
that mismatch but cannot repair it.

The volume-integrated taper diagnostic separately reports a large stiff
matter-radiation exchange magnitude, rising to about `7.02e43 erg s^-1` by
time 100 even though the global radiation and gas energies, the EOS margins,
and all nonlinear solves remain regular.  Because this exchange is absent
from the maximum-Mach cells, it is not the direct cause of their motion.  Its
spatial source and cancellation between the implicit gas/radiation terms
still need to be localized before production use.

Job 348160 localized the fastest exchange cell in every taper MeshBlock at
time 100.  All 104 reports form a very thin shell at ellipsoidal radius
`1.00002--1.00838` and density `2.51e-9--2.95e-9 g cm^-3`, immediately below
the `3e-9 g cm^-3` surface match.  The cells remain close to LTE:
`aT^4/E_r=1.00163--1.00334`.  Their effective Planck opacity is nevertheless
`2.17e4--2.34e4 cm^2 g^-1`, making a sub-percent temperature difference a
`0.039--0.095 s` exchange.  The large signed/absolute powers therefore arise
from a stiff, nearly LTE matching shell and cancel between material and
radiation; they are spatially distinct from the Mach maximum.

Job 348163 added radial force components and the original SCF state to the
time-100 Mach-cell report.  The reported cells have density ratios relative
to the source profile of `0.985--1.43`, but alternating radial force residuals
of `-0.724--+0.350 cm s^-2`.  The global Mach maximum has ellipsoidal radius
1.063, density `2.95e-10 g cm^-3`, and inward radial residual
`-0.560 cm s^-2`.  This confirms that damping is holding a mechanically
misbalanced outer pressure profile rather than merely suppressing one
coherent bulk contraction.

### Total-pressure restart sweep

The first pressure-profile diagnostic compared material pressure with the SCF
total pressure.  That ratio is useful for the EOS/radiation split but is not a
like-for-like support diagnostic.  Commit `b7b63b632` corrects the gate to
report material, radiation, and total pressure separately and applies its
threshold to `(P_gas+E_r/3)/P_SCF`.

Job 348180 then read the same relaxed restarts at times 1, 20, 50, and 100
without advancing them.  The following values are the distribution over the
per-MeshBlock minimum total-pressure ratios in the selected outer band
(`5 rho_atm < rho_source < 3e-9 g cm^-3`):

| time | reports | minimum | 10th percentile | median | 90th percentile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 440 | 0.8384 | 0.8492 | 0.9131 | 0.9410 |
| 20 | 688 | 0.08149 | 0.09982 | 0.8706 | 0.9130 |
| 50 | 720 | 0.07444 | 0.09380 | 0.8171 | 0.8850 |
| 100 | 744 | 0.06612 | 0.08325 | 0.7671 | 0.8482 |

The low-ratio tail is therefore not an omitted radiation-pressure term.  By
time 20, at least ten percent of the reported block minima retain only about
ten percent of the SCF total pressure, while the median continues to erode
through time 100.  This is direct evidence that the HD-polytropic taper is not
a stationary FLD outer solution.

### MESA-cut luminosity source

The prior SCF-FLD gates injected no stellar luminosity.  Commit `8689558f3`
adds an opt-in compact spherical source with an analytically normalized
`(1-r^2/R_h^2)^2` kernel.  The physical input is kept separate from its
spatial regularization: the production value is the WOR-151 MESA-cut
luminosity `116315.98 Lsun`, while the default heating radius is the resolved
soft-core support `4.520740656e12 cm`.  A history column records the actual
mesh-integrated injection.

The standalone C++ integration test reproduces the analytic kernel volume to
`3.4e-14` relative error.  Fresh-map job 348178 then passed one FLD step with
the luminosity enabled.  The L4 AMR mesh injected
`4.452575265e38 erg s^-1`, only `1.01e-7` below the requested
`4.452575714e38 erg s^-1`.  Newton converged in three iterations to maximum
residual `7.64e-9`; each FGMRES solve used two iterations and there was no
warning, backtracking, or EOS-domain violation.  The source implementation is
therefore accepted, but it cannot by itself make the zero-flux HD taper a
radiative equilibrium.  The next initial-data gate must rebuild the outer
profile against the target luminosity, tabulated EOS, opacity, rotation, and
FLD radiation force.

### Rotating radiative-surface reconstruction

Commit `4c08236dc` added an opt-in C++ surface integrator starting from a
chosen density on every SCF angular ray.  It inverts the production material
EOS at fixed `(Pgas,T)`, uses the production Rosseland opacity and rotating
SCF potential, and initializes `Er=aT^4`.  The first version incorrectly
forced the full luminosity through the `3e-9 g cm^-3` match point.  Job 348187
showed that this is not a viable RSG closure: the reconstructed layer was
less than `7e10 cm` thick, removed `0.142 Msun`, raised the initial surface
Mach maximum from `0.00124` to `0.0133`, and raised the maximum force residual
ratio from `0.442` to `0.894`.

Commit `01f39cbb6` therefore uses field 6 of the material-EOS table for
`nabla_ad`.  Where the full FLD luminosity is Schwarzschild unstable, the
temperature gradient is capped at `nabla_ad` and the remainder is assigned to
unresolved convection.  Commit `53d132f18` made this split use the exact
Levermore--Pomraning limiter instead of its diffusion-limit approximation.
At the original match density, job 348195 reconstructed a
`6.74e12--9.87e12 cm` layer and recovered the mapped mass to within `0.018%`
of the original profile.  Only `(2.30--2.63)e-4` of the luminosity is carried
radiatively at the anchor and all accepted integration substeps are
convective.  This quantitatively explains why a static full-luminosity FLD
solution cannot replace the RSG convective envelope at that depth.

The purely adiabatic reconstruction at `3e-9 g cm^-3` over-expanded the outer
shape by about four to six percent.  A Gaussian superadiabatic excess was
then tested with a deeper `1e-8 g cm^-3` anchor.  Peak `delta-nabla=0.55`
(job 348213) produced a plausible `3350--3600 K` edge and an outer shape
within `0.4--2.2%` of the requested rotating surface.  Its one-step gate
348216 converged cleanly (four Newton iterations, two FGMRES iterations per
linear solve, final maximum residual `1.26e-9`), but it is not thermally
acceptable.  The absolute matter-radiation exchange rose to
`6.71e45 erg s^-1` and its minimum timescale fell to `0.0181 s`.

Restart diagnostic 348219 localized that transient to
`rho=(2.4--2.8)e-9 g cm^-3` near the old taper onset, not to the new outer
edge.  The gas and radiation temperatures differ by only one to two percent,
but effective Planck opacities of roughly `1e3 cm^2 g^-1` turn that mismatch
into exchange times of `0.03--0.7 s`.  The superadiabatic gradient makes the
radiative flux divergence change rapidly, while the static Athena model has
no compensating convective-flux operator.  Smaller timesteps would resolve
this transient but would not make the initial state stationary.

The less invasive alternative retains the inner part of the old taper and
matches at `1e-9 g cm^-3` with no superadiabatic excess.  Fresh-map job 348222
ended only `0.49%` outside the polar target and `0.67%` outside the equatorial
target.  Relative to job 348178 its mapped mass differs by `-0.0012%`, taper
mass by `+0.078%`, and maximum force residual is unchanged at `0.441573`.
The interior Mach maximum is unchanged at `3.60e-6`; the outer maximum is
larger (`0.00371` versus `0.00124`) but much smaller than the fully rebuilt
or superadiabatic cases.  Its initial absolute diffusion power is
`2.78e39 erg s^-1`, below the original profile's `1.32e40 erg s^-1`.

One-step job 348223 then passed the thermal gate.  Newton converged in three
iterations, every FGMRES solve used two iterations, and the final maximum
residual was `7.60e-9`.  After the step, the absolute exchange power was only
`3.01e38 erg s^-1` and the minimum exchange time was `3.10e5 s`; the original
job 348178 values were `5.40e38 erg s^-1` and `1.26e4 s`.  Absolute diffusion
power was `2.77e39 erg s^-1`.  This low-density, zero-excess reconstruction is
the accepted short-gate candidate; job 348224 advances it to time 0.1 before
any long relaxation test.

## Open acceptance items

1. Reconstruct the low-density outer density, material pressure, and radiation
   profiles against the full rotating force balance and the WOR-151 cut
   luminosity.  Repeat the strong-
   damping gate, and start a taper only after the corotating removal power is
   demonstrably declining toward zero.  Then require a subsequent undamped
   interval to remain stable.
2. Verify the thin match-shell exchange cancellation on the rebuilt outer
   profile and require its stiffness not to worsen relative to job 348160.
3. Add a focused AMR regression for positivity-constrained FLD ghost
   prolongation and the Newton radiation active set.
4. Keep relative L2 as the production surface-refinement level unless a new
   outer profile changes the controlled A/B result.
