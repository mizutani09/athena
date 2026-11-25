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

This resolves the nonlinear failure but not the outer-layer stability gate.
At `t=30`, the interior and rho-at/above-surface Mach maxima were only
`0.00785` and `0.00760`, while the low-density taper maximum had grown to
`0.942`; `1.94e30 g` exceeded radial Mach 0.1 and `2.61e29 g` exceeded 0.3.
The next physical gate must therefore address the outer taper/refinement rather
than extend this configuration blindly.

## Open acceptance items

1. Add a focused moving-radiation-pulse regression for the upwind implicit
   mixed drift, including transport speed and numerical diffusion.
2. Run an L3/L4 outer-density refinement A/B before any longer table-EOS
   stability extension; the t=30 taper is already nearly sonic.
3. Compare L3/L4 and ideal/table force, Mach, Newton, and FGMRES histories.
4. Replace cell-average closure with a derived, table-thermodynamic profile if
   the first A/B shows appreciable thermal projection noise.
5. Add a focused AMR regression for positivity-constrained FLD ghost
   prolongation and the Newton radiation active set.
