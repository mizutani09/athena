# WOR-122 / WOR-14: temporary thermal support for an RSG convective spin-up

This is an intentionally nonphysical preparation stage. It is disabled in the
normal input (`prep_thermal_tau = -1`). The goal is to keep the initially
unresolved surface transition from cooling before convective transport grows.
The injected energy must be tapered away before interpreting a production run.

## Mechanism

After each implicit FLD solve, each radial shell receives a common offset to
its gas **specific internal energy**. The offset relaxes the mass-weighted shell
mean toward the imported 1D profile at the cell's current density. Angular
thermal fluctuations, density and velocity remain free. A smooth radial window
sets the preparation region. `prep_thermal_tau` and optional time taper set the
relaxation rate. The source updates pressure, FLD's cached gas energy and
opacity; `Eprep_thermal` and `Pprep_thermal` in the HST record energy and power.

The heating is a numerical bridge across the missing initial convective flux,
not a model of nuclear or convective heating. The central physical heat source
remains active. The first test used 790–850 R_sun, full weight at 805–830 R_sun,
1 R_sun bins, and 0.1 code-time relaxation.

The best short trial used `prep_thermal_tau=0.1`,
`prep_thermal_rmin_rsun=790`, `prep_thermal_rfull_rsun=805`,
`prep_thermal_rflat_end_rsun=870`, `prep_thermal_rmax_rsun=900`, and
`prep_thermal_bin_width_rsun=1`. Time taper was disabled. Its exact Slurm
commands are in `$WORK/wor122/prep128_outer_t8.sh` and
`$WORK/wor122/prep128_outer_t20.sh`.

## Gates completed

- Local 64³, one step, PPM: energy-source ledger and change in total gas plus
  radiation energy relative to the unsupported control agree to 2e-10
  relative. A time taper ending before the first step gives exactly zero
  `Eprep_thermal` and `Pprep_thermal`; all old HST columns agree numerically
  with the unsupported control.
- XD 256³, PPM, `t=1` code time (about 0.255 d), job **340119**: normal
  termination, Newton and opacity checks passed. In 813–817 R_sun, the
  mass-weighted geometric mean Tgas ratio to the initial state is 0.9932,
  compared with 0.9101 in the unsupported PPM control. Maximum Mach is
  0.00436, similar to 0.00431 in the control. The preparation power during
  ordinary steps approaches about 0.97 times the central heating luminosity.
  The final duplicate HST record has a shortened time step and should not be
  used as a representative instantaneous power.
- XD 128³, PPM, `t=8` code time (about 2.04 d), job **340122**: normal
  termination and opacity checks passed. In 813–817 R_sun, Tgas ratio is
  0.991 versus 0.829 in the earlier unsupported **PLM** control. In 0–166 R_sun, maximum
  speed is 0.03 versus 0.22 km/s. However, maximum Mach reaches about 0.17
  versus 0.041 in that PLM control; in 850–880 R_sun the maximum speed is
  0.92 versus 0.11 km/s. The different reconstruction order means this
  comparison cannot isolate the source's effect. In the matched PPM control,
  850–880 R_sun maximum speed is 1.02 km/s: the source **reduces** the
  outer flow. Long-run stability remains unverified.
- XD 128³, PPM, same duration and window, but 16 R_sun shell bins, job
  **340124**: maximum Mach increases to 0.181; 813–817 R_sun Tgas ratio
  falls to 0.963 and 850–880 R_sun maximum speed increases to 0.99 km/s.
  Coarsening the correction bins does not solve the disturbance.
- XD 128³, PPM, support shifted inward to 790–830 R_sun with 8 R_sun
  bins, job **340126**: maximum Mach increases to 0.186; 813–817 R_sun
  Tgas ratio is 0.964, 830–850 R_sun Tgas ratio is 1.245, and 850–880 R_sun
  maximum speed is 1.05 km/s. Extra heat diffuses outward and overheats the
  unrestrained outer layers. The source power is about 1.8 Lheat, so this
  option is worse.
- A local 64³ radiation-energy relaxation variant was tried and then
  removed from the final implementation. At `t=1` it left the 813–817 R_sun
  gas temperature at 0.899 of the initial value, and its net energy input was
  small. It directly altered Er without maintaining the gas thermal state.
- XD 128³, PPM, gas support extended to 790–900 R_sun, full at
  805–870 R_sun, job **340128**: normal termination at `t=8` (about 2.04 d).
  The 813–817 R_sun Tgas ratio remains 0.991; maximum Mach falls to 0.149,
  850–880 R_sun maximum speed to 0.80 km/s, and the outer radiative luminosity
  approaches 1.01 Lheat. A local 64³ one-step energy ledger for this variant
  agrees with the total gas-plus-radiation energy change to 7e-11 relative.
  This is the best tested configuration. Its outer speed is lower than the
  matched unsupported PPM control's 1.02 km/s. This is **not yet** a
  long-run gate.
- XD 128³, PPM, same extended support restarted through `t=20` (about
  5.09 d), job **340171**: normal termination, opacity-domain count zero and
  empty stderr. Maximum Mach was 0.260. The source power during ordinary
  steps decreased from roughly 1.0 to 0.97 Lheat, while box radiative
  luminosity remained near Lheat. This suggests a possible slowdown of the
  early outer-flow growth, but convection and a source-free state remain
  unverified. An enthalpy-flux audit followed as job **340188**.
- The job **340188** enthalpy-covariance audit at `t=20` finds only
  0.0045 Lheat in the 800–830 R_sun shell, versus the order-unity missing
  luminosity. The source cannot yet be tapered away. This is a volume-shell
  proxy for convective transport, not an exact spherical-face flux.
- XD 128³, **matched PPM control** without the thermal source, job
  **340180**: normal termination at `t=8`. Maximum Mach is 0.187, versus
  0.149 with extended thermal support. Outer radiative luminosity is about
  1.40 Lheat, versus 1.00 with support. In 813–817 R_sun, Tgas is 0.826 of
  its initial value, versus 0.991 with support; in 850–880 R_sun, maximum
  speed is 1.02 versus 0.80 km/s. The earlier PLM control is useful for
  context but was not a matched reconstruction-order comparison.
- XD 128³, PPM, extended support continued to `t=36` (about 9.17 d),
  job **340189**: normal completion, maximum Mach 0.456. The Mach number
  continued to increase after `t=20`; the 128³ PPM preparation is not yet
  stable on ten-day scales. Job **340200** found the maximum speed at
  850–880 R_sun rose from 1.34 km/s at `t=20` to 2.09 km/s at `t=36`;
  the 0–166 R_sun maximum rose from 0.15 to 0.30 km/s. This is primarily
  an outer-layer disturbance, not a fast central plume. The 800–830 R_sun
  convective gas-enthalpy covariance was only 0.0050 Lheat at `t=36`, so
  the preparation source still cannot be tapered away.
- XD 128³, PLM, same extended support, job **340193**: normal termination
  at `t=8`, maximum Mach 0.0438. The earlier PLM unsupported control at
  `t≈7.85` had maximum Mach 0.0413. Job **340194** found that the support
  retained 0.9915 of the initial gas temperature in 813–817 R_sun versus
  0.8292 without support. The 850–880 R_sun maximum speed was 0.168 versus
  0.114 km/s, so the source raises this PLM speed slightly even while the
  maximum Mach remains similar. Job **340204** completed normally at
  `t=24` (about 6.11 d), maximum Mach 0.136. Job **340206** found that
  813–817 R_sun retained Tgas ratio 0.9917, while 850–880 R_sun maximum
  speed reached 0.523 km/s. The 800–830 R_sun convective gas-enthalpy
  covariance grew from 0.0022 Lheat at `t=8` to only 0.0067 Lheat at
  `t=24`. PLM delays the flow growth but has not established convective
  transport or a source-free equilibrium.
- XD 256³, PPM, extended support, job **340195**: normal termination at
  `t=3` (about 0.76 d), empty stderr, and opacity-domain count zero.
  Maximum Mach was 0.0132, versus 0.0594 in the same-time 128³ supported
  PPM run; box radiative luminosity was 0.955 Lheat. Job **340205** found
  Tgas ratio 0.9929 at 813–817 R_sun and maximum speed only 0.021 km/s
  in 850–880 R_sun. Job **340210** extended this state but hit its 2-hour
  Slurm limit at `t=12.54`, short of its target `t=20`. The last HST point
  at `t=12.52` has maximum Mach 0.0545, box radiative luminosity about
  0.957 Lheat, and relative box-mass change below 2e-8. There is no solver
  failure in the completed steps. Job **340211** was canceled because its
  `afterok` dependency was not satisfied. The last complete HDF pair is at
  `t=11.01`; replacement analysis job **340370** found 813–817 R_sun Tgas
  ratio 0.9888 and 850–880 R_sun maximum speed 0.072 km/s. The signal-time
  `final.rst` was truncated (3,144,983,024 bytes instead of the expected
  3,171,305,824) and cannot be restarted; probe job **340371** failed at
  the restart layout check and its dependent analysis **340372** was canceled.
  The original job also had an absolute cycle limit `nlim=250`, which would
  have stopped it near `t≈13` even with a longer wall time. Job **340373**
  re-ran from the valid `t=3` restart with a 6-hour wall limit and
  `nlim=2000`. It reached `t=17.09` (about 4.35 d) before FGMRES failed:
  200 iterations left relative residual 0.00195 against a 1e-4 target,
  with no reported breakdown. The preceding three solves took 63, 119,
  and 200 iterations, showing an abrupt loss of convergence. Its last HST
  at `t=17.03` had maximum Mach 0.0739, radiative luminosity 0.957 Lheat,
  and relative box-mass change -2.5e-8. The 128³ PPM run had maximum Mach
  about 0.245 at the same time; the 256³ disturbance is much smaller but
  still growing. The `t=16` restart is complete and available. Job
  **340374** was canceled because its dependency could not be satisfied.
  A `t=20` to `t=50` extension was not submitted during the SSH outage.
- Job **340590** restarts the valid `t=16` state with FGMRES restart 40,
  maximum 500 iterations, and two MG preconditioner V-cycles per Krylov
  iteration (formerly 20/200/1). It runs to `t=17.25`, saves HDF states
  around the failure, and checks whether a stronger linear solve passes
  the previously failing time. This changes solver settings only; the
  thermal preparation source and physical initial data are unchanged.
  Job **340593** analyzes the last readable HDF state whether the probe
  succeeds or fails. If the probe succeeds with maximum Mach below 0.15,
  dependent job **340595** continues to `t=20` with the same FGMRES
  settings; job **340596** then measures shell evolution and the convective
  enthalpy-flux proxy. In fact **340590** failed during the first restarted
  step: even 500 iterations left relative residual 7.75e-4 against 1e-4.
  The two preceding linear solves took 32 and 101 iterations. There was no
  new HDF, so afterany analysis **340593** could not run; **340595** and
  **340596** were canceled. The original uninterrupted run had crossed
  `t=16`, so this immediate failure might involve restart-state consistency,
  or the changed preconditioner could itself be harmful. Job **340604**
  reran a few steps from the *same* `t=16` checkpoint with the original
  FGMRES settings 20/200/1 and completed normally at `t=16.15` in 2.5 min,
  maximum Mach 0.0701. Thus the checkpoint is usable and the changed
  FGMRES settings, especially the extra MG preconditioner cycle, are a
  likely cause of the immediate restart-probe failure. Job **340609**
  isolates the larger Krylov restart width by running from `t=16` to
  `t=17.25` with settings 40/500/**1** and HDF saved around the original
  failure point. Dependent job **340612** extends to `t=20` only if 340609
  completes normally, maximum Mach remains below 0.15, and the last 30
  linear solves all use fewer than 200 iterations. Dependent analysis
  **340613** then checks shells and the convective-flux proxy. A separate
  diagnostic job **340619**, dependent on completion of 340609, uses the
  same `t=16` checkpoint with restart 20, maximum 500 iterations and one MG
  V-cycle through `t=17.25`. Comparing 340619 with 340609 distinguishes
  the effect of a larger Krylov restart width from merely raising the
  iteration cap. **340609 completed normally** at `t=17.25` in 20 min;
  maximum Mach was 0.07483, all 96 linear solves met the 1e-4 true-residual
  target, the largest solve took 37 iterations, and the last 30 averaged
  25.1 iterations (maximum 36). The opacity-domain check remained clean.
  The Krylov basis memory reported on rank 0 rose from 85,983,232 to
  169,869,312 bytes when the restart width doubled from 20 to 40. Job
  **340612** started its gated continuation to `t=20`. No automatic longer
  run is scheduled before that gate.
  **340612 completed** at `t=20` (5.09 d) in 37 min 20 s, and analysis
  **340613 completed**. All 168 FGMRES solves met the 1e-4 target (maximum
  33 iterations), and every Newton solve reported `converged` (largest final
  L2 1.08e-11 and max norm 2.45e-9, below the 1e-8 nonlinear target).
  Stderr was empty, the opacity-domain check was clean, maximum Mach was
  0.08668, box radiative luminosity was 0.95687 Lheat, and the mass change
  since the `t=17.25` restart was -5.62e-9. The corresponding 128³ PPM
  prepared run had Mach 0.260 at `t=20`. The t=0-to-20 shell audit found
  Tgas ratio 0.9900 in 813–817 R_sun and 0.9991 in 850–880 R_sun; the
  latter shell's RMS speed was 0.060 km/s. The core (0–166 R_sun) had
  mean outward radial speed 0.079 km/s, a signal to keep monitoring.
  The 800–830 R_sun convective gas enthalpy-flux proxy was only 0.00114
  Lheat at `t=20`; the artificial shell heating cannot yet be tapered on
  evidence of convective heat transport.
  Width-20/maximum-500/MG1 control **340619** also completed at `t=17.25`
  (maximum 124 FGMRES iterations, all Newton solves converged, Mach
  0.07483). This shows that the original 20/200 failure is not reproduced
  from the `t=16` checkpoint with the new output schedule; it does not
  isolate width from the iteration cap on the original uninterrupted
  trajectory. The separate inexact-solver job 340625 belonged to WOR-123;
  it was canceled before starting when the user moved solver-strategy work
  to a separate chat, so the RSG physics continuation could take priority.
  For the RSG physics trajectory, gated **340671** now extends the accepted
  256³ run from `t=20` to `t=28` (7.13 d) on M-bulk-a, 64 ranks, 3-hour
  limit; one dependent M-debug analysis job **340672** audits the last HDF
  state and the t=20-to-28 convective-flux proxy. All large outputs stay
  under `$WORK/wor122`.
  **340671 and 340672 completed**. The run reached `t=28` (7.13 d) in
  1 h 20 min 48 s. All 393 FGMRES solves met the target (largest 39
  iterations), all 131 Newton solves converged (largest final L2
  1.95e-11 and max norm 6.41e-9), stderr was empty, and the opacity-domain
  check remained clean. Maximum Mach rose from 0.08668 at `t=20` to
  0.12084 at `t=28`; relative box-mass change over that interval was
  -1.84e-8, and Lrad/Lheat was 0.95665 at the endpoint. Shell Tgas ratios
  at `t=28` were 0.99189 (813–817 R_sun) and 1.00029 (850–880 R_sun).
  The 850–880 R_sun RMS speed rose from 0.0603 to 0.0844 km/s. Core
  0–166 R_sun mean outward radial speed rose from 0.0791 to 0.1327 km/s.
  The 800–830 R_sun convective gas enthalpy-flux proxy rose only from
  0.00114 to 0.00156 Lstar, still far too small to replace artificial
  preparation heating. This is a smoothly growing disturbance, not a
  stationary configuration or a solver crash.
  A further gated 256³ continuation **340968** runs `t=28` to `t=36`
  (9.17 d) with the same physics and solver settings; dependent analysis
  **340969** follows. One M-debug analysis job **340971** compares core
  axis and diagonal radial flow at `t=20` and `t=28` to test whether the
  growing center signal is directionally localized.
  **340971 completed**: in 0–166 R_sun, mass-weighted radial velocity
  along Cartesian-axis cones (within 15 degrees) versus diagonal cones
  was 0.0778 versus 0.0802 km/s at `t=20`, and 0.1307 versus 0.1342
  km/s at `t=28`. The growing core expansion is nearly isotropic by this
  measure. In 830–900 R_sun, axis-cone velocity rose from +0.1203 to
  +0.1531 km/s, while diagonal-cone velocity changed from -0.0264 to
  -0.0369 km/s. Thus the outer angular flow is strongly structured even
  while the core's mean radial flow has little axis preference. This
  diagnostic does not establish where the outer pattern originates.
  Compact values and the comparison figure are
  `wor122_prep_thermal_20261115/axis_flow_t20_t28.json/png`.
- A targeted convection-seed test raises `velocity_perturb_fraction` from
  1e-4 to 1e-2 while retaining the same 128³ PLM supported setup. Job
  **340236** completed normally at `t=8`; job **340237** measured virtually
  unchanged maximum Mach (0.04377 versus 0.04377) and 800–830 R_sun
  convective covariance (0.002189 versus 0.002187 Lheat). The original
  perturbation envelope vanishes at the stellar surface, so increasing its
  global amplitude does not excite the problematic photospheric region.
- A trial `surface_perturb_fraction=0.01` seeded 0.90–1.00 stellar radii,
  with full amplitude at 0.96–0.985 radius. A local 64³ one-step run
  completed normally at initial Mach 0.0173. The separate XD binary at
  `$WORK/wor122/athena_surface_seed` avoided changing the binary used for
  the 256³ extension. Job **340269** completed normally at `t=8`, maximum
  Mach 0.04376 versus 0.04377 in the baseline PLM run. Job **340270**
  found 800–830 R_sun convective covariance 0.002209 versus 0.002187
  Lheat; maximum speed in that shell increased, with no thermal-transport
  benefit. The trial implementation was removed from the final branch.

Remote raw data and logs are under `$WORK/wor122/prep256_t1/` and
`$WORK/wor122/prep128_t8/`, `$WORK/wor122/prep128_bin16_t8/`, and
`$WORK/wor122/prep128_inner_t8/`, and `$WORK/wor122/prep128_outer_t8/`.
Compact HST and JSON results are also saved in
`tst/nrfld_validation/wor122_prep_thermal_20261115/`. The outward-extended
support was continued from its restart to `t=20` (about 5.1 d) in XD
M-debug job **340171**. The same state was extended to `t=36` (about
9.17 d) in job **340189**. A matched PPM control is job **340180** and a PLM
extended-support check is job **340193**.
`wor122_plot_prep_thermal.py` regenerates `history_comparison.png` and
`shell_comparison.png` from the compact data. The first plot omits unusually
short output-alignment steps from instantaneous power/luminosity curves.

## Next decision

### Inexact Newton / FGMRES strategy (2026-11-16)

The cross-problem convergence and time-to-solution strategy is tracked in
Linear WOR-123; this report holds the RSG-specific evidence and job results.

The 256³ RSG input requests a 1e-4 **relative Schur-system** FGMRES
residual, but accepts a time step only after the **full gas and radiation**
Newton residual has L2 and maximum norms both below 1e-8. The current
FGMRES path throws a fatal error whenever it reaches its iteration cap above
the inner target, before Newton can try that finite correction. At the
340373 failure, the true relative linear residual was 0.001954 after 200
iterations, with no breakdown. This is below a plausible 1e-2 inexact
Newton forcing cap, but the Schur-system norm is **not identical** to the
full nonlinear norm after gas elimination, boundary handoff, and smoothing.
It therefore does not by itself prove the full Newton step would succeed.

The nonlinear driver already reevaluates the coupled residual after each
trial, accepts a trial only if the L2 merit decreases, and commits only when
both L2 and maximum residual pass the strict Newton target. Thus a finite,
moderately inaccurate linear direction can be tested safely as an
**opt-in** inexact Newton experiment; the nonlinear acceptance gate must
remain unchanged. Never treat the inner solver's failure to meet 1e-4 as a
successful *time step* on its own. NaN, Inf, algebraic breakdown, a
non-decreasing nonlinear merit, or failed Newton stopping must reject the
trial; `failure_policy=warn_keep_old` is inappropriate for this experiment
because it skips the implicit FLD update.

First test the existing parameter path before changing solver code. The
prepared XD script for job **340625** would restart the same valid `t=16`
state to `t=17.25` with FGMRES restart 20, cap 200, MG1 and **relative
target 1e-2**. The job was canceled while pending; WOR-123 can requeue it
from `$WORK/wor122/prep256_inexact1e2_t17p25.sh` when solver-strategy work
resumes in its separate chat. Compare normal completion,
all `[NR]` final L2/max residuals, Newton and linear iteration counts,
runtime, max Mach, mass and source-aware energy budgets, luminosity, and opacity-domain
checks against strict job 340609. If it converges and agrees, prototype a
proper inexact Newton policy: a bounded forcing term (for example an
Eisenstat–Walker schedule), report true linear residual and forcing target,
retry a rejected Newton direction with a tighter linear target or wider
Krylov basis, and retain the existing strict nonlinear stopping tests.
Check the actual coupled residual after handoff rather than treating the
Schur residual as an exact forcing ratio. Keep the current strict behavior
as the default until local and 256³ regression tests pass. This follows
the inexact-Newton condition in Eisenstat & Walker (1996,
https://doi.org/10.1137/0917003) and PETSc's documented adaptive
KSP forcing-term approach (https://petsc.org/release/manual/snes/).

The extended support improves Tgas, radiative luminosity and outer-flow speed
relative to the matched PPM control, but that run's Mach and outer speed keep
growing through `t=36`. The PLM extension through `t=24` still accelerates
and has negligible convective heat transport. The 256³ continuation now
reaches `t=36` with all implicit solves converged. Track the energy source,
shell-mean Tgas and Er,
shell velocities, maximum Mach, radiative luminosity and convective-flux
proxy. The artificial source should be tapered only when the convective
flux can carry the missing luminosity; a time-based taper alone would be an
unverified assumption.

### 256³ continuation to t=36 (2026-11-16)

XD **340968** (M-bulk-a, 64 ranks, 1 h 18 min 21 s) and its M-debug
analysis **340969** both completed with exit code 0. The run reached code
time 36, or 9.17 days. All 343 FGMRES solves met their 1e-4 target
(maximum 38 iterations, mean 26.6); all 114 Newton solves converged
(maximum 4 iterations, largest final max norm 9.01e-9). `stderr.log` is
empty, and the opacity-domain, clamp, negative, zero, and nonfinite counts
are all zero. The final restart is 3,171,305,829 bytes and complete.

| Quantity | t=28 (7.13 d) | t=36 (9.17 d) |
| --- | ---: | ---: |
| Maximum Mach | 0.12084 | 0.15479 |
| Mean outward core velocity, r<166 R☉ [km/s] | 0.1327 | 0.1787 |
| RMS speed, 850–880 R☉ [km/s] | 0.0844 | 0.1083 |
| Tgas/initial, 813–817 R☉ | 0.99189 | 0.99191 |
| Tgas/initial, 850–880 R☉ | 1.00029 | 0.99934 |
| Lrad/Lheat at six box faces | 0.95665 | 0.95661 |
| 800–830 R☉ convective enthalpy covariance/Lstar | 0.00156 | 0.00197 |

Box gas mass changed by -1.48e-8 relative over t=28→36. These data show a
continued but smooth speed increase, stable shell-mean temperature, and
good numerical convergence. They do not yet show the convective transport
needed to remove preparation heating. The t=36 HST endpoint uses a short
alignment step, so its instantaneous `Pprep_thermal/Lheat=0.723` is not
representative. Ordinary-step medians are 0.986 over t=24–28 and 0.979
over t=32–36; the *integrated* preparation energy over t=28→36 is
0.946 Lheat times the elapsed code time. Hence there is no evidence that
convection has replaced the artificial source.

One gated extension, XD **341037**, has been submitted to M-bulk-a to run
t=36→52 (13.24 days), with a four-hour limit and the same physics and
FGMRES settings. Its startup gate requires t≥35.99, maximum Mach<0.20,
and maximum FGMRES iterations over the last 30 solves<100. The single
dependent M-debug analysis is **341038** (`afterany`). HDF cadence is four
code time units and restart cadence eight, to limit storage. Files and
scripts are under `$WORK/wor122/prep256_fgmres_r40_t{36,52}/` and
`$WORK/wor122/prep256_fgmres_r40_t52{,_analysis}.sh`. The next decision is based on t=52 shell speed, core flow,
Mach, source power, convective proxy, luminosity, mass, and solver status.

M-debug directional analysis **341039** also completed in 32 s. Its JSON
is `$WORK/wor122/axis_flow_t28_t36.json`. In the core r<166 R☉, mean
radial velocity at t=28 was 0.1307 km/s in 15-degree Cartesian-axis cones
and 0.1342 km/s toward box diagonals; at t=36 these rose to 0.1765 and
0.1802 km/s. The core expansion remains nearly isotropic by this measure.
In the outer 830–900 R☉ shell, the axis-cone mean rose from +0.1531 to
+0.1742 km/s while the diagonal-cone mean fell from -0.0369 to -0.0481
km/s. The outer angular contrast continues to grow, but these shell
averages do not yet identify where the disturbance originates.
