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

The extended support improves Tgas, radiative luminosity and outer-flow speed
relative to the matched PPM control, but that run's Mach and outer speed keep
growing through `t=36`. The PLM extension through `t=24` still accelerates
and has negligible convective heat transport. The 256³ result remains
subsonic through `t=17`, but FGMRES must converge past that point before
extending it. Track the energy source, shell-mean Tgas and Er,
shell velocities, maximum Mach, radiative luminosity and convective-flux
proxy. The artificial source should be tapered only when the convective
flux can carry the missing luminosity; a time-based taper alone would be an
unverified assumption.
