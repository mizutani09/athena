# Mayer §5 profile-offset isolation (2026-11-15)

Worktree `/home/kosuke/simulation/athena-wor115-mayer-offset-20261115`, branch
`codex/wor115-mayer-offset-isolation-20261115`, based on verified WOR-92
commit `ae03e85b5a131b6c5cf7ddb3fe7a6bcb1ef9f91a`. No changes are merged
into `main` or the completed WOR-92 branch. XD experiments use the same
positive-AMR ideal-EOS binary (SHA-256
`b64491aa31da79216c99dbfcc6eb2b9cba805e4c385a4a67d092c303a622b5cb`)
and an isolated root
`/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115`.

## Why the previous two collapse histories cannot identify one cause

The old 14 K ambient path (jobs 338717/338718/338744) used ambient density
and pressure both 1/100 of the core and a pre-implicit background sink. It
reached the first-core density near 29.14 kyr but had pressure-floor cells
near c1095–1097. The validated 420 K ambient path (jobs 339330 through
340334) uses ambient density 1/30 of the core, equal initial gas pressure,
14 K fixed background radiation, and a post-implicit sink with boundary
resynchronization. It reached the same density at 36.871 kyr and had no
accepted pressure-floor cells after the positive-AMR repair. The paths
change multiple controls and cannot be treated as a single-parameter A/B.

At equal code time 1703, the old and current maximum densities are about
`9.12e-17` and `1.04e-16 g/cm³`; at 1900 they are about `3.46e-15` and
`3.11e-16 g/cm³`. The large difference emerges late, after initially
similar growth. This motivates pressure and background-radiation controls.

## Independent local checks

The ideal-gas setup itself uses μ=2.381 and γ=5/3 in both the initial
condition and temperature recovery, with `Egas=P/(γ−1)` and
`P=ρ R T/μ`. Its uniform-sphere density is
`5.251585415881407e-18 g/cm³` and α=Ethermal/|Egrav|=0.41330, matching
the values quoted in Mayer §5 to their stated precision. This excludes an
order-unity initial EOS or unit-conversion factor as the profile-offset
explanation. It does not assert the later thermal trajectories are equal.

The c0 HDF5 from baseline job 339330 has SHA-256
`02389cb4c2cba071bbc9cb041016ce828040a63c38d084b42c6c041f2118bf0c`.
Using same-block centered gradients of its potential, the median radial
acceleration divided by the analytic uniform-sphere acceleration is
`1.00001`, `1.00002`, `0.99999`, and `1.00010` over 300–600, 600–1000,
1000–1500, and 1500–2500 au. The 5–95% envelope across these bands stays
within about 0.05% of unity. This rules out an order-unity gravity unit or
initial-core gravity-strength error; it does not validate late-time gravity
or the treatment of ambient self-gravity.

For the validated c1560 HDF5, a same-block centered estimate of the reduced
FLD gradient gives an absolute LP/Minerbo lambda difference relative to
Minerbo below 0.462% in all sampled interior cells within 7 au. The median
difference there is `6.54e-6`; in 30–300 au the median is 1.44% and maximum
2.44%. These are instantaneous coefficients on the *same* Athena state,
not a time-evolved Minerbo run. They make a direct large limiter effect in
the optically thick first core less likely, without proving it absent.

At c1560 the densest cell has mass `1.03453e-6 M☉`, 3.11 times Mayer's
AREPO target `3.33e-7 M☉`. Within 7 au the median Athena cell mass is
`1.40836e-7 M☉`; among cells with rho≥5e-11 g/cm³, the median is
`6.62323e-7 M☉` and 48.8% exceed twice the Mayer target. Thus refinement
is not uniformly coarser or finer; the densest gas is relatively coarse
under Mayer's mass criterion. This remains a possible profile difference.

At the same final snapshot, no cell within 1000 au has background tracer
fraction≥0.5, and the maximum tracer fraction inside 100 au is only
`3.49e-18`. Thus the prescribed 420 K background reset is not applied
directly inside the measured first core. Changing the background setting can
still alter the core through pressure waves and the infalling envelope; the
snapshot does not isolate those dynamics.

The profile extraction method is not a large multiplicative factor:
the validated c1560 mass-weighted spherical-shell temperatures at 1 and
10 au are 522.07 and 102.60 K; the +x-axis values are 522.12 and
102.18 K. The corresponding differences are <0.5%, far below the Mayer
Figure 7 read-off gaps. The read-off points themselves are approximate
because the paper does not publish a numerical profile table.

The validated final gas and radiation temperatures agree locally in the
first core: at 1, 3, and 5 au, the spherical-shell Tgas/Trad pairs are
522.07/522.06, 384.15/383.70, and 236.08/234.79 K. Thus the cooler
first-core profile is not simply gas falling out of radiative equilibrium
at the endpoint. It can still reflect a different thermal history or
radiative flux before the endpoint.

## XD controls

* Job **340378**, M-test-a one node/64 ranks, changes only
  `problem/background_pressure_ratio` from 1 to 1/30 while keeping ambient
  density 1/30, background radiation 14 K, post-implicit sink, all solver
  flags, and the validated binary. It completed c0→c10 with `PASS` in 1:16,
  960 blocks, mass unchanged within 15-digit HST precision, no pressure or
  Jeans gate violation. Input SHA-256
  `5bd3e640c4a731ac8d87cfd825845db995c460a23be4f1de68ccae78da99d5fd`.
  This 14 K ambient pressure is a sensitivity control, not a claim that it
  matches Mayer's background.
* Job **340381**, M-large-a one node/120 ranks, continues 340378 toward
  code time 2000 or rho_max≥1e-10 with 20-cycle segments, 900 s timeout per
  segment and mass/pressure/Jeans gates. At rho_max=1e-13 g/cm³, the first
  crossing is 28.88075 kyr, compared with 28.87786 kyr on the old 14 K
  diagnostic path and 36.45582 kyr on the validated 420 K path. The two 14 K
  curves agree within 2.9 yr despite different ambient density ratios and
  sink timing. This isolates `background_pressure_ratio` as the dominant
  input control of the *timing* gap between those Athena trajectories. In an
  ideal gas this input also sets the initial ambient temperature and the
  subsequent gas-sink target (420 K versus 14 K); this experiment does not
  separate the initial pressure force from continued thermal forcing. It
  does not identify the numeric background pressure used by Mayer. The job
  reached rho_max=1.0329334e-10 g/cm³ at c1130 and stopped by the planned
  density gate. The first accepted threshold-crossing row is c1113 at
  29.14723 kyr (log-density interpolation: 29.14716 kyr), 57 yr later than
  Mayer's 29.09 kyr and 7723 yr earlier than the validated 420 K trajectory.
  Job 340442 independently restarted c1110→1113 and produced the exact
  first-crossing HDF5; its SHA-256 is
  `dc31eb5a330f1569960725a14da62196ab96ec90569c35227d791d0e0dc7b1b6`.
  At 1/5/10 au, the temperature is 1.159/1.287/1.409 times the validated
  420 K trajectory, but only 0.705/0.603/0.543 times the approximate Mayer
  Figure 7 read-off. Both Athena trajectories put the steepest density drop
  near 6.46 au, consistent with Mayer's reported 6–7 au shock. Job 340381
  had 1202/1202 active Newton solves converged, maximum 20 FGMRES iterations,
  maximum true relative residual 1.994e-8, MaxRSS 54,532,976 KiB, and
  elapsed time 9066 s. Mass drift is below 8e-15; no accepted pressure or
  Jeans gate failed. One accepted HST row in the pressure trial, at code time
  710.9123 (well before dense-core formation), has background gas
  temperature relative error 6.93% immediately after a regrid; all other
  accepted rows through c1130 stay below 1%. This is a known limitation of
  the unpatched baseline operator, so the timing comparison is a strong
  sensitivity result rather than an exact corrected-background A/B.
* Job **340384**, M-test-a one node/64 ranks, changes only
  `problem/background_fix_radiation=true` to `false`, retaining 420 K gas,
  initial Er=a(14 K)^4, and the post-implicit gas sink. It passed c0→c10
  in 1:05; ambient radiation heated from 14 K to a maximum 45.47 K.
  Input SHA-256
  `a7680c20727b91f266ad333af292148de5f27749a84d6e5f766a9de5b257a22b`.
* Job **340386** continued this no-radiation-fix state toward c100 but
  stopped near c26: the maximum radiation temperature reached 57.38 K,
  AMR level increased from 3 at c22 to 7 at c26, and the 64 GiB memory
  request was exceeded (`OUT_OF_MEMORY`, Slurm step MaxRSS 67,078,904 KiB,
  exit 255). It is a failed model-sensitivity control, not a valid physical
  collapse trajectory. No further long no-radiation-fix job is planned
  without a separate stable physical setup and resource estimate.
* A one-parameter background-radiation equilibrium check changes only
  `background_radiation_temperature=14` to `420 K` with the 420 K gas
  background unchanged. Input SHA-256
  `999706ca60677373db091b6a27727cf265151ed76c155233a8d5bdab3df6f866`.
  Initial job **340404** was stopped after c1 because a newly refined
  tracer-marked cell had Tgas relative error 0.435; the uncorrected baseline
  binary confounds the radiation sensitivity with the AMR reset omission.
  Job **340405** reran c0→c2 with the post-AMR reset enabled. At its c1
  accepted state, background gas temperature error was 5.55e-16, but the
  hottest gas and radiation were already 441 K at r≈3008 au near the core
  boundary (versus 420 K at c0), and
  the Jeans minimum rose to 310 cells. The c2 step exceeded the 64 GiB
  request (`OUT_OF_MEMORY`, step MaxRSS 67,079,420 KiB); no valid c2 state
  was accepted. We do not launch a long continuation of this costly and
  as-yet-unverified alternate radiation bath. This is a sensitivity check,
  not an assertion that Mayer used 420 K for the background radiation field.

Mayer §5 states that background gas cells retain their initial temperature,
but does not numerically specify the background radiation temperature or
state a radiation-energy reset. The predecessor §3.1 states ambient density
1/30 and initial gas-pressure equilibrium. Consequently, the 420 K gas
condition is supported by a cross-paper inference, whereas the 14 K
background-radiation reset is a documented Athena assumption. Neither
no-radiation-fix OOM nor a 14 K ambient pressure trial establishes the
actual Mayer choice.

## AMR background reset and resolution control

The validated 420 K path applies the prescribed background gas/radiation
reset after the implicit solve and before AMR. Prolongation can create new
tracer-marked background cells **after** that reset. A controlled c1420→1421
restart with N_J=48 (from 32) and numlevel=13 (from 12) produced a newly
refined cell at r≈2555 au, tracer≈0.653, Tgas≈724 K instead of 420 K;
the maximum accepted background-temperature relative error was 0.93949.
The otherwise identical c1420→1421 restart with the experimental
post-AMR reset enabled yielded a maximum relative error of 5.55e-16 and
left the core temperature near 496 K. Both one-step trials passed gas-mass,
minimum-pressure, and Jeans gates. The invalid uncorrected long job 340392
was canceled after 56 s, before it accumulated a misleading trajectory.

The first source trial reapplied the reset to **all** MeshBlocks after any
AMR update, using an opt-in environment flag. Its isolated binary SHA-256 is
`603345073d820c254d15c7b3c66948137975cda00cdf0ef1815f31eaeec197f2`.
It removed the new-cell temperature error but did more work than required.
The revised opt-in source trial marks blocks created by a refinement-level
change and applies the existing reset only to them, before the established
post-AMR boundary synchronization. Its binary SHA-256 is
`9d786d7c376a574e5b0e3f9ff9cc7379f906bdb2a61124fedbb97bbc04ca7c47`.
The flag remains `ATHENA_WOR115_POST_AMR_BACKGROUND_SINK=1`; default
behavior remains unchanged. The one-step high-resolution job 340424 passed
all gates, with accepted background-temperature error at round-off. At the
same r≈2555 au/tracer≈0.653 cell, Tgas changed from 723.9 to 420.0 K.
The paired HDF audit (`postamr_c1421_pair_audit.json`) finds 57,645 cells
whose pressure changed by >1e-4 relative, all tracer-marked background cells
at r=2324–5830 au and none inside 1000 au. Density is unchanged. The
one-step internal-energy change is 0.0411% of the total. This establishes
local correction, not a large direct change in the first-core interior.

The broad-reset high-resolution continuation job 340398 completed c1421→1621
and reached rho_max=1.0081244e-10 g/cm³. All 220 active Newton solves
converged, FGMRES used at most 17 iterations per solve with maximum true
relative residual 1.9983e-8, and gas mass, pressure, Jeans, and background
temperature gates passed; MaxRSS was 53,342,120 KiB. The first threshold
crossing was 36.870328 kyr, only 0.397 yr before the validated lower-resolution
run. The densest-cell mass fell from 1.03453e-6 to 1.30252e-7 M☉ (0.39×
Mayer's target), yet the 1–100 au temperature profile changed by at most
0.27%. Late resolution alone therefore does not explain the large Figure 7
read-off difference. This does not test resolution from the beginning.

Several broad-reset runs stalled on cna045/cna046/cna150 and were stopped
early or at a bounded 900 s segment timeout. In particular, job 340413
stopped at c1505 on cna150. The same c1500 restart on the **same node** with
the validated old binary (job 340418) and the experimental binary with its
flag OFF (job 340419) completed c1506 in 1:24/1:23. That alone suggested a
source-dependent stall, but a later controlled A/B changed the inference:
the targeted binary also stalled at c1530 on cna150 (job 340427), while the
**same targeted binary, c1526 restart, and inputs** completed c1531 on
cna190 (job 340432) in 1:21. Thus node or MPI/runtime interaction is
supported; neither a hardware defect nor an intrinsic numerical divergence
of the reset is established. To avoid wasted allocation, job 340427 was
canceled after the successful alternative-node reproduction. Job 340433
continued on cna044 to rho_max=1.0022080e-10 g/cm³ at c1561, with all
33 active Newton solves converged, max 17 FGMRES iterations, true relative
residual ≤1.9414e-8, and the same physical gates passed. This latter path
includes the earlier broad-reset history through c1506; a separate pure
targeted c1420→first-core job **340441** completed c1560 and stopped at the
planned density gate. Its 140 distinct accepted HST rows (154 data lines
including restart-boundary duplicates) match the validated reference
exactly in all 39 recorded columns; the maximum accepted background-
temperature relative error was 7.77e-16, compared with 1.467 on the
historical baseline. All 156 active Newton solves converged, FGMRES used at
most 17 iterations and had maximum true relative residual 1.986e-8; mass
drift was 4.44e-16, minimum pressure 5.127e-10 dyn/cm², minimum Jeans
resolution 31.99 cells, MaxRSS 45,839,272 KiB, and elapsed time 1719 s.
Zero-step snapshot job **340443** produced the c1560 HDF5 (SHA-256
`56f1ed163ad1096649cefab62146f335ade8409fcf22c098649d4074c2333944`).
Direct HDF5 comparison with the validated c1560 snapshot found **all
1,294,336 cells identical** for rho, pressure, three velocities, radiation
energy, and tracer. Mesh geometry and AMR levels also match exactly.
Only potential/defect diagnostics differ at small floating-point levels
(potential maximum relative difference 1.18e-7 in 20 cells). The radial
density and temperature curves match exactly. This proves that the late
targeted correction does not perturb the ordinary first-core trajectory.

The original validated 420 K history had three accepted rows with background
temperature error >1% after AMR, at code times 32.252, 581.421, and
2261.232 (relative errors 0.664, 0.929, and 1.467). The targeted c0→c10
job 340434 removed the first anomaly: maximum accepted background
temperature error 5.55e-16, 12/12 active Newton solves converged, true
FGMRES relative residual ≤2.31e-8, gas-mass/pressure/Jeans gates passed.
At c10 its maximum density differs from the original by only about 1.2e-5
relative. This short trial does not measure the accumulated effect of the
later anomalies on the first-core profile.

## Interpretation and remaining uncertainty

The controlled 14 K ambient-pressure/gas-temperature trial closes nearly all
of the **formation-time** gap (29.14716 kyr against Mayer's reported
29.09 kyr), but its first-crossing temperatures at 1, 5, and 10 au remain
only 0.705, 0.603, and 0.543 times Figure 7 approximate read-offs. Those
ratios vary with radius, so one global EOS or unit factor cannot repair the
profile. Mayer's Section 5 gives neither the numeric ambient pressure nor
the radiation-field boundary prescription. Section 3.1 of the predecessor
supports initial gas pressure equilibrium as an inference, which favors the
validated 420 K ambient state over the 14 K control. A paper-faithful
boundary-condition match therefore remains unproven.

The late-resolution trial rules out a large **late** spatial-resolution
effect, but does not test a high-resolution trajectory from c0. The late
post-AMR reset is a real local consistency correction and has no measured
effect on the first-core interior from c1420 onward. The historical c0
trajectory contains three earlier background-temperature anomalies, while
the targeted c0→c10 trial corrects the first; their accumulated impact on
the full collapse remains unmeasured. A full corrected c0 run would consume
roughly a day of XD time under the current configuration. It is the next
appropriate numerical experiment once the paper's radiation/ambient
boundary is specified; launching parallel long runs against undocumented
conditions would not efficiently isolate the remaining physical difference.

The standalone [HTML report](wor115_mayer_offset_report.html) includes the
timing curves, first-core radial profiles, Mayer Figure 7 approximate
read-offs, solver/resource table, and numerical summary. The HDF5 and HST
artifacts are under
`/home/kosuke/simulation/test/rad_nr/wor115_mayer_offset_20261115/`;
the validated reference history and profiles remain in the existing
`wor92_xd_20261107/firstcore_integration_evidence/ideal_mayer5_postsync_c0_20261110/`
folder. All experimental XD jobs for this study are complete or explicitly
stopped; no further long run is queued by this branch.
