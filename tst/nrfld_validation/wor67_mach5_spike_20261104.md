# WOR-67: Mach 5 gas-temperature spike verification (2026-11-04)

## Setup and comparison rule

This study uses the Zhang et al. (2011) Mach 5 semi-analytic reference already stored as `reference/radiative_shock/mach5_semianalytic.csv`. The benchmark domain is [-4000, 2000] cm, the closure is fixed lambda=1/3, and the comparison time is 0.04 s. The numerical profile is translated only by the midpoint of its largest density jump, as in the existing radiative-shock validation. The raw shock position is reported separately. No gas-peak-specific shift is made.

The analytic point peak is 1077.101 K; its half-height width above the downstream temperature is about 3.188 cm. For the existing physical gate, peak error is measured against this **point value**, and spike-amplitude recovery is `(numerical_peak - analytic_downstream)/(analytic_point_peak - analytic_downstream)`. The required limits are peak error <=10% and amplitude recovery >=50%, together with <=2% aligned L1 for density, gas and radiation temperature, <=5% raw shock-position error relative to domain width, <=1% aligned change since the preceding output, and strict Newton/FGMRES convergence.

A second, diagnostic reference averages the analytic conserved gas density, momentum and total gas energy over each numerical cell, then converts the cell averages to temperature with the same ideal-gas EOS. It is a finite-volume-aware comparison, not a strict convergence ceiling. The peak of that projection is also varied over a +/-half-shock-cell uncertainty in density-jump registration. The physical gate never optimizes that shift.

## Established local results

| Case | dx [cm] | Numerical peak [K] | Spike recovery | FV-projected peak at stated registration [K] | FV peak range under +/-dx/2 registration [K] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform N64, CFL .3 | 93.750 | 861.248 | 2.50% | 864.269 | 863.22–937.54 |
| Uniform N128, CFL .3 | 46.875 | 866.658 | 4.94% | 873.120 | 868.99–947.38 |
| Uniform N256, CFL .3 | 23.438 | 878.803 | 10.43% | 891.538 | 882.84–962.78 |
| One-level AMR, finest dx=N256 | 23.438 | 878.920 | 10.48% | 891.538 | 882.84–962.78 |
| Uniform N512, CFL .3 | 11.719 | 905.072 | 22.29% | 928.007 | 909.78–983.46 |
| Uniform N512, CFL .15 | 11.719 | 903.768 | 21.70% | 928.007 | 909.78–983.46 |
| Uniform N1024, CFL .3 | 5.859 | 955.085 | 44.88% | 982.142 | 945.74–1008.53 |
| Uniform N1024, CFL .15 | 5.859 | 951.570 | 43.30% | 982.142 | 945.74–1008.53 |
| One-level AMR, finest dx=N1024 | 5.859 | 946.999 | 41.23% | 982.142 | 945.74–1008.53 |
| Uniform N2048, CFL .3 | 2.930 | 991.437 | 61.30% | 1033.477 | 988.55–1036.79 |

Halving CFL at N512 lowers the final peak by only 1.30 K, whereas changing N256 to N512 raises it by 26.27 K. The N512 peak also falls by about 1.94 K between the saved t=.038 and .04 s outputs, so the global aligned L1 stationarity check alone does not establish local peak stationarity. N256/N512 numerical peaks sit half a cell upstream of the density-jump midpoint, while the FV-projected reference peak sits half a cell downstream. That location difference shrinks in physical units as dx decreases.

The median late-step dt is 4.4325 µs for N512/CFL .3 and 2.2162 µs for N512/CFL .15. The radiation absorption time from the fixed Planck opacity is about 0.8495 µs. Thus this CFL pair also probes dt/t_abs from 5.22 to 2.61. N1024/CFL .3 has almost the same absolute dt as N512/CFL .15, permitting a spatial comparison at nearly matched dt.

N1024/CFL .3 completed in 18,360 cycles and 80.1 wall minutes. Its gas peak
is 955.085 K, a point-peak error of 11.33%, and recovery is 44.88%: both peak
gates still miss by a small margin. Its density, gas-temperature and
radiation-temperature aligned L1 values are 0.116%, 0.073% and 0.052%; raw
shock-position error is 3.06% of the domain width. Every Newton solve
converged strictly. The peak over t=.03–.04 s ranges from 946.276 to
961.333 K; the previous saved t=.038 peak is 5.256 K higher than the final
peak. The median late dt is about 2.217 µs, matching the N512/CFL .15
control. At this matched dt, increasing resolution raises the final peak by
51.32 K, again much more than the N512 CFL effect.
The measured late-dt ratio for N1024/CFL .3 divided by N512/CFL .15 is
0.999986, so this first spatial comparison is matched in dt to better than
0.002%. The audit HTML and JSON provide these pairwise values directly.
The second time-step control completed N1024/CFL .15 in 36,720 strictly
converged Newton steps. Its median late dt is 1.10807 µs, exactly the same
recorded value as N2048/CFL .3, while its spatial grid equals N1024/CFL .3.
Halving dt at N1024 lowers the final peak by 3.515 K and recovery by 1.588
percentage points; doubling N1024/CFL .15 resolution at matched dt raises
the peak by 39.867 K and recovery by 18.008 points. This isolates spatial
resolution as the dominant source of the local peak deficit over these
tested grids and time steps. The N1024/CFL .15 final peak is 951.570 K,
11.65% below the analytic point peak, and its recovery is 43.30%; both peak
gates still fail. Its late t=.03–.04 s peak range is 951.002–964.139 K,
so even its highest late value misses both the 966.411 K recovery and
969.391 K point-error thresholds. Aligned L1 is 0.125% (density), 0.067%
(gas T), and 0.046% (radiation T); the raw shock-position error is 3.06% of
the domain width and the maximum aligned change is 0.039%. All other gates
pass. Its preceding full dt=1.10843 µs step changes the hottest cell's
post-explicit gas temperature 970.036→951.520 K in the implicit update;
the 18.516 K drop is about half of the N1024/CFL .3 full-step drop. The
local gas+radiation energy change agrees with the sampled implicit
diffusion source to 4.5e-13 relative. The shortened final step has
dt=0.64345 µs, with 962.332→951.576 K across the implicit update. The
N1024/CFL .15 job took 140.4 wall minutes, with sampled summed rank
high-water RSS about 313.3 MiB.
Even the highest N1024/CFL .3 history peak in that late interval (961.333 K) is
below the 50%-recovery threshold (966.411 K) and the 10%-error threshold
(969.391 K), so its failure is not due only to choosing the final output
phase. The discrete density-jump center remains at 183.594 cm for all saved
t=.03–.04 s profiles; subcell movement and physical transients cannot be
separated from these sparse snapshots alone.

The N1024 stage diagnostic resolves the output phase: on the last full
dt=2.217 µs, the post-explicit local gas peak is 992.467 K and the
post-implicit peak is 955.012 K, a 37.455 K fall. The final step is
shortened to 0.136 µs to hit t=.04 exactly and gives 957.396 to 955.091 K.
The previous full step therefore describes the normal step-size behavior.
The local implicit gas+radiation energy-budget residual is <7e-13 relative
to the largest implicit diffusion source. Transverse radiation-energy spread
at the sampled shock cells is <2e-10 relative.
At the cell with the largest pre-implicit gas temperature, gas energy density
falls by 0.005863 and radiation energy density by 0.000253 in code units on
that full step. Their sum, -0.006116, agrees with `dt*S_diff_implicit` at that
cell. This identifies implicit radiation diffusion out of the hot cell,
coupled to the gas, as the immediate cause of the post-implicit peak drop;
it is not an unexplained loss of gas+radiation energy.

Uniform N2048/CFL .3 completed at t=.04 in 36,722 cycles and 216.8 wall
minutes. It **passes the established gate**: peak 991.437 K, point-peak error
7.95%, spike recovery 61.30%, aligned L1 0.069% (density), 0.092% (gas T)
and 0.076% (radiation T), raw shock error 3.06% of domain width, and maximum
aligned change since the previous output 0.028%. All 36,722 Newton steps
strictly converged with no relaxed acceptance. The late t=.03–.04 peak
range is 971.042–1008.810 K; its minimum remains above both peak thresholds,
though the 10%-error threshold is only 969.391 K. The analytic spike
half-height width is about 3.188 cm, only 1.09 of these 2.930 cm cells,
so a passing gate does not yet show asymptotic peak convergence.

The N2048 last full step has dt=1.108 µs, versus 2.217 µs at N1024. Its
post-explicit peak falls 1015.925→991.525 K in the implicit stage, a 24.40 K
same-cell drop versus 37.45 K at N1024. At that hottest cell the combined
gas+radiation energy-density change is -0.00439852, equal to
`dt*S_diff_implicit` to about 2e-15 absolute. The final shortened step is
0.158 µs and changes 994.909→991.443 K. The numerical and FV-projected
peaks are both at +1.465 cm from the density jump, but their values differ
by 42.04 K at that registration; the FV projection varies by 988.55–1036.79
K under the explicitly reported half-cell registration sensitivity. No
peak-optimizing translation is used for the physical gate.

One-level AMR with base N512 (finest dx=5.859 cm) also completed at t=.04
and retained both levels (130 final MeshBlocks). Its peak is 946.999 K,
12.08% below the point reference with 41.23% recovery, so it remains
`accuracy_not_met` at this local resolution. Its aligned density/gas/radiation
L1 values are 0.141/0.188/0.146%, raw shock error 3.16% of the domain,
and all 20,143 reported Newton solves strictly converged. The 18,261 global
cycles and 20,143 solves differ because refined levels subcycle. Its late
peak range is 946.408–961.434 K; even the highest value is below the 50%
recovery threshold. The AMR peak at the final output lies half a finest cell
upstream of the density-jump center, while the FV-projected peak lies half a
cell downstream. The analogous uniform N1024 final peak is 8.087 K higher,
but the sign of the AMR–uniform difference changed across earlier saved
times, so a systematic AMR bias is not established by one final sample.
The AMR density-jump midpoint changes from 183.594 cm at t=.038 to
189.453 cm at t=.04, exactly one finest cell, while the uniform N1024
midpoint is 183.594 cm at the final time. This changes which neighboring
cell contains the reported local peak. It may contribute to the 8.087 K
final AMR–uniform difference; it does not explain away the fact that even
AMR's highest late-history peak remains below the physical threshold.

The N64 bitwise control used a fresh current-source build and the previous WOR-10 binary with exactly the same input. Every dataset in their final HDF5 files agrees bitwise at 0.04 s. The production high-resolution uniform runs use the previous binary; this control checks the current source's default physics behavior after the later opt-in diagnostic changes.

## Assessment and remaining work

The existing physical gate is met locally by uniform N2048/CFL .3 and is
missed by N1024 at both CFL values and by one-level AMR with finest N1024
spacing. Two paired comparisons at effectively identical dt show that
halving dx raises the peak by 51.318 K (N512 to N1024) and 39.867 K
(N1024 to N2048), while halving dt at fixed dx lowers it by 1.304 K and
3.515 K respectively. The measured low-resolution spike deficit is thus
dominated by spatial under-resolution in these experiments. The stage
measurements locate the immediate post-explicit peak loss in the implicit
radiation diffusion update and close its local energy budget; they do not
indicate an unexplained energy sink or a solver-convergence failure.

N2048 still has only 1.09 cells across the analytic spike half-height
width. Passing one threshold is not proof of asymptotic convergence or of
uniform, AMR and SMR equivalence. The next complete physical checks are
uniform N4096, one-level AMR with finest N2048 spacing, and an SMR case on
a larger machine. The local sizing pilots quantify their approximate
resources below. If those completed cases disagree after matching dx,
dt, reference registration and output phase, investigate the refinement
interfaces and numerical flux/limiter before changing the acceptance gate.

## Calculations and reproducibility

The combined HTML is `/mnt/d/output/rad/wor67_spike_audit_20261104/index.html`; regenerate its ten completed local cases with `bash tst/nrfld_validation/wor67_rebuild_report.sh`. Simulation inputs, run logs, HDF5, history and resource files are under `/mnt/d/output/rad/wor67_*_20261104`. The baseline N64/128/256 cases are in the WOR-10 output directories; the N512 baseline is `/mnt/d/output/rad/wor10_n512_20261104`. Analysis is `tst/nrfld_validation/wor67_spike_audit.py`.

Uniform N1024 and N2048 use 4x4 transverse cells and cubic MeshBlocks of 4³ cells, with physical transverse extent scaled to retain cubic cells. Stage CSVs in their `output/` directories sample the final explicit RK2 and implicit FLD updates; the preceding full step is also retained. The N512 CFL control uses the same binary and spatial grid as the baseline. All saved `input.used` files are fully materialized without hidden command-line physics overrides. The `provenance.txt` files record binary SHA-256 and CPU affinity. The high-resolution source comparison build is `/mnt/d/output/rad/wor67_current_build_20261104`.

Three MPI jobs initially overlapped on cores 0–3 because separate `mpirun` launches used the same default bindings. Their ranks were moved to disjoint physical cores (0–7 for N2048, 8–11 for N1024, 12–15 for the N512 control), reducing CPU contention. The AMR runner uses explicit no-binding launch plus a restricted core set. The calculation input is unaffected by that runtime scheduling correction.

The full SMR base-N256 and base-N128 attempts, with finest spacing equal to
uniform N1024 and N512 respectively, advanced only to t=0.000183 and
t=0.002039 s before the local throughput made a full t=.04 run impractical.
Their inputs, partial outputs, logs, memory snapshots and stop reasons remain
under `/mnt/d/output/rad/wor67_smr_base{256,128}_20261104`. The N128 SMR
two-rank high-water RSS sum was about 0.78 GiB, versus about 1.47 GiB for
N256. Both had strict solver progress, but neither is a completed physical
validation case. Their freed cores ran N1024/CFL .15, an exact same-grid
temporal control for N1024/CFL .3 and a dt-matched spatial partner to N2048.
The SMR HPC preset remains available in the standard radiative-shock runner.

A uniform N4096/CFL .3 sizing pilot used six MPI ranks on the remaining six
physical cores. Its 100-cycle run took 84.17 wall seconds and advanced only
to t=5.13e-5 s; a separate 10-cycle run measured 199–200 MiB high-water RSS
per rank, about 1.17 GiB summed across six ranks. The late pilot dt was
4.923e-7 s. Holding that early dt and wall rate fixed would require roughly
81,000 more cycles and 19 hours; the steady runtime could differ, so this is
only an order-of-magnitude sizing estimate. Memory is feasible on this WSL
machine, but a full N4096 run is not a practical local follow-up. The pilot
inputs, logs and per-rank resource files are under
`/mnt/d/output/rad/wor67_uniform_n4096_{pilot100,rankmem10}_20261104`.
The standard `radiative_shock_validation.py --preset hpc` runner remains the
path for full uniform N4096, one-level AMR base N2048 and three-level SMR
base N1024 comparisons on a larger machine.

A one-level AMR base-N1024 pilot (finest dx=2.930 cm, the local N2048 uniform
spacing) also completed 100 cycles with six physical-core MPI ranks. It took
85.08 wall seconds and reached t=1.01e-4 s; a six-rank `/proc` snapshot
recorded about 0.62 GiB summed high-water RSS. Its last pilot dt was
9.858e-7 s, giving a rough 40,000 further-cycle and order-10-hour local
projection at the initial measured rate. This is a sizing run only, marked
`exploratory_cycle_limit` by the standard runner, not a physical comparison.
The data are at `/mnt/d/output/rad/wor67_amr_base1024_pilot100_20261104`.

The shock domain has fixed inflow/outflow boundaries. Its integrated mass and energy need not remain constant, so the report records the history's physical-flux spread as a steady-flow diagnostic and, where stage CSVs exist, checks the local implicit gas+radiation energy budget. It does not mislabel open-domain total-energy change as a conservation violation.
