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

The N1024 stage diagnostic resolves the output phase: on the last full
dt=2.217 µs, the post-explicit local gas peak is 992.467 K and the
post-implicit peak is 955.012 K, a 37.455 K fall. The final step is
shortened to 0.136 µs to hit t=.04 exactly and gives 957.396 to 955.091 K.
The previous full step therefore describes the normal step-size behavior.
The local implicit gas+radiation energy-budget residual is <7e-13 relative
to the largest implicit diffusion source. Transverse radiation-energy spread
at the sampled shock cells is <2e-10 relative.

The N64 bitwise control used a fresh current-source build and the previous WOR-10 binary with exactly the same input. Every dataset in their final HDF5 files agrees bitwise at 0.04 s. The production high-resolution uniform runs use the previous binary; this control checks the current source's default physics behavior after the later opt-in diagnostic changes.

## Running calculations and reproducibility

The combined HTML is `/mnt/d/output/rad/wor67_spike_audit_20261104/index.html`; it is regenerated as high-resolution cases finish. Simulation inputs, run logs, HDF5, history and resource files are under `/mnt/d/output/rad/wor67_*_20261104`. The baseline N64/128/256 cases are in the WOR-10 output directories; the N512 baseline is `/mnt/d/output/rad/wor10_n512_20261104`. Analysis is `tst/nrfld_validation/wor67_spike_audit.py`.

Uniform N1024 and N2048 use 4x4 transverse cells and cubic MeshBlocks of 4³ cells, with physical transverse extent scaled to retain cubic cells. Stage CSVs in their `output/` directories sample the final explicit RK2 and implicit FLD updates; the preceding full step is also retained. The N512 CFL control uses the same binary and spatial grid as the baseline. All saved `input.used` files are fully materialized without hidden command-line physics overrides. The `provenance.txt` files record binary SHA-256 and CPU affinity. The high-resolution source comparison build is `/mnt/d/output/rad/wor67_current_build_20261104`.

Three MPI jobs initially overlapped on cores 0–3 because separate `mpirun` launches used the same default bindings. Their ranks were moved to disjoint physical cores (0–7 for N2048, 8–11 for N1024, 12–15 for the N512 control), reducing CPU contention. The AMR runner uses explicit no-binding launch plus a restricted core set. The calculation input is unaffected by that runtime scheduling correction.

The shock domain has fixed inflow/outflow boundaries. Its integrated mass and energy need not remain constant, so the report records the history's physical-flux spread as a steady-flow diagnostic and, where stage CSVs exist, checks the local implicit gas+radiation energy budget. It does not mislabel open-domain total-energy change as a conservation violation.
