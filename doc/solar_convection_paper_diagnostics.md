# Solar-convection diagnostics against published validation practice

The nonmagnetic, 96 km, 6.144 × 6.144 × 3.072 Mm entropy-feedback run is a
long-duration **integration test of FLD with 3D convection**. It is not yet a
solar-calibrated atmosphere or an independent accuracy test of FLD near the
photosphere. Keep the running campaign untouched while analyzing completed
segments.

## Frozen local analysis, 2026-11-08

The report at
`/mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/paper_validation_t20700_22700/index.html`
uses 21 terminal full 3D volumes, one from each completed 100-code-time
segment over `t=20700–22700`. It shows the flux time series, horizontal and
temporal mean stratification, velocity RMS, separate vertical flux components,
Rosseland `tau=2/3` temperature and velocity maps/distributions, and their
horizontal Fourier spectra. `summary.json` lists the source segments and
numbers. The live `t=40000` driver was not stopped or restarted.

| Quantity | Measured value |
| --- | ---: |
| Mean of rolling-500 top / bottom flux | 0.6291 / 0.6355 `F_sun` |
| Effective temperature implied by top flux | 5141 K; solar reference 5772 K |
| Gas temperature on `tau=2/3` | 5182 K; spatial RMS 61 K |
| Vertical speed RMS on `tau=2/3` | 1.23 km/s |
| Upflow area on `tau=2/3` | 49.5% |
| Temperature–vertical-speed correlation there | +0.279 |
| Mid-box convective enthalpy / kinetic flux | +0.680 / −0.083 `F_sun` |
| Interior mass-neutral transport proxy, `z=-1.0–0.85 Mm` | median 0.621 `F_sun` |
| Velocity-pattern Fourier peak | 1.54 Mm horizontal wavelength |
| Newton solves / slow-MG warnings in the chosen segments | 30017 converged / 0 |
| Maximum mass drift in those segments | 0.0286% |
| Change in controlled inflow entropy in this window | +2.30×10⁵ erg g⁻¹ K⁻¹ |

The earlier long-window report `interim_t5000_21200/overview.html` found a
top flux of 0.62896 then 0.62920 `F_sun` in two successive, approximately
4050-code-time windows. Vertical RMS velocity changed by about 0.17%; the
boundary energy ledger residual was 5.7×10⁻⁵ `F_sun` equivalent and all
reported Newton solves converged. This supports dynamically stable convection
at reduced flux. The inflow entropy controller continues to change its target,
so 0.63 is an observed plateau under the current control law, not a proven
fixed-parameter asymptote.

The middle of the box transports energy mainly by rising enthalpy with a
downward kinetic contribution, the expected sign pattern in surface-convection
simulations. Dividing by this run's own top flux gives roughly +1.08 and −0.13
for the mid-box terms. The approximately 0.621 `F_sun` mass-neutral interior
transport is consistent with the 0.629 upper boundary within this diagnostic's
limits. It is **not** an exact interior energy ledger: sparse full-volume output
aliases coherent mean-flow oscillations, cell-centered FLD reconstruction is
approximate, and the proxy excludes mean-flow enthalpy and gravitational flux.
For exact boundary fluxes and global conservation use the HST-based report.

## What the papers test and what remains

* [Rempel (2014)](https://arxiv.org/abs/1405.6814) adjusts the inflow entropy
  to obtain a photospheric flux close to solar and studies domain and resolution
  effects. The current top flux fails that solar-flux target. A thermal-timescale
  response measurement and higher-resolution continuation are still required.
* [Beeck et al. (2012)](https://arxiv.org/abs/1201.1103) compare mean
  stratifications and fluctuations on geometrical and optical-depth surfaces,
  vertical velocity and emerging intensity patterns, and center-to-limb
  variation. We can measure the gas and flow portions from our 3D dumps.
  The dashed 1D curve in our report is **the imported initial profile**, not
  independent observational agreement.
* [Stein & Nordlund (2000)](https://web.pa.msu.edu/people/steinr/papers/realistic_convection.00.pdf)
  use flux partition and direct observable comparisons. Our expected-sign
  enthalpy/kinetic partition supports the presence of physical convection,
  while the absolute flux and surface observables remain unverified.
* [Stein & Nordlund (2002)](https://arxiv.org/abs/astro-ph/0209510) explain why
  radiative transfer is needed at optical depth near unity. Grey FLD does not
  provide directional, wavelength-dependent emerging intensity, spectral
  lines, or limb darkening. The report's `sigma*T_gas^4` is explicitly a
  temperature proxy, not a synthesized intensity or the computed FLD flux.

The next controlled comparisons are: (1) let the current 96 km run reach its
planned endpoint and redo long-window stationarity, (2) compare 48 km then
24 km runs at matched physical conditions and well-relaxed windows (WOR-91),
(3) compare optical-depth profiles to an independent high-resolution solar
reference, and (4) use angle- and frequency-dependent radiation transfer for
surface observables. These discriminate boundary calibration, numerical
resolution, and the grey FLD approximation. A single stable 96 km run cannot
separate them.

## Regenerate the report

From the repository root, with the campaign's completed segment chain and
data symlinks available:

```bash
python3 tst/nrfld_validation/solar_convection_paper_diagnostics.py \
  /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000 \
  --snapshots 21 \
  --data-dir /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/data
```

The script reads only entries already marked complete in `state.json`; the
output directory name fixes the exact `t_start_t_end` window. On another
machine supply `--input` and `--data-dir` for its local copies. Repeated
rolling-500 flux entries overlap, so their average describes the plateau and
is not an independent-sample uncertainty estimate.
