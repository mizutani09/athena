# WOR-75: solar-box initial thermal balance

This is a short-time startup audit of the non-magnetic 6.144 × 6.144 ×
3.072 Mm solar box. It does not establish the long-time calibrated luminosity.
The detailed local report is
`/mnt/d/output/rad/wor75_initial_balance_20261106/comparison/index.html`.

## What is imposed at t=0

The imported 1D profile assumes convection carries nearly all the target
`F_sun = 6.3e10 erg cm^-2 s^-1` at `z ≈ 0`: its model convective flux is
`0.99990 F_sun`. In the actual 3D initialization, every vertical velocity is
zero. Thus the initial advective/enthalpy flux is zero. Reconstruction from
the saved 3D `t=0` radiation and opacity fields gives the interior FLD flux
`1.055e-4 F_sun` at `z=0`, `0.778 F_sun` at `z≈1.92`, and `0.723 F_sun`
at the uppermost interior face `z≈2.88`. The exact accepted HST boundary
fluxes are `5.625e-6 F_sun` at the bottom and `0.996703 F_sun` at the
top. Consequently the initial radiative loss integrated over the box is
about `0.996698 F_sun`, with almost no initial bottom supply. The uppermost
interior face and the Marshak boundary face differ by about `0.274 F_sun`;
the saved volume does not contain the lagged boundary face coefficient, so
the boundary value comes from HST rather than volume reconstruction.

The initial horizontal-mean pressure gradient has a maximum centered
hydrostatic residual of `0.0562 ρg`, near the top. This is a diagnostic
finite difference, not the exact finite-volume momentum residual. Internal
gas plus radiation energy above the top `0.5 Mm` divided by `F_sun` is
`0.02697 h`; above `1 Mm` it is `0.4416 h`. These are reservoir sizes,
not measured thermal relaxation times. The deep model's missing convective
flux and upper boundary mismatch explain why the first photospheric
temperature and emitted flux can move quickly even though the whole box has
a much longer heat time.

## Temporary dark-top counterfactual

Three 96 km, 64×64×32, 16-rank calculations used the same frozen binary,
input, EOS, opacity, initial profile, fixed `s_in=1.581097454457282e9`,
HD2 mass controller (`relax_time=2`, `rate_gain=40`), and CFL `0.3`:

| Case | Top Marshak `alpha` | Interval |
| --- | ---: | ---: |
| Normal | 0.5 | `t=0–40` |
| Dark | `1e-6` | `t=0–10` |
| Restored | 0.5 | restart from dark `t=10` to `40` |

The normal and dark t=0 thermodynamic and interior FLD fields match to
`1e-6`; only the top boundary changes. The dark case initially emits
`1.994e-6 F_sun`. Its mean 3D temperature on the `τ=2/3` surface rises
from `5566 K` to `7303 K` at `t=10`, versus `4661 K` in the normal
case. Restoring the normal boundary produces a short flux spike of about
`1.4 F_sun`. Its mean upper flux over `t=30–40` is `0.379 F_sun`, below
the normal case's `0.444 F_sun`. A temporarily insulated top does not cure
the missing initial convective transport or improve this startup window.
The large short-time bottom flux oscillations and finite run duration mean
that these numbers cannot decide a final equilibrium flux.

Setting the upper photospheric radiation flux exactly to zero would be an
insulating boundary, not a solar vacuum/outgoing-radiation boundary. The
code requires a positive `alpha`, so `1e-6` was used as a controlled
near-zero experiment. It is a useful diagnostic but not a physically
consistent solar initial boundary condition.

All three jobs exited normally. Normal/dark/restored Newton solves were
`478/120/359` converged, with zero fatal or slow-convergence warnings;
maximum FGMRES iterations were `43/19/37`.

The prior matched FLD-profile correction raised the uppermost interior
flux from `0.723` to `0.892 F_sun` at 96 km, yet the t=5–10 top mean
increased only from `0.441` to `0.460 F_sun`; at 48 km the same correction
raised that interior flux from `0.795` to `0.969 F_sun`, while the
t=5–10 top mean rose only from `0.382` to `0.393 F_sun`. This reinforces
that aligning the radiative 1D profile alone does not initialize the 3D
convective heat transport. These older comparisons used separate startup
runs and are supporting evidence, not the dark-top matched comparison.

## Reproduce the audit

From the repository root:

```bash
python3 tst/nrfld_validation/solar_convection_initial_balance_report.py \
  --root /mnt/d/output/rad/wor75_initial_balance_20261106 \
  --opacity-table /mnt/d/output/rad/wor75_initial_balance_20261106/data/opacity_table_rhoT_3p0_6p3.h5 \
  --profile /mnt/d/output/rad/wor75_initial_balance_20261106/data/solar_initial_profile.dat
```

The script reconstructs interior vertical FLD face flux from the t=0
volume with the NR face-opacity cap, three-dimensional face gradient, and
Levermore-Pomraning limiter. It takes the physical boundary fluxes from the
history diagnostics. The report includes `summary.json`, the initial column
plot, and the startup comparison. Each run has a manifest with its command,
hashes, output, and frozen binary. The binary SHA-256 begins `d0d0f78a`.

## Physical follow-up

An exactly stationary 1D, zero-velocity start cannot carry the specified
deep convective fraction of `F_sun` through the 3D equations. A genuinely
better initial condition needs a thermally relaxed 3D convective flow or a
controlled transition from a separate energy transport mechanism. The
next useful check is a matched restart from a relaxed 3D state with the
same `s_in` and Marshak boundary, following total energy storage and
surface flux for several full-box thermal times. Do not tune the top
Marshak coefficient to hide the initial cooling.

## Developed 3D convection seed pilot (2026-11-06)

This experiment tested whether importing a developed convective flow can
remove the zero-velocity startup deficit. The source is the saved `t=9400`
full-3D volume from the fixed-entropy WOR-74 branch, whose preceding
`t=9300–9400` top and bottom means were `0.633` and `0.673 F_sun`.
The target horizontal means are the original `t=0` 1D profile. The seed
builder maps the source density, pressure, and radiation-energy fluctuations
onto those means, removes each layer's mass-weighted mean velocity, and
fades velocity near the upper boundary. A second variant also fades velocity
near the lower boundary. UOV radiation energy is converted from cgs to code
units using `egas_unit` read from the target Athena input. The optional
`initial_seed_file` path reads this exact-grid primitive
state, converts gas pressure through the tabulated EOS, and includes kinetic
energy in the hydro conservative state. With `initial_seed_file=none`, the
same binary reproduces the quiet startup.

The matched 96 km, 64×64×32, 16-rank, CFL 0.3, HD2 relax-time 2/gain 40,
fixed `s_in=1.581097454457282e9` t=0–40 results are:

| Initial state | Convective enthalpy at z≈0, t=0 | Top, t=30–40 | Bottom, t=30–40 | Stored-energy trend, t=30–40 |
| --- | ---: | ---: | ---: | ---: |
| Quiet 1D | `0` | `0.444 F_sun` | `0.118 F_sun` | `−0.337 F_sun` |
| Imported 3D | `0.620 F_sun` | `0.538 F_sun` | `6.549 F_sun` | `+5.787 F_sun` |
| Imported 3D, lower velocity taper | `0.620 F_sun` | `0.539 F_sun` | `6.670 F_sun` | `+5.892 F_sun` |

Top cooling is smaller, but the transplanted states inject far too much
energy at the HD2 lower boundary. The lower taper does not resolve that
mismatch. This is a **failed initial-balance test**, not a solar-luminosity
calibration. All three t=0–40 runs completed with every Newton solve
converged and no slow warning or fatal error. The failure is physical
energy balance rather than solver stability.

The lower-tapered branch was continued to `t=100`. Its `t=80–100` mean
upper and lower fluxes were `0.610` and `2.802 F_sun`; at `t=100` the
lower flux was still `1.923 F_sun`. The imbalance decays but remains
large. A 1.5× velocity candidate was considered but **not run** because
the unscaled boundary injection already failed the balance criterion.
The same branch was continued to `t=200`; over `t=180–200` its upper
and lower mean fluxes were `0.6338` and `0.6857 F_sun`. All 1,435
Newton solves in `t=100–200` converged without slow warnings or fatal
errors. The t=180–200 stored-energy trend was `+0.160 F_sun` and the
endpoint rate `+0.075 F_sun`, so this short window does not prove full
stationarity. It does show the flow returning toward the subsolar
luminosity of its source state after the artificial boundary injection
fades. The volume at `t=200` is not a new solar-flux initial condition.

As a control, the complete relaxed `t=9400` checkpoint was restarted
without transplanting its fields onto the 1D mean. It ran through
`t=9440` with 562/562 Newton solves converged, no warnings, and mean
upper flux `0.632 F_sun` over `t=9430–9440`. This avoids the sharp
startup drop and retains a developed 3D flow, but remains at the
subsolar luminosity of its parent state. Its lower flux over only ten
code-time units is phase-sensitive and not a stationary calibration.

The local report is
`/mnt/d/output/rad/wor75_seed_20261106/comparison/index.html`.
The seed binary, JSON provenance, frozen executable, per-run manifests,
HTML plots, and HST records are in that output root. The generic seed
builder and report are
`tst/nrfld_validation/build_solar_3d_seed.py` and
`tst/nrfld_validation/solar_convection_seed_compare.py`.

The physical next step is to retain the **entire** relaxed 3D state and
calibrate the lower inflow entropy over an adequately long thermal
relaxation. Grafting its velocity and fluctuations onto a hotter 1D mean
does not create a balanced solar-flux state. This belongs with WOR-74's
long-window entropy response and the whole-box heat budget. A shorter
startup cannot substitute for measuring the equilibrium luminosity.
