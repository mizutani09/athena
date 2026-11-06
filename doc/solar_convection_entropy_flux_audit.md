# Why the t=0–5000 entropy feedback did not raise the solar-box flux

This audit concerns the local 96 km, non-magnetic, gray-FLD campaign at
`/mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/`.
No further long calibration run should use the current controller gain until
the long-time response and photospheric closure have been measured.

## Exact boundary budget

The combined HST analysis is `combined/index.html`. Over its latest window,
`t=3750.10–5000`, the mean face fluxes in units of `F_sun=6.3e10 erg cm^-2 s^-1`
are:

| Face term | Mean / `F_sun` |
| --- | ---: |
| Top radiation, including advective radiation | 0.627965 |
| Top hydro | 0.000000 |
| Bottom hydro | 0.629947 |
| Bottom radiation, including advective radiation | 0.00001235 |
| Stored gas+radiation+gravity endpoint rate | +0.002115 |

The top is **not** at `F_sun` after adding hydro: the upper hydro face flux is
exactly zero throughout the late window. The lower `rad_flux_cgs=3.54354335e5`
sets only about `5.62e-6 F_sun` of diffusive radiation at that face; the
remaining radiative advection contributes another `6.73e-6 F_sun`. It does
not impose a solar total flux. The lower hydro flux arises from the
thermodynamic contrast and mass transport of upflows and downflows. The
sampled energy ledger closes to about `5.6e-5 F_sun` in the latest window,
well below the flux deficit. This is a sampled diagnostic, not an exact
Runge–Kutta stage conservation check.

The boundary implementation explains why neither face is forced to emit
`F_sun`: the HD2 lower ghost cells prescribe entropy for *upflows* and copy
interior entropy for downflows; the hydro energy flux then emerges from the
flow. The separate lower FLD condition fixes only the tiny
`rad_flux_cgs`. At the top, the Marshak radiation flux depends on the
computed face radiation energy and its exterior value. The HST
`FhydTop`, `FradTop`, `FadvRadTop`, and `FhydBottom` columns come from the
respective face-flux diagnostics in
`src/pgen/simple_convection_lhllc_fld.cpp`.

The initialized 1D profile gave `F_rad,top/F_sun=0.99670` at `t=0`. It fell
rapidly as the 3D flow and photosphere adjusted. The early profile was a
useful hydrostatic/radiative starting state, but was not a thermally relaxed
3D convective solution. The late optical-depth-2/3 temperature was around
5150–5175 K, consistent in scale with the effective temperature of a
0.628-solar-flux atmosphere (~5138 K for a 5772 K solar reference).

## The entropy command did reach the fluid

The input entropy rose from `1.581097454457282e9` to
`1.5813836703922749e9 erg g^-1 K^-1` in the *computed* final segment,
an increase of `2.86216e5` (0.0181%). The final `state.json` value after
segment 50 is a proposed value for a future segment and was not used at
`t<=5000`.

`tst/nrfld_validation/solar_convection_entropy_transport.py` evaluates the
same tabulated EOS entropy at the first interior cell in all 100 paired
full-volume dumps. It weights upflow/downflow entropy by the magnitude of
vertical mass flux. The reproducible plot and data are
`entropy_transport/index.html` and `entropy_transport/summary.json` under
the campaign root. The two 500-code-time windows show:

| Quantity | t=1000–1500 | t=4500–5000 | Change |
| --- | ---: | ---: | ---: |
| Prescribed `s_in` | 1.581167941e9 | 1.581370544e9 | +202,604 |
| Bottom upflow entropy | 1.581117618e9 | 1.581326748e9 | +209,131 |
| Bottom downflow entropy | 1.579063356e9 | 1.579281817e9 | +218,461 |
| Upflow−downflow entropy contrast | 2,054,262 | 2,044,931 | −9,331 |
| Bottom-cell mean temperature | 21817.7 K | 21822.0 K | +4.3 K |

Thus the imposed change is present in the lower gas, but upflows and
downflows become hotter/higher entropy together. Their contrast, which
largely drives the convective enthalpy transport, remains within ~0.5%.
The net lower hydro flux stays around 0.63 `F_sun` and the top around
0.63 `F_sun`. At `t=5000`, the Python EOS interpolation gives the bottom
area-mean entropy minus `s_in` as −495,842, agreeing with the C++ HST
`BottomSminusSin` to rounding. The measured lower-layer response is real,
not a failure to apply the input or an EOS-table interpolation artifact.

At approximately fixed bottom pressure, the EOS maps the full `+2.86216e5`
entropy change to only `+5.64 K`. The observed late lower-layer temperature
change is of this order. The initial controller gain was derived from a
short-time bottom-*flux* response, so it did not measure how the
upflow/downflow entropy contrast or top radiation respond after recirculation.

## Matched restart perturbation: causal short-time response

`/mnt/d/output/rad/wor74_entropy_audit_20261106/t5000_matched_20code/`
contains three 16-rank HST-only runs from the exact same `t=5000` checkpoint
to `t=5020`. All binary/input/table hashes, grid, CFL, and other boundary
controls match; only `s_in` is changed by `0`, `+1e6`, or `−1e6`. Each run
had 285/285 converged Newton solves, no slow-MG warning, and no fatal.
Open its `index.html` for the comparison plot.

| Change from fixed `s_in` over t=5000–5020 | +1e6 branch | −1e6 branch |
| --- | ---: | ---: |
| Bottom hydro flux / `F_sun` | +0.348102 | −0.329200 |
| Top radiation / `F_sun` | −0.000023 | +0.000022 |
| Stored-energy rate / `F_sun` | +0.339535 | −0.321335 |

This proves that the boundary reacts immediately and that added enthalpy
initially enters storage rather than escaping the top. The symmetric
instantaneous bottom gain is ~`0.339 F_sun` per `1e6` entropy units, close
to the `0.322` short pilot used by the controller. It is **not** a measured
long-term gain for the photospheric flux. Over the full t=0–5000 run,
downflow entropy rose almost as much as upflow entropy, so extrapolating
that short gain to a quasi-steady solar-flux calibration was unjustified.

The HST gas internal energy gives a box Kelvin–Helmholtz scale of about
`20,590` code units (`~128 h` physical); this run reached only about 0.24
of that scale. The feedback formula explicitly relaxes over `20,000`
code units and therefore commanded only `+0.286e6` entropy units by t=5000.
Its slow pace and the unmeasured long-time gain jointly explain why a
solar top flux had not emerged. The current data do not establish the
steady-state `dF_top/ds_in` or the entropy needed for `F_sun`.

### One-crossing matched extension

The fixed and `+1e6` cases were independently restarted from their own
`t=5020` checkpoints with exactly the same entropy setting and numerical
controls, then run through `t=5120`. The interval `t=5000–5120` covers
approximately one vertical crossing. The joined plot and summary are in
`/mnt/d/output/rad/wor74_entropy_audit_20261106/t5000_matched_120code/index.html`
and `window_response.json` beside it. Trapezoidal time means and endpoint
energy rates, always `+1e6` minus fixed, are:

| Code-time window | Δ bottom hydro / `F_sun` | Δ top total / `F_sun` | Δ stored-energy rate / `F_sun` |
| --- | ---: | ---: | ---: |
| 5000–5020 | +0.348106 | −0.0000231 | +0.339535 |
| 5020–5050 | +0.401885 | −0.0000153 | +0.398573 |
| 5050–5100 | +0.341286 | +0.0003432 | +0.339330 |
| 5100–5120 | +0.220229 | +0.0000548 | +0.223526 |
| 5000–5120 | +0.337360 | +0.0001449 | +0.334845 |

The top response begins to appear around `t≈5050`, but remains three
orders of magnitude below the extra bottom input in the full-window mean.
It oscillates, so a short positive interval cannot be interpreted as a
steady response. The bottom response weakens in the last window, consistent
with recirculation or hydrodynamic adjustment, but the HST does not yet
separate upward and downward hydro energy fluxes. Both extension runs reached
`t=5120`: 1444/1444 Newton solves converged in each, with no fatal or
slow-multigrid warning. The upper hydro flux remained exactly zero in both.
This extension strengthens the storage-and-lag explanation while leaving
the eventual `F_top(s_in)` unknown.

The cell-centered entropy audit can be reproduced with:

```bash
python3 tst/nrfld_validation/solar_convection_entropy_transport.py \
  --feedback-state /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/state.json \
  --eos-table /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/data/AthenaNataEosTable_solar_deep.tab \
  --early-window 1000 1500 --late-window 4500 5000
```

## Comparison with published MURaM setups

Rempel (2014, [arXiv:1405.6814](https://arxiv.org/pdf/1405.6814), §2.2)
describes HD2 as symmetric mass fluxes, a fixed mean boundary pressure
with damped pressure fluctuations (`C_dmp=0.95`), and prescribed entropy
in upflows with symmetric entropy in downflows. He states that the
upflow entropy was selected to yield a photospheric flux within a few
percent of solar. Our HD2 follows those broad conditions, but the
profile-derived `s_in` was never independently calibrated for our FLD
physics. Rempel's 6.144×6.144×3.072 Mm models used 32, 16, or 8 km cells,
whereas this local pilot uses 96 km. His published runs started from a
thermally relaxed non-magnetic convection state; ours starts from a 1D
profile. Rempel uses MURaM's short-characteristics radiation solver and
an OPAL-based EOS; our model uses gray FLD with a Marshak top and a NATA
EOS table. These differences can alter the equilibrium flux–entropy
relation; this audit does not isolate which of them dominates the
0.63-solar-flux outcome.

Vögler et al. (2005, [A&A 429, 335](https://www.aanda.org/articles/aa/pdf/2005/01/aa1507.pdf),
§2.3.2) describes a slow luminosity feedback on *specific internal energy*
of upflows, not an additive update of specific entropy. Our controller
instead converts the top-flux error into entropy units using a measured
short-time **bottom** response. It is an exploratory adaptation, not the
published formula or a calibrated long-time transfer function.

## Gate before another long run

1. Measure a *matched*, fixed-entropy long-enough response from the same
   relaxed checkpoint, with both bottom and top fluxes and
   mass-weighted up/down entropy. Add separate upflow and downflow
   hydro-energy face-flux diagnostics if exact cancellation is needed;
   the current HST records their net only. The 120-code-time matched test
   gives a first recirculation check, but is far shorter than the full-box
   thermal time and does not establish a steady response.
2. Separate startup/profile and radiation-closure effects from the
   entropy-control effect. The t=0 1D profile produces solar initial top
   flux, but its 3D relaxed flux is ~0.63 solar. A static/discrete radiative
   balance audit and a resolution/closure comparison should precede
   extrapolation to an XD long run.
3. Only after a measured long-time `F_top(s_in)` response, choose a gain,
   bounds, and an acceptance window for a revised controller. Keep the
   old feedback trajectory as a diagnostic, not a calibrated reference.
