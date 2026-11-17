# WOR-14: Is omitting radiation force justified for this RSG?

## Distinguish the two models

[Chiavassa et al. 2011, Sec. 4.2](https://arxiv.org/html/1109.3619)
explicitly say their CO5BOLD 3D simulations assume `P_rad=0`; radiation
still transports thermal energy. Their Eq. (2) includes gas pressure and
gravity but no radiative momentum force. The paper does not establish that
this approximation is accurate for our opacity, initial structure, or
full-envelope application. The author's motivation should not be inferred
from the omission alone.

By contrast, [Goldberg, Jiang & Bildsten 2022](https://arxiv.org/html/2110.03261v2)
solve RSG envelopes with gas and radiation pressure in Athena++ and report
that radiation provides about one third of the support in an efficient
convective region. Their model is a spherical wedge with an inner boundary,
not a full Cartesian star, so its startup cannot be copied directly. They
used a hot inner boundary or a fixed luminosity to drive convection and
allowed a multi-year transient before assessing a steady state.

## Ratios measured on our initial profile

The following use the imported
`data/ic/st35gm03n07_wor55_lte_fld_trial.dat`, the actual softened
12-Msun potential, `P_rad=E_r/3` as a pressure scale, and
`g_rad=kappa_R F_rad/c` with the *initial* radiative flux. The last ratio
is the acceleration if the **whole** input luminosity were radiative at
the same density, temperature, and opacity; it is a counterfactual.

| Radius [Rsun] | `P_rad/P_gas` | initial `g_rad/g` | full-L `g_rad/g` |
| ---: | ---: | ---: | ---: |
| 300 | 0.610 | 0.496 | 1.08 |
| 500 | 0.205 | 0.220 | 9.1 |
| 600 | 0.229 | 0.148 | 14.5 |
| 700 | 0.184 | 0.257 | 17.88 |
| 750 | 0.0986 | 0.194 | 13.77 |
| 780 | 0.0506 | 0.140 | 10.36 |
| 800 | 0.0209 | 0.0629 | 17.38 |
| 813 | 0.00743 | 0.0250 | 0.651 |
| 817 | 0.00329 | 0.0133 | 0.0196 |

The full-luminosity column is intentionally counterfactual, especially
at 300 Rsun where the local heating fraction is below unity. The direct
acceleration and pressure ratios are the pertinent measurements. In the
photospheric transition near 813 Rsun, ignoring radiation momentum would
change the *initial local force* by only ~2.5% of gravity. In the deep
envelope (300–750 Rsun), it changes support by roughly 15–50% of gravity;
it is not a controlled global approximation for this setup. In addition,
keeping the same profile but forcing the full input luminosity through
700–800 Rsun would make the radiative acceleration many times gravity.
Thus an energy-transport comparison with the CO5BOLD result cannot assume
the same pressure balance.

Reproduction: `wor122_initial_convective_capacity.py` now writes the
pressure/force ratios to `wor122_convective_capacity_20261117.json` and
`doc/figures/wor122_force_budget_20261117.png`.

## First initial-condition gate

The existing 3487-K gray HSE profile
`data/ic/st35gm03n07_wor55_hse.dat` was compared with the present
3227.8-K gray base at the *same* potential mass, requested luminosity,
radius, opacity, and mapped envelope mass. Both map to about 3.01 Msun on
64³. At 700/750/800 Rsun, the 3487-K profile has radiative luminosity
0.0140/0.0143/0.0035 Lstar, against
0.0144/0.0141/0.0036 Lstar in the 3227.8-K base. The photospheric
transition shifts, but the deep initial convective burden does not improve.
The 3487-K base also lacks the validated LTE/FLD exterior of the current
trial, so it is not ready for a 3D run. Merely restoring the nominal
effective temperature is rejected as a remedy.

The next candidate must alter the structure and/or preparation path enough
to carry the required luminosity while keeping radiation-force HSE. Before
any 3D job it must pass: (1) mapped mass, (2) 1D pressure/radiation force
balance, (3) an LTE/FLD-compatible exterior, (4) the radiative luminosity
and convective transport budget, and (5) finite-volume force and FLD face
flux audits on the intended 3D mesh. A short 3D gate then measures actual
temperature, velocity, luminosity, and mass evolution, followed by a
measured taper of any temporary preparation source.
