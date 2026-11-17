# WOR-14: initial convective transport capacity and preparation alternatives

The 128³ conservative-preparation gate (XD 341276) finished normally, but
did not repair the thermal startup. At t=8 its maximum Mach was 0.1217
(additional-heating control: 0.1494), while its six-face radiative
luminosity was 1.371 Lstar (control: 0.971 Lstar). The integrated boundary
loss was 1.051 Lstar versus 0.974 Lstar. Radiation energy fell by 1.37e5
code units in the transfer run, whereas it rose by 1.52e6 in the control.
The local gas temperature ratio at 813–817 Rsun was 0.9524 versus 0.9913;
at 830–850 Rsun it was 1.1433 versus 1.0394. The corrected core velocity
changed from +0.0150 to -0.0022 km/s, but this did not fix the surface.
The 800–830 Rsun convective enthalpy covariance was only 0.00271 Lstar
versus 0.00243 Lstar in the control. Source reallocation is not a viable
long-run cure by itself. Matched XD diagnostics are
`$WORK/wor122/prep128_transfer_t8/{shell_comparison.json,convective_flux_comparison.json}`
from M-debug job 341330 (exit 0, 9 s).

The imported 1D profile requires approximately `Lstar-Lrad` to be carried
by convection, although its 3D initialization has almost zero convective
enthalpy covariance. We estimated an *illustrative* two-stream scale from
the same Tomida material EOS. At fixed profile pressure, hot and cold
streams occupy equal areas and have temperatures `T(1±delta)`. The hot
stream rises at the local adiabatic sound speed; the cold stream's speed
balances mass flux. This is not a rigorous maximum, an MLT prediction, or
a substitute for radiation hydrodynamics. It shows the severity of the
initial transport requirement at plausible contrasts.

| Radius [Rsun] | Required convective L/Lstar | Sound speed [km/s] | Sonic, ±10% T | Sonic, ±30% T | Sonic, ±50% T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 700 | 0.986 | 23.53 | 0.313 | 1.047 | 1.811 |
| 750 | 0.986 | 19.08 | 0.257 | 0.637 | 1.381 |
| 780 | 0.987 | 15.84 | 0.184 | 0.660 | 1.230 |
| 800 | 0.996 | 12.19 | 0.397 | 0.818 | 0.836 |
| 810 | 0.986 | 8.27 | 0.283 | 0.490 | 0.479 |
| 813 | 0.962 | 7.66 | 0.143 | 0.376 | 0.421 |
| 817 | 0.322 | 7.38 | 0.022 | 0.082 | 0.206 |

At 813 Rsun even the extreme ±50% example carries only 0.421 Lstar at
sonic hot-upflow speed, less than half of the prescribed convective part.
Simply increasing uncorrelated velocity noise cannot supply the initial
enthalpy flux. The gray HSE builder selects a superadiabatic excess to
match the envelope mass, but does not independently verify that its
remaining luminosity can be carried by a realizable convection field.
At 800 Rsun the WOR-55 Rosseland opacity is about 25.7 cm²/g. If the
entire Lstar were radiative there, its outward radiative acceleration
would exceed local gravity by roughly a factor 18; convection is mandatory
under that opacity. This is a model construction and startup constraint,
not evidence of an FLD solver failure.

This estimate must not be interpreted as a requirement for a quiet,
everywhere-subsonic RSG. [Chiavassa et al. (2011)](https://www.aanda.org/articles/aa/pdf/2011/11/aa17463-11.pdf)
report supersonic outer-layer velocities, a density inversion associated
with their opacity peak, visible early Cartesian artifacts, and relaxation
over several stellar years. Their RSG radiation step uses short
characteristics designed for strong opacity/temperature jumps. This is a
different surface transport operator from the present FLD, despite a
comparable nominal grid spacing; their successful years-long relaxation
does not validate our current FLD initial state.

The diagnostic can be reproduced with
`tst/nrfld_validation/wor122_initial_convective_capacity.py`; its compact
numbers are in `wor122_convective_capacity_20261117.json` and its figure is
`doc/figures/wor122_convective_capacity_20261117.png`.

## A key difference from the relaxed CO5BOLD model

The top-left panel of Fig. 3 in [Chiavassa et al. (2011)](https://arxiv.org/html/1109.3619)
shows a *relaxed snapshot*: the radiative component is visually about
80–105% of the total luminosity over 500–800 Rsun, with pressure work
contributing roughly -20% to +20%. The same figure's opacity panel is of
order a few cm²/g over much of 500–800 Rsun. These are approximate readings
of a published plot, not digitized data, and they are not a measurement of
that model's *initial* state.

Our imported profile is dramatically different in the outer convective
interior. At 700/750/800 Rsun its radiative luminosities are
0.0144/0.0141/0.0036 Lstar. We therefore evaluated several tables at
the *identical* imported `(rho,T)` states. The local MESA data files are
GS98 solar abundance tables with X=0.70 and Z=0.02. They are a useful
independent reference, but they are not the exact OPAL/PHOENIX merge used
by CO5BOLD and do not have precisely the same metal abundance pattern as
WOR-55.

| Radius [Rsun] | T [K] | WOR-55 κR | Older t7 κR | MESA OPAL GS98 κR | MESA Ferguson GS98 κR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 47119 | 15.30 | 15.65 | 5.76 | outside table |
| 600 | 36085 | 24.18 | 23.86 | 4.03 | outside table |
| 700 | 26402 | 28.86 | 28.19 | 3.03 | 3.00 |
| 750 | 19786 | 21.47 | 21.49 | 4.98 | 5.74 |
| 780 | 15103 | 15.69 | 17.12 | 11.50 | 11.21 |
| 800 | 11062 | 25.71 | 22.62 | 30.67 | 30.58 |
| 813 | 7613 | 0.947 | 0.998 | 0.941 | 1.009 |
| 817 | 5780 | 0.0283 | 0.0310 | 0.0253 | 0.0304 |

Units are cm²/g throughout. The old and new local Optab tables agree
closely along this profile. OPAL/Ferguson is about 4–10 times smaller at
600–750 Rsun, but all tables have a strong opacity drop across the outer
~800–817-Rsun transition. The difference with Fig. 3 is partly an opacity
difference at matched states, and also a difference in the evolved
thermodynamic structure and equations. We cannot attribute it to WOR-55
alone. Reproduction and checksums are in
`tst/nrfld_validation/wor122_opacity_reference_20261117.py` and its JSON;
the plot is `doc/figures/wor122_opacity_reference_20261117.png`.
The XD raw Optab file
`$WORK/wor55/optab/work/rsg_2p5_7p0_p14_3/output/opac_table.h5`
was checked directly. The four source layers bracketing the 700-Rsun
state in `(rho,T)` have κR = 18.20, 23.85, 26.06, and 34.88 cm²/g;
the layers around 750 Rsun have κR = 15.99–25.55 cm²/g. Thus the
~4–10-fold difference from MESA OPAL at those states is present in raw
Optab means and is not caused solely by the final structured-grid
interpolation. This does not identify which microphysical source causes it.
At fixed 1D temperature/radiation gradient in diffusion, replacing only
κR by the MESA OPAL value would raise the 700-Rsun radiative fraction from
0.0144 to approximately 0.137, still leaving most of Lstar to convection.
It would not constitute a new HSE model or remove the 800–817-Rsun cliff.

The physical equations also differ. Equation (2) of Chiavassa et al. has
gas pressure and gravity in momentum, with no radiation-force term shown.
Our RSG calculation explicitly applies `-lambda grad(E_r)` to gas momentum,
and the 1D HSE builder includes radiation acceleration. With the *actual*
softened gravity and WOR-55 opacity, radiating the whole Lstar would give
`g_rad/g=17.88` at 700 Rsun and `17.38` at 800 Rsun. Their local Eddington
luminosities are only 0.0559 and 0.0575 Lstar, respectively. Consequently,
we cannot copy the published radiative-flux profile into our present
radiation-force model without re-solving its pressure, density, and velocity
structure. The high opacity may itself be physical; this is not evidence
that the table is faulty. Our measured curves are
`doc/figures/wor122_lrad_eddington_20261117.png` and the JSON above.

This mismatch narrows the next work: construct a self-consistent 1D/3D
startup whose radiative force and convective transport can both be
supported, and check the remaining difference from the original CO5BOLD
opacity merge before attributing it to microphysics. A Freytag-style
profile can serve as a deliberately different-physics diagnostic only if
radiation-force omission is clearly labeled; it cannot be promoted as a
production solution. Further long XD evolution of the current unbalanced
profile is unlikely to separate these effects.

## First lower-luminosity preparation candidate

The unchanged HSE/gray-profile generator was run with 0.3 Lstar, the
Stefan-Boltzmann-matched 2388-K surface, 830-Rsun radius, and 3-Msun mapped
envelope. It returned 3.00571 Msun on 64³. The LTE/FLD exterior was joined
by 900 Rsun. The radiative 10–90% luminosity transition moved inward to
803.81–806.69 Rsun, but its 2.89-Rsun width remained unresolved on a
128³/256³ uniform mesh. Its outer unfloored HSE residual was 10.9%, versus
1.28% for the target profile. Gas density fell sharply from 5.05e-10 at
830 Rsun to 2.83e-11 g/cm³ at 840 Rsun. A local 64³ one-step gate required
lowering the numerical pressure floor from 1e-15 to 1e-16 code units in the
cold exterior and then converged with zero opacity-domain errors; its maximum
Mach was 0.365, versus 0.00505 in the target model.

XD M-debug **341335** ran the same model at 128³ to t=0.1 (exit 0, 9 s,
Newton convergence and zero opacity-domain errors). Maximum Mach was
0.02393 versus 0.00280 for a matched target-profile one-step run. The
813–817-Rsun mass-weighted gas-temperature ratio improved from 0.9839 to
1.0220, but 850–880-Rsun RMS speed increased from 0.0027 to
0.0195 km/s. Diagnostic **341337** (exit 0, 6 s) saved
`$WORK/wor122/lowL03_128_t01/shell_comparison.json`.

One M-debug continuation, **341338**, was stopped at t=3 code units
(~0.76 day) when maximum Mach reached 0.503 and box radiative output
reached 2.55 times its *reduced* input luminosity. It was a failed short
gate, not a solver crash. No further evolution with this cold exterior
was submitted. Large trial files and scripts are in `$WORK/wor122/lowL03_*`.

Keeping the original 3227.8-K surface while lowering the luminosity to
0.3 Lstar preserved a warm outer atmosphere and yielded a 3.00975-Msun
mapped gray HSE profile. It is intentionally non-Stefan-Boltzmann as a
preparation candidate. The matching LTE/FLD exterior ODE failed to reach
its free-streaming join by 900 Rsun: its flux factor had fallen to 0.0985
from 0.102 at the surface. A warm radiation field with the reduced flux
therefore did not give a stationary exterior under the current FLD closure.
No 3D job was submitted for that inconsistent profile.

An 1800-K Marshak bath at the box boundary was checked locally at 64³,
one step, using the target profile. It reduced six-face output from 1.821
to 1.668 Lstar but left the 813–817-Rsun temperature ratio at 0.9839 and
maximum Mach near 0.00504. Changing the bath at the box edge does not
immediately address the underresolved stellar transition, so this option
was not sent to XD.
