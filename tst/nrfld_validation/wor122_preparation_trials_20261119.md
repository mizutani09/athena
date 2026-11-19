# WOR-14/122: local RSG preparation trials (2026-11-19)

## Purpose

Test whether the photospheric seed can survive long enough for buoyant
transport to develop without the artificial 812.5–817.5 R☉ luminosity source.
These are local 800–860 R☉ Cartesian patch tests, not a stable whole-star
solution. The 1D profile has a separate deep-envelope flux-closure deficit
(`wor122_flux_closure_scan_20261118.md`).

All local tests use Tomida material EOS, WOR-55 gray opacity, profile gravity,
the radiation-force source, and 0.625 R☉ cubic cells unless stated otherwise.
The lower gas boundary is reflecting; horizontal boundaries are periodic.
The transport diagnostic subtracts the mean mass-advection enthalpy flux.
Thus a coherent mean vertical drift cannot masquerade as convection.

## Controlled results

| Preparation | Time [code] | 813–817 R☉ mean T/T0 | 813–817 R☉ corrected convective flux/F★ | 800–813 R☉ mean T/T0 | 800–813 R☉ max Mach | Finding |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bottom F★ over 5 R☉, full FLD, open lower gas boundary | 8 | 0.746 | ≈0 | 1.049 | 0.219 | Surface cools quickly. |
| Bottom F★ over 5 R☉, full FLD, reflecting lower gas boundary | 8 | 0.928 | ≈0 | 1.094 | 0.129 | Boundary closure helps mean state, not convection. |
| No heat, full FLD, reflecting lower gas boundary | 20 | 0.972 | ≈0 | 0.980 | 0.141 | The seed still damps without imposed heat. |
| Shell-mean gas thermal relaxation with bottom F★, full FLD | 8 | 0.989 | ≈0 | 1.111 | 0.115 | Holds mean temperature, but the horizontal mode still damps. |
| Bottom F★ over 5 R☉, FLD diffusion off | 20 | 1.000 | 3×10⁻⁵ | 1.489 | 0.452 | Surface seed survives; deep layer overheats. |
| Bottom F★ over 10 R☉, FLD diffusion off | 20 | 1.004 | 4×10⁻⁵ | 1.216 | 0.154 | Less overheating, but convection remains negligible. |
| Bottom F★ over 20 R☉, FLD diffusion off | 8 | 1.134 | ≈0 | — | — | Heating overlaps and weakens the seeded layer; stopped. |
| Coherent 10% isobaric density seed and 0.2 cₛ buoyancy-aligned vertical velocity, bottom F★ over 10 R☉, full FLD | 8 | 0.749 | ≈0 | — | — | Initial corrected convective flux is 0.0026 F★, then vanishes. |

The 10 R☉ heating case retained about 102% of its initial temperature mode
at t=8, but did not transport an appreciable stellar flux by t=20. Turning
off FLD diffusion removes the interior diffusion and boundary radiation flux;
its reported formal `Frad` output must not be interpreted as transported
energy. The shell-mean relaxation is preparation only and adds/removes gas
energy uniformly per mass in each horizontal layer; it did not preserve
the seed. The normal-FLD restart smoke test from a no-diffusion preparation
completed through t=8 with converged Newton solves, but its corrected
surface convective flux remained ≈0.

The opt-in shell-mean thermal relaxation passed a two-rank MPI startup
smoke test through t=0.2 code units. It allocates extra meshblock restart
data, so it cannot be enabled for the first time from a checkpoint written
without that option; the restart layout check rejects that attempted
toggle. Its matched fresh-start trial already showed no seed preservation.

In the coherent-velocity trial, the imposed surface temperature mode fell
to 1.12% of its initial amplitude by t=4 code units (1.02 days), while the
density mode still retained 79.7%. At t=8 the temperature mode was 1.28%
of its initial amplitude. Thus a large initial upward/hotward correlation
alone does not outrun FLD thermal equilibration. The test is deliberately
stronger than a realistic perturbation and is a negative gate, not a
calibrated convective initial condition.

A separate **coarse** 40×40×60 R☉ source-free/no-diffusion local trial uses
32²×48 cells (1.25 R☉ cubes), the same 10% density and 0.2 cₛ coherent
velocity seed. At t≈20 code units (5.1 days), 813–817 R☉ mean T/T0 is
0.999, the temperature mode has grown to 1.72 times its initial amplitude,
and corrected convective flux has risen from 0.0026 to 0.0041 F★. At
t≈40 code units (10.2 days), the mode is 2.05 times initial and the
corrected convective flux is 0.0046 F★. At t=60 and 80 (15.3 and 20.4
days), it rises to 0.0081 and 0.0151 F★, and maximum Mach rises to 0.226
and 0.308. The temperature mode reaches 2.90 times initial by t=80.
The source-free/no-diffusion preparation is dynamically growing and
numerically stable through 41 days, but its flux still falls far short of the
missing luminosity. A local t=80→160 continuation tested whether the
growth saturates; its CFL was 0.06 instead of 0.03. At t≈100 and 120
(25.5 and 30.6 days), corrected surface convective flux is 0.0233 and
0.0348 F★, with maximum Mach 0.354 and 0.393. At t≈140 and 160
(35.6 and 40.7 days), flux reaches 0.0540 and 0.0768 F★, maximum Mach
0.465 and 0.523, and temperature RMS contrast rises to 20.0%.
The flux has not saturated, but remains about one order of magnitude too
small to replace the missing luminosity even after 41 days of artificial
preparation. A queued XD 64²×96 test with a weaker 2% seed and no coherent
velocity was canceled before it started (see below).

Restoring full FLD and a 10 R☉-deep bottom F★ source from the t≈20
checkpoint gives a direct release test. By t≈24 (about one day later),
813–817 R☉ mean temperature is 0.795 times its release value, its
horizontal temperature RMS falls from 7.11% to 2.06%, and corrected
convective flux falls from 0.00415 to 0.00043 F★. The 800–813 R☉ layer
keeps its mean temperature (ratio 1.003) and about 0.0041 F★ convective
flux, so the immediate failure is concentrated at the radiative transition.
This release fails the physical gate even though the no-diffusion
preparation itself is numerically stable.
By t≈28 (about two days after release), mean temperature is 0.762 of the
release value, temperature RMS has fallen further to 1.12%, and corrected
convective flux is 0.0002 F★. The full-FLD Newton solves converged; this
is a physical transport failure, not an implicit-solver crash.

A second release from the **15-day** t≈60 checkpoint is similar despite
larger prepared fluctuations: after one day of full FLD, the 813–817 R☉
mean temperature is 0.792 of the release value, RMS temperature contrast
falls from 9.84% to 2.34%, and corrected convective flux falls from 0.00807
to 0.00070 F★. Meanwhile the deeper 800–813 R☉ corrected flux rises from
0.0105 to 0.0119 F★. The surface radiative transition is the immediate
loss region; longer preparation has not yet closed its energy deficit.

A third release from the **25-day** t≈100 checkpoint also fails after one
day: the transition temperature is 0.807 of its release value, horizontal
temperature RMS falls from 14.45% to 6.01%, and corrected convective flux
falls from 0.0233 to 0.00454 F★. The deeper 800–813 R☉ flux instead rises
from 0.0282 to 0.0325 F★. The trend improves slightly with preparation
duration, but the radiative transition still cools by roughly 20% in a day.
This third release was stopped after the first diagnostic checkpoint.

The **41-day** t=160 release is less abrupt but still needs a continuation:
after one day of full FLD, the 813–817 R☉ mean temperature is 0.871 of
the release value, temperature RMS decreases from 20.03% to 14.13%, and
corrected convective flux decreases from 0.0768 to 0.0396 F★. The local
maximum Mach increases from 0.523 to 0.851. The deeper 800–813 R☉
corrected flux rises from 0.103 to 0.117 F★, whereas 817–830 R☉ flux
falls from 0.0175 to 0.00056 F★. This is an improvement in flux retention,
but not a stable thermal balance. In the following day, by t≈168,
temperature falls to **0.753 of its release value**, RMS contrast to 2.99%,
and corrected convective flux to **0.00212 F★**. The transition maximum
Mach reaches 0.896. The deeper 800–813 R☉ convective flux remains 0.128
F★, while the 813–817 R☉ layer is thermally erased. This fails the
release gate. The t=164→180 continuation was stopped at t≈168 after the
diagnostic snapshot to avoid spending more time on a rejected state.

An additional **preparation-only** release retains total heating F★ while
moving it gradually from the 812.5–817.5 R☉ shell to the 10 R☉-deep
bottom region over t=160–240. After the first day (t≈164), the transition
mean temperature is 1.056 of its t=160 value and corrected convective flux
is 0.0714 F★, versus 0.0396 F★ for abrupt release. However its horizontal
temperature RMS still falls from 20.0% to 10.5%, max Mach reaches 0.871,
and almost all heating is still artificial and local to the transition.
This is a protected preparation state, not an acceptable source-free run.
By t≈168, its transition RMS contrast falls again to 5.27%, corrected
convective flux to 0.0504 F★, and maximum Mach rises to 1.165. The
overlapping-heater branch is rejected before the 20-day transfer finishes.

A matched variant places the temporary heater at **817.5–825 R☉**, above
the seeded transition. At t≈164, the 813–817 R☉ mean temperature is 1.029
of its t=160 value, RMS contrast is 15.09% (versus 10.48% with overlapping
heat), corrected convective flux is 0.0742 F★ (versus 0.0714), and maximum
Mach is 0.515 (versus 0.871). The 817–830 R☉ mean temperature stays near
its initial value (ratio 1.002). Heater placement therefore materially
reduces seed damping and local acceleration. In the continuation of this
variant, at t≈168, transition mean temperature is 1.008 of
its t=160 value, corrected convective flux 0.0642 F★, temperature RMS
13.03%, and maximum Mach 0.785. By t≈172, corrected flux has fallen to
0.0496 F★, RMS to 10.30%, and maximum Mach has risen to 1.102. At t≈176,
the flux is 0.0351 F★, RMS 8.06%, and maximum Mach 1.190, while the
transition mean temperature is 0.930 of its t=160 value. The branch was
stopped after this diagnostic checkpoint: despite nearly all heating still
being artificial and placed above the transition, it drifts toward the same
cooling and flux-loss state. A third placement at 805–812.5 R☉, below the
seeded transition, gave t≈164 mean T/T0=0.882, corrected convective flux
0.0418 F★, RMS temperature contrast 13.47%, and maximum Mach 0.848. It
does not prevent the initial cooling and is worse than the above-transition
heater. Among the three temporary source placements, none passes a
source-free release gate.

## Further gates

1. Require a source-free full-FLD continuation with bounded temperature,
   luminosity, mass and Mach over multiple surface/dynamical times before
   another whole-star run. Abrupt releases from 5, 15, 25 and 41-day
   preparations have failed that gate. Temporary heat at, above, or below
   the transition improves selected short diagnostics but does not maintain
   the prepared convection.
2. Replace the current 1D profile with a force- and
   flux-closed precursor including the missing deep convective luminosity,
   turbulent support and radiative leakage near the photosphere. The
   MESA-based core/radiative/convective-layer work (WOR-151/155/172) is a
   relevant source of structure and core luminosity; it is not yet a
   validated substitute for this test's gray-FLD initial profile.

The queued XD 64²×96 no-diffusion, no-heat spinup **342942** was canceled
before starting. Its 2% density seed and no coherent velocity were weaker
than the local 10%/0.2 cₛ seed, which still did not survive full-FLD release
after 41 days; running the queued 20-day control would not have advanced
the immediate stability gate. No production RSG job was submitted.

Inputs are `inputs/rsg/athinput.wor122_prep_*`,
`athinput.wor122_nodiff_spinup_40r`, and the coherent-seed inputs. XD script:
`tst/nrfld_validation/wor122_nodiff_spinup_40r_xd.sh`. Compact diagnostics
are in `tst/nrfld_validation/wor122_prep_shell_20261119/`; raw local HDF is
under `/tmp/wor122_*`. The comparison figure is
`doc/figures/wor122_preparation_trials_20261119.png`, generated by
`tools/rsg/plot_wor122_preparation_trials.py`.
