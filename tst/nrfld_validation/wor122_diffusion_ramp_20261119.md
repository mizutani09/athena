# WOR-14/122: gradual FLD restoration from an RSG surface preparation

## Question and setup

The source-free, diffusion-off 800–860 R☉ patch prepared through t=160
code units (40.7 days) has a growing horizontal mode, but only 0.0768 F★
of mass-advection-corrected convective flux at 813–817 R☉. Abruptly restoring
FLD and a bottom F★ source erased that mode. This experiment tests whether
the abrupt switch itself caused the failure.

The opt-in `fld/diffusion_ramp_start_code` and `...end_code` scale all six
implicit face diffusion coefficients by
`q = [0.5 (1 - cos(pi (t - start)/(end - start)))]²`, clipped to zero and
one at the endpoints. The same factor enters the Marshak face coefficient,
so the physical boundary flux and interior diffusion are restored together.
`problem/bottom_heat_follow_diffusion_ramp=true` scales the bottom F★ source
by q. Gas-radiation exchange, radiation force, EOS and opacity are kept on.
This is a nonphysical preparation method; q=1 with no artificial source is
the physical target. The patch has 32²×48 cells of width 1.25 R☉, Tomida
material EOS, WOR-55 gray opacity, profile gravity, reflecting lower gas
boundary, periodic sides, and a Marshak upper radiation boundary.

## t=160 to 176 release

| t [code] | q | T(813–817)/T160 | corrected Fconv/F★ | maximum Mach | top Frad/F★ | bottom heat/F★ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 0 | 1.000 | 0.0768 | 0.523 | — | 0 |
| 164 | 0.0214 | 1.004 | 0.0813 | 0.532 | 0.013 | 0.0214 |
| 168 | 0.250 | 1.009 | 0.0847 | 0.540 | 0.346 | 0.250 |
| 172 | 0.729 | 0.968 | 0.0775 | 0.609 | 0.864 | 0.729 |
| 174 | 0.925 | 0.885 | 0.0451 | 0.924 | 0.982 | 0.925 |
| 176 | 1.000 | 0.803 | 0.0135 | 0.917 | 0.995 | 1.000 |

The first trial at CFL=0.06 failed at t=160.12 with a Newton
`backtracking_exhausted` error when the initial raised-cosine factor was
about 1.5×10⁻⁴. A squared raised-cosine onset and CFL=0.015 converged
through t=176. A matched diagnostic run at CFL=0.06 with the squared onset
still failed at t=161.70 (q≈7.7×10⁻⁴), while mean temperature and convective
flux remained intact. The early failure is therefore a timestep-sensitive
nonlinear-solver limit, not measured thermal collapse. The smaller-timestep
run had converged Newton solves and reached full FLD.

By t=168, the true Marshak top flux exceeds the contemporaneous bottom
heating. The transition then cools and the horizontal transport dies. At
t=176 the 800–813 R☉ shell carries about 0.035 F★ by radiation and
0.148 F★ by corrected convection, while the 813–817 R☉ shell has about
0.905 F★ of radiative flux and only 0.013 F★ of corrected convection.
These shell-centered fluxes are approximate, but the large luminosity gap
and temperature fall are unambiguous. At full FLD the preparation still
fails the physical stability gate. The global top flux approaching F★ at
t=176 does not mean the interior transport has equilibrated.

The `Frad_z` user-output variable in the early ramp runs is the unscaled
formal FLD estimate; the exact top-face history uses the scaled Marshak
coefficient. The source code now scales `Frad_z` by q for subsequent runs.
No production XD job was submitted. Raw restart/HDF/log files are under
`/tmp/wor122_diffusion_ramp_*`; the source input is
`inputs/rsg/athinput.wor122_coherent_diffusion_ramp_40r_local`.

## Longer source-free preparation

The no-diffusion/no-heat control was continued from t=160 to 240. It
converged without a high-Mach runaway, but the 813–817 R☉ corrected
convective flux grew only from 0.0768 to 0.1727 F★, while maximum Mach
grew from 0.523 to 0.672. The flux remains far below the stellar flux.
The continuation is `/tmp/wor122_coherent_nodiff_ext160_240`.

| t [code] | corrected Fconv/F★ | maximum Mach | horizontal T rms/mean |
| ---: | ---: | ---: | ---: |
| 160 | 0.0768 | 0.523 | 0.200 |
| 180 | 0.0995 | 0.563 | 0.208 |
| 200 | 0.1241 | 0.606 | 0.218 |
| 220 | 0.1498 | 0.647 | 0.230 |
| 240 | 0.1727 | 0.672 | 0.244 |

The longer t=240 state was used for one matched gradual-release comparison.
At t=244 (q≈0.022) the transition kept 0.176 F★ of convective flux and its
initial mean temperature. At t=248 (q≈0.251) it still had 0.175 F★ and
T/T240=1.026. However the Marshak top already radiated **0.819 F★**, while
bottom heating supplied only **0.251 F★**. The 817–830 R☉ layer warmed by
5.9% relative to the already overheated t=240 state, and the 830–850 R☉
layer warmed by 12.7% and reached maximum Mach **1.35**. The controlled
comparison was stopped with a graceful SIGTERM at t=248.96; Newton solves
had converged and a final restart was written. This is a physical-state
rejection, not a solver crash. Continuing to full FLD would spend time on
an already nonphysical outer atmosphere.

Further diffusion-off spinup alone is not a credible way to close the
missing flux. The initial model must supply consistent convective/radiative
transport and support before a whole-star run is justified.

The extended preparation also heats the 817–830 R☉ shell from 5024 K at
t=160 to 6943 K at t=240 (+38%) without any heat source. This is internal
energy redistribution in the diffusion-off setup, not evidence that a
physical radiation-cooled photosphere is stable. The t=240 release must be
judged against its own initial state and source-free full-FLD gate.

Compact measured values are in `wor122_diffusion_ramp_20261119.json`; the
comparison figure is `doc/figures/wor122_diffusion_ramp_20261119.png`,
reproducible with `python3 tools/rsg/plot_wor122_diffusion_ramp.py`.

## Next gate

The t=160 and t=240 comparisons isolate the abrupt restart as a secondary
problem. The primary deficit is the missing 3D convective luminosity below
the radiative transition. The already tested Henyey-MLT profile closes the
1D flux, but its radiation handoff is only about 0.9 R☉ wide and its local
turbulent pressure approaches the gas pressure near 816 R☉
(`wor122_henyey_mlt_20261118.md`). A static mapping onto a 7.89 R☉ cell
at 256³ cannot represent either effect.

Before another whole-star run, prepare a resolved local 3D precursor with
actual velocity–enthalpy correlations and their mean Reynolds-stress
support. Audit the mapped finite-volume force and FLD face-flux residuals,
then require a source-free full-FLD continuation for multiple local cooling
times with a bounded transition temperature, mass, Mach, and luminosity.
The existing 1D MLT profile is a structural starting point, not a ready
3D initial condition. Further low-resolution diffusion-off spinup is not
the next gate.
