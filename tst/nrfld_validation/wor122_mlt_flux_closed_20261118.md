# WOR-14: integrated efficient-MLT flux-closed 1D RSG trial

## Result and gate

A trial 1D profile that closes the interior luminosity at each radius was
built and integrated inward from the same 830 R☉, 3227.8 K photosphere as
the current model. With efficient MLT `alpha=1.537`, it has 2.996 M☉ inside
the nominal stellar radius and 3.007 M☉ after 64³ Cartesian mapping.
Local radiative plus convective flux agrees with the imposed luminosity to
better than `10^-10` at the selected audit radii. This is a concrete proof
that a flux-closed 1D profile near the requested mass exists within this
*optimistic* transport prescription.

It is **not suitable for a long 3D run**. Its surface FLD luminosity
10–90% transition is approximately 2.36 R☉ after increasing the profile
sampling to 3072 interior and 1536 atmosphere points. The current
reference profile has 3.35 R☉, and a 256³ Cartesian cell spans 7.89 R☉.
The trial also predicts turbulent pressure of about 0.29 gas pressures at
817 R☉, but the integrated HSE and mapped 3D state do not contain that
support or the velocity–enthalpy correlations that carry the MLT flux.
No XD job was submitted for this rejected candidate.

## Method

The separate experimental generator
`tools/rsg/make_mlt_rsg_hse_trial.py` retains the current Tomida material
EOS, WOR-55 Rosseland opacity, 12 M☉ softened potential, core heating
luminosity profile, radiative force in gas-pressure HSE, and gray
photosphere boundary. At each ODE evaluation it obtains EOS-consistent
`∇ad`, `cP`, and thermal expansion from the entropy table. It solves

`Ftot(r) = Frad(∇; rho,T) + Fconv,efficient-MLT(∇; rho,T,alpha)`

for the local `∇ = d ln T/d ln P`, then integrates the resulting gas-pressure
and temperature derivatives inward. The radiation term uses diffusion
in the interior, with its radiation acceleration coupled to the local gas
pressure gradient. The efficient-MLT coefficient matches the preceding
`wor122_mlt_transport_scale.py` audit (`nu=8`). The model ignores radiative
loss from eddies and turbulent-pressure feedback on force balance. Its
gray exterior is rebuilt with `make_lte_fld_exterior_trial.py` before the
surface-width comparison.

Unlike the old generator, this model does not tune an arbitrary Gaussian
superadiabatic excess to hit a mass. With the photospheric boundary fixed,
its mass is strongly sensitive to the MLT parameter:

| `alpha` | 64³ mapped mass [M☉] |
|---:|---:|
| 1.0 | 0.512 |
| 1.2 | 0.841 |
| 1.4 | 1.637 |
| 1.52 | 2.775 |
| 1.537 | 3.007 |
| 1.56 | 3.342 |
| 1.6 | 4.116 |
| 2.0 | 32.197 |
| 3.0 | 830.049 |

The 830 M☉ result at `alpha=3` is a failure to match the specified envelope
mass under this closure and boundary, **not** evidence of a physical RSG
with such a mass. The mass matching at `alpha=1.537` is a parameter fit;
it does not establish that this alpha is the correct turbulent transport.

At 400/500/600/700/800/817 R☉, the selected fine profile has local
`Lrad/Ltot ≈ 0.44/0.12/0.013/0.006/0.001/0.86`. The corresponding
optimistic `Pturb/Pgas` estimate is approximately
`0.035/0.051/0.044/0.128/0.172/0.29`. The transition from convection
to radiation at the surface is consequently sharp, and the turbulent
support that would accompany this flux cannot be ignored when judging
mechanical consistency.

The same trial permits a first check of the efficient-eddy assumption.
For an eddy length `alpha H`, `tau_eddy v/c` is about 19 at 500 R☉,
1.35 at 800 R☉, 0.087 at 813 R☉, and 0.00092 at 817 R☉. Diffusion-time
coefficients are order unity to several, so the last two values strongly
indicate that photons escape an eddy before it turns over. The optimistic
efficient-MLT closure is especially unreliable exactly where the surface
luminosity switches channels. These estimates use the trial's predicted
velocity and its sampled opacity; they are not a radiation-hydrodynamic
measurement.

## Sampling and limitations

The 768-point trial has a FLD 10–90% width of 2.10 R☉; the 3072-point
trial gives 2.36 R☉. Their masses differ by less than `0.0001 M☉`, and
the resolved-interior sampled gas-HSE residual drops from 1.13% to 0.62%.
The width changes because the continuum FLD gradient is sampled across a
very thin layer; the conclusion that it is much narrower than one 256³
cell is insensitive to this sampling test.

The local closure check is exact for the *chosen formula*, while the
physical prescription is deliberately optimistic. Eddy radiative leakage
could reduce transport near the photosphere; a real turbulent-pressure
gradient would alter the integrated profile and its mass. The FLD exterior
join is experimental, and the interior diffusion formula is not the exact
discrete Athena FLD operator. The trial has no initialized convective
velocity field, so it would immediately lack the convective flux used in
its 1D energy balance if mapped to a static 3D state.

## Next step

The next model needs a surface transport treatment that resolves the
photospheric energy exchange on the intended mesh, plus a consistent
way to build or relax the 3D velocity–enthalpy correlations. Quantify the
effect of eddy leakage and turbulent-pressure gradients in 1D before
changing the physical simulation. A candidate should pass a discrete
Cartesian FLD/force audit and a short source-free test before receiving
a long XD allocation. The surface layer must be resolved or represented
by a defensible subgrid or boundary treatment; adjusting `alpha` alone
does not accomplish that.

Figure: `doc/figures/wor122_mlt_flux_closed_20261118.png`.
Summary and selected profiles:
`tst/nrfld_validation/wor122_mlt_flux_closed_20261118/`.
Reproduce the selected high-sampling profile with:

```sh
python3 tools/rsg/make_mlt_rsg_hse_trial.py \
  --eos data/eos/rsg_tomida2013_material_eos.tab \
  --opacity data/opacity/rsg_gray_opacity_wor55.h5 \
  --alpha 1.537 --points 3072 --atmosphere-points 1536 \
  --output tst/nrfld_validation/wor122_mlt_flux_closed_20261118/alpha1p537_fine_base.dat
python3 tools/rsg/make_lte_fld_exterior_trial.py \
  --base tst/nrfld_validation/wor122_mlt_flux_closed_20261118/alpha1p537_fine_base.dat \
  --eos data/eos/rsg_tomida2013_material_eos.tab \
  --opacity data/opacity/rsg_gray_opacity_wor55.h5 \
  --join-end-rsun 900 \
  --output tst/nrfld_validation/wor122_mlt_flux_closed_20261118/alpha1p537_fine_fld.dat
python3 tst/nrfld_validation/wor122_plot_mlt_flux_closed_20261118.py
```
