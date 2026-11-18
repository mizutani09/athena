# WOR-14: radiatively cooled Henyey-MLT RSG precursor screen

## Decision

Adding a published radiative-loss MLT closure to the 1D RSG precursor
does **not** make this profile suitable for a 256³ 3D run. After fitting
the mapped mass, the LTE/FLD surface luminosity changes over only
approximately **0.93 R☉**, compared with 7.89 R☉ per 256³ Cartesian
cell. The MLT velocity scale also collapses over the same layer. No XD
job was submitted for this candidate.

The earlier suggestion to simply include the local MLT turbulent pressure
in the 1D force equation needs revision. Its sharp surface gradient could
itself be a large artificial force, and a static 3D mapping has no turbulent
velocity field to supply that support. A mechanically consistent 3D
preparation must be evaluated from an actual evolving flow or an explicitly
initialized stress model, not by adding a local-MLT pressure term to a
static atmosphere.

## Source and correction

The closure follows [Ludwig, Freytag & Steffen (1999), Appendix A](
https://adsabs.harvard.edu/pdf/1999A%26A...346..111L):
`Gamma=(nabla-nabla_e)/(nabla_e-nabla_ad)`,
`v²=f1*l²*g*delta*(nabla-nabla_e)/H`, and
`Fconv=f2*rho*cP*v*T*l*(nabla-nabla_e)/H`, with `l=alpha H`.
The Henyey constants are `f1=1/8`, `f2=1/2`, `f4=4*pi²/3`, and
**`f3=8*f4=32*pi²/3`**. The last value uses the explicit correction in
[MESA's MLT controls documentation](
https://docs.mesastar.org/en/stable/reference/controls.html): the
Ludwig et al. table printed `f4/8` for Henyey `f3`, a factor 64 too small.
Trial outputs made with that printed value were discarded before this
report. With the corrected coefficient the source profile's efficient-eddy
assumption remains reasonable in much of the envelope, but fails rapidly
across the surface radiative switch.

The generator `tools/rsg/make_mlt_rsg_hse_trial.py` now has explicit
`--mlt-option efficient|henyey` choices. The efficient branch remains
numerically identical to the preceding trial (data columns compared
exactly after the change). The Henyey branch solves the coupled local
flux equation and eddy temperature contrast during inward integration.
It retains the same EOS, opacity, luminosity, potential, photospheric
state, and radiative force as the earlier 1D trial. Its interior radiation
flux is still a diffusion estimate and the exterior is separately joined
to the experimental LTE/FLD atmosphere.

## Results

At fixed 830 R☉, 3227.8 K photosphere and target 3 M☉ envelope,
`alpha=2.366` gives a mapped mass **3.0040 M☉** (64³ mass audit),
2.9926 M☉ inside the nominal stellar radius, and a sampled maximum
resolved-interior gas-HSE residual of 0.60%. The selected local flux
closure residual is below `10^-10`. The fitted alpha is a model parameter,
not a calibration to observed RSG convection.

| Radius [R☉] | `Lrad/Ltot` | Henyey `Gamma` | Raw `rho v²/Pgas` |
|---:|---:|---:|---:|
| 500 | 0.105 | 4.72 | 0.107 |
| 700 | 0.0052 | 43.0 | 0.275 |
| 800 | 0.0011 | 135 | 0.376 |
| 813 | 0.0126 | 10.3 | 0.626 |
| 815 | 0.0419 | 3.36 | 0.761 |
| 816 | 0.230 | 0.807 | 0.905 |
| 817 | 0.99999 | 0.0010 | 0.00086 |

The transition is abrupt: the 10–90% continuum FLD luminosity widths are
3.35 R☉ for the current imported profile, 2.36 R☉ for the efficient-MLT
mass-matched trial, and **0.93 R☉** for the Henyey-MLT mass-matched trial
with 3072 interior and 1536 atmosphere output points. The Henyey result
is only a few 1D samples wide, so its precise width remains sampling
sensitive. A separate 6144-interior/3072-atmosphere-point check gave the
first outward 10–90% crossing width **0.88 R☉**, with mapped mass
3.0040 M☉; the ~0.9-R☉ conclusion is stable at this level. The FLD
luminosity overshoots before settling outward, so widths are explicitly
measured at the *first outward* 10% and 90% crossings rather than using
monotonic interpolation on the whole array. Recomputing the older
candidate widths with this definition left their quoted values unchanged.
The transition is far below a 256³ cell either way. The continuum
diagnostic is not Athena's discrete Cartesian FLD operator.

For the selected Henyey trial, the raw `rho v²` pressure scale falls
from roughly 743 to 0.002 dyn/cm² over 815–819 R☉. A finite-difference
estimate of its gradient is about `-3.0 rho*g`; even an isotropic
one-third factor gives about `-1.0 rho*g`. The estimate uses a profile
whose force equation excludes that pressure and should **not** be treated
as a solved turbulent HSE. It shows why inserting local MLT pressure
without a jointly relaxed flow would be risky. [Ludwig et al. (1999)](
https://adsabs.harvard.edu/pdf/1999A%26A...346..111L) likewise discuss
the spurious sharp pressure gradient at a local-MLT convective boundary;
[MESA's documented default](
https://docs.mesastar.org/en/stable/reference/controls.html) sets its
`mlt_Pturb_factor` to zero.

## Next physical gate

The local 1D closure is now tested in both efficient and radiatively
cooled forms. Both place the radiation/enthalpy handoff on a layer much
thinner than a 256³ cell, and neither initializes the 3D correlated
convective flux. The next useful 3D preparation should therefore focus on
how the mean thermal state and convective correlations grow *together*
under the actual finite-volume FLD operator. A bounded preparation source
may remain during that development, as the user has allowed, but it must
be removed only after measured convective transport replaces it. If a
different surface treatment or resolution strategy is proposed, first
audit its mapped force and discrete FLD face flux and cost. The earlier
one-quadrant SMR test and whole-shell cell-count estimates are in
`wor122_next_startup_gate_20261117.md`; this thinner Henyey transition
would require still finer coverage.

Figure: `doc/figures/wor122_henyey_mlt_20261118.png`.
Selected profiles and `scan_summary.json`:
`tst/nrfld_validation/wor122_henyey_mlt_20261118/`.

Reproduce the selected 1D candidate with:

```sh
python3 tools/rsg/make_mlt_rsg_hse_trial.py \
  --eos data/eos/rsg_tomida2013_material_eos.tab \
  --opacity data/opacity/rsg_gray_opacity_wor55.h5 \
  --mlt-option henyey --alpha 2.366 \
  --points 3072 --atmosphere-points 1536 \
  --output tst/nrfld_validation/wor122_henyey_mlt_20261118/alpha2p366_fine_base.dat
python3 tools/rsg/make_lte_fld_exterior_trial.py \
  --base tst/nrfld_validation/wor122_henyey_mlt_20261118/alpha2p366_fine_base.dat \
  --eos data/eos/rsg_tomida2013_material_eos.tab \
  --opacity data/opacity/rsg_gray_opacity_wor55.h5 \
  --join-end-rsun 900 \
  --output tst/nrfld_validation/wor122_henyey_mlt_20261118/alpha2p366_fine_fld.dat
python3 tst/nrfld_validation/wor122_plot_henyey_mlt_20261118.py
```
