# WOR-14/122: discrete Henyey-MLT local-patch screen (2026-11-19)

## Question

The mass-matched Henyey-MLT 1D candidate closes its *continuum* radiative
plus convective luminosity but has a roughly 0.9 R☉ surface handoff. Test
whether resolving that handoff in the actual Athena FLD discretization is
enough to prevent initial cooling when the profile is mapped statically.
This is a screen of an initial condition, not a convection simulation.

## Setup

The candidate is
`wor122_henyey_mlt_20261118/alpha2p366_fine_fld.dat` (α=2.366,
Tomida material EOS, WOR-55 gray opacity, 12 M☉ potential). Its density,
gas pressure and gravity were exported with
`tools/rsg/export_rsg_surface_patch_profile.py` to
`data/ic/wor122_henyey_mlt_patch_830.*`. The local plane-parallel patch
covers 800–860 R☉. Horizontal dimensions are 16 cells on each grid, with
periodic sides; the vertical domain has 96, 192 or 384 cells. The top
radiation boundary is Marshak vacuum; the lower gas boundary reflects.
There is no heat source, initial velocity or perturbation. FLD, radiation
force, and gas-radiation coupling are on. The different horizontal widths
make this a vertical-resolution screen, not a comparison of convection
modes. All runs used four local MPI ranks and CFL=0.015.

The initial exact NR-FLD face coefficients were audited in one step; the
gas-pressure flux, radiation-force source and imported gravity were compared
at 815–817 R☉. The same setups were continued for 0.5 code time units
(3.06 hours). Each Newton solve converged; the opacity table recorded no
out-of-domain or nonfinite result. Raw runs and logs are under
`/tmp/wor122_mlt_dx*_r800`, `/tmp/wor122_mlt_dx*_force`, and
`/tmp/wor122_mlt_dx*_short`.

## Results

| Δx [R☉] | Cells across 0.9 R☉ handoff | Initial FLD divergence / Er at 813–817 [code⁻¹] | Eg / abs(initial FLD divergence) [h] | T(0.5)/T(0) at 813–817 | Max Mach at 0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.625 | 1.4 | −34.18 | 39.9 | 0.95058 | 0.0097 |
| 0.3125 | 2.9 | −31.57 | 43.2 | 0.94922 | 0.0111 |
| 0.15625 | 5.8 | −29.49 | 46.8 | 0.95087 | 0.0119 |

The gas-energy timescale is a diagnostic ratio, not a solved cooling law;
the measured gas temperature confirms the sign and scale of the early
change. A fourfold reduction in cell width changes the three-hour cooling
by less than 0.2 percentage points. The flow is still near rest when the
surface transition has already lost about 5% of its temperature.

The initial radiative luminosity in 813–817 R☉ is only
0.24–0.27 L★ on these mapped grids, while it approaches L★ above 817 R☉.
Without the convective flux postulated by the 1D closure, this is a large
finite-volume luminosity divergence. Making the radiative handoff several
cells wide does not create the missing 3D enthalpy transport.

| Δx [R☉] | Mean abs force residual / ρg, 815–817 | Max abs force residual / ρg, 815–817 |
| ---: | ---: | ---: |
| 0.625 | 1.11% | 1.58% |
| 0.3125 | 0.98% | 2.94% |
| 0.15625 | 1.54% | 6.28% |

This force check uses Athena's initial gas momentum-flux divergence and
radiation-force source (`FLD_FORCE_VOLUME`) plus the imported gravity at
each cell. The maximum is sensitive to sampling the sharp transition and
does not converge monotonically. A percent-level discrete force mismatch
still needs attention before a long run, but it does not explain the
resolution-independent cooling that occurs while Mach remains below 0.012.
The static mapping also has no Reynolds-stress support, whereas the local
MLT scale near 816 R☉ is close to the gas pressure; simply inserting that
pressure into static HSE would create a new sharp force gradient.

An illustrative equal-area, isobaric two-stream check using the same EOS
requires hot-stream speeds above the sound speed to carry the missing
luminosity around 813–816 R☉ even for 30–50% temperature contrasts. For
example, at 813 R☉ a sonic pair with ±30% contrast transports only
0.41 L★. This is **not** a general upper bound on 3D convection because
real upflow/downflow areas and entropy contrasts differ. It shows why a
small coherent velocity perturbation cannot be assumed to initialize the
MLT flux.

## Decision and next gate

The static Henyey-MLT mapping fails the local full-FLD start gate at all
three tested resolutions. The dominant early defect is missing correlated
convective transport, followed by the unresolved mechanical role of
turbulence. A new 3D precursor must develop the mean stratification and
velocity–enthalpy correlations together while the surface is supported by
a controlled preparation boundary/source. Its physical continuation must
remove that support and retain shell luminosity, transition temperature,
mass and bounded Mach for multiple local cooling times. The present data
do not justify a whole-star XD run.

Compact measurements: `wor122_mlt_patch_discrete_20261119.json`.
Figure: `doc/figures/wor122_mlt_patch_discrete_20261119.png`, made with
`python3 tools/rsg/plot_wor122_mlt_patch_discrete.py`.
Local reproduction script:
`tst/nrfld_validation/wor122_mlt_patch_resolution_local.sh`.
From the repository root, regenerate the patch profile and run the screen
with a locally configured Athena binary:

```sh
python3 tools/rsg/export_rsg_surface_patch_profile.py \
  --profile tst/nrfld_validation/wor122_henyey_mlt_20261118/alpha2p366_fine_fld.dat \
  --output-prefix data/ic/wor122_henyey_mlt_patch_830 \
  --reference-rsun 830
bash tst/nrfld_validation/wor122_mlt_patch_resolution_local.sh \
  /absolute/path/to/athena /tmp/wor122_mlt_patch_resolution
```
