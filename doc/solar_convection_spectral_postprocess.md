# WOR-109: 3D convection spectral post-processing

The script `tst/nrfld_validation/solar_convection_spectral_postprocess.py` solves
the monochromatic LTE formal-transfer equation along rays through full 3D
snapshots. It is intended as a diagnostic of the completed solar-convection
calculation. It does not change the grey FLD hydrodynamics.

## Inputs and provenance

- A completed campaign `state.json` and paired `prim_full` / `uov_full` Athena++
  HDF5 dumps. The script selects the last N **completed** segments; it rejects
  a mismatch between their times.
- The grey opacity table used by the campaign, with the `source_points` group.
- The matching Optab `mono_XXXXX.h5` files from
  `optab/sample/rhoT_3p0_6p3/output`. The script verifies each selected
  temperature and density **and** its Rosseland mean against the grey table.
  The three arrays `abs`, `line`, and `sca` are extinction coefficients in
  cm^-1. The input `fort.5` enables H-/H2-/atomic continuum, Kurucz and
  molecular lines, Rayleigh scattering, and electron scattering. The local
  Optab source checkout is commit `36128bb30978839cc64a3253a7c1c139ce498bf7`
  and its repository `LICENSE` is GPL-3.0. Upstream atomic/molecular
  database redistribution terms were not audited; the raw 12 GB opacity set
  remains in its local Optab directory and is not committed to Athena++.
- STScI's CALSPEC `sun_reference_stis_002.fits`, downloaded from
  <https://ssb.stsci.edu/cdbs/current_calspec/sun_reference_stis_002.fits>.
  Its flux at 1 AU is scaled to the solar surface using nominal AU and solar
  radius. Optical data include measured spectra; 0.96–2.695 μm is partly a
  model. This reference is for a solar comparison, not a fit target.
- The Pecaut–Mamajek dwarf sequence, downloaded from the
  [author-maintained table](https://github.com/emamajek/SpectralType/blob/master/EEM_dwarf_UBVIJHK_colors_Teff.txt)
  for K1V/K2V temperature context. Its SHA-256 is recorded in `summary.json`.

The report `summary.json` records the SHA-256 of the grey table, the CALSPEC
file, and every Optab file actually read. The monochromatic set need not be
copied into the code repository; only source states intersecting the 3D box
are read.

## Method

For each wavelength, `abs + line + sca` is interpolated logarithmically on
the irregular Optab `(log rho, log T)` grid using Delaunay barycentric weights.
The script fails if a cell lies outside the source-point convex hull. The
Planck source is integrated through vertically uniform cells, with periodic
horizontal interpolation for inclined rays. The ray weights use three-point
Gauss-Legendre quadrature in μ and four azimuths; two vertical substeps are the
default. The lower boundary is a blackbody at the bottom cell temperature.
The script checks isothermal and two-layer slabs against analytic solutions.

Scattering is **thermalized** in the main transfer (`Sλ=Bλ` using total
extinction). A separate 500 nm calculation without scattering opacity and a
near-τ=1 scattering fraction indicate the sensitivity of this approximation.
This is not an iterative scattering solution. Local fluid velocities are not
used to Doppler shift lines. The Optab wavelength grid has resolving power
roughly 5400; the report's sparse wavelength selection supports broad-band
color and flux diagnostics, not precision line profiles. Grey FLD snapshots
cannot recover the temperature structure of a self-consistent non-grey RHD
simulation. No solar agreement is assumed from these calculations.

## Running the post-processor

Use an environment with NumPy, SciPy, h5py, Matplotlib, and Astropy. For example:

```bash
python3 -m venv --system-site-packages /tmp/wor109_venv
/tmp/wor109_venv/bin/pip install astropy
```

Download the solar reference into a `reference/` subdirectory of the output.
Then run from the Athena++ repository root:

```bash
python tst/nrfld_validation/solar_convection_spectral_postprocess.py CAMPAIGN_ROOT \
  --mono-dir OPTAB_ROOT/sample/rhoT_3p0_6p3/output \
  --grey-table CAMPAIGN_ROOT/data/opacity_table_rhoT_3p0_6p3.h5 \
  --solar-reference OUTPUT_ROOT/reference/sun_reference_stis_002.fits \
  --dwarf-sequence OUTPUT_ROOT/reference/EEM_dwarf_UBVIJHK_colors_Teff.txt \
  --output OUTPUT_ROOT --snapshots 3 --wavelengths 160 --substeps 2
```

Open `OUTPUT_ROOT/index.html` first. The report links plots, the CSV spectrum,
the NPZ angular intensities, and the full source manifest. The command uses
160 points in 300–2500 nm plus 60 UV/infrared tail points. The 300–2500 nm
integrated flux is a **lower bound** on bolometric flux. The 100–10000 nm
integral is quasi-bolometric; for a 5100–5800 K blackbody this interval
contains >99.8% of the flux, although the simulated spectrum has its own
upper-atmosphere limitations. It is compared with the FLD upper-boundary
flux, which is a grey RHD quantity and not an exact spectral-transfer target.

To regenerate wavelength-dependent formation heights and vertical
contribution functions, run:

```bash
python tst/nrfld_validation/solar_convection_spectral_formation.py OUTPUT_ROOT
```

This writes `formation.csv`, `contribution_functions.npz`, two plots, and
`formation_summary.json`. The contribution functions are checked against an
independent vertical formal-ray calculation.

The post-processor also writes `movie_sources.json` for the last 48 completed
segments. To regenerate the final 20-second xz/xy/mean-subtracted-xy movies:

```bash
python tst/nrfld_validation/solar_convection_campaign_movies.py \
  --source-summary OUTPUT_ROOT/movie_sources.json \
  --output-root OUTPUT_ROOT --max-frames 160 --fps 8 \
  --end-time 24000 \
  --opacity-table CAMPAIGN_ROOT/data/opacity_table_rhoT_3p0_6p3.h5
```

The resulting `movies/manifest.json` lists every source run and its SHA-256
provenance. The HTML embeds the three movies alongside the spectral plots.

For WOR-109, the completed campaign was stopped between segments at t=24000.
The final restart SHA-256 is
`909a86e74bda7fd7aa1bac35adffc6c160a17157d76789d8121fc5a77f19970f`.
The post-processor used t=23800, 23900, and 24000 on a 64×64×32 box with
96 km cells. The output is
`/mnt/d/output/rad/wor109_solar_spectral_postprocess_20261109/index.html`.

## Initial diagnostic result

- The 300–2500 nm mean emergent flux is approximately **0.610 F_sun**, versus
  **0.630 F_sun** at the grey FLD upper boundary over the same three segments.
  The 100–10000 nm quasi-bolometric synthesis is approximately **0.646
  F_sun**, or **5177 K** in effective-temperature units; the grey FLD flux
  corresponds to **5143 K**. The differences cannot be treated as an exact
  energy-conservation error because the transfer methods and upper
  boundaries differ. The CALSPEC Sun has **0.955 F_sun** in the 300–2500 nm
  band after conversion from 1 AU.
- The ratio of average 400–600 nm to average 800–1200 nm flux is **1.800**;
  the corresponding solar ratio is **2.465**. A 5177 K blackbody is close to
  the synthesized broad color. The empirical K1V/K2V temperatures are 5170/
  5100 K. These are temperature context, not a stellar classification or a
  claim that gravity and line profiles match a K dwarf.
- The 500 nm flux changes by **0.085%** between two and eight vertical ray
  substeps and by **0.071%** between three and five μ quadrature points at
  eight substeps. Halving the 160-point wavelength grid changes the 300–2500
  nm integral by **1.60%**; spectral sampling is the larger numerical error.
- Thermalizing scattering versus omitting its opacity changes 500 nm flux by
  **−0.15%**. Scattering is at most **1.62%** of extinction near τ≈1 among
  these sampled wavelengths, so it is subdominant here.
- At **397.0 and 453.6 nm**, the top half-cell optical depth exceeds 0.1;
  it reaches **2.21** at 397 nm. The domain truncates the line-forming region
  at these points. Individual blue line strengths must not be compared with
  observations until the atmosphere is extended and the transfer recomputed.
- The monochromatic τ=1 surface is inside the 3D box at 159 of 160 sampled
  300–2500 nm wavelengths. At 397.0 nm it is above the simulated top.

The broad color difference from the Sun is largely consistent with the lower
flux-derived temperature. Higher-resolution, non-grey convection and a taller
upper atmosphere are still needed before evaluating detailed solar line
profiles or limb darkening as an observational fit.

## Sources

- Hirose et al. (2022), [Optab](https://doi.org/10.1051/0004-6361/202141076)
  for monochromatic opacities and their units.
- [STScI CALSPEC](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec.html)
  for the solar reference and its observational/model provenance.
- Ibgui et al. (2013), [IRIS](https://doi.org/10.1051/0004-6361/201220468)
  for 3D LTE formal transfer without iterative scattering.
- Rempel (2014), [solar surface convection](https://arxiv.org/abs/1405.6814)
  for the simulation context and resolution sensitivity.
