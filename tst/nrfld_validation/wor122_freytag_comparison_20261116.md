# Why the current RSG startup differs from the Freytag/Chiavassa calculation

The comparison is with Chiavassa et al. (2011), A&A 535 A22, Table 1 and
Sections 2–4: https://www.aanda.org/articles/aa/pdf/2011/11/aa17463-11.pdf.
That paper explicitly reports an initially visible Cartesian imprint, a weak
velocity inherited from a previous model, and **several years** before the
convection becomes irregular and the initial symmetry disappears. The gray
12-Msun, 3-Msun-envelope, 830-Rsun model uses 235^3 cells of width 8.6 Rsun.
Its radiative transfer uses short characteristics with 6–9 substeps per
hydrodynamic step; the paper says its transfer operator is designed to remain
stable across sharp temperature and opacity jumps. It does not claim an
exactly hydrostatic 3-D startup or a quiet photosphere at day seven.

Our current WOR-55 LTE/FLD profile uses the same nominal luminosity, potential
mass, envelope mass, and radius, but a **different** initial atmosphere and
numerical transport. The FLD-matched exterior required 3227.8 K at the
nominal 830-Rsun surface, rather than the comparison model's mean 3487 K;
that boundary temperature has not been independently validated as a physical
photosphere. The uniform 256^3 cell width is 7.89 Rsun, close to the paper's
8.6 Rsun, so the grid-point count by itself does not explain the difference.

The *measured* startup problem is more specific:

- The 1-D radiative luminosity changes from about 0.01 Lstar in the outer
  convective interior to about Lstar over 3.35 Rsun, while the initial 3-D
  convective enthalpy flux is nearly zero. The local cooling time is about
  1.07 days near 816 Rsun. This is an initial energy-flow imbalance, even
  before any solver instability is considered.
- Mapping this profile to a 7.89-Rsun Cartesian grid spreads the FLD face
  flux inward. The initial discrete shell estimate at 800–813 Rsun is
  0.136 Lstar versus 0.00938 in the 1-D profile, and the exact discrete
  diffusion source is about -0.50 Lstar in that shell. The face-flux
  accounting closes; this diagnoses the *initial state on this mesh*, not
  a demonstrated bug in the diffusion solver.
- The same first stage has total force residuals of order 1–5% of gravity
  over 800–850 Rsun and about -11% in the 850–880-Rsun axis sample. The
  observed velocity increment matches the measured force residual. Gas
  pressure discretization dominates the radiation-force discrepancy in
  these bins. The core flow at t=20–28 is nearly isotropic, while the
  outer flow is angularly structured. Stronger core drag reduced core RMS
  speed in a short control without fixing the early surface signal.
- The tested one-quadrant SMR patch did not cover the whole photosphere;
  even its 3.95-Rsun fine cells were wider than the transition. Its short
  gate did not improve the global maximum Mach, so this is not an available
  whole-star remedy yet.
- At 256^3 with shell thermal support and PPM, the completed t=28 state is
  only 7.13 days old (~0.056 dynamical times). Its maximum Mach is 0.121,
  box-mass drift over t=20–28 is -1.84e-8, and all implicit solves pass.
  This is **not yet a failed run**. The artificial thermal source still
  supplies roughly Lstar; the 800–830-Rsun convective enthalpy covariance
  is only 0.00156 Lstar. The source cannot yet be removed, and a stable
  physical year-scale evolution has not been demonstrated.

Primary measurements: `wor122_rsg_discrete_initial_fld_audit_20261115.md`,
`wor122_rsg_initial_surface_force_20261115.md`,
`wor122_rsg_surface_gas_decomposition_20261115.md`, and
`wor122_rsg_prep_thermal_20261115.md` in this directory.

## Decision gates

1. Let the **already submitted** 256^3 t=28→36 continuation (XD job 340968,
   M-bulk-a) and its dependent analysis (340969, M-debug) complete. Track
   maximum Mach, shell RMS speed and temperature, mean core expansion,
   radiative and convective luminosity, source power, box mass, and solver
   status. Do not infer a one-year trend by linearly extrapolating seven days.
2. A preparation model must simultaneously satisfy the target mass, gas plus
   radiation force balance on the *actual* 3-D mesh, and acceptable FLD face
   flux divergence. A wider radiative transition is useful only if its
   pressure, density, temperature, and mass are re-solved together. A
   one-parameter width edit is not assumed valid.
3. When convection carries a substantial fraction of Lstar and the
   surface/core signals no longer grow systematically, taper the artificial
   shell heat over a measured thermal/dynamical time and verify a source-free
   continuation. A quiet, perfectly hydrostatic atmosphere is not the target;
   bounded mass loss and statistically stationary mean radius/luminosity
   over a meaningful fraction of a year are.

## Local transition-width experiment

A temporary opt-in modification of `tools/rsg/make_gray_rsg_hse.py` blended
the radiative flux toward the total luminosity with a smoothstep over a
specified distance below the nominal 830-Rsun surface. The temperature
gradient and gas pressure were re-solved against that flux, followed by
shooting the superadiabatic peak and mapped mass. The code change was
**removed** after the checks below; the evolved Athena++ equations were
never changed. The test illustrates why a one-parameter profile edit is
not an acceptable startup model.

With the same Tomida material EOS, WOR-55 opacity, 3227.8-K nominal surface,
and 64^3 mass feedback, a 16-Rsun window generated a profile with mapped
mass 3.01263 Msun (the existing tolerance is 0.5%). After applying the same
LTE/FLD exterior construction, the 10–90% continuum FLD luminosity
transition was 814.318–817.524 Rsun, or 3.206 Rsun, versus
814.215–817.567 Rsun, or 3.352 Rsun, in the current profile. Thus this
short blend hardly changes the actual problematic layer. The local temporary
profiles are `/tmp/wor122_broad16_{base,lte_fld}.dat`.

A 32-Rsun window failed the 3-Msun interior mass shoot altogether: even at
zero superadiabatic excess the enclosed mass was exp(-1.2746) × 3 Msun,
about 0.84 Msun, and increasing the peak lowered the mass further. A wider
transition therefore requires additional stellar-structure degrees of
freedom, rather than a one-parameter edit at fixed surface state. Its failed
run wrote no profile. An intermediate 24-Rsun window did satisfy the mapped
mass tolerance (3.00953 Msun) and broaden the 10–90% transition to
810.082–819.027 Rsun, a width of 8.945 Rsun. That is only about 1.13
uniform-256 cells, and the finite-difference 1-D HSE residual at
750–830 Rsun worsened from 0.60% to **28.1%**. This candidate was rejected
before any XD job. Its temporary profile is
`/tmp/wor122_broad24_{base,lte_fld}.dat`; 32 Rsun produced no profile.
