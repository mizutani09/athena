# Protostellar collapse figure candidates, 2026-11-16

Open [review.html](review.html) to compare the three candidates. The proposed
main paper pair is `collapse_final_radial.pdf` plus
`collapse_evolution.pdf`; the xy slice is an optional visual supplement.

## Why these diagnostics

- [Mayer et al. (2025), Fig. 7](https://arxiv.org/html/2510.12620v4)
  juxtaposes first-core radial density, gas temperature, radial velocity,
  and a density-temperature phase plot. Its Figs. 8 and 9 show thermal
  evolution for the more general RMHD case. We use these **diagnostic
  quantities**, not its numerical curves as a matched reference: Mayer's
  Section 5 uses ideal EOS and constant opacity.
- [Tomida et al. (2015), Figs. 1, 2 and 4](https://arxiv.org/html/1501.04102v1)
  show central density/temperature history, a density-temperature track,
  and final radial profiles. This motivates the two proposed main figures.
- [Vaytet et al. (2012)](https://arxiv.org/abs/1205.5143) provides context
  for why a grey radiation method and realistic opacity may be useful in
  first-core collapse. We make no quantitative agreement claim from it.

## Proposed captions

**Final spatial structure.** Radial structure at cycle 1080
(`t=29.11046 kyr`, `rho_max=1.04414e-10 g cm^-3`) in the three-dimensional
Cartesian AMR collapse with tabulated H/He EOS and temperature- and
density-dependent Semenov opacity. Panels show shell-mean gas density,
mass-weighted gas temperature together with the radiation-equivalent
temperature `(<E_r>/a_r)^1/4`, and mass-weighted radial velocity. The pale
band marks the `7.507–7.874 au` accretion-shock candidate from the **+x-axis**
profile; it is a guide, not a measured spherical shock surface. The density
break and slower inner flow identify a first-core candidate.

**Evolution.** Maximum gas density over the complete `0–29.11046 kyr`
calculation, maximum gas temperature during the final `28.5–29.12 kyr`,
and the same two domain maxima paired at each history output. The trajectory
shows a nearly isothermal initial stage followed by rapid heating to
`434.524 K`. `rho_max` and `T_gas,max` are independently recorded spatial
maxima, **not** an exactly co-located cell or a Lagrangian fluid-element
track. In the final HDF5 output, the cells of maximum density and maximum
temperature coincide. This figure demonstrates evolution with the chosen
table-based thermodynamics; it does not alone validate the EOS or opacity
tables against an independent physical reference.

**Optional slice.** Gas density and table-EOS gas temperature on the native
leaf-cell plane containing `z=0`, shown in a `64×64 au` view. The renderer
rasterizes cell values to pixels without smoothing. It is useful for showing
the compact approximately spherical core but adds little to the two main
diagnostic figures.

## Scope and provenance

The source is `ff60f8f07842acb00e7753932bcfe30867e3a66e` with `src/`
tree `c4ab200383c7317a2feccc50aa9578d5cf71478d`, identical to the
collaborator tag `protostellar-firstcore-v1`. XD jobs
`338469/338485/338486` comprise the c0-to-c1080 trajectory. The source,
input, EOS table, final HDF5, history, and radial CSV hashes are recorded in
[manifest.json](manifest.json). This run used CFL 0.3 and bounded-inexact
FGMRES with an accepted true relative residual cap of `1e-7`; Newton
converged, gas-mass variation was at most `5.28e-15` relative, the minimum
Jeans resolution was `31.59` cells, and the EOS/opacity domain diagnostics
were clean. The ambient fixing is an intentional source/sink, so total
energy conservation is not claimed. H2 dissociation and the second core
lie outside this result.

The opacity is a Semenov Planck **extinction** table used as an absorption
proxy. The figure shows a successful calculation with that stated model;
it should not be captioned as a physical measurement of opacity or as an
independent validation of its definition. Direct Mayer profile agreement
is outside the paper claim and is tracked separately in WOR-96.

## Rebuild

The committed small CSV files are derived from the verified run. The radial
CSV has LF line endings; its original archived CRLF SHA256 is
`f1d158a89c5b268beb7e2423680177c8a9b0f3fa07e4cfeea087f029a1057fce`.
The final
88 MB HDF5 and EOS table remain external research assets. Run:

```bash
python3 tst/nrfld_validation/collapse_paper_figures_v1/make_figures.py \
  --hdf /path/to/c1080_wor118_full.athdf \
  --eos /path/to/protostellar_tomida2015_material.tab
```

The script checks SHA256 values before rendering. It writes PDF, PNG,
`review.html`, and `manifest.json` in this directory. No solver source is
changed on this branch.
