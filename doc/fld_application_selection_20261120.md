# FLD paper application selection (2026-11-20)

## Decision

Use the tabulated-EOS protostellar collapse as the primary astrophysical
application.  Use the non-magnetic solar local box as a secondary, explicitly
limited numerical application if space permits.  The RSG calculation, rotating
collapse, magnetized collapse, and second-core experiments are not required by
this paper and are not evidence for the claims below.

This selection closes the application-choice gate.  It does not authorize a
merge to `master`, an official release tag, or stronger physical interpretation
without author review.

## Primary application: protostellar collapse

### Fixed figures

The primary paper pair is:

1. `tst/nrfld_validation/collapse_paper_figures_v1/collapse_final_radial.pdf`
   (SHA-256
   `d397daa7f3af45cbf24e48f02f908739418fd435cf2618a60f99f3f6f8549ee6`);
2. `tst/nrfld_validation/collapse_paper_figures_v1/collapse_evolution.pdf`
   (SHA-256
   `a5764dda106f07cf39d2eeaf5a7d07767c74df623113538393f4d583be2bcad5`).

The native-cell Cartesian slice is retained as a diagnostic supplement, not a
main shape-accuracy figure.  It visibly contains a grid-aligned imprint whose
resolution convergence has not been established.  Its SHA-256 is
`c32482e0966343d8e0f3e8956775ca0247b3cf022db73603872749da882a1b74`.
The standalone reviewer is
`tst/nrfld_validation/collapse_paper_figures_v1/review.html`; exact inputs,
external-data hashes, and regeneration instructions are beside it.

### Supported claim

The three-dimensional Cartesian-AMR calculation evolves a non-rotating,
non-magnetized one-solar-mass collapse with the stated tabulated H/He EOS and
density- and temperature-dependent opacity through a first-core candidate.
At the selected endpoint, `t=29.11046 kyr`,
`rho_max=1.04414e-10 g cm^-3`, and `T_gas,max=434.524 K`.  The final radial
structure contains a density break and slower inner flow; the +x profile puts
the candidate accretion-shock interval at `7.507--7.874 au`.

The calculation passed its recorded Newton, bounded-inexact FGMRES, Jeans,
positive-state, table-domain, and gas-mass gates.  Ambient fixing is an
intentional source/sink, so total-energy conservation is not claimed.  Semenov
Planck extinction is used as an absorption proxy and must be named as such.

Do not claim a matched Mayer et al. (2025) reproduction, second-core formation,
independent validation of the EOS/opacity tables, or resolution convergence of
the Cartesian angular structure.  The radial plots are angular averages and
hide density variations of order 20--30% in some shells outside the first core.

### Caption draft: final structure

> Radial structure of a first-core candidate at 29.11046 kyr in a
> three-dimensional Cartesian-AMR collapse evolved with a tabulated H/He
> equation of state and temperature- and density-dependent Semenov opacity.
> The panels show shell-mean density, mass-weighted gas and
> radiation-equivalent temperatures, and mass-weighted radial velocity.  The
> pale band marks the 7.507--7.874 au shock-candidate interval inferred from
> the +x-axis profile and is not a measured spherical shock surface.  Angular
> averaging conceals a grid-aligned density variation of order 20--30% in
> some exterior shells; spherical-symmetry or resolution-convergence accuracy
> is not claimed.

### Caption draft: evolution

> Evolution of the domain-maximum gas density over the complete collapse,
> the domain-maximum gas temperature during the final 0.62 kyr, and the two
> maxima paired at each history output.  The transition from the nearly
> isothermal stage to rapid heating reaches 434.524 K at the selected
> first-core endpoint.  Density and temperature are independently recorded
> spatial maxima rather than a Lagrangian trajectory, although their final
> maximum cells coincide.  The figure demonstrates stable evolution with the
> chosen table-based thermodynamics; it is not an independent validation of
> the tables.

## Secondary application: solar local convection

The defensible result is stable non-magnetic three-dimensional convection and
radiative transport with tabulated EOS, physical Rosseland opacity, and grey
FLD.  Controlled 96 km states bracket one solar flux, but the completed 48 km
open-HD2 branch has a different emergent flux.  Accordingly, the paper may show
the late 96 km flux budget and 3D flow morphology, but must not claim a
grid-converged flux, a quantitative observed-solar atmosphere, non-grey
spectral validation, or a unique asymptotic equilibrium.

The selected compact layout, if the secondary application is retained, is a
temperature/vertical-velocity x-z slice with the Rosseland optical surface,
an x-y flow slice, and the late top/input/storage energy history.  The exact
gate table and proposed caption are in `solar_application_gate_20261120.md`.
Its historical XD evidence remains hash-audited but came from a detached dirty
checkout.  Therefore scientific inclusion and adoption of the clean WOR-174
source option are separate decisions; the result must not be described as a
rerun from this publication candidate.

Suggested caption boundary:

> Non-magnetic local convection evolved with a tabulated equation of state,
> physical Rosseland opacity, and grey flux-limited diffusion.  The
> calculation demonstrates stable three-dimensional radiative-convective
> transport and a controlled near-solar luminosity at 96 km.  It is not a
> non-grey spectral validation, and the different 48 km HD2 flux precludes a
> resolution-convergence claim.

## Source and evidence mapping

| Item | Fixed provenance |
| --- | --- |
| Collapse execution source | `ff60f8f07842acb00e7753932bcfe30867e3a66e` |
| Collapse / publication `src/` tree | `c4ab200383c7317a2feccc50aa9578d5cf71478d` |
| Collapse jobs | XD `338469`, `338485`, `338486` |
| Collapse input | SHA-256 `aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6` |
| Collapse EOS / opacity | SHA-256 `742a3456...ba43` / `6c424454...2f94` |
| Collapse figures | imported as documentation-only descendants; `src/` unchanged |
| Solar scientific gate | WOR-116, limited inclusion, gate commit `8dfaa1a2a` |
| Solar clean option | WOR-174, default off, pending its mature clean-source A/B decision |

The machine-readable mapping is
`tst/nrfld_validation/publication_application_manifest_20261120.json`.
The publication baseline remains a review candidate until its source review is
approved; the application selection itself no longer depends on additional
collapse production runs.
