# WOR-124 paper application package (2026-11-22)

## Scope and source

This package closes the remaining static-figure organization for the rotating,
magnetized protostellar-collapse application.  It does not change or rerun the
solver.  The simulation source remains the frozen tag
`protostellar-firstcore-v2` at `577f273bddee4128d873cd7da7a4b1495b97ba8c`.
The final AMR+1 state is XD job `344378`, cycle 2400, at 29.4721 kyr.

The reproducible figure generator is
[`paper_collapse_application_figures.py`](paper_collapse_application_figures.py).
It consumes the previously audited 75-snapshot movie manifest and the final
c2400 HDF.  The local browsable package is
`/home/kosuke/simulation/test/collapse_rotmhd_20261121/`
`paper_amr12_v2_job344378/paper_application/index.html`.

## Recommended paper figures

1. `final_spatial_distribution_20au.{pdf,svg,png}` is the primary final-state
   figure.  It shows xy and xz density and gas-temperature slices, centered on
   the density maximum, over +/-20 au.  Temperature panels carry density
   contours at `1e-14`, `1e-12`, and `1e-10 g cm^-3`.
2. `central_temperature_density.{pdf,svg,png}` is the primary evolution
   figure.  Every point uses the gas temperature in the exact cell containing
   the density maximum.  It contains 75 real snapshots from c60 to c2400 and
   explicitly marks the c1800 AMR+1 continuation and final endpoint.  The
   source values are in `central_temperature_density.csv`.
3. `final_spatial_distribution_50au.{pdf,svg,png}` is a supplementary wider
   view.  It makes the flattened envelope and axial/equatorial contrast easier
   to see without replacing the better-resolved +/-20 au main figure.
4. The matched-density AMR figure remains the numerical-resolution figure;
   the core diagnostics and 2-D/3-D movies are supplementary material.

Suggested scoped claim: conservative AMR-FLD with a tabulated material EOS,
variable opacity, rotation, and ideal MHD reaches the first-core density regime
stably.  The endpoint should be described as a *first-core density candidate*,
not as proof of a time-stationary hydrostatic object.

## Quantitative endpoint and resolution gate

- `rho_max = 1.0041786e-10 g cm^-3`;
- gas temperature in the density-maximum cell: `336.025 K`;
- volume-weighted inner sonic transition: `6.7384 au`;
- enclosed mass at that transition: `0.0204525 Msun`;
- minimum segment-endpoint Jeans resolution: `31.98` cells;
- all 610 AMR+1 Newton solves and 1,839 FGMRES solves converged;
- no accepted-state opacity-table excursion was found.

Adding one AMR level changes peak density by -0.76%, peak-cell temperature by
-0.68%, mass within 5/10/20 au by -1.43%/-0.64%/-0.52%, and the common-bin
density/temperature/infall-proxy profiles by 1.01%/0.85%/1.70% in relative L1.
This closes the intended resolution-sensitivity gate.

## Literature comparison boundary

- Mayer et al. (2025) report a 6--7 au first-core scale and 300--400 K central
  phase-space temperatures around the `rho > 1e-10 g cm^-3` stage.  The
  present 6.74 au and 336 K agree at the level relevant to this application.
  The magnetic calculation is not a one-to-one replica: it uses 32.25 microG
  (`mu` about 20) and ideal MHD, versus Mayer's 129 microG (`mu=5`) and
  non-ideal MHD setup.
- Commercon et al. (2011) give a representative first-core scale of about
  8 au and `2e-2 Msun`; the present 6.74 au and `0.02045 Msun` are consistent.
- Tomida et al. provide the realistic-EOS radiation-MHD collapse sequence as
  physical context, but their different setup and continuation through second
  collapse are not used as a pointwise validation target.
- Vaytet et al. support grey transport as a controlled approximation in the
  cooler first-core regime.  This supports the FLD scope at 336 K but does not
  establish identical initial-condition evolution.

The comparison supports scale and qualitative morphology.  It does not claim
identical field topology, outflow launching, second-core evolution, or a
time-stationary force balance.

## Cost accounting

The final AMR+1 production segment used five M-large-a nodes / 320 MPI ranks:
1:04:33 wall time, 5.38 node-hours, 610 cycles, and approximately
`1.4786e9` leaf-cell-times-cycle updates.  The Slurm step recorded about
21.1 GiB maximum RSS.

The actual successful initial-to-endpoint lineage is about 16.5 node-hours,
about `1.05e3` rank-hours, and `4.5432e9` leaf-cell-times-cycle updates.  If
the complete lower-resolution production evidence used for the matched
comparison is included, the paper evidence totals about 20.4 node-hours and
`1.31e3` rank-hours.  Development failures, strong-field stress tests, and
rotation-only experiments are excluded from these production numbers.

A short presentation-ready summary is: *the final high-resolution continuation
took about one hour on five nodes; the successful initial-to-final lineage cost
about 16 node-hours; including the resolution comparison, the paper evidence
cost about 20 node-hours.*

## Validation and provenance

Generation completed locally in 10.2 s with 595,780 KiB peak RSS.  The CSV has
75 unique, strictly increasing cycles and times; peak density is monotonic from
`8.49888e-18` to `1.00418e-10 g cm^-3`.  Both spatial scales use the same final
metadata.  All panels, titles, labels, and vector text were checked in the SVG;
the PNG figures and single-page PDFs were also inspected.

Key hashes after final generation:

- movie manifest: `010a08d...` (full value in `figure_manifest.json`);
- final HDF: `943f4714...` (full value in `figure_manifest.json`);
- `central_temperature_density.csv`:
  `334285910f765e58c1fe25294abd89e0fa6495510171d297de39f5b4b86d83bf`;
- full source paths, hashes, first/final records, display assumptions, and
  limitations: `paper_application/figure_manifest.json`.

No additional collapse simulation is required for the scoped paper claim.
Remaining work is manuscript-level selection, caption wording, and any
co-author-requested cosmetic changes rather than unresolved solver validation.
