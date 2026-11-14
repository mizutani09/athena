# WOR-105: distance-weighted diffusion faces and follow-up tests

## Literature and interpretation

- [Eymard, Gallouët & Herbin, *Finite Volume Methods*, §2.3, Eq. 2.28](https://web.lmd.jussieu.fr/~hourdin/HDR/lmdzt/book091001.pdf)
  derives one common face flux for a discontinuous diffusion coefficient.
  Its transmissivity is `1/(d_L/D_L+d_R/D_R)`, where each `d` is the
  center-to-face distance. The paper's symbol `lambda` denotes the
  **diffusion coefficient**, not the FLD flux limiter. Equal distances
  recover a harmonic mean of `D`.
- [Pernice & Philip, *Solution of Equilibrium Radiation Diffusion Problems
  Using Implicit Adaptive Mesh Refinement*, §4](https://doi.org/10.1137/040609069)
  use a harmonic mean of the **diffusion coefficient** between equal-size
  cells, align coarse-side data through interpolation, and sum fine-face
  fluxes onto the coarse face. Their [full manuscript](https://citeseerx.ist.psu.edu/document?doi=3c01cc7198bb8df72404a653e564d720dc16fa50&repid=rep1&type=pdf)
  warns that large material jumps near AMR boundaries require care in
  placing refinement regions. This is a related composite-AMR radiation
  method, not a direct citation for our precise ghost formula.
- [OpenSn's CMFD theory, §6.7.2](https://open-sn.github.io/opensn/theory/iterative.html)
  explicitly eliminates a shared face value and gives the same
  distance-weighted resistance with projected centroid-to-face distances.
- [CASTRO/Zhang et al. 2011, §IV](https://arxiv.org/html/1105.2466v2)
  instead register coarse/fine diffusion-flux differences for deferred
  synchronization with AMR time subcycling. CASTRO's current
  [`kavg(opt=2)`](https://github.com/AMReX-Astro/Castro/blob/371ba40de0ef71963fa8f185d45c7d54e84ec041/Source/radiation/rad_util.H#L264-L295)
  uses a harmonic mean of **opacity** in the thick branch. That is
  different from the harmonic mean of `D=c/(3 sigma)` required by the
  constant-limiter, piecewise-material resistance law.

For a fine width `h` next to a coarse width `2h`, the center-to-face
distances are `h/2` and `h`. In the fixed-limiter diffusion limit,
`D_i=c/(3 sigma_i)`, so the literature formula gives
`F=-c(E_f-E_c)/[3h(sigma_c+sigma_f/2)]`. With the existing two-point
fine ghost `(E_f+2E_c)/3`, this is represented by
`sigma_face=(2 sigma_c+sigma_f)/3`. At equal-size cells, the same
resistance law gives `sigma_face=(sigma_L+sigma_R)/2`.

## Implemented trial

`ATHENA_NRFLD_EXPERIMENTAL_SERIES_RESISTANCE=1` now applies the
distance-weighted rule to **all** faces in radiation-only Cartesian Test C
with the shared-flux path. Previously it applied only on coarse/fine faces.
The flag remains opt-in. The debug face-opacity print now reports the
actual trial coefficient. Arithmetic uses weighted sums to avoid an
overflowing intermediate `2*sigma`.

The flag can also be used with a variable limiter, whose value remains
frozen during each implicit solve. There the effective opacity is passed
to the existing face closure. This is an exploratory FLD definition:
Eymard's linear-diffusion derivation does **not** prove it is equivalent
to solving the two nonlinear FLD half-cells. No production default changed.

## Exact two-material eigenmode, shifted material boundary

The analytic mode from
[`wor105_layered_eigenmode_20261106.md`](wor105_layered_eigenmode_20261106.md)
now accepts a symmetric slab `[a,1-a]`, while the AMR region stays
`[0.25,0.75]`. `a=0.375` puts both material jumps at cell faces
within the fine region; `a=0.125` puts them at cell faces within the
coarse region.
Results below are volume-weighted L1 errors at `t=0.003`, `kappa=4`,
`lambda=1/3`, with double-precision HDF5 output.

| Material left `a` | Density ratio | Base N | Shared `kavg` | Global resistance | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.375 | 0.1 | 64 | 3.268e-5 | **8.789e-7** | 37.2x |
| 0.375 | 0.1 | 128 | 1.519e-5 | **2.252e-7** | 67.4x |
| 0.375 | 10 | 64 | 1.138e-4 | **5.061e-7** | 225x |
| 0.375 | 10 | 128 | 5.157e-5 | **1.329e-7** | 388x |
| 0.125 | 0.1 | 64 | 2.113e-4 | **1.691e-6** | 125x |
| 0.125 | 10 | 64 | 2.762e-5 | **5.939e-7** | 46.5x |

For the fine-cell interior jump, the N64-to-N128 resistance error falls
by factors 3.90 (ratio 0.1) and 3.81 (ratio 10), close to second-order
convergence. With the old trial, which modified only coarse/fine faces,
its output was exactly identical to shared `kavg` at `a=0.375`.
After the all-face change, the originally aligned `a=0.25` N64 results
reproduced the previous series values to the shown digits: `2.221e-6` (ratio 0.1)
and `1.795e-7` (ratio 10). All resistance runs were positive,
strictly Newton-converged, and conserved their periodic integral to
roundoff.
The shifted `a=0.375`, ratio-0.1, N64 series case also ran on four MPI
ranks: its L1 error was `8.789285e-7`, identical to the serial result
to the shown digits, with a diffusion ledger of `3.01e-21`.

## Smooth and variable-limiter stress tests

The existing one-step Gaussian comparison was extended with a smooth
material transition (`interface_width=0.02`), transverse amplitude
0.3, `kappa=4`, and N64. Relative to the projected uniform-fine
solution, the one-step-change L1 errors were `2.309e-6` (shared)
versus `2.378e-6` (resistance) for ratio 0.1, and `1.071e-6`
versus `1.135e-6` for ratio 10. Both conserved. These are small,
float32-output differences, so they do not resolve which face rule is
more accurate for smooth coefficients.

For a variable limiter with `kappa=0.004`, ratio 0.1 or 10, transverse
amplitude 0.3, and N32/N64, both shared methods converged and stayed
positive and conservative. At N64, the one-step-change L1 errors against
uniform fine were:

| Ratio | Shared `kavg` | Resistance |
| ---: | ---: | ---: |
| 0.1 | 2.543e-4 | **1.235e-4** |
| 10 | 4.172e-4 | **1.414e-4** |

However, final-field L1 at ratio 0.1 was `3.103e-4` (shared) versus
`3.740e-4` (resistance), because the initial AMR-to-uniform-fine
projection error is larger than the one-step change. This is not an
analytic FLD accuracy verdict. At N32, four-rank results matched serial.
For the ratio-0.1 variable-limiter run, the relative coarse/fine face
imbalance was `4.39e-15` at `x=0.25` and `8.36e-15` at `x=0.75`;
the old path had 27.6% and 3.91%, respectively.

## Decision

The finite-volume literature directly supports distance weighting for
**linear diffusion** and material jumps. The analytic AMR tests favor
applying that rule at same-level and coarse/fine faces, not only at the
refinement interface. The experimental flag now does this. The
variable-limiter trial demonstrates convergence, positivity, MPI
agreement and conservation, but lacks a two-half-cell nonlinear FLD
derivation or an analytic accuracy reference. Keep it opt-in and do
not generalize it to gas coupling or production defaults yet.

Local outputs: `/tmp/wor105_layered_offset375` (old trial),
`/tmp/wor105_layered_offset375_global_series`,
`/tmp/wor105_layered_offset125`,
`/tmp/wor105_layered_aligned_global_series`,
`/tmp/wor105_layered_offset375_series_mpi4`,
`/tmp/wor105_series_smooth`,
`/tmp/wor105_series_smooth_dense`,
`/tmp/wor105_variable_series_thin`,
`/tmp/wor105_variable_series_thin_dense`, and
`/tmp/wor105_variable_series_face_mpi4`.
