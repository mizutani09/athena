# WOR-122: surface thermal readjustment and initial cell averages (2026-11-15)

This follows the force decomposition in
`wor122_rsg_surface_gas_decomposition_20261115.md`. All comparisons here
use the WOR-55 tabular opacity, Tomida material EOS, the same 1-D LTE/FLD
profile, and a uniform 256³ Cartesian grid. One code-time unit is 0.255
day. Unless stated otherwise, `xorder=3` (PPM) is fixed. Large HDF files
remain under `$WORK/wor122/`.

## What changes during the first 0.255 day?

Analysis job **340080** measured initial-mass-weighted geometric mean
field ratios in complete spherical shells using the saved primitive and
user-variable HDF files. At code time 1, PLM and PPM show nearly identical
thermal changes even though their early force residuals differ:

| Radius [R☉] | PLM ΔTgas | PPM ΔTgas | PLM ΔEr | PPM ΔEr |
| --- | ---: | ---: | ---: | ---: |
| 800–813 | −0.74% | −0.75% | −2.88% | −2.89% |
| 813–817 | −8.93% | −8.99% | −30.41% | −30.60% |
| 817–830 | +1.88% | +1.85% | +7.91% | +7.80% |
| 830–850 | +2.89% | +2.86% | +11.90% | +11.80% |

The transition layer loses radiation energy and cools the gas, while the
adjacent exterior gains radiation energy and warms. The gas and radiation
temperature changes are broadly consistent with tight thermal coupling.
This pattern explains why improving the *initial* momentum force alone
does not settle the subsequent surface motion. The history maximum Mach
rises similarly for PLM and PPM throughout the interval; their boundary
luminosities also agree to within about 0.1% relative at code time 1.

## Diffusion-off diagnostic

Job **340081** repeated the PPM calculation to code time 1 with only
`fld/cut_diff=true`. It finished normally in 1m29s, with Newton convergence
and zero opacity clamps. This is a mechanism control, not a physical RSG
model: the implicit diffusion operator and its Marshak boundary flux are
both removed, while gas-radiation local exchange remains. Analysis job
**340082** measured its complete spherical shells.

| Code time 1 | PPM + diffusion | PPM, diffusion off |
| --- | ---: | ---: |
| 813–817 R☉ ΔTgas | −8.99% | −0.004% |
| 813–817 R☉ ΔEr | −30.60% | +1.15% |
| 813–817 R☉ RMS `|Δv|/(Δt g)` | 4.83% | 0.98% |
| 830–850 R☉ ΔTgas | +2.86% | approximately 0% |
| 830–850 R☉ ΔEr | +11.80% | −0.14% |
| 830–850 R☉ RMS `|Δv|/(Δt g)` | 6.92% | 1.67% |
| Global maximum Mach | 0.004311 | 0.004482 |

Thus diffusion is necessary for most of the rapid **surface-shell**
thermal redistribution and velocity increment. It does not explain every
velocity maximum: the global maximum Mach is slightly higher in this
control. Turning diffusion off is not a viable stability prescription.

The old RSG history callback reported a nonzero formal Marshak luminosity
even when `cut_diff=true`, although the implicit operator applied zero
boundary diffusion flux. Its code-time-1 value of `Lrad/Lheat=0.613`
in job 340081 is **not** transported luminosity and must not be used as
an energy budget. The local source now returns zero for that history
quantity when diffusion is cut. A local 64³ one-step regression compiled
successfully and recorded exactly zero on all six luminosity faces.

## Initial point samples versus cell averages

The pgen initializes conservative fields by evaluating the 1-D profile
at the Cartesian cell center. This is inaccurate in the 3.35 R☉ radiative
transition when a 256³ cell is 7.89 R☉ wide. A positive 4×4×4 Gauss
cubature of the profile over the selected axis cells gives the following
mean cell-average / point-value ratios:

| Radius [R☉] | 128³ Er | 256³ Er | 256³ gas internal energy | 256³ density |
| --- | ---: | ---: | ---: | ---: |
| 813–817 | 1.367 | 1.099 | 1.049 | 0.988 |
| 817–830 | 1.615 | 1.146 | 0.999 | 0.990 |
| 830–850 | 1.091 | 1.019 | 1.048 | 1.045 |

The Er correction is large and resolution dependent. A 4-point per-axis
rule differs from an 8-point rule by at most 0.5% in the inspected 256³
813–830 R☉ cells, with 0.1% RMS difference. The script
`wor122_profile_cell_average_audit.py` and compact shell JSON preserve
the calculation. These values alone do not tell whether cell averaging
will improve the *discrete FLD flux*: the face gradient and limiter also
change when Er changes.

An opt-in `problem/profile_cell_average_trial` now initializes density,
gas internal-energy density, and radiation-energy density using this
positive cubature on uniform grids; it is off by default and rejects
multilevel grids and generic initial-average corrections. A local 64³
one-step PPM pair passed Newton and opacity gates but increased
`Lrad/Lheat` from 1.821 (point sample) to 2.970 (cell average), despite a
small Mach reduction. XD job **340087** repeated the one-step pair's
cell-average side at 256³ and completed in 53 s with zero opacity clamps.
Its `Lrad/Lheat` rose from the matched point-sample 0.78184 to 0.79758,
while maximum Mach changed from 0.0004414 to 0.0004378. The initial HDF
reproduces the offline cubature ratios in selected surface cells.

Job **340088** extended only this 256³ cell-average candidate to code time
1 and completed normally in 3m10s with Newton convergence and zero opacity
clamps. Analysis job **340091** read all 4096 blocks of both the one-step
and time-1 HDF pairs. Compared with the matched point-sampled PPM run:

| Gate | Point sample | Cell average |
| --- | ---: | ---: |
| time 0.1: 800–813 R☉ RMS `|Δv|/(Δt g)` | 1.71% | 8.44% |
| time 0.1: 813–817 R☉ RMS `|Δv|/(Δt g)` | 1.00% | 3.75% |
| time 0.1: 817–830 R☉ RMS `|Δv|/(Δt g)` | 1.74% | 7.37% |
| time 1: boundary `Lrad/Lheat` | 1.08347 | 1.17651 |
| time 1: maximum Mach | 0.004311 | 0.004243 |
| time 1: 813–817 R☉ ΔEr | −30.60% | −23.93% |
| time 1: 813–817 R☉ ΔTgas | −8.99% | −9.43% |
| time 1: 830–850 R☉ ΔEr | +11.80% | +18.70% |
| time 1: 830–850 R☉ RMS `|Δv|/(Δt g)` | 6.92% | 6.92% |

The cell average is a better representation of the imported conservative
profile, but **it is not a discrete steady state**. It increases the
first-step force response in the inner transition by factors of roughly
4–5, raises boundary luminosity by 8.6% relative at time 1, and increases
warming of 830–850 R☉. Its slightly smaller maximum Mach and reduced
relative Er loss in 813–817 R☉ do not make it a stable prescription.
The opt-in remains a diagnostic trial, off by default; it must not be
selected for a long RSG run on this evidence.

## Interpretation and next gate

The original 1-D LTE/FLD profile is close to continuum HSE, but mapping
onto the 3-D grid does not make it simultaneously stationary under the
discrete Godunov force and discrete FLD diffusion. The present evidence
does **not** show that the radiative transient is a solver bug. A reliable
initial model needs an audit of the actual discrete FLD face flux and its
divergence, including the Marshak boundary, and a surface grid with
several radial cells through the 3.35 R☉ transition. Only after these
gates should relaxation/damping or longer evolution be compared.

The thermal comparison figure is
`doc/figures/wor122_rsg_surface_thermal_response_20261115.png`.
Compact outputs are in `tst/nrfld_validation/wor122_surface_force_20261115/`
(`shell_evolution256.json`, `nodiff_shell_evolution256.json`,
`nodiff_global_t1_256.json`, `cellavg_global256.json`,
`cellavg_shell_evolution256.json`, and `cellavg*axis.json`).
The cut-diffusion run is under `$WORK/wor122/ppm256_nodiff_t1/`; the
cell-average trial uses an isolated source snapshot at
`$WORK/wor122/athena_cellavg/`, with outputs in
`$WORK/wor122/cellavg256_t01/` and `$WORK/wor122/cellavg256_t1/`.
Local 64³ smoke outputs are under
`/home/kosuke/simulation/test/wor122_{point,cellavg}64_t01/`.
