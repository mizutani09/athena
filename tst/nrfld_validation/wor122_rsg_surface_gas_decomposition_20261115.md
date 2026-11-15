# WOR-122: mapping versus Godunov force at the RSG surface (2026-11-15)

This follows `wor122_rsg_initial_surface_force_20261115.md`. The question is
where the discrete gas-force error appears when a nearly hydrostatic 1-D
LTE/FLD profile is put on a Cartesian 3-D grid. The existing uniform 128³
and 256³ one-step XD2000 outputs were reanalyzed. Two additional **one-step**
uniform 256³ controls held the initial model, EOS, FLD, gravity, boundaries,
mesh, and timestep fixed and changed only `time/xorder`: job **340071** used
donor-cell reconstruction (`xorder=1`), and job **340072** used PPM
(`xorder=3`). The previous job **339996** used PLM (`xorder=2`). All three
256³ initial HDF primitive fields are bitwise equal in the selected axis
block. Jobs 340071/340072 completed normally in 43/45 seconds, respectively,
with Newton convergence and zero opacity clamps. The initial kinetic-to-gas
pressure ratio is at most 6.7×10⁻¹⁰ in the inspected 800–850 R☉ layer;
initial advective momentum flux is therefore negligible beside the pressure
force.

## Force decomposition

For every inspected cell, `wor122_rsg_surface_gas_decomposition.py` compares
four radial gas-support accelerations divided by local gravity:

1. **1-D derivative:** the continuum `−dP/dr` from the imported pressure
   profile, evaluated at the cell radius.
2. **Sampled 1-D, centered:** the same 1-D pressure evaluated at Cartesian
   neighboring cell centers, then centrally differenced along x, y, z.
3. **Mapped HDF, centered:** the saved initial 3-D pressure centrally
   differenced using actual neighbor cells, including across MeshBlocks.
4. **Godunov:** the code's actual first-stage gas momentum-flux divergence
   from `FLD_FORCE_VOLUME`.

All four use the **same mapped cell density** in their acceleration
denominator. Hence step 1→2 measures Cartesian sampling and the centered
derivative's truncation error; step 2→3 measures the pressure change from
mapping `ρ` and internal energy separately through the tabular EOS; step
3→4 measures the difference between a centered pressure difference and the
actual limited-face reconstruction plus HLLC momentum flux. Step 3→4 is
*not* a direct measure of HLLC error, because the finite-volume and centered
operators have different stencils. At the outer edge of the 128³ selected
block, 256 cells without both domain neighbors were excluded; none lies
in the reported radial shells.

The 256³ axis-block gas support is:

| Radius [R☉] | 1-D derivative | sampled 1-D | mapped HDF | actual PLM+HLLC |
| --- | ---: | ---: | ---: | ---: |
| 800–813 | 95.95% g | 96.41% g | 96.26% g | 96.69% g |
| 813–817 | 98.03% g | 91.84% g | 91.79% g | 100.17% g |
| 817–830 | 99.50% g | 95.52% g | 95.67% g | 100.25% g |
| 830–850 | 100.97% g | 114.82% g | 114.57% g | 94.91% g |

The mapped pressure differs from the linearly interpolated profile pressure
by only about +0.23% in the 813–817 R☉ shell and −0.52% in 830–850 R☉.
More directly, the mapped-minus-sampled **force** is −0.05% g and −0.25% g
in those shells, respectively. Thus the EOS mapping of pressure is not the
leading term. The large force changes occur when a 3.35 R☉ transition is
sampled on a 7.89 R☉ Cartesian grid and when the limited Godunov face flux
is formed. In 128³, the sampled 1-D gas support in 813–817 R☉ is only
78.93% g; it rises to 91.84% g at 256³ but is still far from the
continuum 98.03% g. This supports radial underresolution as a major cause.

## Controlled reconstruction comparison

The same 256³ axis block gives the following **total** radial acceleration
after adding FLD radiation force and gravity. Positive is outward. The
control changes only `xorder`, so the changed gas momentum flux identifies
strong sensitivity to face reconstruction. Radiation-force differences are
smaller than the gas-force differences in these bins.

| Radius [R☉] | donor cell, order 1 | PLM, order 2 | PPM, order 3 |
| --- | ---: | ---: | ---: |
| 750–800 | +0.18% g | −0.37% g | −0.10% g |
| 800–813 | −4.16% g | +0.79% g | +0.87% g |
| 813–817 | −5.02% g | +1.77% g | −1.76% g |
| 817–830 | +2.59% g | +0.65% g | −0.66% g |
| 830–850 | +11.57% g | −5.05% g | +1.93% g |
| 850–880 | +30.34% g | −10.88% g | −8.30% g |

For all three orders, the force residual still agrees with the measured
first-step `Δv_r/(Δt g)` within 0.032 percentage point over the inspected
shells. PPM reduces the 830–850 R☉ residual and the 750–800 R☉ RMS, but
the 813–817 R☉ residual merely changes sign, the 800–813 R☉ RMS increases
relative to PLM, and 850–880 R☉ remains substantially inward. Increasing
reconstruction order alone does not establish discrete HSE or a stable RSG.

At code time 0.1 (about 2200 s), maximum Mach is 5.53×10⁻⁴ for donor,
4.38×10⁻⁴ for PLM, and 4.41×10⁻⁴ for PPM. Boundary `Lrad/Lheat` is
0.78233, 0.78185, and 0.78184, respectively. The nearly identical early
luminosity does not prove later cooling will agree.

## Full-sphere first-step response and short continuation

Analysis job **340075** read *all* 4096 blocks of each 256³ one-step HDF pair
and measured the mass-weighted RMS of `|Δv|/(Δt g)` in full spherical shells.
The selected-axis block follows the general pattern, although the full
sphere gives a more robust comparison:

| Radius [R☉] | donor, time 0.1 | PLM, time 0.1 | PPM, time 0.1 |
| --- | ---: | ---: | ---: |
| 0–166 | 1.90% | 2.43% | 0.37% |
| 750–800 | 0.74% | 0.43% | 0.25% |
| 800–813 | 4.07% | 1.05% | 1.71% |
| 813–817 | 3.89% | 2.10% | 1.00% |
| 830–850 | 8.77% | 3.98% | 1.68% |
| 850–880 | 20.20% | 7.62% | 4.56% |

Thus PPM reduces the global first-step surface response in most shells,
but worsens 800–813 R☉ relative to PLM. The core improvement is large;
it should not be confused with a demonstration of long-term stability.

Job **340073** then extended matched uniform 256³ PPM to code time 1
(0.255 day) with `dt≤0.1` on M-debug. It completed normally in 3m13s,
with all Newton solves converged, zero opacity clamps, and positive saved
radiation energy. Analysis job **340076** compared the whole-sphere HDF
velocity increment with the existing uniform 256³ PLM control:

| Code time 1 | PLM | PPM |
| --- | ---: | ---: |
| Maximum Mach | 0.00429681 | 0.00431118 |
| Boundary `Lrad/Lheat` | 1.084022 | 1.083471 |
| Relative gas-mass change | −7.12×10⁻⁹ | −3.89×10⁻¹⁰ |
| Minimum saved `Er` [code] | 1.916×10⁻⁶ | 1.915×10⁻⁶ |
| Full-sphere 813–817 R☉ RMS `|Δv|/(Δt g)` | 4.91% | 4.83% |
| Full-sphere 830–850 R☉ RMS `|Δv|/(Δt g)` | 3.83% | 6.92% |

The two runs have nearly the same maximum Mach and radiative luminosity.
Despite the better *first step*, PPM has a larger accumulated velocity
increment in 830–850 R☉ by time 1, especially near the axes. The ratio
uses initial density as the mass weight and the initial gravity magnitude
to normalize the velocity increment, so it is a short-time diagnostic,
not an instantaneous force at time 1. Its reversal is sufficient to reject
the claim that simply replacing PLM by PPM stabilizes this setup. The
relative mass changes are tiny for both cases; PPM's smaller value does
not change this assessment. This is an early gate, not a year-scale run.

## Next gate

The next useful implementation is a **discrete** HSE initializer or
well-balanced hydrostatic face reconstruction coupled to the chosen
FLD-force and gravity operators. Its validation should compare actual
first-stage residuals over the *whole* sphere, including axis and diagonal
directions, and should resolve the 3.35 R☉ surface transition with several
radial cells. A localized thin patch can be tried only after the coarse/fine
force and FLD flux are audited. Stronger core damping would not remove this
surface force mismatch. The present data do not isolate the PLM limiter from
the HLLC solver within step 3→4, and cannot establish a standalone solver
defect.

Local compact JSON is in `wor122_surface_force_20261115/`; large raw HDF
and log outputs remain under `$WORK/wor122/force256_donor_axis/`,
`force256_ppm_axis/`, `$WORK/wor122/ppm256_t1/`, and the previous
`force*surface*/` directories. The decomposition figure is
`doc/figures/wor122_rsg_initial_surface_gas_decomposition_20261115.png`;
the full-sphere first-step/time-1 comparison is
`doc/figures/wor122_rsg_global_surface_velocity_20261115.png`.
