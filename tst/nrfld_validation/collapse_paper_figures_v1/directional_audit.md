# Angular structure and literature check (2026-11-16)

## Measurement

The final `z=0` slice uses native leaf-cell values, not a smoothed or
interpolated field. We compared the four Cartesian-axis directions with the
four xy diagonals in 0.5 au wide rings, each using 10-degree angular wedges.
The quoted differences are `(axis - diagonal) / their mean`. We also checked
six Cartesian-axis cones against eight 3D body-diagonal cones (15-degree half
angle), volume-weighting the native cells in each radial shell. The 3D cones
have a finite radial width, so their values are a check on the 2D result,
not an exact point-by-point comparison at the same radius.

| Final c1080, 2D ring range | Mean absolute gas-temperature difference | Mean absolute density difference |
|---|---:|---:|
| 1–5 au | 2.2% | 3.5% |
| 5–10 au | 1.2% | 11.1% |
| 10–20 au | 1.5% | 19.8% |
| 20–30 au | 0.8% | 22.9% |

The radiation-equivalent temperature has essentially the same directional
difference as gas temperature. The 3D cone comparison at c1080 gives a 1.9%
gas-temperature difference in 1–5 au (axes cooler), while in 10–15 au the
axes are 1.1% hotter and 17.0% denser. At the accretion-shock region
(7–9 au) the 3D density sign reverses: axes are 29.0% less dense, with only
0.5% gas-temperature difference. One 0.5-au plane ring near the shock
reaches roughly 50% density difference, which is more sensitive to the
location of the steep density jump than the shell averages. We therefore
report ranges and methods, not a single universal angular-error value.

The pattern is not solely a color-map effect. It is also not solely an
EOS-table evaluation effect, because `Tgas` and `Trad` agree closely in their
angular pattern. The c300 snapshot is nearly isotropic in temperature
(<0.05% in 10–30 au) and has about 1–2% density differences, whereas the
c500 snapshot already shows 17–32% density differences in several shells.
For c500 at 10–15 au, both cone groups are entirely on AMR level 9 and their
mean radii differ by only 0.02 au, yet density differs by 30.5%; current
refinement level alone does not explain it. In c1080 the xy-axis and xy
diagonal refinement transitions also occur at different radii. A grid-related
imprint from Cartesian hydrodynamics, gravity, refinement, or their coupling
is plausible in this spherical, unmagnetized, nonrotating setup, but this
audit does not isolate which mechanism produces it or establish its
resolution convergence. Do not label the temperature pattern as a physical
anisotropy or as a proven defect in one specific solver component.

## Literature and figure assessment

- **Evolution figure:** Its near-isothermal stage followed by heating after
  the central density reaches around `10^-13 g cm^-3` is qualitatively
  consistent with [Tomida et al. (2015), Fig. 2 and Sec. IV.1](https://arxiv.org/html/1501.04102v1)
  and [Mayer et al. (2025), Figs. 8–9](https://arxiv.org/html/2510.12620v4).
  Those are not matched-condition curves: Tomida's models include magnetic
  fields and rotation, and Mayer's Figs. 8–9 show its later RMHD model. Our
  plotted `rho_max` and `Tgas,max` are independently sampled domain maxima,
  not one Lagrangian trajectory.
- **Final radial figure:** The density break, much slower inner velocity,
  and smooth temperature transition through the shock agree qualitatively
  with [Mayer et al. (2025), Fig. 7](https://arxiv.org/html/2510.12620v4)
  and [Commerçon et al. (2011), Fig. 6](https://arxiv.org/html/1102.1216).
  Our candidate shock radius is 7.5–7.9 au, versus approximately 6–7 au
  for Mayer's ideal-EOS, constant-opacity example and 8 au for Commerçon's
  3D example. Our final `Tgas,max=435 K` and `Mcore≈0.030 Msun` are of the
  same order as Commerçon's 396 K and 0.021 Msun, but their 3D comparison
  snapshot is at central density `2.7e-11 g cm^-3`, below ours
  (`1.04e-10 g cm^-3`), and initial conditions and microphysics differ.
  These numbers do **not** establish quantitative validation.
- **Final xy slice:** Its compact hot region is consistent with first-core
  formation, but its axis/diagonal pattern is **not** validated by the cited
  literature. Commerçon et al. explicitly report a very small cell-density
  spread at fixed radius in their nonrotating 3D AMR calculation. The
  directional differences in [Tomida et al. (2015), Figs. 4 and 7](https://arxiv.org/html/1501.04102v1)
  involve an MHD/rotating setup, so they cannot explain our fourfold pattern.
  Mayer's spherical-case Fig. 7 contains radial profiles, not an angular
  comparison. Thus the xy map should remain supplementary/diagnostic and
  should not be used to claim a well-converged spherical solution.

The radial means remain useful as a first-core *formation* demonstration,
with the angular limitation stated. A publication claim about 3D spherical
accuracy would need a controlled resolution or grid-geometry comparison at
the same physical stage.

## Reproducibility

The figure branch does not change solver source. The final c1080 HDF5 SHA256
is `13767189008f53ffb9e802b7ef66dcd97faff7045adbdd59051d19269a0354ac`;
the EOS table SHA256 is
`742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`.
Earlier snapshots from the same XD trajectory are c300 SHA256
`4837bf9e066e6d4a54c74b145ae6fe31b48cec4f200ebac10db8755b336cff28`
and c500 SHA256
`bba751bb6a74e6f9c405e15de4cd8e8de53a69a9dbf2aa0d12dae9e3eef3fe0b`.
`audit_directional_anisotropy.py` regenerates the final xy-ring CSV,
summary, [comparison plot](directional_anisotropy.pdf), and
[ray plot](directional_rays.pdf) from c1080. `audit_3d_cones.py` regenerates
`cones_c300.json`, `cones_c500.json`, and `cones_c1080.json` from the given
snapshots. These outputs record the exact per-shell counts, radii, levels,
density, gas temperature, and radiation-equivalent temperature.
