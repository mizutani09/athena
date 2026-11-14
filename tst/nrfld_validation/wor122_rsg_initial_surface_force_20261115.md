# WOR-122: initial RSG surface-force diagnosis (2026-11-15)

## Question and setup

Does the rapid surface motion begin with a discrete force imbalance, and
which term causes it? Three uniform-grid, one-step XD2000 controls used the
same WOR-55 LTE/FLD RSG profile, Tomida EOS, Freytag potential, central
heating, `xorder=2`, and a 0.1-code-unit step (about 2200 s). They used
source snapshot `6f2f819e`, binary `bin/athena_wor122_6f2f819e`, and
`inputs/rsg/athinput.wor122_rsg_smr_onepatch` with static refinement off.
All three finished normally with Newton convergence and no opacity clamps.

| Job | Mesh / selected block | Queue | Elapsed | Remote output under `$WORK/wor122` |
| --- | --- | --- | ---: | --- |
| 339997 | uniform 128³, +x axis, GID 457 | M-debug | 13 s | `force128_surface_axis/` |
| 339996 | uniform 256³, +x axis, GID 3656 | M-debug | 38 s | `force256_surface_axis/` |
| 339998 | uniform 256³, positive diagonal, GID 3647 | M-debug | 36 s | `force256_surface_diagonal/` |

The root-cell widths are 15.79 and 7.89 R☉, respectively. The 1-D
radiative-luminosity transition at 813–817 R☉ is about 3.35 R☉ wide, so
neither mesh resolves it radially. The selected blocks cover only small
parts of the star; the diagonal block has few outer cells. These are
targeted local diagnostics, not global angular averages.

`FLD_FORCE_VOLUME` gives the actual first-stage gas flux-divergence force
and radiation force in each selected cell. The analyzer adds the exact
analytic source gravity at the cell center, divides by cell mass and local
gravity magnitude, and bins by radius with mass weights. Thus positive
radial acceleration means outward. The initial/final HDF velocities
independently give the observed first-step acceleration. The comparison
1-D forces are the continuum derivatives and radiation term evaluated from
`data/ic/st35gm03n07_wor55_lte_fld_trial.dat` at the sampled cell radii.

## Measured local acceleration

All values below are mean radial acceleration divided by local `g`, in
percent. The velocity column is the independently measured first-step
`Δv_r/(Δt g)` from the 256³ axis run. The 1-D continuum profile is close to
hydrostatic balance in these sampled shells.

| Radius [R☉] | 128³ axis | 256³ axis | 256³ diagonal | 256³ observed | 1-D continuum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 750–800 | +0.60 | −0.37 | −0.06 | −0.38 | +0.01 |
| 800–813 | +6.47 | +0.79 | −0.42 | +0.79 | +0.01 |
| 813–817 | −5.26 | +1.77 | +2.07 | +1.76 | −0.05 |
| 817–830 | −4.15 | +0.65 | +0.38 | +0.65 | −0.13 |
| 830–850 | −1.43 | −5.05 | −1.95 | −5.04 | +0.63 |
| 850–880 | −23.39 | −10.88 | −4.11 | −10.88 | +1.26 |

In the 800–813 R☉ shell, the 128³ to 256³ axis comparison reduces the
mass-weighted RMS of total acceleration from 9.66% to 1.56% of `g`.
At 813–817 R☉ the **sign changes** and 256³ still has a 1.88% RMS
residual. At 830–850 R☉ the 256³ axis residual is worse than 128³;
this is not monotone convergence. The 256³ axis-versus-diagonal difference
is large in the outer layers. The diagonal sample there has only 130 cells
at 830–850 R☉ and 35 at 850–880 R☉, so the magnitude must not be
generalized to a full sphere.

The predicted force residual and observed `Δv_r/(Δt g)` agree within
0.011 percentage point in the 256³ axis shell means, and within 0.036
percentage point across all three selected blocks. This directly connects
the initial discrete imbalance to the first velocity increment. It does not prove
that the same term dominates after the surface has evolved or cooled.

## Which force is wrong?

Radiation pressure is present in the force balance. In the 256³ axis
813–817 R☉ shell, mean outward support is 100.18% of `g` from gas
pressure and 1.59% from radiation, versus continuum-profile values
98.03% and 1.92%. Therefore the gas-force excess is about +2.14% of `g`,
the radiation-force deficit about −0.33% of `g`, and the net discrete
outward residual is +1.77% of `g`. At 830–850 R☉, gas support is 94.91%
versus a continuum 100.57%; radiation contributes only about 0.05%,
giving the −5.05% inward discrete residual. The 128³ 800–813 R☉ shell
has gas support 102.54% versus continuum 95.86%, which explains most of
its +6.47% net residual.

The gas-force discrepancy contains both mapping of the steep 1-D profile
to 3-D cell averages and Godunov reconstruction/flux discretization. This
experiment does not separate those contributions or establish a solver bug.
The radiation-force term is smaller than the gas-force error in these
surface bins, though it is physically significant farther inside. Stronger
central drag cannot remove this initial *surface* force error. The first
step also runs before heating can materially alter these outer layers.

## Consequences and next gate

The surface profile is close to **continuum** HSE but is not in the
**discrete** HSE of the coupled 3-D update. Additional damping can suppress
the velocity transient, but the unbalanced force persists and reappears
when damping relaxes. A candidate initial model should be corrected against
the actual finite-volume gas flux, FLD force, and discrete gravity on the
intended mesh. Before a long run, it should pass these local gates:

1. Quantify mapped cell-average pressure versus face-reconstructed gas
   force separately, then adjust the 3-D pressure/energy profile or use a
   well-balanced hydrostatic reconstruction. Preserve the tabular EOS and
   radiation equilibrium while doing so.
2. Cover the full radiative transition with several **radial** cells and
   compare matched 128³, 256³, and finer surface-resolved grids. A single
   one-quadrant patch and a 7.89 R☉ cell do not resolve a 3.35 R☉ layer.
3. Require a small first-stage residual over the entire sphere, directional
   consistency, positive states, zero opacity clamps, Newton convergence,
   and a first-day luminosity/Mach/mass check before another long-duration
   calculation. Damping must be tested only after the force is balanced.

No long-duration job was submitted in this investigation. Compact shell
JSON is in `wor122_surface_force_20261115/`; local raw log copies are in
`/home/kosuke/simulation/test/wor122_surface_force_20261115/`, and the
large HDF snapshots remain under the listed `$WORK` directories. Reproduce
the bins with `wor122_rsg_surface_force_bins.py`. The two-panel plot is
`doc/figures/wor122_rsg_initial_surface_force_20261115.png`.
