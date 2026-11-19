# WOR-120: conservative scalar-positivity experiment (2026-11-20)

## Scope

- Branch: `wor-120-hd-4wz-scalar-positivity`
- Base prototype: `e167f88e7` (`wor-120-hd-4wz-scalar-wz`)
- Status: research experiment; not merged into the production HD branch

The component-wise scalar WENO-Z prototype is fourth-order on smooth profiles but
creates small negative cell averages next to a discontinuous tracer.  The existing
EOS floor then clips those values and increases the globally integrated scalar by
about `1e-5`.

This experiment applies an a-posteriori conservative correction after every scalar
flux-divergence stage.  Within each MeshBlock it sets negative scalar cell averages to
zero and pays for the correction by scaling the positive scalar mass in the same
block, using physical cell volumes.  Thus an already positive solution is bitwise
unchanged and a repairable block retains its integrated scalar mass.  The correction
also covers the extra SSPRK(5,4) register update.

## Results

The finite-safe `g++-simd` build passes.  All tests use RK4, `xorder=4wz`,
`NGHOST=4`, and the same inputs as the parent prototype.

| one-period test | scalar L1 | scalar max | relative scalar-mass change |
|---|---:|---:|---:|
| uniform 100 x 100 | 2.631117e-2 | 7.017494e-1 | -1.19e-12 |
| static refinement, 20 x 20 blocks | 2.494538e-2 | 8.350342e-1 | +8.82e-11 |

The parent prototype changed scalar mass by `+9.03e-6` and `+1.07e-5`,
respectively.  The conservative repair therefore improves the defects by about 7.6
and 5.1 orders of magnitude without materially changing discontinuous-profile error.

The smooth one-dimensional Gaussian remains identical to the parent prototype:

| cells | L1 error | order to next resolution |
|---:|---:|---:|
| 64 | 1.97983e-4 | 4.52 |
| 128 | 8.60955e-6 | 4.22 |
| 256 | 4.61659e-7 | - |

The correction is inactive in these positive smooth solutions, so it does not hide a
loss of asymptotic order.

The exact MPI SCF configuration (self-gravity, two scalars, `NGHOST=4`) also builds.
A four-rank, surface-level-0 RK4 + 4wz smoke advances one cycle on 400 MeshBlocks
(1,638,400 zone-cycles) without a NaN, multigrid warning, or topology change.  Both
integrated scalar masses are unchanged at the 12-digit history precision.

## Remaining limitation

A MeshBlock whose total scalar mass is non-positive has no positive state with the
same block integral.  The local correction deliberately leaves such an exceptional
block unchanged rather than creating mass; the EOS floor can still act later.  This
explains why the multilevel result is near, but not exactly at, roundoff conservation.
Eliminating that last case rigorously requires a shared-face low/high-order flux
limiter or an inter-block conservative redistribution.  The experiment is therefore
useful evidence, but it does not yet replace the robust scalar-PLM branch.
