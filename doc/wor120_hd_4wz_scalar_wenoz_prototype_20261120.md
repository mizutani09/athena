# WOR-120: bounded 4wz passive-scalar prototype (2026-11-20)

## Scope and design

- Branch: `wor-120-hd-4wz-scalar-wz`
- Worktree: `/home/kosuke/simulation/athena-wor120-hd4wz-scalar-wz`
- Base: production HD 4wz commit `251b6cee6`
- Status: research prototype; not merged into production

The production 4wz Hydro path and the pre-existing passive-scalar path used different
fourth-order algorithms. This prototype gives scalars the same numerical contract as
Hydro:

1. Recover point scalar concentration from evolved scalar and density volume averages.
2. Reconstruct point concentrations with component-wise Berta WENO-Z.
3. Upwind with the point density flux saved by the Hydro Riemann solve.
4. Convert point scalar fluxes to face averages with the transverse Laplacian.
5. Use Hydro's troubled/coarse-fine mask for PLM fallback and add a scalar-curvature
   sensor so a tracer discontinuity is detected even in a smooth background fluid.

The Hydro algorithm, restart representation, and evolved conservative scalar remain
unchanged. The scalar curvature sensor examines all active dimensions so a transverse
tracer edge also disables the transverse flux correction.

## Smooth convergence

A one-dimensional Gaussian concentration is advected through one periodic domain with
RK4 + 4wz. At 64 cells and finer the scalar sensor is inactive in the resolved smooth
profile.

| cells | L1 error | order to next resolution |
|---:|---:|---:|
| 32 | 2.28447e-2 | 6.85 |
| 64 | 1.97983e-4 | 4.52 |
| 128 | 8.60955e-6 | 4.22 |
| 256 | 4.61659e-7 | - |

The coarsest value is intentionally fallback dominated. The asymptotic values recover
the original unconstrained prototype's fourth-order behavior; at 64--256 cells they
also beat the unrelated legacy fourth-order PPM scalar errors.

## Discontinuous and multilevel tests

The 100 x 100 Zalesak slotted-cylinder profile completes one RK4 rotation without a
NaN or runtime failure.

| mesh | cycles | scalar L1 | scalar max | relative scalar-mass increase |
|---|---:|---:|---:|---:|
| uniform | 843 | 2.63135e-2 | 7.018e-1 | 9.03e-6 |
| static refinement, 20 x 20 blocks | 2,506 | 2.49380e-2 | 8.350e-1 | 1.07e-5 |

The uniform L1 error is about 6% smaller than RK4 + PLM (`2.79765e-2`). The multilevel
case exercises point-flux storage on transverse stencil rows, the coarse-fine Hydro
mask, scalar restriction/reflux, and the scalar point recovery.

## SCF integration smoke test

The exact SCF build used by WOR-120 (MPI, self-gravity, passive scalars, `NGHOST=4`)
compiles after the scalar changes.  A four-rank, surface-level-0 SCF restart advances
one RK4 cycle with 400 mesh blocks and 1,638,400 zone-cycles.  It finishes without a
NaN, fatal error, or multigrid warning, and the printed stellar mass is unchanged at
history precision.  The super-time-stepping primitive-update path deliberately keeps
the pre-existing cell-average scalar recovery: it must not consume the Hydro point
mask or point density from an unrelated STS stage.

## Remaining blocker

The slotted-cylinder scalar mass should be conservative to roundoff. RK4 + PLM changes
it by only about `6e-8` relatively, while this prototype still increases it by about
`1e-5`. The increase occurs when a small negative evolved concentration is reset by
Athena++'s existing scalar floor. Normal-direction, all-direction, and one-cell-dilated
curvature fallbacks reduce the defect substantially but do not eliminate it; dilation
also increases diffusion and cost. Local clipping of WENO face states destroys
asymptotic fourth-order convergence and was rejected.

The next numerical step is a conservative positivity-preserving flux limiter that
blends the high-order scalar flux toward the already available PLM flux by a shared
face coefficient. Until that limiter passes smooth, discontinuous, and AMR tests, the
validated scalar-PLM branch is the robust option and this prototype must not replace
the production scalar path.
