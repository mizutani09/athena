# WOR-177 aggressive core approximation: scope and validation plan

## Decision

The `mcut=7.0026122505 Msun`, `R_support=64.98117948 Rsun` model is an
experimental low-cost model for RLOF through contact and the early interaction
with the outer envelope.  It is not yet a replacement for the accepted
`mcut=6.279768207 Msun`, `R_support=4.054775 Rsun` model in calculations of a
deep plunge, envelope unbinding, or final separation.

The distinction is important: the isolated-star jobs test numerical stability,
not whether the two core approximations give the same binary evolution.

## Constructed model

- The retained MESA table starts at `7.0026122505391051 Msun` and
  `64.98117947774728 Rsun`; a `gamma=5/3` modified-Lane--Emden continuation
  supplies the gas inside this radius.
- The fitted core particle is `6.5190013 Msun`; the gas inside the support is
  `0.4762341 Msun`, for `6.9952355 Msun` inside the support in total.
- The SCF central density is `6.53288236e-6 g cm^-3`, a factor 737 below the
  accepted diffuse-core model and about `1.27e8` below the original MESA
  central density.
- The support is 6.97% of the approximately `931.9 Rsun` stellar radius.
- The `nr=16384`, `nmu=16`, `lmax=12` SCF solution converged in 21 iterations;
  its scalar virial residual is `1.5234e-6` and Poisson L2 residual is
  `7.795e-9`.
- The surface taper changes the piecewise-linear gas mass by `1.448e-4` and the
  pressure-volume integral by `6.217e-5`.

The model deliberately cuts into the convective envelope.  This is the main
physical risk, even though the initial outer profile is very well preserved.

## Outer-structure comparison

The accepted and aggressive tapered SCF profiles use the same radial grid.  In
the resolved envelope outside `R_support`, restricting the comparison to cells
with both densities at least `1e-9 g cm^-3`, the aggressive/accepted differences
are:

| quantity | RMS log10 difference | maximum absolute log10 difference |
|---|---:|---:|
| density | `5.397e-5 dex` | `4.284e-4 dex` |
| pressure | `1.162e-4 dex` | `8.276e-4 dex` |

The spherical moment of inertia differs by `+2.21e-6` globally and by
`+4.36e-5` when only the material outside the aggressive support is integrated.
These small differences show that the initial monopole and outer mass
distribution are preserved.  They do not test the adiabatic response to mass
loss, the tidal Love number, convective modes, or the drag once the companion
approaches the modified region.

## Athena++ mapping and cost

| model | central level | finest dx | cells/support | map force L2 |
|---|---:|---:|---:|---:|
| aggressive | L3 | `4.296875 Rsun` | 15.123 | `0.0038394` |
| aggressive control | L4 | `2.1484375 Rsun` | 30.246 | `0.0009915` |
| accepted reference | L7 | `0.2685547 Rsun` | 15.10 | `0.005232` |

Both aggressive mappings have mass errors below `7.4e-7` and ideal-gamma
energy errors below `4.2e-8`.  Pure CFL scaling predicts about 16 times fewer
cycles at L3 and 8 times fewer at L4 than the accepted L7 model.  Scaling the
measured accepted cost of `49.26 node-hour/t_dyn` gives provisional costs of
about `3.1 node-hour/t_dyn` at L3 and `6.2 node-hour/t_dyn` at L4.  The submitted
jobs must replace these estimates with measured costs.

Long isolated-star runs were submitted on XD on 2026-11-24:

- job `347029`: root64, surface L2, central L3, 3 nodes, target `1.5 t_dyn`;
- job `347030`: root64, surface L2, central L4, 3 nodes, target `1.5 t_dyn`.

At the last check both jobs were pending in the scheduler.  The launch commit is
`3c93475e9` on branch `wor-177-aggressive-core`.

## Why this can be adequate for RLOF through contact

Outside a spherical support, matching the enclosed mass preserves the core's
monopole gravitational field.  During the onset of RLOF the quantities that
directly control mass loss are concentrated near the surface and Lagrange
points: stellar radius, pressure/density scale heights, surface entropy,
rotation, the Roche geometry, and the ability to resolve the L1/L2/L3 streams.
Reichardt et al. find that the duration and absolute transfer rate in the RLOF
phase are resolution dependent, while transfer through the Lagrange points
drives the orbital evolution and the subsequent inspiral is much less
resolution sensitive [Reichardt et al. 2019](https://arxiv.org/abs/1809.02297).

Replacing a giant core by a point mass plus a hydrostatic continuation is an
established way to obtain stable multidimensional envelopes
[Ohlmann et al. 2017](https://arxiv.org/abs/1612.00008).  This supports the
method, but it does not establish that an arbitrary cut radius is harmless.

## Why this approximation can fail

The aggressive boundary is in the convective envelope.  It can therefore alter
the donor's response to mass removal, tidal quadrupole, mode spectrum,
synchronization, and the binding energy available during a plunge.  In a
9.4-Msun RSG calculation, Moreno et al. obtained similar orbital evolution for
10- and 20-Rsun cuts but a different evolution for a 40-Rsun cut that entered
the convective envelope.  They explicitly identify the missing drag and the
proximity of the final orbit to the cut as concerns
[Moreno et al. 2022](https://doi.org/10.1051/0004-6361/202142731).

Massive-RSG SPH calculations likewise replace the unresolved inner star by a
point mass and matched gas, but use a cut near the bottom of the convective
envelope rather than deep within it
[Lau et al. 2022](https://arxiv.org/abs/2111.00923).  Thus the present 65-Rsun
cut is more aggressive than the most directly comparable published production
models and needs an explicit binary sensitivity test.

## Acceptance sequence

1. **Isolated numerical convergence.** Compare L3 and L4 over `1.5 t_dyn`.
   Require bounded mass and mechanical-energy drift, no growing core-localized
   nonradial mode, a completed first radial turn, and agreement of the main
   diagnostics between L3 and L4.  This selects the numerical resolution only.
2. **Static physical audit.** Compare the accepted and aggressive models in
   exterior enclosed mass, binding energy outside 65 Rsun, moment of inertia,
   fundamental radial period, and a static-tide/Love-number proxy.  The moment
   of inertia check already passes strongly; the response quantities remain.
3. **Short binary A/B/C gate.** Run the accepted L7 model, aggressive L3 model,
   and aggressive L4 control with the same point-source companion, orbit, and
   spin through RLOF/contact.  Compare `Mdot_L1`, L2/L3 loss, `a(t)`, `Jdot`,
   donor torque, volume radius, surface shape, and bound/unbound mass.
4. **Use boundary.** The aggressive model may become the production model for
   RLOF/contact if L3/L4 are numerically converged and the binary observables
   agree with the accepted model.  Do not use it for the deep plunge when the
   companion or important gas wakes probe the support region.  A conservative
   warning boundary is `a < 3 R_support` (about 195 Rsun); inside
   `R_support` the softened model is unambiguously not the original star.

The binary tolerances should be fixed before examining the A/B/C outcome.  A
reasonable initial gate is 5% for separation and cumulative torque, 10% for
cumulative transferred mass, 1--2% for donor radius, and 10--20% for the
L2/L3 loss fraction.  These are project decision tolerances, not universal
accuracy claims.
