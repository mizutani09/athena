# WOR-31: Test C multilevel controls (2026-10-24)

## Scope

This audit uses the periodic, radiation-only Test C.  It is not a collapse
validation.  The SciPy calculation is an independent uniform-grid reference;
the Athena++ comparisons use existing smoother modes.  No production solver
default, gravity path, post-MG smoothing count, or implicit-update count is
changed.

## Rejected AMR-boundary tangent experiment

An opt-in experiment replaced only the fine-side correction ghost fill by the
frozen minmod tangent selected by the base state.  The global finite-difference
Jv diagnostic remained at roundoff, but the V-cycle became much worse:

- shifted AMR, q=1e4: the first V-cycle residual grew by about 16;
- shifted AMR, q=1e7: the first V-cycle residual grew by about 1.7e4;
- at q=1e4, the common handoff residual was about 16.1 at MG end, 22.9 after
  boundary fill, and 3.68 after 16 fine sweeps.

Changing only this side of the interface does not create a consistent
composite Jacobian: the coarse interface row, flux correction, and coarse
hierarchy remain unchanged.  The experiment was therefore removed.  It is
evidence against adopting a boundary-only tangent fix, not evidence that the
state/correction interpolation mismatch is harmless in every coupled case.

## Galerkin causal control

`test_c_galerkin_reference.py` keeps the fine operator, restriction,
trilinear prolongation, and red-black smoother fixed.  It changes only the
coarse hierarchy from coefficient rediscretization to `A_H = R A_h P`.

| nominal q | rediscretized first V-cycle | Galerkin first V-cycle |
|---:|---:|---:|
| 5e5 | 0.936206935 | 0.860069348 |
| 7e5 | 1.020718730 | 0.948510966 |

For q=7e5, the recursive residual factors (levels L0--L3) are
`1.021, 1.125, 1.229, 1.399` with rediscretization and
`0.949, 0.762, 0.801, 0.897` with Galerkin operators.  The high-frequency
power fraction of the restricted right-hand side grows from about 0.18 at L0
to 0.87 at the deepest restriction.  The direct solve on the 8x1x1 bottom
grid has a relative residual near 3e-16, so bottom-solve accuracy is not the
failure.

Replacing only the first coarse level is insufficient (q=7e5 factor
1.0129); replacing the first two levels makes the cycle weakly convergent
(0.9912), and the full Galerkin hierarchy reaches 0.9485.  Thus the result is
multilevel and cannot be reduced to one bad top-level correction.

This does not prove that a production Galerkin implementation is the only
fix.  `R A_h P` has a wider stencil and would be a substantial generic-MG
change.  It does show causally that the current rediscretized hierarchy, in
combination with the transfer and smoother, is sufficient to turn this
uniform Test C cycle from convergent to divergent.

The stencil count makes that limitation concrete.  The rediscretized
hierarchy retains 7 nonzeros per ordinary 3-D row, whereas the same R/P pair
produces 27 nonzeros per row on the first three Galerkin coarse levels.  A
different average of the six face coefficients alone cannot represent these
edge/corner couplings.  This explains how `A_H` can agree closely with `RAP`
on smooth test modes yet behave differently after repeated restriction has
created substantial high-frequency content.

## Existing-smoother controls

At q=7e5 with the strict NR tolerance 1e-10:

- `jacobi-rb`: first V-cycle factor 1.105, immediately rejected;
- `jacobi`: first V-cycle factor 1.243, immediately rejected;
- `jacobi-double`: first V-cycle factor 0.733 and no initial rejection, but a
  30-cycle cap leaves a linear relative residual of about 4.6e-3.

In this implementation `jacobi-double` is two full-cell Jacobi updates per
configured smoothing pass: `redblack_=true` causes the generic driver to call
the full-cell `fsmoother_=0` update twice.  It is therefore a materially more
expensive smoother, not a minor red/black ordering variant.

With `jacobi-double`, a 200-cycle cap, and NR tolerance 1e-9, all four
variable-limiter controls complete:

| nominal q | uniform fine | shifted AMR |
|---:|---:|---:|
| 7e5 | 2 NR iterations, 0 rejects | 12 iterations, 1 reject |
| 1e6 | 2 NR iterations, 0 rejects | 25 iterations, 18 rejects |

At q=5e5 and the same tolerance/cap, `jacobi-rb` fails immediately on the
uniform-coarse and shifted-AMR layouts, while `jacobi-double` completes all
four layouts.  Both modes take 2 Newton iterations on uniform-fine and 11 on
aligned AMR.  Their final Er arrays are bitwise identical in those two common
passing layouts.  Local elapsed times were 1.12 versus 1.20 s (uniform-fine)
and 1.63 versus 1.78 s (aligned AMR); the previously failing shifted case took
3.19 s with `jacobi-double`.  These tiny single-rank timings are comparative
controls, not production performance estimates.

At 1e-10 the uniform runs stop very close to tolerance (final normalized max
about 2.2e-10 and 2.6e-10), while q=7e5 shifted AMR converges in 13
iterations.  This is a useful diagnostic and possible operational fallback,
but it is expensive and q=1e6 shifted AMR remains marginal.  It is not yet a
validated collapse default.

An omega=0.8 control was also rejected.  In the actual q=5e5 Athena++ case it
changed the first-cycle factor from 0.993 to 1.093; the recursive coarse solve
factor worsened from 1.197 to 1.392.  An idealized offline optimum therefore
did not transfer to the actual hierarchy.
Single-update weighted Jacobi at omega=0.5, 2/3, and 0.8 also failed the q=7e5
uniform and shifted controls on the first V-cycle.  Weighted `jacobi-double`
at omega=0.8 retained the uniform pass but worsened shifted AMR to 25 Newton
iterations and 18 rejected cycles; omega=2/3 failed shifted AMR immediately.
The unweighted double update is the best of the tested existing modes.

## Current interpretation and next gate

The strongest local evidence now points to incompatibility among the
rediscretized multilevel operator, transfer, and smoother under large
diffusion number.  AMR adds a separate interface-consistency problem, but it
is not necessary for the variable-limiter uniform failure.  Blanket minmod
correction transfer, boundary-only frozen-minmod tangent, red-black order
reversal, coefficient-cache changes, and omega=0.8 have not fixed it.

Before changing generic multigrid, compare `jacobi-double` and `jacobi-rb` on
the same short collapse restart with identical strict/relaxed thresholds and
record time, linear caps/rejections, NR iterations, conservation, and output
differences.  In parallel, a Test-C-only Galerkin or Krylov-preconditioned
prototype can establish whether robust coarse correction is achievable
without changing production defaults.

Raw/reference results:

- `tst/outputs/wor31_galerkin_levels_20261024.json`
- `tst/outputs/wor31_smoother_jacobi-rb_20261024`
- `tst/outputs/wor31_smoother_jacobi-double_20261024`
- `tst/outputs/wor31_smoother_jacobi_20261024`
- `tst/outputs/wor31_jacobi_double_200_20261024`
- `tst/outputs/wor31_jacobi_double_relaxed_20261024`
- `tst/outputs/wor31_omega100_20261024`
- `tst/outputs/wor31_omega080_20261024`
