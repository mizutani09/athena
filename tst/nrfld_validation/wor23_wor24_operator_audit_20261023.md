# Test C operator audit — WOR-23 / WOR-24 (2026-10-23)

## Outcome

The fixed-lambda, shifted-AMR Test C has a directly reproduced **operator
mismatch**. The NR residual uses minmod state ghosts, while MG solves with
trilinear correction ghosts. Solving that mixed system with a small dense
direct solver, without any V-cycle or coefficient restriction, reproduces the
large negative first correction measured in Athena.

This identifies a concrete mechanism for this reduced case. It does **not**
establish that every collapse failure, or even every Test C failure, has that
cause. No evolution algorithm, gravity code, smoother count, implicit-update
count, or acceptance threshold was changed in this work.

## Scope and reproducibility

- `tst/outputs/wor23_wor24_audit_20261023/`: final audit, 27 simulations.
- Executable/source snapshot: `tst/outputs/wor23_jv_20261023/build_source/`.
  This is a dirty-worktree snapshot, incrementally rebuilt after diagnostic
  fixes; do not use the initial incomplete build manifest as its final hash.
  The actual binary SHA256 is in `audit.json` and each collection manifest.
- Final binary SHA256:
  `e76b85e7e711331ecb4f62bf195df4f0d466be04e40a93dfae2de1e39d72b1b7`.
  Final snapshot `src/` hash excluding generated `defs.hpp`:
  `e8e8228230c0c5cd6f335c740e6c92497e68d78a3055e4e91d4fb9a41da54604`.
  All non-generated snapshot source files match the working tree at handoff.
- One MPI rank, periodic Cartesian x-only field represented in 3D; no gas
  evolution, gravity, matter-radiation exchange, or mixed-frame terms.
- Domain `[0,1] x [0,.125] x [0,.125]`; root `64 x 8 x 8`, blocks `8^3`;
  uniform-fine control `128 x 16 x 16` (64 blocks); AMR has 36 blocks.
- Gaussian `1 + .1 exp(-.5((x-.5)/.04)^2)`, constant density/opacity.
  Aligned refinement `[.25,.75]`, shifted refinement `[.125,.625]`.
- `dt=2e-4`; q is `dt D / (1/128)^2`, using lambda=1/3 to define nominal D.
  NR threshold `1e-7`, no relaxed acceptance, max 64; linear cap 30.
- Per simulation: 30 s, 4 GiB address-space limit, 32 MiB output-file limit;
  measured peak RSS below 254000 KiB and runtime below 3 s in the final audit.
- Twelve OFF/ON pairs have exactly identical NR result dictionaries, history
  bytes, and HDF datasets (including initial output for failed cases). This
  establishes instrumentation neutrality for these cases, not MPI/general-EOS
  coverage. Main `bin/athena` was not rebuilt.

See README for fresh isolated-build and reference commands. The existing
`test_c_controlled.py` also exposes `--global-jv`, `--handoff`, and
`--jv-iteration`. Tests deliberately include NR failures; a successful audit
does not mean those failures disappeared.

## 1. Where the returned direction changes

At q=1e7, first linear solve, report `||b-A delta||_V / ||b||_V` using the same
NR leaf-cell coefficients, RHS, and volume weighting at all four stages:

| Layout | MG end | Retrieved | After delta ghost update | After existing fine sweeps |
|---|---:|---:|---:|---:|
| Uniform fine | 1.005e-11 | 1.005e-11 | 1.005e-11 | 1.005e-11 |
| Aligned AMR | 5.282e-11 | 5.282e-11 | 1.045e-7 | 2.700e-8 |
| Shifted AMR | 1.926e-11 | 1.926e-11 | 1.815e-2 | 4.686e-3 |

The shifted case's first change is confined to the coarse/fine-face category:
its region-normalized residual goes from `3.727e-10` to `0.5087` while interior
and same-level categories are unchanged. Region denominators differ; do not
add or directly compare those numbers as global contributions. Its largest
post-exchange absolute residual is ~1791 at x=.62109375 next to x=.625.
Double-product versus extended-precision difference-form discrepancies are
~4e-12 relative globally, far smaller than .018. Roundoff does not explain this
jump. The default 16 fine sweeps reduce it, but do not recover MG accuracy.

`native_row_rel` evaluates the stored MG row on the **current stage's ghost
values**; after exchange it is not the original MG boundary operator's norm.
The diagnostic adds no synchronization at MG end: it measures exactly what
RetrieveResult is about to copy. In this case retrieval itself changes nothing.

## 2. An earlier problem: the full step is already negative

At MG end in the shifted case, before the ghost-policy change:

- correction mean = -9.330263706; range [-9.420263839, -9.319605624];
- raw full-step minimum = -8.321535478; all 18432 active cells are nonpositive;
- source mean = -2.875603980 (initial residual RHS, not physical source heating).

Thus an accurately solved MG correction is not automatically a useful physical
Newton update. The first full/half/etc. trials hit the positivity floor; only
scale .0625 is accepted, with very small improvement. The next direction fails
all allowed trials. This is why it would be incomplete to describe the .018
handoff jump as the entire failure mechanism.

## 3. Independent direct reference: no multigrid iteration

`test_c_composite_reference.py` collapses the x-only geometry to 96 leaf cells
(128 for uniform). It reproduces coarse ghost averaging and two fine-side
ghost rules from the code:

- NR state / returned delta: minmod-limited coarse reconstruction;
- MG internal: x-only reduction of trilinear interpolation, `.75*u_c + .25*u_neighbor`.

For the initial state u0, it forms

```
b_NR = dt D L_minmod(u0)
A_MG = I - dt D L_trilinear
A_MG delta = b_NR
```

The dense direct solution predicts shifted q=1e7 correction mean
`-9.33026371` and full-step minimum `-8.32153548`, matching Athena to ~5.1e-9.
All six fixed-lambda controls match RHS mean, correction mean/min/max and
full-step minimum within the documented mixed tolerance 2e-7.
The worst difference is ~9.6e-8 in the tiny aligned-case mean at q=1e7.

Using the same MG operator consistently for the state equation,
`A_MG u_new = u0`, gives a positive shifted-case minimum ~1.01007522.
A separate conservative nonuniform finite-volume reference gives ~1.01002541
and volume-integrated energy conservation to ~2.7e-11. These are controls, not
proposals to replace the production solver without further validation.

The MG trilinear leaf operator preserves constants (`L*1=0`) but is not
volume-conservative on this AMR mesh (`max|V^T L|=80`, unit diffusion). These
are different properties. The conservative reference has the latter defect
~1.4e-14. The minmod initial residual's nonzero volume integral agrees with the
Athena RHS mean to ~2.3e-13; it is not caused by Newton stopping early.
The inconsistent ghost operators, nonconservation, and large constant-like
correction therefore deserve investigation before blaming coefficient
restriction alone. This reference reproduces the first step, not the complete
nonlinear iteration or all intermediate MG levels.

## 4. Boundary-inclusive Jv: a separate distinction

`A v` here uses NR coefficients and the returned-direction ghost policy.
`FD` uses the raw radiation residual after the real NR state boundary exchange
on `u+h v`. A third control replaces direction ghosts by their finite-difference
tangent. There is no gas elimination in this uncoupled case.

- Shifted q=1e7 actual direction, first/second solves: relative Jv error
  ~8.69e-9 / 9.42e-9 at h=1e-4. It grows as h shrinks, consistent with subtraction
  noise; the tangent control is ~1e-14. There is no evidence here for a large
  minmod-tangent mismatch dominating this failure.
- Aligned q=1e7 **second** direction: Jv error ~.08385 globally, ~.1351 in the
  coarse/fine category; essentially unchanged across h=1e-4...1e-6. The tangent
  control removes it to ~6e-17. This confirms the distinction between
  `G(delta)` and `G'(u)delta`, but this case nevertheless converges.
- Hence the presence of a Jv mismatch alone is not a success/failure classifier.
  Do not confuse this NR-tangent test with the MG/NR ghost-policy mismatch above.

The diagnostic records the resulting ghost/tangent difference, not internal
minmod branch identifiers. Full coupled 2-variable/Schur, physical boundaries,
multiple MPI ranks, and the collapse failure state remain untested.

## 5. Variable-limiter failure is another mechanism

Correction to the earlier WOR-22 interpretation: the code computes limiter
face coefficients at the **start of the implicit step**, and lags them during
Newton. Spatially variable coefficients do not imply they change during each
Newton iteration. `closure=frozen` is the production residual test;
`closure=refreshed` is a hypothetical nonlinear-closure reference only.

At nominal q=5e5, uniform/aligned/shifted converge in 3/4/7 iterations.
At q=7e5 all fail. In the uniform case the first V-cycle is rejected, the
rollback returns a zero direction, and the relative linear residual remains 1.
There is no AMR boundary in that case. Jv on that actual zero direction is
explicitly skipped, not reported as a successful check. Spatial-coefficient
contrast and the coarse correction remain the next controlled investigation
for this separate failure (WOR-25).

## Next steps and cautions

### Minmod transfer control (2026-10-23)

Two superficially similar changes have different meanings and results.

1. Applying minmod directly to every coarse-grid **correction** in the uniform
   MG hierarchy makes the first V-cycle worse.  The q=5e5 factor changes from
   0.936206935 to 1.01013602, so a case that previously passed would now be
   rejected.  At q=7e5 it changes from 1.020718730 to 1.07362766.  This control
   has no AMR boundary and rules out blanket replacement of the linear
   trilinear correction prolongation by minmod as a general remedy.
2. At an AMR coarse/fine boundary, the operator needed by Newton is the
   state-dependent tangent `G'(u) delta`, not the nonlinear reconstruction
   `G(delta)`.  In the independent 96-cell shifted q=1e7 reference, the current
   mixed operator gives an updated minimum -8.32153548.  A dense solve using
   the minmod tangent at the base state gives 1.00966246; directly solving the
   piecewise-linear `G(delta)` control gives 1.00997290.  Both remove the
   negative direction in this reduced model, but the latter is no longer a
   fixed linear solve and therefore is not a drop-in linear-MG prolongation.

Artifacts are `minmod_prolongation_reference.json` and
`composite_minmod_tangent_reference.json`.  The promising implementation path
for the fixed-lambda shifted AMR family is a base-state-dependent *linearized*
coarse/fine boundary operator (or a common conservative face operator), not a
global nonlinear minmod transfer.  This still requires a full Athena Test C,
conservation/positivity, MPI, and coupled-collapse validation.

1. For the fixed-lambda AMR family, inspect matched NR/MG boundary policies and
   a conservative composite face-flux formulation; validate positivity and
   conservation as well as residual reduction. Do not blindly replace every
   minmod call with unlimited interpolation.
2. For the variable-coefficient uniform family, investigate the first rejected
   V-cycle and coarse operator, independent of AMR ghost inconsistency.
3. Extend only supported changes to coupled collapse and MPI controls. The
   present evidence does not certify long collapse runs or relaxed thresholds.

WOR-23/24 remain open for their broader coverage requirements. Their local
Test C stages are implemented and measured; this report separates those
completed measurements from the remaining validation.
