# WOR-22: controlled local reproduction (2026-10-19)

## Outcome and scope

A small radiation-only problem reproduces Newton backtracking failure on static
AMR while matched uniform grids converge. This is a **new reduced reproducer**,
not a replay of the HPC collapse cycle-7 restart and not a confirmed root cause.
No evolution solver, gravity, smoother setting, or implicit-update count was
changed. The existing Test C pgen and validation isolated-build helpers are reused.

The primary reproducer has constant density and constant opacity, frozen gas,
fixed lambda=1/3, no matter-radiation exchange, no gravity, no radiation force,
and no mixed-frame source. Therefore those physical effects and a material
interface are not necessary to reproduce this particular failure. AMR transfer,
the linear operator, its solve, and the NR handoff still need to be distinguished.
The large diffusion number is an intentional solver stress test: fixed lambda
does not represent physical free streaming at the resulting small opacity.

## Provenance and limits

- HEAD: `b5c91fdef2a90bf96d60fa21dd49b4c36cb4bbaf`, **dirty working tree**.
- Source snapshot hash: `bc688cb345bcd50842c27393d3827c58a797386603876a722abffd9135ef7564`.
- Binary SHA256: `1a3c3244e9a148bf39175169391d8ffde27cb68c73ab4740454c553261815158`.
- Snapshot/build/patch/configuration: `tst/outputs/wor22_controlled_20261019/`.
- Configuration: `--prob=nrfld_diff_amr_interface -nrmgfld -hdf5 -mpi --cflag=-O3`.
  This is deliberately an HD radiation-only diagnostic, not the production
  magnetic/scalar-enabled collapse configuration.
- Existing checkout `bin/athena` and configuration were not overwritten.
- One MPI rank; two build jobs; each simulation limited to 30 s, 4 GiB address
  space, 32 MiB per output file; core dumps disabled. All measured cases below
  completed in under 3 s, peak simulation RSS below 120 MiB; no timeouts/OOM.
- Simulation arithmetic is double. HDF5 fields are single precision; history
  uses `%.16e`. Conservation numbers below come from history, not rounded HDF5.
  Profile averages are accumulated in double. No transverse variation was
  resolved in the saved field data (not a double-precision symmetry proof).

## Controls

Domain `[0,1] x [0,.25] x [0,.25]`, periodic on all sides. MeshBlocks `8^3`.
Uniform coarse: `32 x 8 x 8` (4 blocks). Uniform fine: `64 x 16 x 16` (32 blocks).
AMR: base `32 x 8 x 8`, one extra level, 18 leaf blocks for either layout.
`amr_aligned` refines `[.25,.75]`; `amr_shifted` refines `[0,.5]`.
Both have the same number of cells and the same finest dx as uniform fine.
The refined *locations* differ, so full-fine output is the spatial reference.

The material slab, when contrast > 1, remains `[.25,.75]` in both layouts.
In the shifted layout its left interface is fine interior and right interface
is coarse interior. This does not hold the resolution of both interfaces fixed.
The constant-density control removes that ambiguity entirely.

`hydro/active=fixed`, `time/nlim=1`, `time/tlim=2e-4` give exactly the same dt
for every grid, verified in NR logs. Define q=dt*D_low/(1/64)^2. Opacity is varied
to set q; D=c_code/(3 sigma_R). Actual initial sigma_R ranges, mesh levels,
block counts, and finest intervals are saved in manifests. MG coefficient
diagnostics are in each run log.

Linear threshold `1e-10`, cap 30 V-cycles; NR max 64, relaxed acceptance disabled.
Existing smoother defaults are preserved, including post-MG sweeps (0 uniform,
16 AMR). Thus the comparison includes that existing algorithmic difference.

## Results

`wor22_matched_20261019/manifest.json`: pulse center .25, width .08, NR threshold
1e-10. At q=1 and q=1e4 all 16 cases converged:

| q | Density contrast | Uniform coarse/fine NR | AMR aligned NR | AMR shifted NR |
|---|---|---|---|---|
| 1 | 1 | 1 / 1 | 4 | 3 |
| 1 | 10 | 1 / 1 | 6 | 3 |
| 1e4 | 1 | 2 / 2 | 12 | 7 |
| 1e4 | 10 | 2 / 2 | 8 | 7 |

At q=1e8, NR threshold 1e-10, uniform grids stopped at residual max ~1e-8
with backtracking exhausted. This is distinct from AMR cases stalled at O(1).
Floating-point/residual accuracy is a candidate for the uniform residual floor,
not yet proved. Repeating with threshold 1e-7 allows the uniform controls to pass.

`wor22_stiff_tol7_20261019/manifest.json` (same pulse, q=1e8, threshold 1e-7):
uniform coarse/fine converge in 2 iterations for both contrasts. AMR aligned
converges in 10 iterations for contrast 1 but fails for contrast 10; shifted
AMR fails for both contrasts before accepting its first NR step.

### Additional periodic-tail control: recommended diagnostic case

The original Gaussian centered at .25 is not periodic at the domain ends and
the shifted AMR has a coarse/fine interface across the periodic seam. This is
a potential confound. To suppress it, a separate comparison uses pulse center
.5 and width .04: endpoint values are equal and Gaussian tails are below
double-precision resolution relative to the unit background near the seam.
Initial radiation and density are identical across layouts within this series.

`wor22_periodic_control_20261019/manifest.json`: q=1e8, threshold 1e-7:

| Contrast | Grid | NR result | Rejected V-cycles | Final max |
|---|---|---|---|---|
| 1 | uniform coarse | converged, 2 | 0 | 6.37e-9 |
| 1 | uniform fine | converged, 2 | 0 | 3.52e-8 |
| 1 | AMR [.25,.75] | converged, 4 | 3 | 2.31e-8 |
| 1 | AMR [0,.5] | backtracking exhausted, 1 | 1 | 1.0 |
| 10 | uniform coarse | converged, 2 | 0 | 7.08e-9 |
| 10 | uniform fine | converged, 2 | 0 | 2.29e-8 |
| 10 | AMR [.25,.75] | converged, 5 | 4 | 1.41e-8 |
| 10 | AMR [0,.5] | converged, 10 | 9 | 5.91e-8 |

Use the **contrast=1, shifted AMR** case for WOR-23/24. Final L2 is .4496185688;
it is not merely missing a very tight tolerance. One logged max location is
x=(.4140625,.0078125,.0078125), gid=9, relative to an xp coarse neighbor.
Max=1 can be attained by multiple cells; this does not isolate one unique cause.
The periodic topology still exists and its role is not ruled out.

### Conservation observation

At q=1e4, contrast=1, initial pulse center .25, the relative volume-integrated
radiation energy change is ~1e-15 on uniform grids, -2.6428126e-4 on aligned AMR,
and -2.0736261e-4 on shifted AMR, despite NR convergence to 1e-10.
The largest magnitude among the q<=1e4 runs is 3.4920460e-4 (aligned, contrast10).
This is a separate conservation warning, not a demonstrated explanation for
the NR failure. Converged residuals alone do not establish conservative AMR
transport. Diagnose coarse/fine flux balance in WOR-24/27.

## Reproduction

Fresh isolated build and the strongest constant-density comparison:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_controlled.py \
  --output tst/outputs/wor22_reproduce \
  --diffusion-numbers 1e8 --contrasts 1 --nr-threshold 1e-7 \
  --pulse-center .5 --pulse-width .04 --verbosity 2
```

Exit status 1 is expected when the intended failing case is reproduced.
Per-case raw status, NR status, dt, final cycle/time, field finiteness, and
history are recorded separately. An abort is not treated as an environmental
success; these failures contain an explicit NR failure before termination.

Reuse the preserved binary without rebuilding:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tst/nrfld_validation/test_c_controlled.py \
  --output tst/outputs/wor22_reproduce_saved \
  --binary tst/outputs/wor22_controlled_20261019/build_source/bin/athena \
  --diffusion-numbers 1e8 --contrasts 1 --nr-threshold 1e-7 \
  --pulse-center .5 --pulse-width .04 --verbosity 2
```

No collapse restart is required: the pgen deterministically reconstructs the
initial state. The exact historical HPC restart has **not** been replayed.
The current 12h run has not been inspected or changed.

## Superseded exploratory runs and next tasks

`wor22_controlled_20261019` and `wor22_stiff_20261019` requested shifted refinement
[.125,.625], which Athena expanded to [0,.75] (25 blocks). Keep their build and
raw outputs as provenance, but use the matched 18-block series for conclusions.
Earlier HDF analysis used float32 averaging, which produced spurious apparent
transverse deviations ~1e-6. The final harness and reanalysis use float64;
the corrected manifests show zero resolved transverse deviation.

Next: WOR-23 global Jv with actual ghost exchange; WOR-24 stage-by-stage handoff
residual and conservative interface flux checks; WOR-25 coarse operator.
The pure-radiation reproducer bypasses gas Schur cancellation, so WOR-26 is not
necessary to explain this reproducer, but remains relevant to coupled collapse.

## Addendum: diffusion-number and interior-boundary controls

The first shifted layout touches the periodic domain seam. Detailed output put
one maximum at x=0.0078125 in a fine block with a coarse periodic x-minus
neighbour. This is relevant to the periodic production mesh, but it confounds a
generic refinement interface with the periodic coarse/fine path.

`wor22_qsweep_20261019` holds the grid and central pulse fixed and changes only
q. Uniform fine converges in 1--2 NR iterations with no rejected V-cycle for
q=1e4...1e8. Aligned AMR converges in four iterations throughout, but rejected
cycles appear at q>=1e6. Shifted AMR uses 9,9,10,10 iterations at q=1e4...1e7,
with rejected cycles increasing from 0 to 9, and fails at q=1e8. Thus strong
diffusion alone is insufficient in this test; it amplifies an AMR-layout effect.

To remove the periodic coarse/fine seam, `wor22_interior_amr_20261019` uses a
64-cell root mesh and the shifted refined region [0.125,0.625]. Both refinement
boundaries are internal. The central pulse at x=0.5 is 0.125 from the right
boundary; the aligned [0.25,0.75] case places it 0.25 from either boundary.
Both AMR layouts contain 36 leaf blocks with the same dx range; uniform fine
contains 64 blocks at the same finest dx.

| q | uniform fine | AMR aligned/far | AMR shifted/near |
|---|---|---|---|
| 1e6 | converged 1 | converged 4 (2 rejects) | converged 6 (1 reject) |
| 1e7 | converged 2 | converged 4 (3 rejects) | failed after 1 accepted step |
| 1e8 | converged 2 | converged 4 (3 rejects) | failed before accepting a step |

This confirms an interaction among large `dt*D/dx^2`, AMR, and placement of a
radiation gradient relative to refinement boundaries in the reduced problem.
It does not yet identify whether ghost linearization, coefficient restriction,
coarse correction, post-MG boundary exchange, or flux mismatch is the cause.
Because diffusion is elliptic, the largest normalized residual can move into a
flat region away from the originating interface; max location alone is not a
causality proof.

Using the repository Mayer input units and the failed production dt=0.0297804,
an estimate with lambda=1/3 gives q~3.4e3 (core) and 3.4e5 (ambient) on the root
grid; four refinements increase these to ~8.7e5 and 8.7e7, and eight refinements
to ~2.2e8 and 2.2e10. Actual density and flux limiter must be measured at the
failed cell. These estimates overlap the reduced test's degradation range.

### Variable-limiter control

The production input uses the variable limiter, so the fixed-lambda result is
not sufficient by itself. With the same interior layouts and variable limiter,
all grids converge through nominal q=5e5, while the iteration counts at q=5e5
are 3 (uniform), 4 (aligned AMR), and 7 (shifted AMR). At q=7e5 all three fail;
at q>=1e6 all tested grids fail. Here nominal q is based on lambda=1/3 and the
input opacity; the actual face coefficient is limiter-dependent and must be
read from diagnostics. **Interpretation corrected 2026-10-23:**
`CalculateCoefficientsOnce` evaluates the limiter at the beginning of the
implicit step; its face coefficients are lagged throughout the Newton solve.
These failures therefore do not demonstrate nonlinear coefficient variation
*during Newton*. Spatially varying lagged coefficients and the linear solve
remain candidates even on the uniform grid. The evidence supports an
aggravating AMR effect, not the stronger claim that a refinement boundary is
the sole necessary cause.
