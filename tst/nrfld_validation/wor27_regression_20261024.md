# WOR-27: baseline regression gate (2026-10-24)

## Scope

This gate validates the current dirty `nr_mgrad_dev` source snapshot at commit
`9737ed77`.  Both calculations use isolated source copies and record the dirty
source hash, executable hash, generated inputs, logs, actual timesteps and
termination state.  The parent checkout was not reconfigured.  These tests do
not validate a long AMR collapse or promote `jacobi-double` to the production
default.

## Constant-coefficient diffusion convergence

Command:

```bash
python3 tst/nrfld_validation/diffusion_convergence.py \
  --max-cores 2 --output-root validation \
  --run-name wor27_diffusion_20261024
```

All Newton solves converged and all runs reached the common final time 0.02.
The actual timestep sequence, positivity and finite fields were checked.

| measurement | observed orders |
|---|---|
| fixed diffusion number, versus dx | 1.9702, 2.0007 |
| time-extrapolated spatial error, versus dx | 1.9851, 2.0048 |
| fixed N=32 successive dt differences | 0.9890, 0.9945 |

The first row uses N=16,32,64 and dt proportional to dx squared.  It therefore
shows the expected combined second-order path but is not, by itself, an
independent time-order measurement.  Richardson cancellation at each fixed
grid gives the second row, establishing spatial order.  The third row holds
the grid fixed and establishes first-order implicit time convergence.

Artifacts:

- `validation/wor27_diffusion_20261024/manifest.json`
- `validation/wor27_diffusion_20261024/summary.json`
- `validation/wor27_diffusion_20261024/convergence.png`

## Matter--radiation coupling

Command:

```bash
python3 tst/nrfld_validation/run_suite.py --case couple --max-cores 2 \
  --output-root validation --run-name wor27_couple_20261024 --no-plots
```

The run completed 100 cycles to code time 0.2485785864.  Fields remained
finite.  The maximum logarithmic gas-energy error was
`4.2907082e-4`, below the configured `5e-3` limit.  The test passed.

Artifacts:

- `validation/wor27_couple_20261024/couple/manifest.json`
- `validation/wor27_couple_20261024/summary.json`

## Test C smoother cross-check

The preceding Test C audit supplies the current candidate comparison.  With
variable limiter q=5e5, NR tolerance 1e-9 and a 200-cycle linear cap,
`jacobi-double` completes all four layouts while `jacobi-rb` fails immediately
on uniform-coarse and shifted AMR.  On the common passing uniform-fine and
aligned-AMR layouts, the final Er arrays are bitwise identical.  This supports
testing the existing `jacobi-double` as an operational fallback, but the
short-collapse restart A/B comparison, conservation comparison and production
timing remain open.

A second, smaller candidate is documented in
`wor25_nonmonotone_window_20261024.md`.  Allowing one transiently worsening
V-cycle (`linear_divergence_patience=2`) makes all eight q=5e5/7e5 Test C
layout controls pass, while the final HDF5 datasets of the three controls that
already passed are bitwise unchanged.  The runtime default remains one; this
candidate also requires the same collapse-state gate.

The candidate was then run through the two baseline regressions using the
same previously built executables.  Diffusion again gives fixed-diffusion-number
orders `1.9702, 2.0007`, independently extrapolated spatial orders
`1.9851, 2.0048`, and fixed-grid temporal orders `0.9890, 0.9945`.  All values
are identical to the default-policy baseline above.  Coupling again completes
100 cycles with maximum logarithmic gas-energy error `4.2907082e-4`.  Both
candidate regressions pass.

Candidate artifacts:

- `validation/wor27_diffusion_patience2_20261024/manifest.json`
- `validation/wor27_couple_patience2b_20261024/couple/manifest.json`

## Remaining gate

Use one identical pre-failure collapse restart and identical dt/tolerances for
`jacobi-rb` and `jacobi-double`.  Record solver termination reasons, Newton and
linear iterations, rejected cycles/backtracks, wall time, history columns and
field differences.  Do not interpret creation of a restart or raw MPI status
alone as numerical success.  No generic MG, gravity, post-MG smoother count or
implicit-update count should change in this comparison.
