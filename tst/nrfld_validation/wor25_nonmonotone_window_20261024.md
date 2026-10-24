# WOR-25: bounded non-monotone V-cycle window (2026-10-24)

## Question

The existing linear solve rolls a V-cycle back as soon as that cycle increases
the linear residual.  The independent reference showed that the failing
variable-limiter case has the sequence `1.020719, 0.818972, 0.667085, ...`
relative to its initial residual when iteration is allowed to continue.  This
test asks whether allowing exactly one worsening cycle removes the artificial
hard failure without changing cases whose first cycle already decreases the
residual.

The production-compatible default remains `linear_divergence_patience=1`.
The candidate is selected explicitly with `linear_divergence_patience=2` and
retains an immediate growth bound of two relative to the best residual.  No
smoother, transfer, coarse operator, post-smoothing count, implicit update,
Newton tolerance, gravity option or physical term was changed.

## Controlled matrix

Both matrices used the same isolated executable, variable limiter, contrast 1,
NR threshold `1e-9`, at most 30 V-cycles, and the four Test C layouts.  The
nominal fine-grid diffusion numbers were `5e5` and `7e5`.

| q | layout | patience=1 | patience=2 | NR iterations with patience=2 |
|---:|---|---|---|---:|
| 5e5 | uniform coarse | fail at first rejection | pass | 5 |
| 5e5 | uniform fine | pass | pass | 5 |
| 5e5 | aligned AMR | pass | pass | 11 |
| 5e5 | shifted AMR | fail at first rejection | pass | 11 |
| 7e5 | uniform coarse | fail at first rejection | pass | 6 |
| 7e5 | uniform fine | fail at first rejection | pass | 6 |
| 7e5 | aligned AMR | pass | pass | 11 |
| 7e5 | shifted AMR | fail at first rejection | pass | 10 |

Every candidate run had zero rejected V-cycles and finite positive fields.
For the three cases which pass under both policies, every dataset in the final
Athena HDF5 output is bitwise identical.  Thus the option is inactive on these
already-monotone paths.

Artifacts:

- `tst/outputs/wor25_patience1_matrix_20261024/manifest.json`
- `tst/outputs/wor25_patience2_matrix_20261024/manifest.json`
- isolated executable under `tst/outputs/wor25_patience1_20261024/build_source/bin/athena`

## Interpretation and remaining gate

This establishes a direct failure in the stopping policy for the reduced
problem: a bounded, transient residual increase is converted into a zero
correction and an NR failure when patience is one.  It does not show that every
collapse failure has the same mechanism, nor does it repair the measured
rediscretized-versus-Galerkin hierarchy weakness or the separate shifted-AMR
boundary mismatch.

Before changing the default, run the option on the same pre-failure collapse
restart and compare it with the default and `jacobi-double`.  Old restart
headers may not contain the new parameter, and Athena command-line overrides
cannot add an absent key.  `job_mayer5_smoother_ab_pdebug.sh` handles this with
the supported `-r RESTART -i MINIMAL_OVERLAY` merge; it does not edit the binary
restart or replace unrelated restart parameters.
