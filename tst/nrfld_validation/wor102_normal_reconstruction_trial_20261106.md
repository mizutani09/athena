# WOR-102 normal reconstruction trial (2026-11-06)

Branch: `codex/wor102-normal-reconstruction-trial-20261106`, based on the
conservative tangential reconstruction prototype at `5bcc2862`. This branch
tests an additional normal-direction correction; it is not selected for the
main WOR-102 prototype.

With fine width `h`, the coarse center is at `-h`, the first fine center at
`h/2`, and the next fine center at `3h/2` relative to the common face. A
quadratic interpolation through those three centers gives the face normal
gradient `(-8 E_c/15 + E_f/3 + E_ff/5)/h`. The corresponding first-fine
ghost is `8 E_c/15 + 2 E_f/3 - E_ff/5`. A weight `theta` blends this ghost
with the selected symmetric two-point ghost. The coarse face uses coefficient
`(4/3 - 4 theta/15) Dbar` and the D-weighted effective fine value
`(1-w) E_f + w E_ff - tangential_offset`, with
`w = (theta/5)/(2/3 - 2 theta/15)`. This makes the two integrated face
contributions cancel algebraically for every `theta` in `(0,1]`. The same
linear formula is used for the Newton correction. The FLD limiter remains
fixed during the nonlinear solve.

The trial is opt-in with `ATHENA_NRFLD_EXPERIMENTAL_NORMAL_RECONSTRUCTION=1`
and `ATHENA_NRFLD_EXPERIMENTAL_NORMAL_WEIGHT` (default 1). It is restricted
to radiation-only static Test C with shared face coefficients. The
`wor102_profile_comparison.py` script accepts `--normal-reconstruction` and
`--normal-weight` for local tests.

For N32 one-step tests with assembled frozen Jv, the shared/existing update
L1 error ratios were:

| Case (opacity / amplitude / density ratio) | Tangential only | Tangential + full normal | Full normal diffusion ledger |
| --- | ---: | ---: | ---: |
| 0.4 / 0 / 1.15 | 1.134 | 0.384 | -4.64e-20 |
| 0.4 / 0.3 / 10 | 0.591 | 0.586 | -1.17e-20 |
| 4 / 0.3 / 1.15 | 0.469 | 0.430 | -1.27e-21 |
| 4 / 0.8 / 0.1 | 0.306 | 0.194 | -1.39e-20 |
| 4 / 0.8 / 10 | 0.545 | 0.516 | -5.29e-22 |
| 0.04 / 0 / 0.1 | 0.769 | 1.185 | -4.16e-19 |
| 0.04 / 0.3 / 0.1 | 0.317 | 0.715 | -3.34e-19 |
| 0.04 / 0.3 / 1.15 | 0.844 | 0.839 | -1.12e-19 |
| 0.04 / 0.8 / 0.1 | 0.323 | 0.738 | -2.98e-19 |
| 0.04 / 0.8 / 10 | 0.780 | 0.826 | -1.05e-19 |

All N32 cases converged in one Newton step and retained positive Er. The
N64 ratios were `0.399` (thick/strong, versus tangential-only `0.168`),
`0.686` (moderate, versus `0.707`), `0.274` (thin/strong, versus `0.198`),
and `0.491` (standard/flat, versus the existing solver). All four converged,
had positive Er minima, and had diffusion ledgers of order `1e-19` or less.

Partial weights did not produce one setting that improved both flat and thin
strong cases relative to tangential-only. At N32, `theta=0.25` gave ratios
`0.949` (standard/flat), `0.278` (thick/strong), `0.406` (thin/strong), and
`0.864` (thin/flat); `theta=0.5` gave `0.758`, `0.249`, `0.496`, and `0.964`.
The tangential-only ratios for those four cases were `1.134`, `0.306`,
`0.323`, and `0.769`. Thus a positive normal weight helps the flat case but
increases error in the thin cases tested. The quadratic ghost also has a
negative coefficient on the second fine cell, which may reduce monotonicity;
no broader positivity or production validation was performed. The normal
correction is not recommended for the main prototype without a justified
regime-dependent weight and corresponding validation.
