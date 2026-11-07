# WOR-92: why the strict CFL=0.3 FGMRES residual stops decreasing

## Finding

For the early Tomida table-EoS + Semenov opacity collapse at CFL=0.3, the second FGMRES solve in cycle 0 misses its true relative residual target of `2e-8`: 300 iterations leave `3.8796577724e-8`. The direct cause in this test is **finite-precision representation of the FGMRES correction in the solution vector, amplified by the diffusion-dominated operator**. This is more specific than attributing the failure only to “poor convergence,” a bad Hessenberg residual estimate, or a coarse-fine face defect.

The trace computes the requested correction `δx`, the realized double-precision update `fl(x+δx)-x`, the predicted residual `r_before-A(δx)`, and the true residual `b-A(fl(x+δx))`. It also evaluates `A(realized)-A(δx)` and `A(x_after)-A(x_before)-A(realized)` separately. The diagnostic performs no FGMRES state mutation. The diagnostic and uninstrumented baseline have identical first and second iteration counts and true residuals.

| Second solve, relative to `||b||` | Mixed AMR, table κR | Mixed AMR, fixed κR=0.0652 | Uniform level 1, table κR |
| --- | ---: | ---: | ---: |
| Projected residual | 1.8435841e-8 | 1.9388561e-8 | 6.5293624e-11 |
| Independently predicted `||r_before-A(δx)||` | 1.8435841e-8 | 1.9388479e-8 | 6.5293624e-11 |
| Recomputed true residual | **3.8796578e-8, fail** | **1.9855654e-8, pass** | **9.1188031e-8, fail** |
| `||true-predicted||` | 3.4105661e-8 | 4.3254383e-9 | 9.1188008e-8 |
| `||A(realized)-A(δx)||` | 3.3711449e-8 | 4.3208114e-9 | 9.1188008e-8 |
| `||A(x_after)-A(x_before)-A(realized)||` | 5.0329905e-9 | 1.6205653e-10 | 0 |

The individual gap vectors can cancel, so their norms should not be added as scalars. The near match of projected and independently predicted residuals shows that the FGMRES correction is internally consistent. The dominant discrepancy arises **when the correction is represented in the existing double-precision iterate**. In the mixed-AMR table run the requested correction has norm `3.5856920e-12`, while the cellwise difference between requested and realized corrections has norm just `1.3722745e-18`; applying the diffusion operator makes that representational difference `3.3711449e-8` relative to `||b||`. The current Schur operator has diagonal magnitudes up to `3.93e10` and reaction-to-diagonal fraction as low as `2.62e-11` in this case (earlier coefficient audit). The gap is distributed across interior cells (`2.298e-8`), same-level block faces (`1.681e-8`), and coarse-fine faces (`1.877e-8`); these disjoint region norms correspond to about 45%, 24%, and 30% of the squared gap. It is not exclusively an AMR interface effect.

The uniform-grid control is decisive: the final requested correction has norm `2.00114e-18`, but **all 262,144 active cells** are unchanged when it is added to the double-precision iterate. Thus the predicted residual is `6.53e-11`, yet the true residual remains `9.12e-8`. Its independently assembled componentwise backward error was `8.889e-17` in the earlier uniform audit, below double epsilon. A further Krylov iteration cannot improve a correction that the stored state cannot represent. The fixed-κR control has a much smaller update gap and just passes the strict true-residual gate; fixing κR is a conditioning diagnostic, not a justified opacity replacement.

## Improvement trials and decision

| Isolated update trial, mixed-AMR table κR | First true residual | Second true residual | Strict `2e-8`? |
| --- | ---: | ---: | --- |
| Original double updates | 1.2189076e-8 | 3.8796578e-8 | No |
| Kahan compensation across Krylov basis updates | 1.2163878e-8 | 3.8816403e-8 | No |
| Form restart update in `long double`, cast once to `Real` | 1.2150815e-8 | 3.1622392e-8 | No |

Kahan compensation does not solve this case. Forming each restart correction in extended precision improves the outcome but cannot satisfy the target when the final solution is still stored in double precision. Neither trial is a validated production fix. Earlier tests also found that long-double Hessenberg/Givens, more MG V-cycles, altered smoother weight, and deeper Arnoldi did not meet this strict second-solve target; merely raising the 300-iteration cap is therefore not supported.

The practical route already supported for an **early local pilot** is the general-EOS exact local gas root plus a bounded inexact-Newton linear gate using the independently recomputed **true** residual, with actual Newton decrease and final nonlinear tolerances enforced. A `1e-7` true-residual cap reproduced the same 30 early CFL=0.3 cycles as the previous floor trial, including bitwise identical ten-cycle state arrays; 21 of 62 linear solves exceeded the strict `2e-8` gate, with maximum `6.21057e-8` [inexact-Newton audit](wor92_inexact_newton_20261107.md). The local root accurately solves the mixed-frame table gas exchange, but does not by itself remove this linear precision barrier. The cap is a pilot policy, not proof of the complete collapse: CFL=0.15 failed at cycle 1 with `1.461e-6`, tighter Newton tolerance failed at cycle 0, and the CFL=0.3 continuation stopped at cycle 78 due to a trial opacity lookup below the table domain. Energy and AMR-face discrepancies also require an accuracy budget.

If strict `2e-8` convergence is required, the next implementation experiment should target the operator and state precision together: assess scaled variables or stronger Schur preconditioning to reduce amplification, then test whether higher-precision correction/state storage is necessary. Raising the inner tolerance alone is acceptable only with a bounded true-residual gate, Newton convergence, step rejection or `dt` reduction on failure, and independent conservation/physics checks. An adaptive Newton forcing term may avoid oversolving early linear systems; it needs a matched-time CFL and long-run validation. Semenov-versus-Optab definitions remain a separate opacity-model question.

## Reproduction and artifacts

The [JSON](wor92_roundoff_mechanism_results_20261107/summary.json), [CSV](wor92_roundoff_mechanism_results_20261107/solves.csv), [figure](wor92_roundoff_mechanism_results_20261107/roundoff_mechanism.png), [logs/inputs/diagnostic patches](wor92_roundoff_mechanism_results_20261107/evidence.tar.gz), and [audit generator](wor92_roundoff_mechanism_audit_20261107.py) record the exact runs. The copied-build changes are diagnostic or rejected trials; no solver source was committed. The underlying report-branch Git hash before this audit is `34ebf35869833aedb35cedf8306adc38ccbd0e0a`. Baseline binary SHA-256: `5b259bf3c0a4b768a6a35aa9f2d9cebfd3b9151f78b80ed5583069a772ffb691`. Final step-trace binary: `7d1a32d675881cc6c923dac9799c4313f753b1ecb97507b113623cf8fbdfa216`. Kahan binary: `aff547b72fd46a94c312615dedbbf73c0fd44107f12b701b13ce25af0c945090`. Extended-update binary: `095e02ede7ceb47c527dd3adcb6bd4147dbd7ffbe2572ef96fe78ea3656c5f5c`. Full hashes also appear in the JSON.

Regenerate all tables and the figure from the report worktree:

```bash
python3 tst/nrfld_validation/wor92_roundoff_mechanism_audit_20261107.py
```

Representative mixed-AMR diagnostic command from `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, with a fresh output directory:

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_STEP_TRACE=1 \
  timeout 120s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_step_trace_trial/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_component_freeze_cfl030_strict_3cycles.used \
  -d /path/to/fresh/output
```

The fixed-κR control uses `input_table_fixed_rosseland_high_cfl030_strict_3cycles.used` and `ATHENA_WOR92_FIXED_ROSSELAND=1`; the uniform control uses `input_uniform_l1_restart40_mg2_strict.used`. For Kahan or extended-update controls, replace the binary with the corresponding copied build and `ATHENA_WOR92_STEP_TRACE=1` with `ATHENA_WOR92_KAHAN_X=1` or `ATHENA_WOR92_LONG_DOUBLE_UPDATE=1`. **Omit** `ATHENA_WOR92_FIXED_ROSSELAND` entirely in table-opacity runs. The input and patches are archived. The measured first `dt` is `31.114819909` code units. No XD2000 access, merge, or production solver commit occurred. WOR-92 remains **In Progress**.
