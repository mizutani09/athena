# WOR-102: conservative coarse/fine FLD prototype (2026-11-06)

Branch: `codex/wor102-conservative-fld-20261106`. The prototype is opt-in via
`ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1`. It is guarded to the periodic,
Cartesian, cubic, static Test C problem with radiation-only, uncoupled,
fixed-limiter diffusion. The default solver path is unchanged.

## Interface stencil

For coarse width `H` and fine width `h=H/2`, the centers are `3H/4` apart.
At each fine face, use the two-point flux `D (E_c-E_f)/(3H/4)`. The fine ghost
is `(E_f+2E_c)/3`; the coarse face coefficient is multiplied by `4/3`, and
the coarse ghost receives the average of the four first-layer fine cells.
Consequently, for a common face `D`, the coarse face contribution equals the
area sum of the four fine face contributions with opposite sign. The same
ghost operation is applied to the Newton iterate and to each Krylov direction.

FGMRES uses a finite-difference secant of the frozen-coefficient radiation
residual for `Jv`, with the existing MG V-cycle as a preconditioner. The
secant is evaluated in long double to avoid cancellation. The implementation
does not yet differentiate a refreshed, nonlinear flux limiter.

## Local results

Build: `python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld
--nghost=2 --cxx=g++ -mpi`, then `make -j8`.

The existing eight-case conservation audit passed with the prototype under
both MG and FGMRES (`--require-conservation`, tolerance `1e-10`). For the
AMR-aligned, constant-density case:

| Mode | Relative `Er` change | Newton final L2 | FGMRES iterations |
| --- | ---: | ---: | ---: |
| Default, MG | `-1.0299531437e-5` | `2.014e-13` | — |
| Prototype, MG | `+8.4461015e-14` | `2.598e-13` | — |
| Prototype, difference Jv + FGMRES | `+1.5237812e-15` | `1.066e-16` | 7 |

The x1=`0.25` interface ledger was `-6.56619e-7` (sum of opposing sides)
in the default run and `-3.70e-20` with the prototype. The frozen global
Jacobian check at the coarse/fine face gave relative L2 `1.39e-12` for
epsilon `1e-4`. FGMRES linearity checks gave additivity `1.22e-15` and
homogeneity `4.25e-15`.

Changing the pulse and refinement plane to x2 or x3 reproduced the default
drift (`-1.02995e-5`), while the prototype drift was respectively
`+1.52e-15` and `0`. Four-rank MPI runs on x2/x3 gave `+1.31e-15` and
`-2.18e-16`. For x1, serial, two-rank, and four-rank runs produced the same
history values and seven FGMRES iterations. The density-ratio `0.1` and
`10` one-step probes also converged and had `|ΔEr/Er| <= 3.1e-15`.

## Limits before production adoption

The proof of face cancellation assumes a *shared* diffusion coefficient.
The current prototype reuses each side's existing face coefficient, and
agreement at the tested density contrasts does not prove this for arbitrary
opacity or a variable limiter. It is still limited to static periodic Test C;
gas/radiation coupling, physical boundaries, regrid/restart, and production
Mach 5 have not been validated. A production implementation needs an
explicitly shared face coefficient/flux, a nonlinear residual Jv (or an
equivalent validated Jacobian), profile accuracy against uniform fine, and
performance measurements before the XD A/B run.
