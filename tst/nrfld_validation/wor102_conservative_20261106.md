# WOR-102: conservative coarse/fine FLD prototype (2026-11-06)

Branch: `codex/wor102-conservative-fld-20261106`. The prototype is opt-in via
`ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1`. It is guarded to the periodic,
Cartesian, cubic, static Test C problem with radiation-only, uncoupled,
fixed-limiter diffusion by default. The default solver path is unchanged.
An additional `ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1` flag allows
variable-limiter trials. `ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1` turns on
fine-face coefficient exchange; leaving it off reproduces the defect.
`ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0` switches FGMRES to the assembled
frozen-coefficient operator for a cost comparison. The finite-difference
operator is the default under the prototype flag. For the coefficient and
variable-limiter flags, `1` enables and `0` or unset disables the feature.

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

The fixed-limiter prototype remains limited to static periodic Test C. The
variable-limiter extension below explicitly exchanges shared coefficients,
but is also limited to this test problem. Gas/radiation coupling, physical
boundaries, regrid/restart, and production Mach 5 have not been validated.
A nonlinear residual Jv (or an equivalent validated Jacobian) and performance
measurements remain before an XD A/B run.

## Follow-up local tests: variable limiter and profile accuracy

Test C now accepts `problem/transverse_amplitude` to vary the radiation field
along the coarse/fine face. With a variable limiter, a density ratio of `0.1`
and transverse amplitude `0.8`, the prototype converged but its AMR-aligned
`ΔEr/Er` was `+4.97550e-7` (AMR shifted: `+6.91023e-7`). Uniform coarse and
fine runs remained within `8.2e-14` of conservation. The aligned interface
ledger sums were `+1.19299e-8` at x1=`0.25` and `+1.97900e-8` at x1=`0.75`;
their sum matches the nonzero integrated diffusion contribution. Keeping the
fixed limiter in the same transverse test reduced the drift to `5.3e-15` or
less over density ratios `0.1`, `1`, and `10`. Thus the current coarse/fine
coefficient evaluation is the remaining conservation defect, not Newton
nonconvergence. For this strong transverse case, the frozen Jv check was
`5.00e-12` but the refreshed-limiter Jv check was `1.65e-6` at the interface.

Reproduce the failing test after an MPI or serial Test C build with:

```sh
ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
python3 tst/nrfld_validation/wor101_conservation_audit.py \
  --binary bin/athena --output /tmp/wor102_variable_transverse \
  --linear-solver fgmres --variable-limiter \
  --transverse-amplitude 0.8 --contrasts 0.1
```

For the constant-density one-step profile, a VTK comparison against uniform
fine (`64×16×16`) gave volume-weighted mean absolute error `5.1461e-5` for
the default AMR stencil and `5.5864e-5` for the conservative prototype;
maximum absolute error was respectively `3.9542e-4` and `6.0964e-4`. The VTK
values are float32, whose rounding is much smaller than these differences.
At the next x1 resolution (`64` coarse cells, `128×16×16` uniform reference),
the mean errors were `1.3893e-5` and `1.5407e-5`; maxima were `1.9372e-4`
and `1.9443e-4`. Both mean errors decreased by factors of `3.70` to `3.62`
when x1 resolution doubled, but the symmetric two-point face retained a
roughly `9–11%` mean-error cost for these two meshes. Before adoption,
consider a tangential reconstruction that preserves one shared face flux.

## Shared fine-face coefficient trial

The selected extension evaluates the existing full-gradient FLD closure on
each fine subface. It sends first-layer averages of `D_f` and `D_f E_f` from
fine blocks to the adjacent coarse block using a dedicated boundary variable.
The coarse face then uses `D_c=(4/3) average(D_f)` and the D-weighted fine
ghost `E_g=average(D_f E_f)/average(D_f)`. Fine faces retain their individual
`D_f`. This makes coarse and fine integrated face contributions opposite for
the frozen coefficient field, including transverse variation. The same
weights restrict FGMRES directions. The exchange occurs after the initial
face closure calculation and after each Newton trial update, including across
MPI ranks. It allocates 12 additional Real components per active/ghost cell
while enabled; the memory and communication cost require later measurement.

The variable-limiter eight-case audit with transverse amplitude `0.8` and
density ratios `0.1` and `1.15` passed the `1e-10` conservation gate under
shared coefficients. For the difficult aligned `0.1` case, `ΔEr/Er` changed
from `+4.97550e-7` (shared exchange off) to `+6.09512e-15` (on). Its
x1=`0.25` face ledger sum changed from `+1.19299e-8` to `+2.12e-20`; the
x1=`0.75` sum changed from `+1.97900e-8` to `-3.58e-20`. It converged in one
Newton step and eight FGMRES iterations. Five timesteps gave cumulative
`ΔEr/Er=+1.43671e-14` (serial) and `+1.45848e-14` (four ranks), versus
`+1.01988e-6` with the shared exchange off. One-step x2/x3 face trials
also converged in serial and four ranks with `|ΔEr/Er| < 7e-15`. Turning
difference Jv off retained convergence and conservation in this frozen-D
trial (eight FGMRES iterations); no timing conclusion is drawn from one run.
Static-AMR restart after cycle 2 reproduced the direct five-step final time
and `Er` volume integral exactly, both at one rank and when restarting the
one-rank checkpoint on four ranks.

Reproduce the passing gate with the three opt-in flags and the preceding
audit command, adding `ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1`,
`--contrasts 0.1 1.15`, and `--require-conservation`.

The legacy `global_jacobian_check` does not incorporate the new coefficient
exchange and is explicitly rejected while shared exchange is on. A new
Jacobian check against the exchanged residual, nonlinear limiter refresh,
coupled gas/radiation, adaptive regrid, physical boundaries, and profile
accuracy for this variable-limiter scheme remain before production adoption.
