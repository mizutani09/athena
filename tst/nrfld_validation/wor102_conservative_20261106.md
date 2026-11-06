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
The coupled and adaptive-regrid diagnostic flags are described below and are
also off unless set to `1`.

| Switch | Enabled effect | Cost / current gate |
| --- | --- | --- |
| `ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1` | Conservative two-point interface and FGMRES | Master opt-in; unset uses existing solver |
| `ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1` | Exchange per-fine-face `D` and `D E_r` | 12 Real components per cell, boundary exchange each Newton coefficient refresh |
| `ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0` | Use assembled frozen Schur operator instead of secant Jv | Required for experimental coupled runs; `1` or unset selects secant in radiation-only trials |
| `ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1` | Permit variable-limiter Test C | Trial gate; coefficient exchange remains independently switchable |
| `ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1` | Permit coupled Test C | Requires `DIFFERENCE_JV=0` |
| `ATHENA_NRFLD_EXPERIMENTAL_REGRID=1` | Permit adaptive Test C | More blocks and boundary exchanges during refinement |
| `nrfld/global_jacobian_check=true` | Run global Jv diagnostic | Additional residual/coefficient evaluations; default false |

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

The prototype remains limited to periodic Cartesian Test C. Coupled and
adaptive runs below are controlled probes, not production validation.
Physical boundaries, moving AMR interfaces, and production Mach 5 remain
untested. A nonlinear residual Jv (or an equivalent validated Jacobian),
profile accuracy for variable closure, and performance measurements remain
before an XD A/B run.

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

The `global_jacobian_check` now incorporates the face-moment exchange. For the
strong transverse case, the coarse/fine frozen-coefficient Jv mismatch was
`5.06e-12` at epsilon `1e-4` in both serial and four-rank runs. The
refreshed-limiter mismatch was `1.67e-6`, confirming that the current Jv
describes the solver's lagged closure rather than a fully nonlinear limiter.
Production coupled gas/radiation, physical boundaries, and profile accuracy
for this variable-limiter scheme remain before adoption.

## Coupled one-step and five-step diagnostics

`ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1` additionally permits Test C gas/radiation
coupling, only with `ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0`. The current
secant Jv is defined for the radiation-only residual and omits gas elimination.
An attempted local Schur secant matched the assembled operator to approximately
`5e-16` initially, but drifted to `1.4e-9` in later Krylov calls, causing
FGMRES to miss its `1e-10` target after 200 iterations. Increasing the secant
step to `0.5` did not remove cancellation in the stiff coupled residual.
The coupled trial therefore uses FGMRES with the assembled Schur operator.
The selected coupled difference-Jv path remains outstanding.

At density ratio `0.1`, transverse amplitude `0.8`, and variable limiter,
one step converged both with and without shared coefficient exchange. The
implicit gas+radiation energy ledger's domain-summed diffusion contribution
was `+1.62019e-7` without exchange and `-3.21e-18` with exchange; the local
equation balance was below `5e-17` in both runs. Thus the difference was a
face conservation defect, not an unconverged local equation.

That deliberately stiff case failed at cycle 2 in the existing solver and
both prototype variants, so it is unsuitable for a multi-step comparison.
With density ratio `1.15`, transverse amplitude `0.3`, and opacity `0.004`,
all variants converged for five steps. The five implicit ledgers summed to
`+9.64841e-3` without shared exchange and `-3.80e-15` with it. The maximum
single-step diffusion contribution fell from `4.79494e-3` to `3.60e-15`.
The corresponding Newton iteration counts were `22,14,6,7,7` and
`22,12,4,4,4`. Four-rank MPI reproduced the shared variant's iterations,
and its final radiation integral differed from serial by about `1e-15`.
The coupled difference-Jv remains to be completed. Rebuilding after the
coupled guard and shared Jv audit change, the eight variable-limiter
conservation cases again passed with `|ΔEr/Er| <= 8.19e-14`.

An initial local 20-step timing check on the variable-limiter Test C case
(density ratio `0.1`, transverse amplitude `0.8`) kept all runs converged.
With shared exchange off, the measured wall time was `2.00 s` and cumulative
`ΔEr/Er=1.05e-6`; with exchange on and difference Jv on it was `2.13 s` and
`2.61e-14`. Disabling difference Jv while retaining exchange measured
`2.07 s`. Each mode used 156 total FGMRES iterations. These are single-run,
small-grid measurements and are not a production performance estimate.

## Adaptive regrid and restart probe

`ATHENA_NRFLD_EXPERIMENTAL_REGRID=1` permits the same opt-in scheme on Test C
adaptive AMR. With density ratio `0.1`, transverse amplitude `0.3`, and
variable limiter, the local five-step run refined from 4 to 18 blocks at
startup and returned to 4 blocks after cycle 2. All Newton solves converged.
The cumulative radiation energy change was `3.61e-14` in serial and
`3.64e-14` on four ranks. Restarting a serial cycle-2 checkpoint on four
ranks reproduced the continuous four-rank run's final time and radiation
integral exactly. This covers one controlled refine/derefine transition;
more complex moving AMR boundaries remain untested.
