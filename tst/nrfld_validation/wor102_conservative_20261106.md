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
| `ATHENA_NRFLD_EXPERIMENTAL_NONLINEAR_JV=1` | Refresh the variable limiter at each Newton trial and differentiate that residual | Radiation-only static AMR, shared coefficients and FGMRES only; four closure evaluations per Jv |
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
secant is evaluated in long double to avoid cancellation. A separately gated
trial below differentiates a refreshed, nonlinear flux limiter.

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
untested. The separately gated nonlinear Jv is validated only in local Test C
cases. Production-scale profile accuracy and performance measurements remain
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

## Follow-up: block-face conservation and profile accuracy

At base x1 resolution `64` (144 AMR leaf blocks), the shared-coefficient
trial exposed a smaller defect that the earlier 18-block case missed. With
density ratio `0.1`, transverse amplitude `0.8`, and `dt=5e-5`, the x1
coarse/fine face sums canceled to about `1e-20`, but the domain diffusion
ledger was `+2.94820e-11`. An opt-in extension of
`ATHENA_NRFLD_INTERFACE_LEDGER=1` to all six block faces located the entire
defect on y block boundaries adjacent to the left x1 coarse/fine interface.
Independent variable-limiter evaluations there differed across otherwise
same-level fine blocks. The shared-coefficient path now copies the positive
side's face `D` to the negative side using the **existing** face-moment
exchange. This adds no new exchange; the feature remains under the shared
coefficient switch. Afterward, summed x/y/z block-face contributions were
`-5.2e-21`, `0`, and approximately `0`, respectively, and the volume
diffusion ledger was `-4.48e-21`. Four-rank MPI reproduced these face sums
and its final Er integral differed from serial by `8.3e-17`.

The profile comparison uses the same physical domain and timestep for both
resolutions: x1 `[0,1]`, x2/x3 `[0,0.25]`, `dt=5e-5`, base x1 `N=32/64`,
and uniform-fine x1 `2N`. It compares each AMR leaf value to the uniform-fine
values **averaged over that leaf**. To reduce the effect of differently
sampled initial data, the primary metric is mean absolute error in the
one-step **change** of Er. VTK output is float32; measured errors below are
far above its rounding scale. The reference is a finer numerical solution,
not an analytic truth, and the initial discretizations still differ.

| Density ratio, transverse amplitude | Base N | Existing update L1 | Shared update L1 | Shared / existing | Existing diffusion ledger | Shared diffusion ledger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1, 0.8 | 32 | `4.612e-4` | `3.845e-4` | `0.834` | `+1.47e-6` | `+1.22e-20` |
| 0.1, 0.8 | 64 | `3.566e-4` | `1.922e-4` | `0.539` | `+1.09e-6` | `-4.48e-21` |
| 1.15, 0.3 | 32 | `7.618e-5` | `9.684e-5` | `1.271` | `-1.44e-7` | `+5.49e-21` |
| 1.15, 0.3 | 64 | `2.094e-5` | `4.214e-5` | `2.013` | `-7.95e-8` | `-7.67e-21` |

Thus the conservation fix improves profile accuracy for the strong case but
reduces it for the moderate case. The two-point prototype without shared
face coefficients has almost the same update errors as the shared variant,
while retaining nonzero diffusion ledgers (`+6.42e-9` at N64 in the strong
case). Face coefficient exchange is required for conservation; the profile
tradeoff comes mainly from the chosen symmetric two-point coarse/fine
discretization. Adoption should await a higher-accuracy conservative face
reconstruction and tests over more gradients and optical depths. The
single-run wall times are not a reliable overhead estimate; shared exchange
remains switchable for later repeated timing.

The new same-level synchronization passed the previous eight-case
conservation gate, x2/x3 one-step face ledgers (`|sum| < 2e-20`), the adaptive
five-step refine/derefine probe (`|ΔEr/Er|=3.61e-14`), and the coupled five-step
trial (Newton iterations `22,12,4,4,4`, cumulative diffusion ledger
`-5.06e-15`). At N64, the frozen global Jv relative mismatch was
`1.29e-11` on same-level block faces and `6.63e-12` on coarse/fine faces
at epsilon `1e-4`. The refreshed-limiter mismatch in the lagged solver is
nonzero; a separately gated nonlinear Jv trial is described below.

Reproduce both profile conditions with:

```sh
python3 tst/nrfld_validation/wor102_profile_comparison.py \
  --binary bin/athena --output /tmp/wor102_strong --resolutions 32 64 --plot
python3 tst/nrfld_validation/wor102_profile_comparison.py \
  --binary bin/athena --output /tmp/wor102_moderate --resolutions 32 64 \
  --contrast 1.15 --transverse-amplitude 0.3 --plot
```

## Additional closure and optical-depth configurations

The same one-step, N32 update-error comparison was repeated at fixed
`dt=5e-5` for opacity `0.04` (thin), `0.4` (standard), and `4` (thick),
transverse amplitude `0`, `0.3`, or `0.8`, and density ratio `0.1`, `1.15`,
or `10`. Every case converged. The ratio below divides the shared-face
update L1 error by the existing-stencil update L1 error; a value below one
means the shared-face trial was closer to the uniform fine solution. The
reference uses the lagged closure in this table.

| Opacity / amplitude / density ratio | Existing update L1 | Shared update L1 | Ratio | Existing diffusion ledger | Shared diffusion ledger |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.4 / 0 / 1.15 | 4.057e-6 | 4.601e-6 | 1.134 | -1.803e-7 | -1.028e-20 |
| 0.4 / 0.3 / 10 | 6.691e-5 | 8.228e-5 | 1.230 | -1.166e-7 | -6.204e-25 |
| 4 / 0.3 / 1.15 | 9.757e-6 | 1.270e-5 | 1.301 | -1.430e-8 | -2.958e-22 |
| 4 / 0.8 / 0.1 | 7.917e-5 | 1.151e-4 | 1.454 | +4.053e-9 | +1.390e-22 |
| 4 / 0.8 / 10 | 1.874e-5 | 2.344e-5 | 1.251 | -3.897e-9 | +5.630e-23 |
| 0.04 / 0 / 0.1 | 6.391e-5 | 4.916e-5 | 0.769 | -3.215e-6 | -1.055e-19 |
| 0.04 / 0.3 / 0.1 | 8.225e-4 | 2.979e-4 | 0.362 | -3.371e-7 | +5.445e-20 |
| 0.04 / 0.3 / 1.15 | 2.975e-4 | 3.478e-4 | 1.169 | -1.398e-6 | -1.352e-20 |
| 0.04 / 0.8 / 0.1 | 1.996e-3 | 7.679e-4 | 0.385 | -4.666e-6 | -7.470e-20 |
| 0.04 / 0.8 / 10 | 1.052e-3 | 1.211e-3 | 1.151 | -2.641e-7 | -4.642e-20 |

At N64, the thick/strong/ratio-0.1 case had a shared/existing error ratio
`2.102`, while the thin/strong/ratio-0.1 case had a ratio `0.183` with the
assembled frozen Jv. In that thin case the existing solver's diffusion
ledger was `+1.162e-5` and the shared-face ledger was `+1.482e-19`. The
default frozen secant Jv stalled in the second FGMRES solve (200 iterations,
relative residual `2.96e-3`); the assembled frozen Jv converged in two
Newton steps and 18 total FGMRES iterations for the shared-face variant.
Thus conservation is robust in this sweep, but accuracy and solver behavior
depend substantially on the physical parameters.

## Experimental nonlinear-limiter Jv

`ATHENA_NRFLD_EXPERIMENTAL_NONLINEAR_JV=1` is a separate opt-in trial. The
lagged solver computes the limiter once for a timestep. This trial instead
recomputes it after each Newton trial or rollback, and FGMRES applies a
two-scale centered difference of the same refreshed residual. Richardson
combination removes the leading centered-difference error. Each Jv therefore
requires four coefficient/face exchanges and residual evaluations, in
addition to the existing frozen MG preconditioner. The flag is gated to
periodic Cartesian Test C, radiation-only, static AMR, shared coefficients,
variable limiter, and `nrfld/linear_solver=fgmres`.

For standard opacity, density ratio `0.1`, amplitude `0.8`, N32, the uniform
fine nonlinear reference converged in three Newton steps / 18 FGMRES
iterations; the AMR shared-face case converged in four / 29. Its one-step
update L1 error was `3.849e-4` against that **nonlinear** reference and its
diffusion ledger was `+4.06e-21`. A four-rank run reproduced the final Er
integral to the printed precision. The independent one-sided global Jv
diagnostic at the coarse/fine face gave relative L2 `2.59e-8` for a smooth
direction at epsilon `1e-4` and `2.72e-7` for the first Newton direction;
the lagged/frozen Jv mismatch on the same Newton direction was `2.32e-3`.
The FGMRES linearity check gave additivity `8.58e-9` and homogeneity
`5.17e-14`. These values support the local nonlinear derivative but do not
establish production robustness.
Five N32 steps at the same physical parameters converged in Newton counts
`4,4,3,3,3`; the cumulative Er integral changed by `-1.96e-15` relative.

For the difficult N64 thin/strong/ratio-0.1 AMR case, the nonlinear trial
converged in nine Newton steps, with FGMRES counts `19,11,10,10,10,10,10,10,10`
and final L2 `2.92e-12`; its diffusion ledger was `-8.40e-20`. A single
centered difference at a smaller perturbation had stalled at the 200-iteration
FGMRES limit, motivating the two-scale derivative. The nonlinear model costs
many more closure exchanges than the frozen variants. The lagged uniform
fine reference is not a valid accuracy comparator for it; use
`--nonlinear-jv` to run a matching nonlinear uniform fine reference.
That N64 nonlinear reference converged in five Newton steps / 47 FGMRES
iterations. Against it, the nonlinear shared-face AMR update L1 error was
`3.075e-4` (interface mean `1.085e-3`). The single-run wall times were
`37.0 s` for the uniform fine reference and `45.6 s` for the AMR case;
the corresponding frozen assembled-Jv cases took `2.70 s` and `1.96 s`.
These timings include process startup and output, and quantify only this
small local case. The nonlinear AMR result is close to the frozen result
(`2.949e-4`) but solves a different discrete nonlinear equation.

Reproduce the matching nonlinear-reference comparison with:

```sh
python3 tst/nrfld_validation/wor102_profile_comparison.py \
  --binary bin/athena --output /tmp/wor102_nonlinear_thin \
  --resolutions 64 --opacity 0.04 --contrast 0.1 \
  --transverse-amplitude 0.8 --nonlinear-jv
```
