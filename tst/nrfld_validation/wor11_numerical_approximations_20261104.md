# WOR-11: NR-FLD approximations and conservation audit (2026-11-04)

## Scope and result

This is an audit of the current double-precision Cartesian NR-FLD implementation,
not a claim that every FLD model error is below a specified tolerance. The
production validation cases use 3D meshes (often thin slabs) with cubic cells.
This audit starts from clean commit
`209e2239872a36b8ec278d5472db7b8724a18ea1`. The historical full-suite
manifest under `validation_suite_complete_20261103` records commit
`7a44d202c1fe9a3dbe267619ceff67ab016e1fae` **with a dirty source tree**;
its case results are prior evidence, not a fresh full-suite execution of this
branch. The suite manifest and per-case build snapshots preserve provenance.
This audit adds a fail-fast
geometry check and a few precision-safe constants on
`codex/wor11-numerics-audit`; the final commit is recorded in Linear.

| Component | Current discretization / approximation | Consequence |
| --- | --- | --- |
| Closure | Levermore-Pomraning `lambda=(2+R)/(6+3R+R²)`, `R=|grad Er|/(sigma_R Er)`; `chi=lambda+(lambda R)²` in `src/fld/fld.hpp`. Large-`R` formulas avoid overflow. | `lambda→1/3` and `chi→1/3` for diffusion; `lambda R→1` and `chi→1` for streaming. These are FLD approximations, so a sharp optically thin front is not an exact transport solution. |
| Face opacity | `FaceOpacity` caps the harmonic mean by `(4/3)/dx`, then takes the lesser of that and the arithmetic mean. | This Howell-Greenough style rule regularizes sharp jumps and transparent faces; it changes the face diffusion coefficient relative to a pure harmonic mean. |
| Hydro/radiation transport | Reconstructed `Er, lambda, alpha` share hydro faces. HLLC uses `p_g+lambda Er` in wave construction. Gas flux subtracts `(1+lambda)Er v_contact`; radiation advects `alpha Er v_contact`, `alpha=(3-chi)/2`. | In uniform states the gas subtraction recovers the pure-gas enthalpy flux. Away from thick/thin limits, `alpha` and `1+lambda` differ. This is a split mixed-frame approximation. |
| Force and work | `-lambda grad Er` acts explicitly on gas momentum. The hyperbolic mixed-frame work `-lambda v·grad Er` is explicit, and `q v·grad Er`, `q=2 lambda sigma_P/sigma_R`, is implicit by default, with an explicit comparison switch. | The implementation is an O(v/c) mixed-frame split. It does not evolve an independent radiation momentum equation or discretize a separate full tensor `P:grad v` term. Its pressure/work accuracy must be judged as a coupled discretization. The retired `cut_Pnablav` parameter is rejected. |
| Matter/radiation exchange | Planck/Rosseland opacities and face closure/diffusion/drift coefficients are fixed for each implicit stage. Newton updates `c sigma_P(aT⁴-Er)`, the EOS derivative, local gas solve, Schur diagonal and RHS at each iterate. | The nonlinear solve converges for fixed stage coefficients; it does not Newton-differentiate `lambda(R)` or opacity. Strongly temperature-dependent opacity therefore needs a separate timestep/lagging convergence check. |
| Linear solve | Coupled gas+radiation defect uses a directly formed total-energy residual to avoid cancellation. FGMRES solves the fine-grid linear system with a multigrid V-cycle preconditioner; smoothing is red-black Gauss-Seidel. Uniform Galerkin and multilevel rediscretized paths have different operator contracts. | A small FGMRES residual alone does not guarantee a conservative nonlinear AMR operator. Newton acceptance/backtracking and the multigrid handoff remain separate checks. |
| Geometry | Implicit diffusion and multigrid use `dt/dx1²` for all active axes. The Newton driver already rejects nonuniform mesh generators; the new constructor guard rejects non-Cartesian and non-cubic active cells. | Rectangular-cell results would otherwise use inconsistent directional scaling. Full anisotropic/metric-aware support requires changing residual, Jacobian, MG, boundary flux, and diagnostics together. |

Relevant implementations: `src/fld/fld.hpp`,
`src/fld/calculate_radiation_face_states.cpp`,
`src/hydro/rsolvers/hydro/hllc_fld.cpp`,
`src/fld/add_mixed_frame_source.cpp`,
`src/nr_multigrid/NRFLD.cpp`, and
`src/linear_solver/linearMG/linearMG.cpp`. See also the separate coefficient
cache audit `wor30_coefficient_update_audit_20261023.md` and Schur audit
`wor26_schur_rhs_20261104.md`.

## Measured limits

| Limit / test | Evidence | Interpretation |
| --- | --- | --- |
| Thick diffusion | `/mnt/d/output/rad/validation_suite_complete_20261103/cases/diffusion/`: coupled space-time refinement orders 2.0118 and 2.0058; time-extrapolated spatial orders 2.0123 and 2.0057; temporal difference order 1.00017. | The intended second-order spatial and first-order temporal behavior is observed in this smooth cubic-cell case. |
| Marshak diffusion wave | Same suite, `cases/marshak/`: passed analytic profile/L1 checks with zero applied spatial shift; the fixed-time-step L1 table is monotone but includes time error. | The analytic comparison works in this setup; raw fixed-dt rates alone are not a proof of second-order space accuracy. |
| Thin streaming | Same suite, `cases/streaming/`: both low-opacity front cases passed their quantitative gates; fitted front speed was about 0.906–0.970 c depending on opacity and timestep. WOR-59 separately measured 50% lag decreasing from 2.871 cm at N128 to 1.469 cm at N512 when both space and time were refined. | Finite front delay/broadening is a measured FLD/discretization limitation; a pass is not exact-front agreement. |
| Strong local exchange | `wor52_lte_exchange_20261104.md`: 40 independent EOS root controls, maximum relative root error 1.192e-15; stiff `dt c sigma_P≈6.9e12`; coupled full-FLD one-step implicit energy ledger ≤2.40e-11 relative. | The local LTE solver and gas/radiation pairing remain stable in these extreme controls. The full calculation includes a physical Marshak boundary loss. |
| Fixed radiation bath | Same suite, `cases/couple/`: low/high analytic ODE tests pass; the fixed bath is externally held and is deliberately not a closed gas+radiation energy test. | It verifies coupling accuracy, not total-energy conservation. |
| Radiative shock | Same suite, `cases/radiative_shock/`: strict solver convergence and small global profile error, but Mach 5 spike accuracy gate fails at the suite resolution. | Global L1 and Newton convergence cannot substitute for resolving the spike. HPC resolution study is WOR-67. |

The closure regression
`python3 tst/regression/run_tests.py nrmgfld/stable_fld_closure --hide_make`
passed in both double and single precision after replacing a diagnostic
default `1e99` (outside `float`) with `numeric_limits<Real>::max()`. It
checks the thick/thin limits, zero opacity, invalid input, and extreme finite
`R`. This is an algebraic probe; it does **not** validate a full single
precision NR-FLD executable. A separate full single-precision build still
fails in `calculate_radiation_face_states.cpp` because `std::max(Real,
double)` is ill-typed. Production validation remains double precision.

## Conservation and solver tolerance

The implicit exchange changes gas and radiation with opposite signs. The
mixed-frame explicit source also applies a pair, except when radiation energy
is clipped at its floor; its opt-in ledger prints the floor contribution. The
implicit stage ledger measures gas change + radiation change + diffusion +
Marshak boundary flux. It does not include the preceding explicit hydro
stage. Thus each ledger must be interpreted at its own stage boundary.

Fresh source-isolated periodic-box reproduction:

```bash
python3 tst/nrfld_validation/closed_box_validation.py \
  --output-root /mnt/d/output/rad \
  --run-name wor11_closed_box_current_20261104 --max-cores 4
```

The 8³ periodic, nonuniform-state, fully coupled box ran to code time 1.
All five cases had strict Newton and FGMRES convergence and positive finite
final fields. Source snapshot SHA256
`5e9e063afd271373485ed1af2fe7646465c78daeda8843d9940ea9a128d94ddf`;
binary SHA256
`dacd346b134c0a578d4d9993170d07fdfbd6982f601efde47c7c72dfd3a92a23`.

| CFL / Newton target | Cycles | Maximum mass drift | Maximum total-energy drift |
| --- | ---: | ---: | ---: |
| 0.2 / 1e-6 | 28 | 1.057e-7 | 1.720e-7 |
| 0.2 / 1e-8 | 28 | 1.057e-7 | 1.720e-7 |
| 0.2 / 1e-10 | 28 | 1.057e-7 | 1.720e-7 |
| 0.1 / 1e-8 | 56 | 4.452e-8 | 7.248e-8 |
| 0.4 / 1e-8 | 15 | 2.726e-7 | 4.436e-7 |

This passes the declared 1e-5 conservation gate. Drift decreases with
timestep and is insensitive to Newton tolerance here; the exact source of
the ≈1e-7 floor is not localized. No claim of machine-precision global
conservation is warranted. The post-guard rerun
`/mnt/d/output/rad/wor11_geometry_guard_20261104/` produced the same
five pass results and numerical drifts. The final-source rerun
`/mnt/d/output/rad/wor11_final_closed_box_20261104/` also passed all five
cases with the same drifts. Its source snapshot SHA256 is
`d78c52d547646e269a24ac45b2e1aa8fdc09c2a77ce8171a67ce2d985cb2b26d`
and its binary SHA256 is
`5c11d9b1ad416a4607e18b7436f0f218ddf94dc97589d17d055dae8611ffcdf3`.

The known AMR counterexample is separate: `wor22_reproduction_20261019.md`
reports an approximately 2–3e-4 relative energy loss in a periodic,
constant-coefficient AMR diffusion control while uniform drift is around
1e-15. `wor23_wor24_operator_audit_20261023.md` independently reproduces
the mismatched coarse/fine ghost operators and nonzero volume-left-null
defect. Tightening Newton tolerance cannot fix that operator conservation
defect. See WOR-23/24/54; WOR-25 is assigned to another environment.

## Changes and checks from this audit

1. `NRFLD::NRFLD` now rejects non-Cartesian or non-cubic active-cell
   geometry before allocating the linear MG solver. The cubic periodic
   box passed unchanged. A rectangular `x2` extent was rejected with the
   specific geometry diagnostic before evolution. Athena's general
   `ATHENA_ERROR` path returned process status 0 in this negative probe, so
   the check was verified by diagnostic text and absence of evolution output,
   not by exit code.
2. The diagnostic end-time default and radiative-shock missing-position
   sentinel use the largest representable `Real` instead of `1e99`.
   The closure probe now compiles and passes for both precisions. Two
   `TINY_NUMBER` arguments in the mixed-frame source were cast to `Real`
   after a full single-precision build exposed template type errors.
   Double-precision results remain unchanged.

## Remaining decisions

- Resolve the AMR operator/ghost-policy conservation defect under
  WOR-23/24/54, with WOR-25 handled separately. Compare a volume-conservative
  leaf operator and the production operator before making physics claims from
  AMR energy drift.
- Complete WOR-67 at uniform/AMR/SMR matched finest resolution and timestep.
  Keep raw shock position and unshifted spike metrics alongside aligned
  profile L1.
- Quantify opacity/limiter lagging error for a temperature-dependent opacity
  with timestep refinement. The present thick diffusion and local exchange
  results do not isolate this effect; WOR-59 already quantified thin-front
  time/space errors for nearly constant, low opacity.
- Supporting rectangular or curvilinear cells requires a coordinated
  directional/metric-aware operator change. Until then the new guard makes
  the restriction explicit. A full single-precision executable also needs a
  separate compile and numerical audit; the production evidence here is
  double precision.
