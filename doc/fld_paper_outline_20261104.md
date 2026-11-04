# WOR-12: FLD implementation and validation paper plan

Status: planning inventory, 2026-11-04. This file specifies a proposed structure,
source-to-figure traceability, and claim-dependent missing evidence. It is not
manuscript text, a final figure interpretation, or a declaration of novelty.
The author decides the equations, physical interpretation, final figures, and
publication scope.

## Evidence baseline and scope

- Repository planning baseline: clean `nr_mgrad_dev@209e2239872a36b8ec278d5472db7b8724a18ea1`.
  The separate WOR-11 audit and geometry guard are on
  `codex/wor11-numerics-audit@3ded4d77`; they are **not** in this baseline.
- Historical integrated report:
  `/mnt/d/output/rad/validation_suite_complete_20261103/index.html`
  (`suite.json`: 12 cases, 11 `passed`, radiative shock
  `accuracy_not_met`). This `full` report imports completed child runs; it
  is not a new twelve-case computation from one clean binary. Its parent
  manifest records base commit `7a44d202` and a dirty tree. Child manifests
  and isolated build snapshots are the provenance for individual results.
- Most recent local Mach-5 resolution audit:
  `/mnt/d/output/rad/wor67_spike_audit_20261104/index.html`. Uniform
  N2048/CFL 0.3 passed the local spike gate; the matched AMR/SMR and N4096
  HPC series remain open in WOR-67.
- Fresh WOR-11 final-source closed-box reproduction:
  `/mnt/d/output/rad/wor11_final_closed_box_20261104/index.html`; 5/5 cases
  passed. Its manifest records a **dirty WOR-11 working tree snapshot**
  containing the subsequently committed changes, source snapshot SHA256
  `d78c52d547646e269a24ac45b2e1aa8fdc09c2a77ce8171a67ce2d985cb2b26d`.
- Candidate figures below are existing evidence, **not** frozen publication
  artifacts. Before any numerical claim is submitted, reproduce the selected
  cases at an immutable source commit and preserve input, binary/source hash,
  analysis revision, units, and raw data next to each final figure.

The natural current scope is a numerical-method and validation paper. A
general statement that “FLD, Newton–Schur, or multigrid is new” would conflict
with the prior methods in [Zhang et al. (2011), CASTRO II](https://ccse.lbl.gov/Publications/wqzhang/castro2.pdf)
and [Tomida & Stone (2023)](https://arxiv.org/abs/2302.13903).
Whether this implementation makes a publishable distinct contribution is a
scientific decision after the comparison and missing checks below.

## Proposed section outline

| Section | Content to prepare | Local source/evidence | Claim boundary |
| --- | --- | --- | --- |
| 1. Motivation and contribution statement | Define target applications and the exact implementation question. Compare against CASTRO gray FLD and prior Athena++ framework/MG work. | Literature table below; WOR-12 and WOR-11 | Do not call the mixed-frame split, Newton elimination, or MG itself novel. |
| 2. Equations and closure | State evolved gas/radiation variables, units, frame, O(v/c) terms, `R`, Levermore–Pomraning `lambda(R)`, `chi`, `alpha=(3-chi)/2`, Planck/Rosseland opacity, and regimes. | `src/fld/fld.hpp`, `src/fld/fld.cpp`, `src/fld/calculate_radiation_face_states.cpp`; WOR-11 audit | Write the exact implemented equations; separate full tensor radiation pressure work from its implemented split approximation. |
| 3. Time and space discretization | Trace explicit RK hydro/radiation-face transport and source stages, then the single implicit backward-Euler FLD update after all explicit stages; explain reconstruction, face opacity, radiation force, mixed-frame drift placement and primitive synchronization. | `src/main.cpp` near the implicit call, `src/task_list/time_integrator.cpp`, `src/hydro/rsolvers/hydro/hllc_fld.cpp`, `src/fld/add_mixed_frame_source.cpp` | Explicit RK order does not imply the whole FLD update is second order in time. |
| 4. Nonlinear and linear algorithm | Show gas/radiation residuals, direct total-energy residual, local gas elimination/Schur system, stage-lagged closure/opacity, Newton line search and stopping rule, right-preconditioned FGMRES with MG V-cycle, uniform versus AMR coarse operators. | `src/nr_multigrid/NRFLD.cpp`, `src/newton_raphson/`, `src/linear_solver/linearMG/`; WOR-26/30/52/54 | Distinguish linear residual, nonlinear residual, and physical conservation. No claim of a conservative AMR operator while WOR-23/24/25/54 remain open. |
| 5. Physical and mesh boundaries | Document periodic/fixed/outflow/Marshak cases actually exercised, the Robin flux formula and stage energy ledger; identify supported Cartesian cubic cells. | `src/nr_multigrid/NRFLD.cpp` (`MarshakFaceActive`, `MarshakFlux`), `src/fld/fld.cpp`, test inputs; WOR-11/WOR-69 | Athena++ general curvilinear mesh support does not imply NR-FLD curvilinear support. |
| 6. Verification and validation | Organize tests by what they verify: closure/local exchange; diffusion convergence and Marshak boundary; moving fronts and shocks; multidimensional geometry/AMR; conservation/restart/mixed-frame. | Figure and test matrices below | A test PASS means its own defined gate passed, not all model or AMR claims. |
| 7. Performance and application (conditional) | If included, report same-tolerance MG/FGMRES time-to-solution, iteration counts, RSS and rank scaling; then a converged RSG/collapse application with physical budgets. | WOR-20, WOR-15/21/54/68 | Do not use old WSL MG timing as current FGMRES performance; do not use a short successful restart as a long-run application validation. |
| 8. Limitations and reproducibility | List FLD thin-front lag, stage-lagged opacity, Cartesian cubic restriction, AMR energy defect, Mach-5 spike resolution, full single-precision build gap, and exact data/command record. | WOR-11/23/24/25/59/67/69; manifests | Separate diagnosed limitations from hypotheses. |

An implementation flow diagram would be a new schematic derived from
`src/main.cpp`, `src/task_list/time_integrator.cpp`, `NRFLD.cpp`, and
`linearMG.cpp`: explicit RK stages → opacity/state refresh → one implicit
backward-Euler gas/radiation solve → conserved/primitive synchronization.
Its exact arrows and equation labels should be checked against the final
committed source before drawing it as a paper figure.

## Literature comparison to write up

| Primary source | Established method relevant here | Comparison that this paper can make |
| --- | --- | --- |
| [Zhang et al. 2011, CASTRO II](https://ccse.lbl.gov/Publications/wqzhang/castro2.pdf) | Gray mixed-frame FLD, explicit Godunov hyperbolic step, implicit backward-Euler diffusion/exchange, Newton temperature elimination and Schur complement, hypre multigrid; published tests include radiative shocks, pulse advection, and static equilibrium (§6.6–6.8). | Give a term-by-term and test-definition comparison. This code has Athena++ face/reconstruction/task-list integration and right-preconditioned FGMRES-MG controls, but numerical superiority or first implementation is unproven. CASTRO's AMR time refinement/boundary treatment must not be described as identical to ours. |
| [Stone et al. 2020, Athena++ framework](https://arxiv.org/abs/2005.06651) | MeshBlock/AMR/task-list framework and HD/MHD solvers, including nonuniform and curvilinear capabilities of the framework. | State which framework facilities NR-FLD actually uses. Its implicit operator is formulated for Cartesian cubic active cells; the separate WOR-11 branch adds a guard against unsupported geometry. Framework-wide coordinate support is not FLD support. |
| [Tomida & Stone 2023, Athena++ self-gravity MG](https://arxiv.org/abs/2302.13903) | AMR multigrid for the Poisson self-gravity equation, conservative Laplacian treatment and parallel scaling tests. | Explain reused MG hierarchy/task infrastructure versus the separate nonlinear, variable-coefficient radiation problem. Do not transfer their conservation or scaling results to this FLD operator. |

These are comparison prompts, not final priority/novelty conclusions. A
literature search for additional contemporary radiation solvers and a
source-level comparison are needed before the Introduction claims
uniqueness or a performance advantage.

## Test-to-claim and figure inventory

Let `S=/mnt/d/output/rad/validation_suite_complete_20261103/cases`.
Paths in the table are relative to `S`. All twelve rows have a child
`summary.json` and `index.html`; eleven have a root `manifest.json`, while
`couple` keeps manifests under `simple/<low|high>/` and
`cases/<low|high>/dt_*/`. “Main candidate”
identifies a possible manuscript slot, not a result already approved for
publication.

| Test / draft placement | Existing figure or report | Measured evidence | Limitation / required caption check |
| --- | --- | --- | --- |
| Coupling, main or supplement | `couple/figures/couple_dt_sweep.pdf`, `couple/figures/couple_error_by_step.pdf` | Simple low/high max log gas-energy errors 6.23e-4 / 2.20e-3; all eight fixed-dt sweep cases passed, including first-step versus later-step accounting. | Radiation is a fixed bath. Do not use it to claim gas+radiation energy conservation. Show measured `dt/t_couple` and the first-step error separately. |
| Diffusion order, main | `diffusion/figures/convergence.pdf`, `diffusion/figures/profiles.pdf` | Coupled space-time orders 2.0118, 2.0058; time-extrapolated spatial orders 2.0123, 2.0057; temporal difference order 1.00017. | State which series changes both dx and dt, which cancels leading time error, and which measures time order at fixed N. |
| Marshak analytic boundary, main | `marshak/figures/comparison.pdf`, `marshak/figures/spatial_convergence.pdf` | At `tau=0.3`, extrapolated spatial L1 orders: radiation 1.982/1.940; gas 1.976/1.982. Applied spatial shift = 0. | The fixed-dt L1 curve contains time error; label the extrapolated series. The finite domain and initial 1 K gas state differ slightly from the semi-infinite zero-energy analytic idealization. |
| Thin streaming, supplement/limitation | `streaming/figures/two_panel.pdf`, `streaming/figures/front_trace.pdf`; `/mnt/d/output/rad/wor59_convergence_20261104/index.html` | For `chi_R=1e-7 cm^-1`, 50% front lag is 2.871 cm at N128, `dt=dx/(20c)`; WOR-59 reports 1.469 cm at N512 with joint refinement. | `x_0+ct` is a kinematic sharp-front comparator, not an analytic FLD profile. Include width and leading-edge error; no postprocessing shift. |
| Rad-EOS shock tube, supplement or main cross-check | `shock_tube/figures/final_profiles.pdf`, `shock_tube/figures/final_errors.pdf` | At FLD N128 versus independent 1D Rad-EOS hydro N4096: rho relative L1 6.11%, temperature 0.821%, Er 1.304%; all Newton solves strict. | Reference is an independently adapted high-resolution hydro limit, not an exact solution. Preserve the N2048/N4096 reference-difference estimate and conservative projection; no alignment. |
| Radiative shocks, Mach 2 main; Mach 5 conditional | `radiative_shock/figures/mach2_aligned.png`, `radiative_shock/figures/mach5_raw.png`; WOR-67 audit | Mach 2 at N256 meets gate. Mach 5 at N256 has aligned Tgas L1 0.233% yet gas-peak error 18.4% and spike recovery 10.4%, so the suite says `accuracy_not_met`. Local uniform N2048 gives peak error 7.95% and recovery 61.3%. | Show raw shock position and aligned shape separately; alignment is by density-jump midpoint only. The N2048 local result is provisional until matched HPC uniform/AMR/SMR refinement (WOR-67). |
| Advected pulse, main or supplement | `advection_pulse/figures/pulse_profiles.pdf`, `advection_pulse/figures/pulse_radiation_errors.pdf` | Moving/static comparison after prescribed periodic `v_0 t` translation: Tgas relative L1 4.19e-6, Er 1.59e-5 at final time. | The applied translation is part of the Galilean comparison and must be specified. It is not a fit to maximize agreement. |
| Static equilibrium, supplement | `static_equilibrium/figures/equilibrium_history.pdf`, `static_equilibrium/figures/equilibrium_maps.pdf` | Stored final fields show zero drift at output precision; max velocity 7.52e-8 cm/s over 12,168 cycles. | All 12,168 implicit calls needed zero Newton updates and zero FGMRES calls; this is an equilibrium-preservation test, not evidence of nonlinear-solver convergence. |
| 3D thermal wave, conditional multidimensional figure | `thermal_wave/figures/radial.pdf`, `thermal_wave/figures/amr32_slice.pdf` | Tgas contrast L1 decreases 1.63%→0.868% from uniform N32→N64. AMR base N32/fine N64 matches uniform-fine fields at stored precision; energy drift about −3.81e-7. | The 10 K front is still 59.72 cm versus analytic 55.48 cm at N64. AMR/uniform agreement in this case does not clear the separate coarse/fine conservation defect. |
| Closed periodic box, main conservation figure | `closed_box/figures/conservation.png`, `closed_box/figures/cfl_dependence.png`; fresher WOR-11 report | At CFL 0.2, mass/total-energy maximum drifts ≈1.06e-7/1.72e-7 across Newton targets 1e-6–1e-10; drift decreases with dt. Final WOR-11 branch repeats 5/5 cases. | State energy definition and stage/boundary accounting. This is not machine-precision conservation; source of the ≈1e-7 level is not isolated. Use fresh committed provenance for final figure. |
| AMR restart/MPI, supplement | `fgmres_amr_restart/figures/difference.png`, `fgmres_amr_restart/figures/regrid.png` | Direct/restart/2-rank comparisons give zero stored relative L1/max difference in the six-cycle test with topology change and strict FGMRES. | Tests restart and rank reproducibility in this setup, not arbitrary AMR conservation or production collapse stability. |
| Mixed-frame placement, supplement | `mixed_frame/figures/jacobian_ledger.png`, `mixed_frame/figures/pulse_n64_cfl0.03.png` | Both implicit and explicit `q v·grad Er` placements complete the selected shock/pulse controls; one-step Jacobian/ledger diagnostics pass. | Mark the explicit placement as an experimental comparison. A difference in runtime or an individual profile is not proof of generally better accuracy. |

The suite may support a shorter main paper by keeping diffusion, Marshak,
coupling, the Mach-2 shock, and closed-box conservation in the core, with the
remaining tests in a compact validation table or supplement. That selection
depends on the final paper's actual contribution and the author’s judgment.
Do not assemble a Mach-5 “agreement” panel from the N256 run without its
failed spike metric.

Algorithm controls outside the twelve-case suite can support a Methods table
or supplement. They should remain separate from the physical benchmark
figures:

| Control | Existing source | What it establishes / does not establish |
| --- | --- | --- |
| Closure extreme limits | `tst/regression/scripts/tests/nrmgfld/stable_fld_closure.py`; WOR-11 audit | Algebraic thick/thin and transparent-face checks pass in double/single precision. Full single-precision NR-FLD build still has type errors. |
| Direct coupled Schur RHS | `tst/nrfld_validation/wor26_schur_rhs_20261104.md` | Direct total-energy residual removes measured `Fg+Fr` cancellation at a stiff RSG cell; the short trajectory did not measurably change. |
| Coefficient cache/update timing | `tst/nrfld_validation/wor30_coefficient_update_audit_20261023.md` | Representative cache on/off controls match, including MPI/AMR/mixed/general EOS. This is not a timestep-lagging accuracy study. |
| Local LTE general-EOS exchange | `tst/nrfld_validation/wor52_lte_exchange_20261104.md` | 40 scalar root controls had max relative gas-root error 1.192e-15; a stiff full-FLD one-step stage ledger closed within 2.40e-11 relative after accounting for Marshak loss. This is not proof of long RSG evolution. |

Candidate tables:

1. **Algorithm contract:** evolved variables, split order, closure and
   opacity lagging, Newton/linear tolerances, geometry and boundary support.
   Source: code paths in Sections 2–5 and WOR-11 audit.
2. **Validation matrix:** benchmark, grid/dt, reference definition,
   metric/gate, source hash, strict solver status, and pass/limitation.
   Source: each child `summary.json` and `manifest.json`; do not count
   imported `full` children as new calculations.
3. **Cost and scaling:** only after WOR-20 reruns current FGMRES and MG
   controls at matched solution accuracy, with wall time, rank RSS and
   iteration counts.

## Claim-dependent missing evidence and order

| Priority | Claim that depends on it | Next evidence and owner |
| --- | --- | --- |
| Required before freezing numerical figures | All selected panels describe one identifiable source and settings. | Re-run selected cases from a clean frozen commit after branch integration; record immutable input, binary/source/analysis hashes, actual dt, EOS/opacity and raw HDF5. The 2026-11-03 `full` report remains historical evidence. WOR-9/56/66 runners provide the mechanism. |
| Required for resolved Mach-5 claim | Gas-temperature spike and shock bulk converge with dx and controlled dt. | WOR-67: matched uniform/AMR/SMR N2048 then N4096 HPC runs, peak/width and raw position, aligned L1, late dt, strict NR/FGMRES, boundary flux and resource use. Keep the N256 failure visible. |
| Required for general AMR conservation claim | The composite FLD update is conservative and its Jacobian/FGMRES operator is consistent at coarse/fine faces. | WOR-23/24/54 and WOR-25 (other environment): reconcile minmod state ghosts, correction ghosts, volume-left-null/face flux and true residual; re-run a closed periodic AMR control. Existing Test C reports ≈2–3e-4 energy loss despite tight Newton tolerance. |
| Required for application claim | RSG/collapse reaches the stated physical interval with strict solve and acceptable budgets. | WOR-15/21/54/68 plus table-EOS/opacity provenance: resolve cycle-106 non-descent/restart sensitivity, track floor/boundary source terms, and compare profiles at fixed physics. A short restart success is a solver control only. |
| Required for efficiency/scaling claim | Current FGMRES configuration improves time-to-solution or scales as claimed. | WOR-20: matched-accuracy MG/FGMRES, one-node rank sweep, then HPC scaling; include variance, iteration counts, rank RSS, compiler/MPI/binding. Older WSL timings are not current-speed evidence. |
| Important for model-error discussion | Stage-lagged limiter/temperature-dependent opacity does not change the reported observable beyond the chosen tolerance. | Design a fixed-grid dt sweep with a strongly temperature-dependent opacity and a higher-accuracy reference; separate timestep, nonlinear, and opacity-freeze effects. WOR-11 lists this as unmeasured. |
| Scope declaration | Geometry/precision capabilities are stated accurately. | WOR-69 covers directional/metric-aware non-cubic and curvilinear cells. Full single-precision NR-FLD build remains unsupported even though the algebraic closure probe passes. Do not list either as validated support. |

## Freeze checklist for the final figure set

1. Choose the paper's exact claim set; retain only figures whose gate
   supports those claims. Mark all other curves as diagnostics or limitations.
2. Freeze a clean source commit after merging required branches. Run the
   selected matrix using the same documented solver family and compiler
   settings; if cases need different EOS/problem builds, record each binary
   separately.
3. Generate vector PDF from the analysis scripts when available; regenerate
   PNG-only plots from saved CSV/HDF5 as needed. Preserve native units,
   reference construction, cell-centre versus finite-volume projection,
   and every applied shift in the caption material.
4. For each panel, retain a row with command, input SHA256, source/binary
   SHA256, analysis script SHA256, output path, physical metric and solver
   status. The author can then verify captions directly against
   `summary.json` and `manifest.json`.
