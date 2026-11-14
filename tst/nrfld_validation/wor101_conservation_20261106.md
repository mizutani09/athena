# WOR-101: implicit FLD conservation at an AMR coarse/fine face

## Decision

The cause is the lack of a shared diffusion flux at coarse/fine faces in the
committed Newton FLD operator. A conservative production fix must make the
coarse residual use the area sum of the fine-face fluxes and linearize that
same coupling for the Newton and multigrid solve. A post-solve energy adjustment
or a change to the ghost interpolation alone is **not accepted**: either would
change the converged state without making the solved residual conservative.

The new local audit is a regression target for that operator work. No solver
discretization is changed in this branch. The existing optional ledger now
also works with the radiation-only periodic control.

## Controlled local result

Command (after `python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld
--nghost=2 --cxx=g++ && make -j8`):

```sh
python3 tst/nrfld_validation/wor101_conservation_audit.py \
  --binary bin/athena --output /tmp/wor101_local/audit_20261106_v2
```

One step at `dt=2e-4`; all physical boundaries periodic; radiation-only;
fixed flux limiter; density contrast either 1.0 or 1.15. The pulse is centered
on the x=0.25 coarse/fine boundary in the aligned AMR layout. The shifted AMR
layout places that pulse inside the fine region. The same code uses a uniform
mesh for controls.

| Density contrast | Uniform coarse ΔEr/Er | Uniform fine ΔEr/Er | AMR aligned ΔEr/Er | AMR shifted ΔEr/Er |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | +2.18e-15 | +1.09e-15 | −1.02995e-5 | −1.58588e-6 |
| 1.15 | +2.18e-16 | +6.53e-16 | −1.01810e-5 | −1.56404e-6 |

For contrast 1.0, the x=0.25 left coarse-face contribution is
`−1.5787343574e−7` and the right fine-face area sum is `−4.9874523771e−7`.
Their sum, `−6.5661867346e−7`, matches the one-step whole-domain radiation
change `−6.5661853910e−7` to the Newton/roundoff scale. Both contributions
have the same sign, so they cannot represent a single shared face flux.
The x=0.75 mismatch is only `+1.3245e−13`. The direct operator ledger gives
`−6.5661854100e−7` with local equation balance `1.78e−15`; the final Newton
L2 residual is `2.01e−13`. Thus poor nonlinear convergence does not explain
the drift. A flat radiation field gives exactly zero operator and energy
change on the same AMR mesh.

The constant-density case has constant physical opacity and `lambda=1/3`,
so a variable diffusion coefficient, material interface, or limiter branch is
not required to trigger the defect. Moving the pulse away from the AMR face
reduces but does not eliminate it. The periodic boundaries exclude physical
energy escape.

The XD Mach 5 N2048 ledger independently localized the late-time production
defect to an AMR coarse/fine face near x=195.3125 cm: face-sum mismatch about
6.4% of the sum of the two absolute face contributions. The local periodic
control isolates the same mechanism without a shock, hydrodynamics, or XD.

## Source mechanism and implementation boundary

`NRFLD::CalculateCoefficientsOnce` evaluates the FLD face coefficient from
each block's state and ghost fields. `NRFLD::CalculateCoefficients` applies
each cell's face term to the residual. `NewtonRaphson::ProlongateBoundary`
uses the generic coarse/fine ghost interpolation and restriction. Across a
refinement face, those one-sided stencils do not produce opposite area-weighted
fluxes. The Newton radiation boundary variable constructs its base class
with flux correction disabled, and the Newton task list does not send or
receive a radiation flux correction. Local Newton residual convergence is
therefore compatible with a globally nonconservative result.

A safe fix needs all of the following together:

1. Compute one physical flux per fine face from a stated coarse/fine face
   reconstruction and coefficient rule.
2. Use the fine-face area sum with opposite sign in the coarse-cell residual,
   including every Newton trial and final defect evaluation.
3. Represent the resulting coarse/fine cross-block derivatives in the linear
   solve or use a verified matrix-free Jacobian with a compatible preconditioner.
4. Carry the face data across MPI ranks and AMR topology changes, and preserve
   gas/radiation source cancellation, positivity, and physical boundary terms.
5. Require periodic uniform/AMR energy closure, face-pair closure, solver
   convergence, rank invariance, and the Mach 5 N2048 A/B comparison before
   adopting it. The N4096 run remains in WOR-100.

Changing only `NRBoundaryVariable`'s flux-correction flag is insufficient:
the face arrays are not populated with the implicit FLD flux, the Newton task
list has no correction communication, and a correction after convergence
would invalidate the residual. A global radiation-energy offset would conceal
the defect and alter the state; it is not a conservative discretization.

## Reproduction artifacts

- Audit runner: `tst/nrfld_validation/wor101_conservation_audit.py`
- Local summary: `/tmp/wor101_local/audit_20261106_v2/summary.json`
- Per-case input, log, and history: same directory, under each case name
- Previous XD interface audit:
  `tst/nrfld_validation/wor67_interface_balance_20261106.md`

The audit's `--require-conservation` flag is intended to fail on the current
AMR solver; it becomes a passing regression gate when a conservative operator
is implemented.

## Follow-up scope check (2026-11-06)

The smallest apparent patch was tested: set the Newton radiation boundary
variable's `fflux` constructor argument from `false` to `true`, rebuild, and
rerun all eight periodic audit cases. Every whole-domain radiation-energy
change was numerically identical to the baseline, including the aligned AMR
case's `−1.029953143683e−5`. The flag was restored. It only allocates/enables
the boundary-variable machinery; the Newton task list never populates the
implicit face-flux arrays or calls `SendFluxCorrection` and
`ReceiveFluxCorrection`.

The production correction spans at least four coupled parts:

1. A common coarse/fine face-flux rule for all three directions, variable
   opacity and limiter, and the coarse/fine face-area sum.
2. The same rule in the initial Newton residual, every line-search defect,
   and the committed-state ledger. The two residual evaluations currently
   live in separate NRFLD paths.
3. A matching cross-block Jacobian or matrix-free action and a multigrid
   preconditioner that still converges with the new operator.
4. MPI exchange, AMR topology changes, and regression checks in radiation-only
   and coupled Mach 5 runs.

This is a multi-module solver change rather than a small boundary flag or
interpolation edit. A rough engineering estimate is several hundred lines,
about 1–3 weeks for an implementation and the needed local/MPI/XD validation;
the first single-rank constant-coefficient prototype could be shorter but
would not be safe to adopt for production.
