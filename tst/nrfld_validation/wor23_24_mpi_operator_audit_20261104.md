# WOR-23 / WOR-24: MPI Test C operator diagnostics (2026-11-04)

## Change

The existing optional boundary-inclusive `global_jacobian_check` and
`handoff_diagnostics` now aggregate their local cell sums, counts, extrema,
and worst-cell location over MPI ranks. They print once on rank 0. The Jv
direction scale and baseline ghost-change statistic use global maxima before
the epsilon sweep. No diagnostic arrays or collectives are used when the
options are disabled. This extends the **uncoupled, periodic, only-radiation**
Test C diagnostic; it does not add coupled Schur or physical-boundary Jv.

The controlled runner accepts `--global-jv --handoff --mpi-ranks 2`. It keeps
the deterministic input, one fixed step, strict NR tolerance and saved build
provenance. A nonzero exit code for the shifted-AMR case is an expected
numerical failure, not a diagnostic crash.

## One versus two ranks

Using the *same new executable*, q=1e7, dt=2e-4, uniform fine / aligned AMR /
shifted AMR were run with one and two ranks. All three have the same NR status:
converged in 2, converged in 4, and backtracking exhausted after 1 iteration.
Every corresponding Jv/handoff record has the same region and stage. The
largest absolute change in their global `l2_rel` Jv values is below 2e-23;
the largest change in `common_rel` handoff values is below 2e-18. In the
aligned case's *second* actual Newton direction (`--jv-iteration 1`), the
frozen-coefficient Jv mismatch at h=1e-4 is 0.0838518 globally and 0.135052
on coarse/fine faces in both rank layouts. The finite-difference tangent
control reduces it to ~6e-17. This reproduces the earlier local result with
MPI rather than identifying a new failure cause.

Two-rank diagnostics ON/OFF gave exactly matching NR-result dictionaries,
history bytes and every saved HDF5 dataset for all three layouts. The
shifted-AMR first linear solve again has `||b-Aδ||_V/||b||_V`:

| Stage | Common relative linear residual | Minimum full update |
| --- | ---: | ---: |
| MG end | 1.92597e-11 | -8.32153548 |
| Retrieved | 1.92597e-11 | -8.32153548 |
| After delta boundary exchange | 1.81471e-2 | -8.32153548 |
| After 16 fine sweeps | 4.68592e-3 | -8.32139164 |

The negative full-step direction exists inside the solved MG system before
handoff. The boundary update introduces the first **linear residual jump**;
these are separate findings. The first jump is concentrated at coarse/fine
faces, as in the one-rank audit.

As a post-MG-only control, existing `amr_correction_prolongation=minmod`,
`tangent`, and `constant` were tested with one rank and identical inputs.
All fail with backtracking exhausted. All have the **same** MG-end negative
minimum -8.32153548 and MG-end residual 1.92597e-11. After boundary exchange,
common residual is 0.01815 for minmod versus 0.02865 for tangent/constant.
Changing the returned correction's ghost interpolation alone cannot repair the
earlier negative MG direction. A matched operator at the state/linear solve
level still needs to be tested, together with conservation and positivity.

## Artifacts and remaining scope

- `/mnt/d/output/rad/wor23_24_mpi_audit_20261104/` — two-rank ON
- `/mnt/d/output/rad/wor23_24_mpi_off_20261104/` — two-rank OFF
- `/mnt/d/output/rad/wor23_24_one_rank_control_20261104/` — one-rank ON
- `/mnt/d/output/rad/wor23_jv_iteration1_rank{1,2}_20261104/` — second direction
- `/mnt/d/output/rad/wor23_24_{minmod,tangent,constant}_20261104/` — post-MG controls

The builder's `manifest.json` records binary/source hashes and effective
inputs. The main checkout binary was not overwritten. The original
single-rank operator audit and 96-cell direct reference are documented in
`wor23_wor24_operator_audit_20261023.md`.

Still open: coupled two-variable/Schur Jv, physical boundaries, real collapse
state, limiter branch identifiers, and a conservative common AMR face operator.
The MPI result establishes that the reduced Test C discrepancy is reproduced
across rank partitions; it does not justify claiming the collapse case solved.
