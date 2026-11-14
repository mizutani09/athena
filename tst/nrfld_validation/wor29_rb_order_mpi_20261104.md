# WOR-29: red/black order and MPI control (2026-11-04)

## Effective color order

`linearMG::Smooth` XORs its incoming color with the V-cycle offset. MeshBlock
task lists pass 0 then 1; the default (`symmetric_rb_sweeps=false`) therefore
alternates the starting color between cycles and uses the same order on the
pre/post legs. Before this audit, octet/root smoothing passed `coffset_` then
`1-coffset_`; the second XOR canceled the offset, always starting with color 0.
The coarsest root solve behaved the same way. `symmetric_rb_sweeps=true` passes
0 at octet/root pre/post stages, and the V-cycle toggles the offset between
legs; the post leg reverses the pre order. This is a smoother pairing, not a
proof that the entire AMR preconditioner is symmetric.

This branch adds opt-in `nrfld/consistent_rb_colors=true` for the root and octet
calls. It passes 0 then 1, including at the coarsest root, so the existing
`Smooth` XOR makes their effective starting color match MeshBlock levels.
Default is false until a representative collapse regression is run. Gravity MG
retains its old behavior; only linearMG sets this flag. The setting is exposed
by `test_c_controlled.py --consistent-rb-colors true|false`.

## Fixed-limiter, two-rank Test C

Source root `209e2239` plus this branch's diagnostics; q=1e7, dt=2e-4,
constant density, only radiation, root 64x8x8, uniform fine or one-level
aligned/shifted AMR. All paired cases use the same input, two MPI ranks,
strict NR threshold 1e-7 and the same binary within each color-policy pair.
The default, reverse-post, and consistent-color options give:

| Layout | Historical default | Reverse post | Consistent colors |
| --- | --- | --- | --- |
| Uniform fine | converged, NR=2 | converged, NR=2 | converged, NR=2 |
| Aligned AMR | converged, NR=4, 3 rejected V-cycles | same | same |
| Shifted AMR | backtracking exhausted, final L2=0.658334119 | 0.658334119 | 0.658334119 |

The aligned final energy drift is -5.1434268e-5, -5.1434279e-5 and
-5.1434276e-5, respectively. Thus neither order change corrects the known
AMR operator mismatch. In variable-limiter two-rank controls, q=5e5 passes
uniform/aligned/shifted with NR iterations 3/4/7 in both historical and
consistent-color modes. At q=7e5 all three fail in both modes. The changes
in converged final residuals are below ~4e-4 relative; no robust improvement
is demonstrated. The earlier one-rank uniform reference found that reversing
post order worsened first-cycle factors from 0.9362 to 0.9783 (q=5e5) and
1.0207 to 1.0571 (q=7e5); see `wor29_rb_order_20261023.md`.

Raw manifests and logs:

- `/mnt/d/output/rad/wor29_mpi_false_valid_20261104/`
- `/mnt/d/output/rad/wor29_mpi_true_20261104/`
- `/mnt/d/output/rad/wor29_mpi_consistent_20261104/`
- `/mnt/d/output/rad/wor29_mpi_variable_{false,true}_20261104/`

The `wor29_mpi_false_20261104` directory contains a rejected setup attempt
with `handoff_diagnostics=true` on two ranks before MPI support was added;
it is not a numerical result. Its isolated binary was reused for the valid
historical-default and reverse-post controls.

With FGMRES and the same q=1e7 two-rank controls, both color policies pass all
three layouts. Uniform/aligned/shifted require 13/29/66 total FGMRES
iterations in both; Newton needs 2/2/3 iterations. Shifted-AMR's minimum final
Er is ~1.009973 in both modes. FGMRES recovers a usable solve in this reduced
case, but its shifted-AMR radiation-energy drift remains about -5.20e-5 and
the AMR discretization mismatch is not removed. Raw artifacts are at
`/mnt/d/output/rad/wor29_mpi_fgmres_{false,true}_20261104/`.

**Decision:** keep `symmetric_rb_sweeps=false` and the historical color policy
as production defaults for now. The new consistency flag is a controlled opt-in
for diagnosis. Reverse-post order worsened the prior uniform MG control and
neither MPI contrast provides a compensating benefit. The consistent-color
policy is logically cleaner but has no demonstrated accuracy or convergence
benefit beyond tiny floating-point changes; a representative collapse run is
still required before changing that default. This does not rule out a more
appropriate smoother/order for a revised conservative AMR operator.
