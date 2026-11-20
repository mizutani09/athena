# WOR-124 local-face FGMRES preconditioner follow-up (2026-11-20)

## Scope and decision

This follow-up tests a defect-corrected MG right preconditioner for the
conservative local-face fallback introduced on
`exp/wor124-local-face-monotone-20261120`.  The implementation is retained as
an opt-in experiment.  It is not enabled by default.

Selected strong-field recipe:

```text
ATHENA_NRFLD_FGMRES_TRUE_DEFECT_CORRECTIONS=1
nrfld/fgmres_mg_vcycles=3
```

The recipe passes the c964-to-c970 gate and reduces the local-face path by
10.4% relative to the uncorrected local-face baseline.  The global two-point
retry remains substantially faster, so this result does not justify making
the local-face path or the defect correction a default.

## Provenance

- base: `aacb8a739` (`exp/wor124-local-face-monotone-20261120`)
- implementation: `721566c29`
- XD gate script: `66b133b5b`, parameterization: `bb37ad7b8`
- branch: `exp/wor124-local-face-two-point-pc-20261120`
- isolated worktree:
  `/home/kosuke/simulation/athena-wor124-local-face-two-point-pc-20261120`
- XD source snapshot:
  `/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/source_local_face_true_defect_pc_20261120`
- restart: the immutable cycle-964 checkpoint from XD job 343340
- XD: M-test-a, 5 nodes, 320 MPI ranks, CFL 0.15, table EOS and opacity

## Why this correction was tested

The AMR V-cycle does not call the tangential face reconstruction used by the
FGMRES true operator.  Turning tangential reconstruction off only inside the
preconditioner is therefore a no-op: the native MG inverse is already an
approximation closer to the two-point interface operator.  The remaining
cost comes from the mismatch between that MG approximation and the mixed
local-face/tangential true operator.

The tested linear right preconditioner applies

```text
z0 = M^-1 r
d  = r - A z0
z  = z0 + M^-1 d
```

where `A` is the same frozen true operator used by outer FGMRES.  Outer
matvecs, true-residual convergence, and the nonlinear residual are unchanged.
The environment variable accepts 0 through 3 corrections and defaults to 0.

## Local regression

Optimized Test C, N64x16x16, two MPI ranks:

| corrections | FGMRES iterations | relative residual | CPU time |
|---:|---:|---:|---:|
| 0 | 8 | 8.358153778e-11 | 4.4309 s |
| 1 | 5 | 3.436876292e-12 | 5.0223 s |
| 2 | 4 | 6.783459309e-13 | 5.5122 s |

All three runs used one Newton iteration and produced byte-identical history
files (`sha256=9e1c19ef13132a6a36fddb6adac980ca91e74492bfd4690ac375516583d6a0ab`).
This showed better conditioning but no small-case time benefit.

Both the optimized Test C target and a debug table-EOS MHD MPI target built
successfully.

## XD c964-to-c965 tuning gate

| job | corrections | V-cycles per MG inverse | iterations | solver elapsed | result |
|---:|---:|---:|---|---:|---|
| 343897 | 0 | 5 | 628, 394, 407 (1429) | 221 s | PASS, control |
| 343894 | 1 | 5 | 427, 267, 277 (971) | 258 s | PASS, slower |
| 343901 | 1 | 3 | 428, 267, 276 (971) | 188 s | PASS, selected |

Five V-cycles for both MG inversions performed too much work despite the
32% reduction in outer iterations.  Three V-cycles preserved the iteration
reduction and improved elapsed time by 14.9% relative to the control.

All cases reproduced the same unsafe direction at c964: 287 crossing cells
in 28 blocks, adding 1792 local two-point face cells.  Each completed one
local retry, no global retry, two Newton updates, and reached the cycle limit.

## XD c964-to-c970 continuation

Selected job 343904 passed:

- FGMRES iterations: 428, 267, 276, 283, 299, 173, 176, 280, 284, 173,
  174, 267, 267; total 3347.
- Uncorrected local-face job 343876: total 4936.
- Solver elapsed: 645 s versus 720 s; 75 s or 10.4% faster.
- Slurm elapsed: 11:16 versus 12:35.
- MPI-step MaxRSS: 34,187,636 KiB versus 33,507.5 MiB, effectively unchanged.
- Seven nonlinear solves and 12 Newton updates in both runs.
- Maximum final true residual L2: 6.39e-9 versus 1.15e-8.
- Maximum final true residual max norm: 6.5333941e-6 versus 6.5333946e-6.
- No fatal error, non-finite value, global retry, opacity-domain violation, or
  non-positive audited cell.
- Opacity audits are identical: 2,412,544 cells, rho range
  2.875800028e-20 to 9.121865860e-14 g cm^-3 and temperature range
  6.999962807 to 22.40895462 K.
- Printed history mass is identical.  Global energy-like histories agree to
  3.8e-16 relative; derived extrema agree to about 2e-7 relative.  A near-zero
  diagnostic differs by 6.3e-14 absolute (6.9% relative), which is not a
  conservation discrepancy.

The final restart is not byte-identical, as expected for a different Krylov
trajectory, but the physical histories and nonlinear convergence agree within
the values above.

## Remaining limitation

The previously validated global full two-point retry (job 343472) still needs
only 208 solver seconds and 767 FGMRES iterations through c970.  The local
face path preserves tangential reconstruction away from the 1792 affected
face cells but remains about 3.1 times slower than that robust global retry.
Consequently:

- keep the defect correction opt-in;
- use one correction with three V-cycles only when evaluating the local-face
  physical trajectory;
- retain the global retry as the practical robust baseline;
- do not run a longer production collapse from this experimental branch
  without a separate physical reason to prefer the local-face trajectory.

## XD artifacts

- job 343894: `rootfix_local_face_true_defect_pc/job_343894_c965`
- job 343897: `rootfix_local_face_true_defect_pc/job_343897_c965`
- job 343901: `rootfix_local_face_true_defect_pc/job_343901_c965`
- job 343904: `rootfix_local_face_true_defect_pc/job_343904_c970`
- baseline job 343876: `rootfix_local_face_monotone/job_343876`

All paths are below
`/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/`.
