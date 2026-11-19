# WOR-120: HD 4wz production decision (2026-11-20)

## Decision

Accept `wor-120-hd-4wz-memory2` for the production **HD-only** SCF relaxation
stage with:

- `NGHOST=4`, 32-cubed MeshBlocks, and the accepted surface-L3 AMR hierarchy;
- RK4 plus Berta WENO-Z (`time/xorder=4wz`) for the hydrodynamic update;
- conservative PLM for passive-scalar transport;
- FMG-only self gravity (`mgmode=fmg`, `threshold=-1`, `niteration=0`);
- three 110 GiB M nodes, 168 MPI ranks, and a four-hour allocation for a
  `0.1 t_dyn` segment.

The result does not approve 4wz for FLD, MHD, or relativity.  Passive scalars are
explicitly second order; fourth-order tracer transport would require a separate
Berta-compatible reconstruction, fallback mask, and convergence contract.

## Provenance and isolation

The sustained 4wz run was XD job `343579`, using branch commit
`59ddc1d814d4553d340658370f5b2f92dd91f135`, executable SHA-256
`042fcd1701f4215c038a38ec80746ca7671163ba06472eb8c9aa8219b7e9811e`, and seed
restart SHA-256
`1b1d43db0c2636740c3c41163896455e0c0844d1d240a606cf8a5f17da55237a`.
The seed was created once and read successfully by both methods.  The original BSCF
repository remains unchanged and clean.

Job `343567` is not scientific data: a submission-time `--ntasks=1` override caused
one-rank startup and it was discarded after 78 s.  The launcher now rejects any
allocation other than 168 tasks on three nodes.  Corrected job `343579` ran with
168 tasks, 56 per node.

## Sustained same-restart A/B result

Both methods start from the same 2,416-block restart and end at exactly code time
`47.006` (`0.1 t_dyn`).  RK2+x2 uses 4,752 cycles; RK4+4wz uses 4,753.  Neither run
creates or destroys a MeshBlock.  The fixed hierarchy is 264 blocks at L2, 1,976 at
L3, 56 each at L4 and L5, and 64 at L6.

All nine HD acceptance gates pass for 4wz:

| quantity | 4wz maximum/final | limit |
|---|---:|---:|
| stellar-mass drift | `2.88e-10` maximum (`-7.79e-11` final) | `1e-4` |
| angular-momentum drift | `1.56e-5` | `1e-3` |
| mechanical-energy drift | `7.06e-6` | `1e-2` |
| radial Mach RMS | `6.88e-3` | `3e-2` |
| radial KE / `|W|` | `2.35e-5` | `5e-4` |
| nonradial perturbation KE / `|W|` | `2.25e-5` | `5e-4` |
| mean-radius drift | `5.02e-6` | `1e-2` |
| RMS-radius drift | `1.81e-5` | `1e-2` |
| absolute virial residual | `2.51e-3` | `2e-2` |

The final-half (`0.05--0.1 t_dyn`) signed virial residual lies between
`-7.795e-4` and `-7.719e-4`.  Its least-squares change over the window is
`-5.84e-6` (an absolute-residual increase of `5.84e-6`), smaller than x2's
`8.26e-6` fitted change and negligible relative to the `0.02` gate.  The residual
is settled rather than showing secular runaway.

The final direct A/B differences are:

| quantity | x2 | 4wz | symmetric relative difference |
|---|---:|---:|---:|
| stellar mass (g) | `3.041554724963e34` | `3.041554724726e34` | `-7.79e-11` |
| angular momentum | `8.348491637524e53` | `8.348485063160e53` | `-7.87e-7` |
| mechanical energy (erg) | `-1.629489352729e48` | `-1.629480650070e48` | `5.34e-6` |
| radial Mach RMS | `6.7919e-3` | `6.8817e-3` | `1.31e-2` |
| local radial Mach maximum | `42.3025` | `42.2935` | `-2.13e-4` |
| mean stellar radius (cm) | `2.843900518934e13` | `2.843907726918e13` | `2.53e-6` |
| RMS stellar radius (cm) | `3.159431793397e13` | `3.159437976755e13` | `1.96e-6` |

The local Mach maximum is not a standalone rejection gate because a tiny-mass,
low-sound-speed surface cell controls it.  Its close agreement throughout the run
(0.02% at the final time), together with the low mass-weighted RMS and kinetic-energy
ratios, shows that it is not a 4wz-specific instability.

At the final time, the Hydro fallback covers 25.16% of active cells: 18.05% from the
Berta sensor and 7.11% from the AMR safety band.  Only 18.75% of stellar mass lies in
fallback cells, so 81.25% of stellar mass remains on the high-order Hydro path.

## Measured production cost

| quantity | RK2+x2 | RK4+4wz | 4wz/x2 |
|---|---:|---:|---:|
| wall time | 2,282.78 s | 8,804.15 s | 3.857 |
| Athena main-loop time | 2,219.15 s | 8,630.41 s | 3.889 |
| zone-cycles/s | `1.6953e8` | `4.3600e7` | 0.2572 |
| allocated CPU-hours | 106.54 | 410.85 | 3.857 |
| allocated node-hours | 1.9025 | 7.3367 | 3.857 |
| average RSS/rank | 1,736,675 KiB | 1,819,547 KiB | 1.0477 |
| maximum RSS/rank | 4,441,052 KiB | 4,840,504 KiB | 1.0899 |

The 4wz run finishes in 2 h 26 min 44 s for its MPI step, leaving more than 1.5 h
inside the four-hour request.  Three 110 GiB M nodes are therefore the minimum tested
and accepted 4wz layout.  The earlier two-node P fallback was cancelled once this
layout passed startup and the old OOM point.

## Restart and result artifacts

The accepted continuation restart is:

```text
/work/mizutaniks/wor120/hd4wz_scf_l3_memory2_t01_4wz_343579/
  scf_rsg_relax.final.rst
SHA-256 c3e4249de92293493b2d665a8668625c847d7c9e99def9bc74af7c47b280e389
```

It is 8.1 GiB and was written after normal termination at `t=47.006`.  The complete
history, diagnostic CSV, summary, gates, timing, stderr, and provenance files are in
the same directory.  Machine-readable A/B outputs are in:

```text
/work/mizutaniks/wor120/hd4wz_scf_l3_memory2_t01_compare_manual_20261120/
  ab_comparison.json
  cost_comparison.json
```

## Handoff

Use the memory2 branch and the accepted restart for subsequent HD work.  Keep the
original x2 restart and history as the regression reference.  FLD remains a separate
integration stage: do not infer an FLD acceptance decision from this HD-only result.
