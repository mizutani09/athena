# WOR-120: 4wz passive-scalar transport experiment (2026-11-20)

## Scope

- Branch: `wor-120-hd-4wz-scalar-plm`
- Worktree: `/home/kosuke/simulation/athena-wor120-hd4wz-scalar-plm`
- Numerical change: retain fourth-order Berta WENO-Z for Hydro, but transport passive
  scalars with conservative PLM when `time/xorder=4wz`.
- Isolation: this branch is not merged into the production branch; BSCF is unchanged.

The experiment tests a specific discrepancy in the original 4wz integration. Hydro
uses point-value WENO-Z plus the density/pressure and coarse-fine fallback mask, while
passive scalars previously followed Athena++'s unrelated legacy fourth-order PPM and
flux-correction path. The stellar tracer drives the surface AMR condition, so a
tracer-only difference can change the mesh even when the fluid solution is stable.

The branch skips the legacy fourth-order scalar cell-average conversion and calls
`PassiveScalars::CalculateFluxes` with order 2 under Berta 4wz. Hydro, gravity, the
time integrator, and the restart format are unchanged.

## Verification

The exact SCF configuration (`NGHOST=4`, two passive scalars, MPI, multigrid gravity)
builds and completes a four-rank surface-L0 one-cycle smoke with 400 MeshBlocks and no
NaN, runtime error, or multigrid warning. The existing three-dimensional Hydro 4wz
regression is unaffected: sound and entropy waves retain finest-grid RMS-L1 errors
`1.23819e-10` and `8.71055e-11`, with orders 4.146 and 4.139.

A smooth one-dimensional Gaussian scalar advection comparison confirms the intended
tradeoff. The legacy fourth-order scalar path converges at fourth order, whereas the
PLM experiment approaches second order (rates 1.61, 1.75, and 1.83 from 32 through
256 cells). PLM is therefore a robustness option for the SCF tracer, not a general
fourth-order scalar implementation.

## B16 sustained SCF result

XD job `343506` repeats the B16 surface-L2, 100-cycle A/B gate with the scalar-PLM
binary. Its x2 control is unchanged to the expected run-to-run level. The 4wz result
removes the topology growth seen with legacy scalar transport.

| B16, 100 cycles | x2 | 4wz + scalar PLM |
|---|---:|---:|
| wall time | 34.53 s | 68.62 s |
| initial/final/maximum MeshBlocks | 2,472 / 2,472 / 2,472 | 2,472 / 2,472 / 2,472 |
| maximum mass drift | 0 | 4.09e-10 |
| maximum mechanical-energy drift | 4.78e-4 | 4.92e-4 |
| maximum radial Mach RMS | 2.48e-3 | 2.49e-3 |
| final total/stellar-mass Hydro fallback | 0 / 0 | 59.10% / 37.80% |

The legacy-scalar B16 4wz run grew from 2,472 to 2,808 MeshBlocks, reached a maximum
mechanical-energy drift of `1.69e-2`, and took 88.28 s. Scalar PLM keeps the original
topology, reduces that energy excursion by a factor of about 34, and reduces wall time
by 22%. The final x2/4wz mechanical energies differ by only `6.07e-6`.

This result identifies legacy fourth-order tracer transport as the main trigger for
the B16 regrid/energy offsets; it does not indicate a secular instability in Hydro
4wz. The selected B32 production layout already passes its 100-cycle gate without
this change, so the ongoing baseline surface-L3 gate remains scientifically useful.
Before merging scalar PLM, either accept its explicit second-order tracer contract or
complete and validate a bounded Berta WENO-Z scalar path with the same fallback mask.
