# WOR-26: coupled Schur RHS cancellation audit (2026-11-04)

`CalculateDefect` evaluates the coupled total-energy residual directly as
`Δeg + ΔEr - dt*diffusion`, while `CalculateCoefficients` previously formed
`Fg + Fr`. The latter adds two separately rounded, large exchange residuals.
An opt-in `ATHENA_NRFLD_SCHUR_AUDIT=1` diagnostic compares both forms at the
same cell/iterate and reports their values, local energy scale, the
`CalculateDefect` total-energy scale, worst cell, exchange residual, and
assembled Schur RHS.

The unmodified production RSG general-table-EOS build was run for one actual
hydro step at CFL 0.01, 0.03, 0.1, and 0.3. At CFL 0.3 the worst cell was
`gid=0 (i,j,k)=(18,17,17)`: `Fg+Fr=0.018047332763671875`, direct
`Ftotal=0.018048743275385963`, difference `1.4105e-6` or `2.0581e-8`
of its `68.5335` local energy scale. The difference is `1.3490e-8` of the
`104.5635` total-energy scale used by `CalculateDefect` in that cell. Its
exchange residual was `1.956e10`. The assembly discrepancy is thus comparable
to the requested `1e-8` Newton residual tolerance, so the RHS evaluation was
changed to `-Ftotal + Fg/dFg_deg`. No EOS root, closure,
Jacobian, or radiation-only branch was changed.

After the change, the opt-in diagnostic reports exact agreement between the
assembled RHS and the direct formula at the worst cell of each coefficient
evaluation. All four RSG one-step cases and the `t=2` full-FLD series strictly
converged with the implicit energy ledger below `1e-8`. One-step Newton
iterations were unchanged (`6,6,5,5`); total FGMRES iterations were unchanged
(`42,43,36,37`); the final HDF fields were identical to the legacy RHS at
stored precision. Thus this corrects an assembly inconsistency and removes a
numerical risk; it did **not** measurably improve this short RSG trajectory or
establish that RHS cancellation caused the earlier collapse failure.

A separate stiff, coupled `nrfld_sound` run used 2 MPI ranks, a static level-1
AMR interface, FGMRES, and two steps. Both levels were present, both Newton
steps converged (final maximum `2.90e-11` on step 2), and all 324 audit rows
matched the direct assembled RHS. The worst `Fg+Fr` discrepancy in this
non-LTE sound-wave case was only `6.93e-17` of local energy. The coupled,
only-radiation, and fixed-radiation branches passed
`nrmgfld/coupled_residual_norms`; the coupled branch now asserts the RHS audit.

Reproduce the RSG check after `scripts/configure_rsg.sh`:

```text
python3 tools/rsg/validate_lte_cfl.py --output /mnt/d/output/rad/<unique-name> \
  --one-step --energy-ledger --schur-audit --mode full
```

Raw runs: `/mnt/d/output/rad/wor26_audit_20261104` (legacy RHS),
`wor26_direct_20261104`, `wor26_direct_t2_20261104`,
`wor26_verified_20261104`, `wor26_final_scale_20261104`, and
`wor26_amr_sound_20261104`.
