# Rotating RSG FLD startup validation (WOR-182)

Date: 2026-11-25

## Configuration

- q = 0.25 RLOF synchronous rotation
- full mixed-frame FLD with radiation force
- 4wz hydrodynamics and SSPRK(5,4)
- FGMRES with one multigrid V-cycle per preconditioner application
- central L4 mesh, surface L2 refinement

The startup and restart launchers are:

- `analysis/wor182/xd_q025_corot_fld_startup_gate.sh`
- `analysis/wor182/xd_q025_corot_fld_restart_gate.sh`

## Startup gates

Both L3 (XD job 347691) and L4 (347695) completed through cycle 110 and
time 0.01, including the dt=0.00389 step that failed with non-SSP RK4.
L4 cost 203.6 s versus 184.8 s for L3, a factor of 1.10.  At the end of
the short gate, L4 reduced the interior radial Mach maximum from 1.26e-4
to 4.69e-5 and the relative gas-plus-radiation energy residual from
1.05e-7 to 3.11e-9.

## Exterior pressure-floor failure

The first L4 CFL=0.1 continuation (347696) reached time 1 but was not a
valid stability result.  At cycle 131 the implicit solve switched to relaxed
acceptance, and at time 0.868 the low-density-side Mach maximum jumped from
3.17 to 1.2e6.  The dense-interior radial Mach remained 2.24e-3.

A same-checkpoint, same-dt reproduction (347712) located the nonlinear defect
near ellipsoidal radius 1.047.  Representative post-step state:

- rho = 9.73e-10 g cm^-3
- P = 6.67e-16 dyn cm^-2
- |v_r| = 1.23e3 cm s^-1
- c_s = 1.07e-3 cm s^-1

The requested hydro pressure floor was 1.0e-4 dyn cm^-2.  The ideal-gas
implicit FLD correction used `TINY_NUMBER` as its gas-energy floor, while the
general-EOS path converted the EOS pressure floor to an internal-energy floor.
The implicit update could therefore undo the ordinary hydro pressure floor.

Commit `17e04faa4` uses `pfloor/(gamma-1)` for the ideal-gas implicit floor.
The fixed same-step run (347716) retained P=1.0e-4 dyn cm^-2 and reduced the
surface Mach maximum from 1.216e6 to 3.237 without changing the interior Mach.
The mass with radial Mach above 0.3 was 1.74e28 g (8.8e-6 Msun), so the
remaining supersonic motion was confined to a very small exterior mass.

The floor-constrained Newton solve still ended by relaxed acceptance with a
gas maximum residual of 0.148 (configured relaxed maximum: 0.2).  Longer-term
acceptance therefore requires both a fixed-code continuation and an audit of
whether the constrained residual should be treated explicitly rather than as
an unconstrained Newton residual.
