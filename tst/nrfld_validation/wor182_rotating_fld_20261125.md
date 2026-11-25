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

The pressure-floor-only repair exposed a second issue.  The cycle-131 solve
ended by relaxed acceptance with a gas maximum residual of 0.148 (configured
relaxed maximum: 0.2), and cycle 132 failed with a gas residual of 0.942.  The
offending cell was exactly at the gas-energy lower bound, while the
unconstrained Newton direction pointed below that bound.  This is a
complementarity condition, not a failed unconstrained root.

Commit `690256155` implements the lower-bound active set consistently in the
residual, Jacobian, and state update.  An active cell keeps gas energy at the
pressure floor and solves the radiation equation directly; if its residual
later points back into the feasible region, it leaves the active set.  This is
important for mixed-frame FLD because the direct radiation row must retain the
full nonsymmetric drift stencil.

The exact restart and timestep that previously failed completed in XD job
347731.  FGMRES/Newton converged in five iterations, with final maximum
residual 5.23e-9 and gas maximum residual 1.00e-14.  The pressure remained
1.00e-4 dyn cm^-2.  Job 347732 then continued from time 0.868 to 1.0 at
CFL=0.1; all four implicit solves converged in five iterations without relaxed
acceptance, with final maximum residuals between 5.21e-9 and 5.34e-9.

At time 1.0 the interior radial Mach maximum was 2.37e-3.  The low-density
surface-taper maximum was 3.74, but it occurred in the pressure-floor exterior;
the stellar mass with radial Mach above 0.3 was 1.75e28 g (8.8e-6 Msun).
Relative to time zero, mapped mass changed by 3.47e-12, mapped angular momentum
by 1.75e-7, and the volume-integrated gas-plus-radiation energy by -1.26e-5.
The latter is a state change rather than a closed-system conservation residual,
because this setup includes radiation transport and boundary fluxes.

## CFL ramp

After the constrained solve passed at CFL=0.1, two short restart gates tested
the requested production CFL without imposing a timestep cap:

- job 347734, CFL=0.2, time 1.0--1.3: five Newton solves, all converged in
  five iterations; maximum FGMRES count 23 and final nonlinear maximum
  residual 5.96e-9;
- job 347736, CFL=0.3, time 1.3--1.6: three Newton solves, all converged in
  five iterations; maximum FGMRES count 22 and final nonlinear maximum
  residual 6.26e-9.

At time 1.6 the interior radial Mach maximum was 2.58e-3 and the mass above
radial Mach 0.3 was 1.79e28 g.  The exterior pressure remained exactly at
1.00e-4 dyn cm^-2.  Thus the earlier failure was not an intrinsic CFL=0.3
limit; it was the inconsistent floor and constrained-residual treatment.

## Long CFL=0.3 continuation

XD job 347737 continued the repaired L4 model from time 1.6 to 50 at CFL=0.3.
All 398 implicit solves converged without relaxed or fatal acceptance.  The
mean and maximum Newton counts were 5.95 and 6, the mean and maximum FGMRES
counts were 20.97 and 27, and the largest final nonlinear residual was
8.10e-9.  This confirms that FGMRES with the multigrid preconditioner is
robust for this nonsymmetric mixed-frame system.

At time 50 the history diagnostics gave an apparent interior radial Mach
maximum of 0.194 and a surface maximum of 149.  The stellar mass with radial
Mach above 0.3 was 8.07e31 g, or 0.28% of the mapped stellar mass.  The mapped
mass drift was 6.8e-10, the mapped angular-momentum drift was 1.16e-5, and the
gas-plus-radiation energy change was -5.44e-4.  The mean stellar radius changed
by 5.37e-5 and radial kinetic energy remained 5.68e-6 of the gravitational
binding-energy magnitude.  The large maximum therefore did not describe a
global expansion or loss of equilibrium.

## Atmosphere radiation-force A/B test

XD job 347743 repeated the continuation through time 20 while suppressing the
explicit radiation force below 101 times the atmosphere density floor.  It was
indistinguishable from the unmasked run at the reported precision:

| time 20 diagnostic | unmasked | masked |
| --- | ---: | ---: |
| interior radial Mach maximum | 0.089903 | 0.090009 |
| surface radial Mach maximum | 34.078 | 34.139 |
| stellar mass with radial Mach > 0.3 | 9.24e28 g | 9.26e28 g |

Consequently, direct radiation acceleration of the pure numerical atmosphere
is not the source of the growing surface motion.

## Full-force balance diagnostic

The original force diagnostic contained gas pressure, gravity, core gravity,
and centrifugal acceleration, but not the radiation acceleration.  It could
not measure equilibrium for a radiation-supported model.  Commit `0985c1279`
adds the acceleration computed by the exact face-gradient and flux-limiter
discretization used by the explicit FLD source and reports the residual of all
five forces.

Short restart probes at the beginning (XD job 347761) and end (347762) of the
long continuation gave:

| time | old relative L2 residual | full relative L2 residual | full local maximum |
| ---: | ---: | ---: | ---: |
| 1.721688 | 0.35643 | 9.12e-4 | 0.776 |
| 50.1217 | 0.35673 | 1.59e-3 | 0.668 |

Thus the mass-weighted stellar force balance is accurate to 0.09% initially
and 0.16% at time 50.  Radiation pressure supplies the roughly 36% force that
was missing from the old diagnostic.  The high local maxima nevertheless show
that a small subset of cells is poorly balanced.

## Equatorial localization

Equatorial primitive and force slices were generated immediately after the
time-1.6 and time-50 checkpoints (XD jobs 347765 and 347766).  The match
ellipsoid used by the history diagnostic has equatorial radius 927.5 code
units.  It includes the innermost surface-transition cells, so its label
"interior" must not be interpreted as the bulk stellar interior.

At time 1.601, the largest radial Mach number at normalized equatorial radius
q <= 1 was 2.59e-3.  Its full relative force residual was 1.03e-2.  In radial
bins below q=0.95, local full-force imbalance stayed at approximately 2% or
less.  A representative transition cell at q=1.097 instead had:

- radial Mach number 0.0919;
- density 2.23e-14 g cm^-3 and pressure 2.49e-2 dyn cm^-2;
- outward gas-pressure acceleration 0.761 in code units;
- inward gravity 0.00393, radiation acceleration 0.000407, and centrifugal
  acceleration 0.000792;
- full relative force residual 0.990.

At time 50.001, the largest q <= 1 radial Mach number, 0.159, was at q=0.990:
the innermost surface shell.  The maximum below q=0.9 was only 0.0041 (and was
0.016 in the innermost q < 0.1 bin).  At q=1.099 the radial Mach number was
0.389 and the outward gas-pressure acceleration was 0.212, versus 0.00392
from gravity, 0.000164 from radiation, and 0.000794 from rotation.  The full
relative residual there was 0.964.

The disturbance is therefore launched by an excessive outward gas-pressure
gradient in the partially stellar surface-matching layer and propagates
inward from the surface.  It is not a deep-core instability, an implicit FLD
solver failure, or direct acceleration of pure atmosphere cells.

## Initialization inconsistency exposed by the slices

`scf_map::CellAverage` already evaluates the LTE gas/radiation split at every
quadrature point and separately averages `gas_internal_energy` and
`radiation_energy`.  However, the FLD branch of `scf_rsg.cpp` ignores those
averages whenever a cell contains any stellar quadrature weight.  It instead
computes a new temperature and LTE split from the cell-averaged density and
total pressure.  This operation is nonlinear and treats a cut cell containing
mostly atmosphere as wholly stellar for the split.  The resulting gas
pressure need not equal the volume average of the locally balanced gas
pressure, exactly where the slices find the order-unity error.

This suggested a controlled test using the already averaged
`mapped.gas_internal_energy` and `mapped.radiation_energy`.  A low-memory C++
mapping audit subsequently falsified that explanation: the offending cells
on the surface-L2 mesh have stellar quadrature volume fractions of exactly
zero or one, and pointwise versus post-average LTE splitting changes their
pressure acceleration by less than 1e-5 relative.  Commits `36f93ac4d` and
`b2ee32ecf` therefore revert the trial production change rather than retaining
an ineffective extra code path.

## Atmosphere-interface resolution

The same mapping audit identifies the actual discrete mismatch.  Along the
equator, the last stellar cell has density 4.23e-11 g cm^-3 and gas pressure
3.83 dyn cm^-2, while the next cell is the 1e-14 g cm^-3 numerical atmosphere
at 1.16e-2 dyn cm^-2.  The stellar cell itself has a pressure acceleration of
0.662 cm s^-2, comparable to gravity, but the centered pressure gradient
divided by the atmosphere density gives 330 cm s^-2 in the first atmosphere
cell.  The polar value is 45.3 cm s^-2.  The finite gamma=5/3 taper is
continuous analytically but its scale height collapses to zero at its outer
edge, so surface L2 cannot represent the final pressure drop.

Commit `4c49de19d` extends `scf_surface_taper.cpp` with an optional nonzero
atmosphere-matched endpoint.  It uses the hydrostatic family

```
rho(h) = rho_atm + C h^n
P(h)   = P_atm + rho_atm h + C h^(n+1)/(n+1),  n = 1/(gamma-1),
```

which preserves `dP/dh = rho`, exactly matches the original SCF state at
rho=3e-9 g cm^-3, and approaches the numerical atmosphere continuously at
h=0.  With zero endpoint density the old generator remains bit-for-bit
identical (SHA-256 `3accd919...` for the accepted rotating profile).

A scan selected rho_atm=1.7e-11 g cm^-3 at 10,000 K as the first candidate.
On the surface-L2 mapping it reduces the first-atmosphere-cell pressure
acceleration from 330 to 0.550 cm s^-2 at the equator and from 45.3 to 0.147
cm s^-2 in magnitude at the pole.  Its taper changes the original SCF mass by
1.82e-4, versus 2.53e-4 for the previous zero-endpoint taper; the difference
between the two tapered models is only 7.1e-5 of the gas mass.  This is a
numerical-atmosphere regularization, not a claim that the physical stellar
photosphere has this density.

The next acceptance test is a fresh-map L4 startup A/B with initial and final
force slices.  It must verify that the full local force residual and surface
Mach improve without degrading the bulk-star L2 balance or implicit FGMRES
convergence.  Only after that short gate should the candidate be continued to
time 1 and then time 50.
