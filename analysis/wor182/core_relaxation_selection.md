# Numerical-core relaxation selection

## Scope

The relaxation acts only on the velocity perturbation relative to rigid
corotation.  After the global startup taper ends, it is retained inside the
modified numerical core (`r < 65 R_sun`, with a `35 R_sun` smooth taper).  It
is not a physical viscosity and must not be judged by the material-pressure
force residual alone.  In an FLD model that residual omits radiation pressure.

The production diagnostic is the runtime full-force residual, evaluated after
the FLD face state is current:

```text
gas-pressure force + radiation force + gravity
```

Restart-only snapshots do not refresh the FLD face state and therefore cannot
be used to evaluate this quantity.

## Selection experiment

Compare `relaxation_core_tau = 5, 2, 1` from identical checkpoints over the
same interval:

| model | checkpoint | comparison endpoint |
|---|---:|---:|
| q=0.25 RLOF corotation | 300 | 325 |
| nonrotating | 150 | 175 |

The global relaxation timescale remains 5 code times, but its temporal taper
has already ended at both checkpoints.  Thus only the independently selected
core timescale differs.

Select the *largest* (weakest) timescale that satisfies all of the following:

1. CFL 0.3 completes with FGMRES, without nonlinear/linear breakdown or a
   timestep collapse.
2. The final global and `r < 1.5 R_support` full-force L2 residuals do not
   exceed the tau=5 control.
3. Radial KE, nonradial perturbation KE, interior Mach number, and the stellar
   mass fractions above radial Mach 0.1 and 0.3 are reduced or remain bounded.
4. The source-adjusted total-energy change remains smooth.  Artificial
   corotating-frame KE removal should stay below `2e-4 |W| / t_dyn` during the
   comparison; lower is preferred when two choices control the mode equally.
5. Exterior-envelope motion is not used as a rejection by itself.  The model
   is convective, so localization and mass-weighted energies matter more than
   an isolated maximum Mach number in the surface taper.

The accepted value must then be rerun for at least one dynamical time in both
rotation states.  A checkpoint is suitable for the binary startup only after
that gate.  Persistent core damping in a binary is a separate approximation:
it can absorb waves and angular momentum, so the CE calculation needs either
an undamped handoff or an explicit damping-sensitivity control.

## Baseline at the common endpoint

The tau=5 controls give:

| model | RMS Mach | interior max Mach | radial KE / abs(W) | full-force L2 | KE removed / abs(W) |
|---|---:|---:|---:|---:|---:|
| rotating, t=324.913 | 0.01834 | 0.04927 | 3.96e-6 | 0.00629 | 4.50e-6 |
| nonrotating, t=174.502 | 0.02610 | 0.06930 | 5.75e-6 | 0.00172 | 3.02e-7 |

The comparison jobs are `350509`/`350511` (tau=2) and `350530`/`350531`
(tau=1).  Job `350522` is the one-code-time schema and solver smoke test.
