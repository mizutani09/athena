# WOR-120: HD 4wz sensor-threshold experiment (2026-11-20)

## Scope

- Branch: `wor-120-hd-4wz-sensor`
- Base: `wor-120-hd-4wz` commit `1f3bd1412`
- Experiment commit: `77f728bab`
- Production default: unchanged at `hydro/berta_sensor_threshold=1.0e-2`

This branch only makes the Berta density/pressure curvature threshold an input
parameter. It is deliberately separate from the production branch. The BSCF source
tree is not used or modified by this change.

The parameter is read once per `Hydro` object and must be finite and positive. The
existing hard-coded value remains the default, so inputs that omit the new key retain
the previous numerical path. The SCF input states the default explicitly to preserve
provenance in copied run directories.

## Validation

The complete `hydro4/hydro_linwave_3d` regression passed with the parameter omitted,
therefore exercising the default-value path. The 4wz results are unchanged:

| wave | Nx1 | RMS-L1 | finest-pair order |
|---|---:|---:|---:|
| sound | 64 | 1.23819e-10 | 4.14614 |
| entropy | 64 | 8.71055e-11 | 4.13906 |

The pre-existing RK4+4c and SSPRK5_4+4 controls also passed. A one-cycle 4wz smoke
test completed with an explicit value of `1.0e-2`; an explicit value of zero was
rejected at startup with
`hydro/berta_sensor_threshold must be finite and positive`.

## Use in the SCF campaign

The accepted L2 B32 result at the conservative default already leaves 86.2% of the
stellar tracer mass on the high-order path. Threshold relaxation is therefore not a
prerequisite for the pending 100-cycle and surface-L3 acceptance gates.

If those gates pass, sensitivity runs may compare `1e-2`, `2e-2`, and `5e-2` from an
identical restart. A larger threshold labels more cells as smooth. It may reduce PLM
fallback, but it must not be promoted on fallback percentage alone. Adoption requires
finite states, no new gravity warnings, bounded Mach growth and conservation drift,
and no material loss of robustness over a run long enough to include repeated AMR
updates.
