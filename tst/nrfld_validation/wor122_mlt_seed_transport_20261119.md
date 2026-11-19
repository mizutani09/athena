# WOR-14/122: Henyey-MLT seed survival and outer-boundary controls (2026-11-19)

## Decision

The mass-matched Henyey-MLT surface model does not pass a two-day local
full-FLD preparation gate. A coherent buoyancy seed grows if radiative
diffusion is artificially disabled, but loses nearly all of its temperature
contrast with full FLD. Closing the outer radiation flux reduces cooling
at 813–817 R☉ but overheats the upper atmosphere. Fixing the temperature
at the **top face** leaves the rapid cooling almost unchanged. Neither
boundary control establishes useful convective transport. No whole-star
or long XD job is justified from these states.

## Matched local tests

All four runs use the profile exported in
`data/ic/wor122_henyey_mlt_patch_830.*`, the 800–860 R☉ patch,
16×16×96 cells (0.625 R☉ spacing), four local MPI ranks, Tomida EOS,
WOR-55 gray opacity, profile gravity, radiation force, reflecting lower
gas boundary, and no heating. A 2% coherent isobaric density perturbation
and buoyancy-aligned 0.2 cₛ velocity seed are identical at t=0. Each runs
to code time 8 (2.04 days), with snapshots at approximately 0, 4 and 8.

| Case | FLD diffusion | Top radiation boundary |
| --- | --- | --- |
| Full | on | Marshak vacuum, α=0.5 |
| No diffusion | off | Marshak setting irrelevant because transport is off |
| Nearly closed top | on | Marshak vacuum, α=10⁻⁸ |
| Fixed top | on | Fixed temperature, 2431.18 K at the top ghost boundary |

The fixed-temperature case constrains only the top boundary at 860 R☉;
it does not freeze the whole 817–860 R☉ volume.

## Two-day result

| Case | Mean T/T₀, 813–817 | Coherent T-mode / initial | Mean T/T₀, 850–860 | Corrected convective proxy / F★, 813–817 | Maximum Mach, 813–817 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 0.7009 | 0.0793 | 1.1104 | 0.00001 | 0.311 |
| No diffusion | 0.9997 | 3.4681 | 1.0002 | 0.00231 | 0.192 |
| Nearly closed top | 0.8899 | 0.3559 | 2.3021 | 0.00018 | 0.313 |
| Fixed top | 0.6992 | 0.0783 | 0.9906 | 0.00001 | 0.311 |

The corrected proxy is the horizontally averaged enthalpy flux after
subtracting mean mass advection; a coherent box drift is not counted as
convection. The no-diffusion result is a **preparation-only** control and
its small flux is still far below the roughly 0.75 F★ missing from the
initial mapped radiative flux. The 813–817 R☉ mean temperature is already
0.75 T₀ after one day with full FLD. With diffusion disabled it remains
at T₀ while the seed grows, directly separating buoyant growth from the
radiative damping in this numerical setup.

The nearly closed case has top radiative flux only 2.8×10⁻⁷ F★ at t=8;
its interior FLD still moves energy out of the transition and heats the
850–860 R☉ layer to 2.30 T₀. The fixed-top case keeps that outer layer
near its initial temperature but does not protect the transition.
Consequently an outer boundary change by itself does not correct the
initial 3D flux deficit. These four short controls do not prove that all
possible volumetric outer supports or lower-luminosity precursors fail.

All four runs reached t=8 with zero `FATAL`, empty stderr, and Newton
convergence on every positive-dt solve. Each log additionally has a
`dt_zero_initialization` record; this is not a failed solve. The full
run, for example, has 138 converged positive-dt solves. Mass drift from
full-domain snapshots is at most approximately 5×10⁻⁶. The outcome is a
physical thermal-transport failure, not an implicit-solver or lower-gas-
boundary failure.

## Next gate

A further temporary support should be screened against the actual 3D
temperature-mode survival and corrected convective flux, not only the
horizontal mean temperature. A useful precursor must develop a sustained
convective contribution comparable to the missing luminosity while
retaining a bounded atmosphere, then survive full FLD after the support
is removed. The present Henyey model's sharp handoff and missing turbulent
support make it a poor direct 3D initializer. Revising the underlying
stratification and luminosity history, potentially with the MESA
core/radiative/convective-layer work in WOR-151/155/172, is now more
promising than another seed-amplitude or top-face-boundary sweep.

Compact data: `wor122_mlt_seed_transport_20261119.json`.
Figure: `doc/figures/wor122_mlt_seed_transport_20261119.png`.
The local runs and full diagnostic JSON are under
`/tmp/wor122_mlt_seed_{full,nodiff,closedtop,fixedtop}/`.
Reproduce all four from the repository root with a locally configured
Athena binary:

```sh
bash tst/nrfld_validation/wor122_mlt_seed_transport_local.sh \
  /absolute/path/to/athena /tmp/wor122_mlt_seed_transport
```

The script uses `tools/rsg/analyze_rsg_entropy_mode.py` and
`tools/rsg/analyze_rsg_surface_transfer.py`; both now accept local height
coordinates or absolute radius coordinates for this surface patch.
