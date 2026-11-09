# WOR-74: coarse-grid diagnosis and XD context (2026-11-09)

## Data and scope

- The local 96 km rolling-feedback campaign is complete only through `t=24000` in this workspace. `state.json` says `stopped`, with a stop signal between segments. The separately reported `t≈40000` continuation is not present here.
- The XD Dropbox snapshot is `/AstroResearch/FLD_solar_XD2000_20261105/wor91_24km_results_20261109/`. Its 24 km fixed-entropy history reaches `t=1103.10`; its published HTML analysis ends at `t=884.909`. The two HST files downloaded for this audit match `MANIFEST.sha256`: initial `4ef92f58482548d6c80ea57037cee1ab7c175a987dc6bb813cfee9041bf89ea1`; continuation `fe3c41d91bf4f0f086663fce952bfcfa5b0900d10eb124a2f6f513771c438b60`.
- The 24 km and 96 km histories use different `s_in` policies and are at different stages of thermal relaxation. Their flux difference is **not** a clean resolution effect.

## Completed 96 km diagnosis

The last two nonoverlapping 2000-code-time windows in the local 64×64×32 calculation are:

| Window | Top / F_sun | Bottom / F_sun | Stored-energy trend / F_sun |
| --- | ---: | ---: | ---: |
| 20000–22000 | 0.62902 | 0.63654 | +0.00793 |
| 22000–24000 | 0.62949 | 0.63576 | +0.00743 |

The sampled face-flux/gravity ledger residual is about `5.5e-5 F_sun` in both windows. All 57,189 Newton solves in the last 40 segments converged; there were no slow-MG warnings. Mass drift stayed below 0.029%. This is a persistent low-flux, nearly balanced state over these windows; it does not establish an infinite-time equilibrium.

The EOS-based interior lower-cell diagnosis compares early `t=13100–17100` with late `t=21000–24000`: the imposed inflow entropy rose by `+851,468 erg g^-1 K^-1`; the mass-flux-weighted upflow and downflow entropies rose by `+871,752` and `+897,500`. Their contrast fell from `2.0645e6` to `2.0388e6` rather than increasing. A separate full-volume vertical profile uses 20 early and 21 late completed snapshots. At the lowest cell, its up/down entropy contrast changed from `2.037e6` to `2.000e6`, while up/down mass circulation was nearly unchanged. These are cell-centred Eulerian measurements, not tracked plumes or exact boundary Godunov fluxes.

From the same `t=24000` restart and frozen binary, five 10-code-time branches changed only `rad_top_alpha` or `s_in`. The common `t=24000.1467–24010` window spans 0.086 vertical crossing time:

| Case | Top / F_sun | Bottom / F_sun | Stored-energy trend / F_sun |
| --- | ---: | ---: | ---: |
| Baseline, alpha 0.5 | 0.63231 | 0.46535 | -0.17649 |
| Alpha 0.25 | 0.52011 | 0.40115 | -0.08335 |
| Alpha 1.0 | 0.68661 | 0.50703 | -0.21726 |
| s_in +1e6 | 0.63232 | 0.78010 | +0.12907 |
| s_in -1e6 | 0.63230 | 0.15651 | -0.47667 |

The `±1e6` entropy branches change the lower net flux by approximately `±0.31 F_sun` but the upper flux by only `±1e-5 F_sun` in this short interval. Their stored-energy rate changes in the same direction as lower input. The top alpha branches establish that the top flux responds to the Marshak boundary coefficient; they do not identify an appropriate physical alpha. At fixed `s_in`, the lower face has directional absolute hydro fluxes of roughly `+158` and `-157 F_sun`; the net is their difference. These large opposing values must be interpreted with a common specific-energy reference, not compared directly with solar convective flux.

## XD 24 km context

The XD history and the local 96 km early history have the same box dimensions but different cell sizes and inflow entropy settings. Over matching `t=1000–1103`:

| Series | Top / F_sun | Bottom / F_sun | Stored-energy trend / F_sun |
| --- | ---: | ---: | ---: |
| XD 24 km, profile entropy fixed | 0.5584 | 0.497 | -0.0711 |
| Local 96 km, entropy feedback | 0.6266 | 0.647 | +0.0423 |

The XD 24 km flow is still cooling over this latest window, and the shared result is not a long thermal-time calculation. Its separate matched-restart `±1e6` test at `t=940–960` finds lower-face responses near `±0.3 F_sun` and almost no upper-face response, consistent with the local short-time branch. The XD investigation also reports a rapid initial upper-cell cooling after the initially solar-like top flux. Its discrete top-face residual and the longer response remain open.

## Reproduce the reports

Run the scripts from the repository root. The data roots below are local to this workstation; download the two XD HST files from the Dropbox snapshot and check their SHA-256 before using them.

```bash
python3 tst/nrfld_validation/solar_convection_boundary_sweep.py \
  --root /mnt/d/output/rad/wor74_coarse_sweep_20261109 \
  --header-from /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/feedback_001_t100/simple_convection_general_6Mm_box.hst \
  --input /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/feedback_240_t24000/athinput.generated \
  --restart /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/feedback_240_t24000/simple_convection_general_6Mm_box.final.rst \
  --binary /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/athena.frozen \
  --output /mnt/d/output/rad/wor74_coarse_sweep_20261109/report
python3 tst/nrfld_validation/solar_convection_xd_context.py \
  --xd-initial /mnt/d/output/rad/wor74_coarse_diagnostics_20261109/xd_24km/long128.hst \
  --xd-continuation /mnt/d/output/rad/wor74_coarse_diagnostics_20261109/xd_24km/continuation_337953.hst \
  --local-root /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000 \
  --local-input /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/feedback_001_t100/athinput.generated \
  --output /mnt/d/output/rad/wor74_coarse_diagnostics_20261109/xd_24km
python3 tst/nrfld_validation/solar_convection_wor74_overview.py \
  --campaign /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000 \
  --header-from /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/feedback_001_t100/simple_convection_general_6Mm_box.hst \
  --root /mnt/d/output/rad/wor74_coarse_diagnostics_20261109 \
  --sweep-report /mnt/d/output/rad/wor74_coarse_sweep_20261109/report
```

The main local report is `/mnt/d/output/rad/wor74_coarse_diagnostics_20261109/index.html`. It links to the entropy, vertical-profile, perturbation and XD-context reports. The calculations themselves are in `/mnt/d/output/rad/wor74_coarse_sweep_20261109/`; `summary.json` records the restart and binary hashes.

## Next discriminating experiment

1. Obtain the separate 96 km `t≈40000` campaign state, manifests, HST and last 3D snapshots. Verify the last complete restart, entropy limit, solver status, and two independent long-window energy ledgers before drawing a final fixed-point conclusion.
2. Use a single relaxed 96 km checkpoint for bounded, matched branches. A lower-flux or up/down entropy-contrast controller can be tested against fixed `s_in`, but the short-time `0.31 F_sun / 1e6` lower-face response must **not** be used as a steady upper-face gain. Run long enough to see whether the downflow entropy rises with inflow entropy and whether the net lower input persists. Do not force the downflow ghost entropy without a separate boundary-energy audit.
3. Repeat the promising physics with **the same explicit `s_in`**, EOS, top boundary and temporal maturity at 48 and 24 km. The currently shared XD 24 km case does not meet that comparison requirement.
