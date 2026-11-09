# WOR-74: solar flux calibration on the 96 km box

## Physical and numerical target

Use the completed `t=24000` 64×64×32 checkpoint of the 96 km box. Its last
500-code-time means are `F_top=0.62989 F_sun`, `F_bottom=0.63429 F_sun`, and
the stored-energy trend is `+0.01140 F_sun`. Here `F_sun=6.3e10 erg cm^-2
s^-1`, and bottom total includes hydro, radiation diffusion, and radiation
advection. The prescribed lower radiation flux alone is only
`3.54354335e5 erg cm^-2 s^-1`, so the inflow entropy governs almost all of
the lower energy supply.

For the current top Marshak coefficient `alpha=0.5` and external radiation
energy zero, the observed top radiation energy is about `2.65 erg cm^-3`.
With the same closure, a solar flux requires an effective boundary radiation
energy of approximately `F_sun/(alpha*c)=4.20 erg cm^-3`, corresponding to a
radiation temperature near 4855 K. This is a target for the boundary face,
not an instruction to rescale the top cell artificially. The initial 1D
profile specifies `T_top=4850.985 K`, but the relaxed 3D top cell is near
4327 K.

At `t=24000`, the measured top-face diffusion flux divided by
`0.5 c ×` the top-cell mean radiation energy is `0.9971`; at `t≈24300` in
the `+1e7` branch the same ratio is `0.9944`. The small departure is expected
because the actual boundary law uses a face state, whereas these ratios use
the adjacent cell mean. This checks that the low upper flux follows the low
upper radiation energy; it does not establish why the surface energy settled
there.

Rempel's HD2 condition prescribes the entropy of bottom upflows while leaving
downflow entropy symmetric; the prescribed value is chosen to obtain the
desired radiative loss at the surface. See [Rempel (2014), Section 2.2](https://arxiv.org/pdf/1405.6814).
We retain that physical boundary structure and calibrate only its input
entropy. Changing the upper Marshak coefficient is a diagnostic, not the
luminosity control.

## Matched restart experiment

The frozen executable SHA-256 is
`d0d0f78ae0762580dae674d665556ff6ebaa89df1a2fdf09ccb1a706e3e20d7f`.
All branches share the same `t=24000` checkpoint, grid, EOS and opacity data,
CFL 0.3, mass-balance settings, and 16 MPI ranks. The baseline holds
`s_in=1.583570672066638e9 erg g^-1 K^-1`; the perturbation holds
`s_in+1e7`. A 10-code-time short branch is followed by a 300-code-time
continuation. These are transient response tests, far shorter than the
estimated whole-box thermal time of about 20,600 code units.

The `+1e7` continuation completed at `t=24310`: 4,358/4,358 Newton solves
converged, no slow multigrid warning, minimum available memory 18.27 GiB.
Over `t=24210–24310` it had `F_top=0.63778 F_sun`, `F_bottom=0.66099
F_sun`, and a stored-energy trend of `+0.03698 F_sun`. The lower entropy
contrast rose from `1.56e6` at the shared checkpoint to `5.04e6` at
`t≈24100`, then fell to `1.96e6` by `t≈24300` as downflow entropy caught up.
The matched fixed-entropy baseline is complete. Both runs have zero fatal and
slow-multigrid warnings; 4,279 and 4,358 Newton solves, respectively, all
converged. The sampled energy-ledger residual magnitude is below
`7e-5 F_sun` in each window, so the large temporary mismatch between upper
and lower flux is accounted for as stored energy at this diagnostic accuracy.

| Code-time window | Branch | Top / F_sun | Bottom / F_sun | Stored-energy trend / F_sun |
| --- | --- | ---: | ---: | ---: |
| 24010–24150 | fixed s_in | 0.62833 | 0.62098 | -0.01111 |
| 24010–24150 | s_in +1e7 | 0.62107 | 4.26212 | +3.83622 |
| 24150–24310 | fixed s_in | 0.62929 | 0.64441 | -0.00494 |
| 24150–24310 | s_in +1e7 | 0.63667 | 0.74490 | +0.08903 |

Over the later 160-code-time window, the perturbation raises the lower flux
by `+0.10049 F_sun` and the upper by only `+0.00738 F_sun`; the stored-energy
trend difference is `+0.09397 F_sun`. Thus it still increases energy input
after the strongest initial transient, but has not supplied the missing
`≈0.37 F_sun` or established a new equilibrium. The full thermal response
cannot be inferred from this 310-code-time test. At `t≈24300`, the bottom
cell's mass-flux-weighted entropy contrast is `1.96e6` in the perturbed run
versus `2.26e6` in the baseline: downflow entropy has caught up enough that
the interior contrast alone is not a reliable controller signal.

The branch output is
`/mnt/d/output/rad/wor74_entropy_range_20261109/`. The analysis script
`tst/nrfld_validation/solar_convection_wor74_long_branches.py` produces a
large-text HTML report, plots, and hash-checked JSON once the matched
baseline completes.

## Lower-flux controller pilot

The previous controller uses a rolling *upper* flux error, whose response is
delayed by the box thermal time. A new opt-in `--feedback-face bottom` mode in
`run_solar_entropy_feedback_long.py` uses a 500-code-time rolling mean of the
measured lower total flux. At each 100-code-time restart, it updates

`delta_s = clip[(100/1000) (1-F_bottom/F_sun) (1e6/0.322326259),
                -5e5, +5e5] erg g^-1 K^-1`,

with a cumulative bound of ±`3e7` from the original profile entropy.
At the mean pressure of the `t=24000` bottom cell, an EOS-table inversion
maps `+3e7` to roughly `+600 K` in an isolated (P,s) state; this sets a
finite exploration range, not a prediction of the eventual 3D temperature.
The numerical coefficient is a conservative integral-controller scale based
on a *transient* lower-face response, not a measured steady upper-face gain.
This separation is important because a fixed entropy step causes a large
initial input and then downflow entropy responds. The pilot will be assessed
from lower/upper flux, stored-energy trend, mass drift, and solver status;
reaching `F_bottom≈F_sun` for one segment is not a calibrated result.

The pilot started at 2026-11-09 17:31 JST at
`/mnt/d/output/rad/wor74_bottom_flux_controller_20261109/`, forked from the
last complete `t=24000` restart. It is configured to stop at `t=26000` for
review. Its `state.json`, `index.html`, and `memory_monitor.csv` provide
restart status and resource use. A subsequent thermal-time continuation is
needed before claiming sustained `F_top≈F_sun`.

The first controlled segment `t=24000–24100` completed in 10.17 minutes.
Its upper/lower fluxes are `0.62896/0.64919 F_sun`; the 500-code-time
rolling lower total is `0.64117 F_sun`, so the next bounded update is
`+111,325 erg g^-1 K^-1`. All 1,412 Newton solves converged, there were no
slow-multigrid warnings, mass drift was 0.0273%, and the lowest available
memory was 18.12 GiB. The second segment started automatically. At this
measured speed, the remaining 19 segments would finish around 20:55 JST on
2026-11-09, subject to runtime variation and resource guards.

To resume the prepared pilot from this worktree, pass exactly the recorded
settings (the `--allow-binary-change` flag is required because the older
reference manifest predates the diagnostic-only frozen executable):

```bash
python3 tst/nrfld_validation/run_solar_entropy_feedback_long.py \
  --binary /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/athena.frozen \
  --base-input /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/athinput.base.validated \
  --data-dir /mnt/d/output/rad/wor74_feedback_20261106/from_t5000_96km_rolling500_t20000/data \
  --reference-manifest /mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot/face_gravity_hd2_mass_tau2_rate40_current_64x64x32_np4_t4255_5000_resume/manifest.json \
  --output-root /mnt/d/output/rad/wor74_bottom_flux_controller_20261109 \
  --initial-s-in 1.581097454457282e9 --target-time 26000 \
  --segment-code 100 --flux-average-code 500 --tau-kh-code 1000 \
  --feedback-face bottom --response-fsun-per-1e6 0.322326259 \
  --max-entropy-change 500000 --max-total-entropy-change 30000000 \
  --ranks 16 --resume --allow-binary-change
```
