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
The same-time fixed-entropy baseline is needed before attributing late
differences to the perturbation.

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
The numerical coefficient is a conservative integral-controller scale based
on a *transient* lower-face response, not a measured steady upper-face gain.
This separation is important because a fixed entropy step causes a large
initial input and then downflow entropy responds. The pilot will be assessed
from lower/upper flux, stored-energy trend, mass drift, and solver status;
reaching `F_bottom≈F_sun` for one segment is not a calibrated result.

The pilot is prepared at
`/mnt/d/output/rad/wor74_bottom_flux_controller_20261109/`, forked from the
last complete `t=24000` restart. It is configured to stop at `t=26000` for
review. Its `state.json`, `index.html`, and `memory_monitor.csv` provide
restart status and resource use. A subsequent thermal-time continuation is
needed before claiming sustained `F_top≈F_sun`.

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
