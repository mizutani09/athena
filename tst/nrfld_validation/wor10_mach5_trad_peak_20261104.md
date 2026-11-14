# WOR-10: Mach 5 radiation-temperature peak audit (2026-11-04)

## Result and definitions

The normal HDF5 profile is written **after** the implicit FLD solve in `src/main.cpp`. The stage CSV diagnostic samples the central transverse cell after explicit RK2 and after the implicit solve. Enabling that diagnostic leaves the final HDF5 profile bitwise unchanged in the N128 and N256 controls.

The *spatial wiggle* is the largest fall from a local T_rad maximum to the next minimum in the first 300 cm downstream of the numerical density jump. This is the small non-monotone feature in the saved solution. The *reference residual* is the largest positive difference between numerical and semi-analytic T_rad within 60 cm of the density jump. These are different quantities. For reference comparisons, both profiles are centered on their **density** jumps; there is no extra T_rad or peak alignment. A translation cannot change a temperature difference between two numerical cells.

All runs use the Mach 5 fixed λ=1/3 problem and reach 0.04 s with strict Newton/FGMRES convergence. The [Zhang et al. (2011) §6.6 benchmark](https://ccse.lbl.gov/Publications/wqzhang/castro2.pdf) also uses λ=1/3 and a [-4000, 2000] cm domain. The existing semi-analytic CSV has a T_gas peak of 1077.10 K with an approximately 3.19 cm half-height width.

| Cells across 6000 cm | dx [cm] | T_rad wiggle [K] | Reference residual [K] | T_gas peak [K] |
| --- | ---: | ---: | ---: | ---: |
| N64 | 93.750 | 0.034607 | 0 | 861.248 |
| N128 | 46.875 | 0.108032 | 0.205333 | 866.658 |
| N256 | 23.438 | 0.039368 | 0.236807 | 878.803 |
| N512 | 11.719 | 0.022705 | 0.289433 | 905.072 |

The wiggle is ~0.108 K at its largest in this sweep, about 1.3e-4 of the ~855 K T_rad background. It is not monotone across all resolutions, but it decreases from N128 through N512. The reference residual does not follow the same trend. Even N512 has cells wider than the analytic gas spike. The N512 gas peak remains 172 K below the semi-analytic peak, so WOR-67's physical accuracy gate remains unmet. The earlier one-level AMR run had a 0.025818 K wiggle at finest dx=23.438 cm and a T_gas peak of 878.920 K.

## Explicit versus implicit update

The final timestep is shortened to land exactly at 0.04 s. The previous full step is more representative of the usual update size; both are recorded in `stage_metrics.csv`.

| Final shortened step | N128, dx 46.875 cm | N256, dx 23.438 cm |
| --- | ---: | ---: |
| dt [s] | 5.119e-6 | 4.576e-6 |
| T_gas peak after RK2 → after implicit [K] | 876.154 → 866.663 | 898.567 → 878.808 |
| T_rad peak after RK2 → after implicit [K] | 858.035 → 855.591 | 863.551 → 855.542 |
| T_rad spatial wiggle after RK2 → after implicit [K] | 0 → 0.108044 | 0.172542 → 0.039408 |

At N128 the implicit step *creates* the small post-step wiggle, whereas at N256 it *reduces* a larger explicit-stage wiggle. It lowers the gas peak by 9.49 K (N128) or 19.76 K (N256) in that last step; the maximum same-cell gas temperature drop at N128 is 11.16 K. On the preceding full N256 step, the peak falls from 916.36 to 878.52 K. Thus output timing matters greatly for the apparent gas peak, yet the saved peak still reflects the actual end-of-step state, and the gas spike remains under-resolved.

At the explicit radiation hotspot, the first-stage transport raises T_rad by 5.45 K (N128) or 9.53 K (N256); the explicit mixed-frame source changes it by only 0.054 or 0.046 K. The implicit update then lowers it by 5.48 or 9.51 K. RK2 stage averaging means these increments must not be summed as a single cell budget. The implicit per-cell gas+radiation energy change agrees with its diffusion source to <1.5e-13 relative in these last-step samples, and the conserved and primitive gas temperatures agree within 4e-13 K. This excludes an output-conversion error and a gross implicit energy imbalance.

At N128, χ_R dx=39.79 and κ_P dx=0.00184. The cell diffusion time is about 1.9e-7 s and the radiation absorption time about 8.5e-7 s, both shorter than the typical 1.77e-5 s explicit CFL step. This explains why the two stages can show markedly different peaks without implying a failed implicit solve.

## Controls and interpretation

| Control at N128 unless stated | T_rad wiggle [K] | Finding |
| --- | ---: | --- |
| CFL 0.3 / 0.15 / 0.075 | 0.108032 / 0.037476 / 0.136841 | No monotone temporal convergence of the tiny local feature at fixed dx. |
| Right boundary 1250 / 2000 / 2750 cm | 0 / 0.108032 / 0.019226 | Changing only downstream extent at fixed dx changes the feature. |
| Right boundary 5000 / 8000 cm | 0.275635 / 0.154236 | Wider domains also change shock location and profile; no simple boundary trend. |
| Reconstruction order 2 / 1 | 0.108032 / 0 | First order smooths the feature but also broadens and shifts the shock and lowers T_gas peak. |
| Newton/FGMRES thresholds 1e-8 / 1e-10 | 0.108032 / 0.108032 | Final T_rad arrays are bitwise equal. |
| Mixed-frame drift implicit / explicit | 0.108032 / 0.108032 | No visible effect at HDF5 precision. |
| Mixed-frame terms on / off | 0.108032 / 0.114685 | Small effect, not the main cause. |

The right-boundary controls keep x_min=-4000 cm and dx=46.875 cm; nx1 changes with x_max. They are full initial-value calculations, so the changed shock position and pre-shock state prevent attributing the difference solely to a local boundary stencil. First-order reconstruction is likewise a diagnostic, not a candidate production fix: it suppresses the wiggle through extra diffusion while losing gas-spike information. A variable λ would change the published comparison problem and is not an appropriate control for this issue.

The evidence supports a small **discretization- and domain-sensitive** non-monotonicity around an under-resolved shock, with a phase-dependent explicit/implicit contribution. It does not identify a solver defect or a single unique term to change. We therefore make no FLD physics change or create a duplicate fundamental-fix issue. WOR-67 already tracks the necessary high-resolution gas-spike and related dx/dt/boundary verification. If that study finds a non-vanishing T_rad feature under convergence, it should open a specific numerical-scheme issue with that evidence.

## Reproduction

The combined HTML and copied subreports are at `/mnt/d/output/rad/wor10_complete_audit_20261104/index.html`. All simulations and original inputs are under `/mnt/d/output/rad/wor10_*_20261104`. The manual boundary/CFL controls have `command.txt` and `input.effective` next to the original `input.used`; `run.log` records convergence. The diagnostic scripts are `mach5_trad_peak_audit.py`, `mach5_split_stage_audit.py`, and `mach5_wor10_report.py` in this directory. The N128/N256 stage runs set `fld/stage_diagnostics=true`, sampled the last 0.5/0.2 ms, and used the same production executable and input otherwise.

To reconstruct the aggregate from the saved subreports:

```bash
python3 tst/nrfld_validation/mach5_wor10_report.py \
  --output /mnt/d/output/rad/wor10_complete_audit_20261104 \
  --audit spatial=/mnt/d/output/rad/wor10_spatial_sweep_20261104 \
  --audit time_step=/mnt/d/output/rad/wor10_dt_sweep_20261104 \
  --audit right_boundary=/mnt/d/output/rad/wor10_boundary_sweep_20261104 \
  --audit reconstruction=/mnt/d/output/rad/wor10_xorder_audit_20261104 \
  --audit solver_tolerance=/mnt/d/output/rad/wor10_tolerance_audit_20261104 \
  --audit mixed_frame=/mnt/d/output/rad/wor10_mixed_frame_audit_20261104 \
  --stage N128=/mnt/d/output/rad/wor10_stage_n128_20261104/report \
  --stage N256=/mnt/d/output/rad/wor10_stage_n256_20261104/report
```
