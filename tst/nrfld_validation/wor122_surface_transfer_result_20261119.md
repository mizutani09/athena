# WOR-14/122: long local-surface source-transfer gate (2026-11-19)

## Decision

XD job **342523** completed normally but **did not pass** the physical release gate. Moving the artificial 812.5–817.5 R☉ luminosity source to the bottom over code time 80–160 did not create a self-sustaining convective enthalpy flux. The nominal enthalpy flux at the transition becomes positive only because of a coherent mean vertical flow; after subtracting mass advection, its horizontal covariance remains ≤`2.0×10⁻⁴ F★`. The radiative flux leaving the top declines to `0.428 F★` by code time 200, and the local box loses 5.35% of its mass. The transition temperature first rises, then falls, then overshoots strongly. No year-scale full-star continuation should be based on this preparation.

## Run and numerical checks

- Input: `inputs/rsg/athinput.wor122_rsg_patch_wide_transfer`, 40×40×60 R☉ Cartesian periodic patch, 64×64×96 cells (Δ=0.625 R☉), r=800–860 R☉, Tomida EOS and WOR-55 opacity, profile gravity, radiation force, Marshak top. The volumetric luminosity is held in the surface shell until t=80, moved by a smooth cosine to the bottom through t=160, then held at the bottom to t=200. `F★=8.389×10⁹ erg cm⁻² s⁻¹`.
- XD M-bulk-a **342523**: `COMPLETED`, exit 0, elapsed 02:39:24, physical span 50.93 days, 5941 cycles. All 5941 Newton steps reported `converged`; no FATAL, stderr empty, opacity out-of-domain/clamped/nonfinite counters all zero. Full snapshots at t=0/40/80/120/160/200 remain on `$WORK/wor122/surface_patch_wide_transfer/`.
- The source-power history has `(Pbottom+Psurface)=42.4454` code units throughout (16 times the 10×10 R☉ patch power, as required by area). This verifies the programmed transfer; it does not by itself establish a closed energy budget, since the gas boundary is open.

| Code time (days) | Surface source fraction | Tgas/T0, 813–817 | Top Frad/F★ | Raw Fenth/F★, 813–817 | Mass-advection-corrected Fenth/F★ | Box mass drift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (0) | 1.00 | 1.000 | diagnostic unavailable | 0.000 | ≈0 | 0 |
| 40 (10.2) | 1.00 | 1.075 | 0.855 | 0.038 | 0.000015 | −0.27% |
| 80 (20.4) | 1.00 | 1.303 | 0.811 | 0.044 | 0.000018 | −2.07% |
| 120 (30.6) | 0.50 | 1.220 | 0.769 | 0.008 | 0.000019 | −4.71% |
| 160 (40.7) | 0.00 | 0.770 | 0.622 | 0.276 | 0.000003 | −5.15% |
| 200 (50.9) | 0.00 | 1.826 | 0.428 | 0.358 | 0.000198 | −5.35% |

Temperatures and interior fluxes are full-3D horizontal means, then averaged over the 813–817 R☉ heights. The corrected enthalpy flux at each height is `〈(e_gas+P_gas)v_z〉 − 〈e_gas+P_gas〉〈ρv_z〉/〈ρ〉`. It removes the bulk mass flow, and is a convective proxy rather than a complete luminosity decomposition. The final raw-to-corrected flux ratio is about 1800, so raw `Fenth` cannot be interpreted as convection in this run. The initial random vertical-velocity rms at the transition is 0.211 km/s, but the horizontal rms falls to 0.060 km/s by t=80; horizontal temperature rms is only 0.19% then. The initial 1D profile is locally superadiabatic (`∇−∇ad≈0.69–0.77` at 800–817 R☉), yet the chosen seed does not build meaningful turbulent heat transport during this run. The local pressure-scale-height dynamical time is about 19 days, so the first 20 days cover only roughly one such time.

At t=160, after complete source transfer, the 800–813 R☉ maximum Mach reaches about 1.00, with mean vertical speed +4.64 km/s. At t=200 the 813–817 R☉ mean speed is +2.38 km/s and the 817–830 R☉ maximum Mach about 0.80. These are coherent vertical motions accompanying large thermal and density rearrangement, not evidence of developed convection.

## Boundary finding

The legacy `HydroReflectInnerX3` in `nrfld_conv_sun.cpp` is **not** an impermeable reflecting boundary: it copies the interior normal velocity with the same sign into lower ghost cells and supplies an inflow internal energy when velocity points upward. At t=40–120 its lowest active cells have mean downward velocity about −0.55 to −0.74 km/s and mass-flux density roughly `−(0.8–1.1)×10⁻⁴ g cm⁻² s⁻¹`, while the uppermost active-cell flux is around `+10⁻⁷ g cm⁻² s⁻¹`. Thus lower-boundary outflow is the plausible dominant cause of the 5.35% mass loss (≈`1.82×10²⁷ g`); these cell-centered values are indicators, not exact boundary-face flux integrals.

An opt-in mirror-gas lower boundary, `problem/bottom_hydro_reflecting=true`, has now been added without altering the legacy default. In matched 10×10 R☉, t=20 (5.09-day) supported local tests, it reduces mass drift from `−6.68×10⁻⁴` to `−2.43×10⁻⁶`, while transition temperature (`1.01266` vs `1.01271 T0`), raw enthalpy flux (`0.03791` vs `0.03790 F★`), and top radiative flux (`0.88033 F★` in both) are effectively unchanged. It isolates the boundary mass leak but does not create the missing convective flux in five days.

## Coherent entropy-seed gate

The new opt-in `problem/density_perturb_frac=0.02` applies a horizontally coherent, isobaric density perturbation in the active surface layer. Gas energy and radiation LTE energy are recomputed from the perturbed density and unchanged pressure; the old input behavior remains the default. A closed-boundary, supported 10×10×60 R☉ local run completed to t=20 (5.09 days), 411 cycles, with no opacity-domain or Newton failures. In the 813–817 R☉ layer, the horizontal gas-temperature RMS fraction **fell from 0.8495% to 0.1036%**, and the mass-advection-corrected enthalpy flux at t=20 was only `6×10⁻⁶ F★` versus raw `0.0381 F★`. Horizontal vertical-velocity RMS fell from 0.210 to 0.075 km/s. The history mass drift was `−2.41×10⁻⁶` (the float32 full-domain HDF sum gives `−4.13×10⁻⁶`). The perturbation therefore did not establish convective transport on this short, 10 R☉ horizontal scale. A shorter wavelength may be smoothed by radiation and numerical diffusion; this observation alone does not distinguish the two or rule out convection on larger scales.

XD M-debug **342865** tried a 160×160×60 R☉, 64×64×96-cell scale test but failed before time stepping: the FLD multigrid requires cubic cells. The corrected **342866** 40×40×60 R☉, 64×64×96-cell, t=8 (2.04-day) test completed in 2:16 with 0.625 R☉ cubic cells, no opacity-domain failures, and Newton convergence. In the 813–817 R☉ layer, the horizontal temperature RMS fraction fell from 0.8493% to 0.1020%, and the corrected convective proxy at t=8 was only `5.30×10⁻⁶ F★` versus raw `0.0271 F★`. Top radiative flux was `0.890 F★` and history mass drift `−9.64×10⁻⁷`.

The wider **342892** 80×80×60 R☉, 128×128×96-cell gate completed in 11:06, exit 0, at the same 0.625 R☉ cell size and t=8. Its transition-layer horizontal temperature RMS fraction fell from 0.8494% to 0.0959% (11.3% retained, versus 12.0% at 40 R☉), and its corrected convective proxy was `7.39×10⁻⁶ F★` versus raw `0.0269 F★`. Its top radiative flux was `0.890 F★` and history mass drift `−9.63×10⁻⁷`. Thus doubling the horizontal scale from 40 to 80 R☉ did not materially change the damping over two days. This points away from simple horizontal-wavelength limitation; the vertical structure of the imposed perturbation, coupled thermal relaxation, or numerical dissipation remain candidates. The comparison alone does not identify which mechanism dominates.

No further long source-transfer job is justified with this rapidly decaying seed. The next useful gate is a short, controlled energy and perturbation-budget calculation to distinguish radiative thermal smoothing from hydro dissipation, followed by an initial-state or preparation-forcing revision if needed. The local patch remains a transport diagnostic, not a global 12 M☉ star-in-a-box solution.

Compact full-3D time/height profiles and history: `tst/nrfld_validation/wor122_surface_transfer_20261119/analysis.json`, `entropy_seed_10rsun.json`, `entropy_seed_40rsun.json`, and `entropy_seed_80rsun.json`; reproduce with `tools/rsg/analyze_rsg_surface_transfer.py` from the respective full HDF (for the two-snapshot runs, `--snapshots 0,1`). The local 10 R☉ output is `/tmp/wor122_rsg_patch_closed_entropy_seed_full_t20/`; XD outputs are under `$WORK/wor122/surface_patch_wide_transfer/`, `$WORK/wor122/surface_patch_wide_entropy_seed/`, and `$WORK/wor122/surface_patch_80rsun_entropy_seed/`. Figure: `doc/figures/wor122_surface_transfer_gate_20261119.png`, made with `tools/rsg/plot_rsg_surface_transfer.py`.
