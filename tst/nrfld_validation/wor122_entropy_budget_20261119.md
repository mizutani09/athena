# WOR-14/122: surface-seed damping controls (2026-11-19)

## Question and result

Why did the 2% coherent isobaric density/buoyancy seed in the RSG local surface patch lose almost all of its temperature contrast in about two days? Matched short controls identify **two major drivers: the imposed volumetric surface heating and FLD transport**. Initial random velocity noise is not responsible, and doubling the linear grid resolution does not change the heating-induced loss. These experiments do not assign an additive percentage to each driver because changing the transport also changes the mean thermal background.

## Matched controls

All runs use a 10×10×60 R☉ periodic Cartesian box at r=800–860 R☉, 2% sinusoidal isobaric density seed, Tomida EOS, WOR-55 opacity, profile gravity, radiation force, closed lower gas boundary, and t=8 code units (2.04 days). The baseline resolution is 16×16×96 (Δ=0.625 R☉). Full-domain HDF snapshots at t=0, approximately 4, and 8 are analyzed in the 813–817 R☉ layer. The temperature-mode statistic is the horizontal projection of `T/〈T〉−1` onto the imposed `sin(2πx/Lx)sin(2πy/Ly)`, averaged over those heights and divided by its t=0 value. This tracks the original coherent pattern, whereas a raw RMS also includes newly generated structure.

| Control | Surface heat | Initial random vz | FLD diffusion | Top Marshak | t=8 temperature-mode retention | Density-mode retention | Adiabatic-residual retention | Tmean/T0 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline | on | on | on | α=0.5 | **5.36%** | 12.0% | 7.59% | 0.991 |
| Nearly closed top | on | on | on | α=10⁻⁸ | **10.88%** | 14.8% | 17.5% | 1.223 |
| Diffusion off | on | on | off | absent | **23.65%** | 46.7% | 43.4% | 1.281 |
| Diffusion off, source only | on | off | off | absent | **23.84%** | 51.4% | 44.4% | 1.280 |
| Diffusion off, velocity only | off | on | off | absent | **101.89%** | 97.5% | 100.7% | 1.000 |
| Diffusion off, static seed | off | off | off | absent | **100.22%** | 102.1% | 100.6% | 1.000 |

The nearly closed top still permits interior FLD diffusion; `fld/cut_diff=true` removes both interior diffusion and the Marshak boundary flux while retaining local gas-radiation exchange. In this problem's history output, a nonzero formal `Frad_top` can still be reported in `cut_diff` runs even though it is not a transported flux; it is excluded from all energy conclusions here. The baseline top radiation flux at t=8 is 0.890 F★, versus approximately `3×10⁻⁷ F★` with α=10⁻⁸. All six runs reached the time limit, had zero opacity-domain/clamp/nonfinite counters, and no Newton failure.

The adiabatic-residual mode is `δln P−Γ₁ δln ρ`, with local Γ₁ inferred from `c_s²ρ/P`. It helps distinguish a pressure/density wave from a thermodynamic perturbation but is an **Eulerian linear proxy, not specific entropy**. The optional entropy user output is NaN because this setup has no pgen entropy table; it is not used. The transition-layer mean temperature differs substantially among controls, so differences in retained fractional mode cannot be interpreted as independent additive energy-loss terms.

## Resolution gate and causal interpretation

The no-diffusion, no-source, zero-velocity control was repeated on XD M-debug **342912** at 32×32×192 cells (Δ=0.3125 R☉): temperature-mode retention was **100.23%**, versus 100.22% at the baseline resolution. With surface heating on and velocity noise off, the high-resolution XD M-debug **342915** run retained **23.92%**, versus 23.84% at the baseline resolution. Both XD jobs completed with exit 0, in 36 and 40 seconds respectively; the heavy HDF remains under `$WORK/wor122/entropy_hydro_fine/` and `$WORK/wor122/entropy_hydro_source_fine/`.

Thus the tested mode is not disappearing because the static seed is underresolved, and its heating-induced attenuation does not improve when cell width is halved. Surface heating alone reproduces nearly all of the **nonradiative** mode loss, while initial velocity noise alone does not. FLD interior transport and the open radiative top add further loss under the heated condition. The exact path by which spatially uniform shell heating weakens the seed (EOS-dependent heating, gas-radiation exchange, and induced gas motion) remains to be separated; these controls do not prove a unique microscopic mechanism or establish a self-sustaining convection solution.

The immediate preparation change to test is an outer thermal support that does not deposit the full luminosity inside the seeded 812.5–817.5 R☉ layer, followed by gradual release after a buoyant flux is measurable. Keep the closed lower gas boundary. Use a short release gate with the full-3D mass-advection-corrected enthalpy flux before another long transfer job.

Reproduction: local inputs `inputs/rsg/athinput.wor122_entropy_budget_*` and `athinput.wor122_entropy_hydro_*`; XD scripts `tst/nrfld_validation/wor122_entropy_hydro_*_xd.sh`; mode diagnostic `tools/rsg/analyze_rsg_entropy_mode.py`; compact JSON in `tst/nrfld_validation/wor122_entropy_budget_20261119/`; plot `doc/figures/wor122_entropy_budget_20261119.png`. The local raw runs remain under `/tmp/wor122_entropy_budget_*/` and `/tmp/wor122_entropy_hydro_*/`.
