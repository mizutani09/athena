# WOR-92: background reset and early CFL sensitivity

The [matched-time CFL audit](wor92_matched_time_20261107.md) found large pressure and radiation-energy differences even though all Newton and FGMRES solves converged. This follow-up isolates the protostellar problem generator's **background temperature reset**. In the current input, `problem/fix_background_temperature=true`, `problem/background_fix_radiation=true`, and `problem/background_sink_after_implicit=false`. The source function `ApplyBackgroundHeatSink` overwrites the gas internal energy and radiation energy of tracer-selected background cells with their 14 K target (`T_core × background_pressure_ratio / rho_ambient_factor`) at every step. The assignment itself has no dt factor. It is an imposed reservoir/boundary model, so its timestep sensitivity matters for physical interpretation.

## Short same-state test

All runs restart the identical cycle-100 checkpoint (`t=2266.625011212`), use the same isolated 2-rank/120-block binary and table input, and stop at the exact CFL=0.3 first-step end time `t=2275.4309789960685`. CFL=0.3/0.15/0.075 take 1/2/4 steps. Only the listed background-reset options differ. Every Newton stage converges; opacity diagnostics show zero table-outside/clamped lookup.

| Background treatment | CFL=0.3 Erad (10^42 erg) | CFL=0.075 Erad | Relative change 0.075 vs 0.3 | Gas internal-energy change |
|---|---:|---:|---:|---:|
| Current: gas + radiation reset before implicit | 2.77235 | 1.91849 | **−30.80%** | −10.67% |
| Reset disabled | 3.49846 | 3.51758 | +0.55% | −3.80% |
| Gas reset only, radiation free | 3.47111 | 3.39279 | −2.26% | −4.48% |
| Gas + radiation reset after implicit | 2.21596 | 1.76120 | −20.52% | −12.24% |

The intermediate CFL=0.15 result for the current treatment is `2.26569e42 erg`; with the reset off it is `3.49957e42 erg`. The current treatment therefore creates a strong step-count dependence in a very short interval. Turning off the radiation reset removes most of this particular **radiation-energy** sensitivity, while an approximately 9% maximum-temperature difference persists with the reset entirely off. The background reservoir is a dominant contributor, not a complete explanation for the gas/transport sensitivity.

This is an isolation test, **not an adopted production change**. At CFL=0.3 the background-tracer gas/radiation temperature errors after the short step rise to roughly 20% with the reset off or gas-only, compared with roughly 13% under the current before-implicit treatment. Moving both resets after implicit gives final background errors at machine zero but retains a 20.5% radiation-energy CFL difference. Neither simple alternative simultaneously preserves the intended 14 K reservoir and demonstrates time convergence. A physically specified outer-boundary/reservoir treatment, or a dt-consistent relaxation with a justified timescale, must be tested before choosing a production configuration.

## One-step source ledgers

The existing opt-in mixed-frame explicit ledger was enabled for one step from the same checkpoint at CFL=0.3 and 0.075. Its 120 block reports show zero `radiation_floor_volume` in both runs and equal/opposite gas-radiation explicit source exchange to printed accuracy. The implicit energy ledger's maximum rank relative balance is `2.295e-6` at CFL=0.3 and `1.222e-7` at CFL=0.075. These diagnostics rule out explicit radiation clipping as the large short-step energy difference. They do not by themselves attribute every gas/radiation energy change among hydro transport, implicit diffusion/coupling, gravity, boundaries, and the background reset.

## Reproduction and evidence

Run `python3 tst/nrfld_validation/wor92_background_sink_20261107.py` to regenerate the [summary JSON](wor92_background_sink_results_20261107/summary.json), [case CSV](wor92_background_sink_results_20261107/cases.csv), [paired-difference CSV](wor92_background_sink_results_20261107/deltas.csv), [figure](wor92_background_sink_results_20261107/sink_controls.png), and [full logs/history archive](wor92_background_sink_results_20261107/evidence.tar.gz). JSON/CSV include the binary, input, checkpoint, each log/history/restart SHA-256. The binary SHA-256 is `0e6af159f3c545d765667ab45a3b7feaad7a3ddbb0c05e8897633822c86f5c1b`, built from source commits `6da81d20`, `ff632195`, and `14633c31` on `codex/wor92-domain-guard-trial-20261107`.

From `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, use the same base command and environment as the [matched-time report](wor92_matched_time_20261107.md), a fresh `-d` directory, `time/nlim=1000 time/tlim=2275.4309789960685`, and `time/cfl_number=0.3`, `0.15`, or `0.075`. For the four treatments, pass respectively:

```text
problem/fix_background_temperature=true problem/background_fix_radiation=true problem/background_sink_after_implicit=false
problem/fix_background_temperature=false
problem/fix_background_temperature=true problem/background_fix_radiation=false problem/background_sink_after_implicit=false
problem/fix_background_temperature=true problem/background_fix_radiation=true problem/background_sink_after_implicit=true
```

For one-step explicit ledgers, use `time/nlim=101 time/tlim=10000.0 fld/mixed_frame_ledger_limit=1`. For implicit ledgers, use `time/nlim=101 time/tlim=10000.0` with `ATHENA_NRFLD_ENERGY_LEDGER=1`. Both ledgers were run at CFL=0.3 and 0.075 in fresh directories. No XD2000 access or merge occurred.

WOR-92 remains **In Progress**. Next: quantify the reset's actual per-step energy removal, test a physically justified reservoir treatment, and repeat matched-time CFL and mesh comparisons before first-core claims.
