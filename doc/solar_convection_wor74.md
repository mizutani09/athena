# WOR-74: inflow-entropy sensitivity and feedback preparation

The non-magnetic 6.144 × 6.144 × 3.072 Mm, 96 km solar-box run reached
`t=5000`. In its last `t=4150–5000` window, the mean top flux was
`0.6257 F_sun`, the mean bottom hydro flux was `0.6236 F_sun`, and the
stored-energy trend was `+0.0005 F_sun`. The measured interval is nearly
balanced at a reduced flux. Its late state is the common restart for WOR-74.

## Common restart and controlled variables

- Restart: `/mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot/face_gravity_hd2_mass_tau2_rate40_current_64x64x32_np4_t4255_5000_resume/simple_convection_general_6Mm_box.final.rst`
- Input: `inputs/radiation/athinput.simple_convection_lhllc_fld_general_6Mm_box`
- EOS, opacity, and 1D profile: the three files linked under
  `/mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot/data`.
- Baseline effective HD2 upflow entropy:
  `1.581097454457282e9 erg g^-1 K^-1`. The matched variants set
  `s_in_from_profile=false` and add `-1e6`, `+1e6`, or `+2e6` in the same
  units. The baseline keeps `s_in_from_profile=true`. This starting value is
  the tabulated-EOS entropy of the imported 1D profile at the first bottom
  cell center. It has **not** been calibrated against the measured 3D flux.
  The offsets are small sensitivity probes (±`1e6` is ±0.063% of the
  baseline), not inferred solar-flux solutions.
- All cases retain CFL `0.3`, `64×64×32`, HD2 mass relaxation time `2`,
  mass-rate gain `40`, the same binary and source checkpoint. The prepared
  plan checks the binary, input, and external-data hashes against the parent
  run before producing commands.

The first prepared interval ends at `t=5500`. It is a *fixed-entropy
sensitivity sweep* measuring the response to small changes and checking
solver stability. It is not an entropy-feedback run. **Do not infer a
calibrated entropy from a transient upper flux.** The full-box internal-energy divided
by `F_sun` corresponds to about `2×10^4` code-time units. Continue promising
branches from their own checkpoints until at least two late windows, each at
least one vertical crossing time long, show stable flux and energy storage.
Then repeat the most promising values at 48 km and, if possible, higher HPC
resolution. WOR-74's acceptance rules also require top flux
`0.9–1.1 F_sun`, inter-window difference `≤0.05 F_sun`, energy-storage
trend `≤0.1 F_sun` in magnitude, mass drift `≤1%`, and all NR solves
converged. Compare optical-depth-aligned thermodynamics, mean profiles, and
bottom hydro and mass fluxes as well as the top flux.

## Relation to the MURaM flux feedback

[Rempel (2014)](https://arxiv.org/pdf/1405.6814), §2.2, states that the entropy of lower-boundary upflows is
chosen to give a photospheric energy flux within a few percent of solar. The
paper does not give an entropy update equation. The earlier MURaM method
paper, [Vögler et al. (2005)](https://www.aanda.org/articles/aa/pdf/2005/01/aa1507.pdf), §2.3.2, **does** prescribe a feedback: it
updates the *specific internal energy* of inflows from the upper radiative
flux error on the box Kelvin–Helmholtz time, rather than holding that inflow
energy fixed. Its equation (20) is

`epsilon_in <- epsilon_in * [1 + (dt/tau_KH)*(F_sun-F_top)/F_sun]`,

with `tau_KH = E_internal/(A*F_sun)`. The present Athena++ HD2 boundary
holds `s_in` fixed and has no corresponding luminosity feedback. Its existing
feedback changes the lower pressure to control *mass*, not energy flux.

Before using the fixed-sensitivity results as a final calibration, implement
and verify a slow luminosity feedback for this tabulated-EOS entropy
boundary. The MURaM formula must not be applied by multiplying `s_in`:
specific entropy and specific internal energy have different thermodynamic
meanings. Convert through the EOS or use a measured entropy-to-flux response
with a conservative update timescale, preserve the controller state on
restart, and compare against an identical fixed-entropy control. The upper
flux lag and the `~2e4`-code-time box heat time require a long-window check
of both the top flux and stored energy. The fixed `t=5000–5500` sweep is the
response-measurement stage for that controller, not the feedback itself.

### Bounded local feedback pilot

`tst/nrfld_validation/run_solar_entropy_feedback.py` provides a restart-based
outer-loop test. From the same `t=5000` state it runs both a fixed-entropy
control and a feedback branch in four matched 25-code-time segments through
`t=5100`. The same restart schedule isolates entropy changes. After each
non-final segment it averages the top radiative flux over the last 20
code-time units and applies

`delta_s = (segment_duration/tau_KH)*(1-<F_top>/F_sun)*(1e6 / response)`,

where `tau_KH=2e4` code time and `response=0.322326259 F_sun` per `1e6`
entropy units is the **measured short-time bottom-flux response** from the
`+1e6` smoke comparison. This converts the slow flux error into entropy
units; it is a provisional controller gain, not a measured long-time
photospheric derivative. The absolute update is capped at
`5e4 erg g^-1 K^-1` per segment. Both branches use 16 MPI ranks and keep
only HST and restart outputs. Every two seconds the driver records WSL
available memory, swap use, and summed Athena rank RSS. It stops the
current process group if available memory falls below 3 GiB, used swap rises
above 4 GiB, or total wall time reaches 27 minutes.

The exact local command is:

```bash
python3 tst/nrfld_validation/run_solar_entropy_feedback.py \
  --restart /mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot/face_gravity_hd2_mass_tau2_rate40_current_64x64x32_np4_t4255_5000_resume/simple_convection_general_6Mm_box.final.rst \
  --header-from /mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot/face_gravity_hd2_initial_flux_64x64x32_np4_t10/simple_convection_general_6Mm_box.hst \
  --base-input inputs/radiation/athinput.simple_convection_lhllc_fld_general_6Mm_box \
  --data-dir /mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot/data \
  --output-root /mnt/d/output/rad/wor74_feedback_20261105/feedback_t5000_5100 \
  --initial-s-in 1.581097454457282e9
```

The driver writes a summary HTML, JSON, flux and entropy plot, and a
`memory_monitor.csv`. This checks the controller and restart continuity;
100 code-time units are much shorter than the thermal relaxation time.
An initial matched two-segment `t=5000–5004` smoke test completed. The first
fixed and feedback segments were identical in top and bottom flux. After a
positive `115.84 erg g^-1 K^-1` entropy update, the second segment's bottom
hydro flux differed by at most `3.9e-5 F_sun`; the top differed by at most
`5.1e-12 F_sun`, as expected on such a short interval. Peak summed Athena
rank RSS was `1.09 GiB`, minimum WSL available memory was `17.95 GiB`, and
swap use stayed near `0.20 GiB`. The smoke report is
`/mnt/d/output/rad/wor74_feedback_20261105/controller_smoke_matched_t5000_5004/index.html`.

The full 16-rank `t=5000–5100` matched pilot also completed in **21.96 wall
minutes**, within the 27-minute guard. Its report and raw summary are
`/mnt/d/output/rad/wor74_feedback_20261105/feedback_t5000_5100/index.html`
and the adjacent `summary.json`. The first 25-code-time HST files in the
fixed and feedback branches have identical SHA256 hashes. The controller
applied three positive entropy increments of `1445.40`, `1470.94`, and
`1426.57 erg g^-1 K^-1` at `t=5025`, `5050`, and `5075`; the total applied
change was `4342.91 erg g^-1 K^-1`, or `2.75e-6` of the initial entropy.
The late `t=5075–5100` means were:

| Branch | Top radiation / `F_sun` | Bottom hydro / `F_sun` |
| --- | ---: | ---: |
| Fixed entropy | 0.627433335 | 0.556569377 |
| Slow feedback | 0.627433053 | 0.557982875 |

Thus the bottom hydro-flux difference was `+0.00141350 F_sun`, whereas the
top radiative-flux difference was only `-2.82e-7 F_sun`. The feedback has
started to act at the lower boundary; this short test **does not establish
solar-flux calibration or statistical stationarity**. The box heat time is
about `2e4` code units, two orders of magnitude longer than this pilot, and
the last 25-code-time window spans only about 0.22 vertical crossing times.
Both branches reported `1445/1445` converged Newton solves, no slow-MG
warnings or fatal errors, and maximum segment mass drift below `0.028%`.
During the run the peak summed Athena rank RSS was `1123 MiB` (`1.10 GiB`),
minimum WSL available memory was `18374 MiB` (`17.94 GiB`), and peak used
swap was `203 MiB`; swap did not grow. The 657 monitor records are in
`memory_monitor.csv` next to the report. A longer thermal-time test should
check the top-flux response, stored-energy trend, and controller gain before
calling the entropy calibrated.

### Long feedback trajectory started from t=0

The separate `tst/nrfld_validation/run_solar_entropy_feedback_long.py`
driver starts the 96 km box from the same 1D profile as the fixed baseline,
but sets the known profile-derived entropy explicitly at `t=0` and applies
the slow feedback between 100-code-time restart segments. It keeps the
HD2 mass-rate gain `40`, relaxation time `2`, CFL `0.3`, and 16 MPI ranks.
The 96 km short-time bottom response `0.322326259 F_sun` per `1e6` entropy
units is a provisional gain. The top radiative flux is averaged over each
completed segment; the update uses `tau_KH=2e4` code-time units and a
`5e4 erg g^-1 K^-1` per-segment cap. This is an exploratory long transient,
not a calibrated solar solution. The earlier fixed run changed mass-control
settings during startup, so it is not an exactly matched control trajectory.

The new campaign root is
`/mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/`.
The first `t=0–100` segment completed with `1309/1309` converged Newton
solves, zero slow-MG warnings, and a `0.0459%` maximum late-window mass
drift. Its simulation wall time was `561.39 s`; the driver spent `571.19 s`
including plots and analysis. The segment-mean top radiation and bottom
hydro fluxes were `0.488587` and `1.135891 F_sun`. The next entropy is
`1.581105387610293e9 erg g^-1 K^-1`, an increment of `7933.153`.
These are startup transients, not steady calibration measurements. Peak
summed Athena rank RSS was `1.108 GiB`, minimum WSL available memory
`18.030 GiB`, and peak used swap `0.284 GiB`. The driver stops if WSL
available memory falls below `3 GiB`, used swap exceeds `4 GiB`, or disk
free space falls below `50 GiB`.

The t=100 checkpoint passed its SHA256 and report checks. The same branch
was resumed in the background toward `t=5000`. The driver writes an atomic
`state.json`, a large-font `index.html` with a progress plot and case links,
`memory_monitor.csv` every five seconds, and a `driver.log`. A completed
segment can be resumed with the same arguments plus `--resume`; the target
may be increased to `t=20000` later without repeating completed segments.
If an interrupted case directory already exists, the next attempt uses a
new retry name and restarts from the last verified completed checkpoint.
For live checks:

```bash
tail -n 3 /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/driver.log
tail -n 3 /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/memory_monitor.csv
python3 -m json.tool /mnt/d/output/rad/wor74_feedback_20261105/from_t0_96km_t5000/state.json | head -40
```

The first segment's total wall time is `9.52 min / 100 code-time`; the
`t=5000–5100` pilot needed about `10.8 min / 100 code-time` for simulation
without slices. With current sparse plots and possible throughput changes,
allow roughly **8–12 wall hours to t=5000**, and **32–50 wall hours to
t≈20000** (the approximate whole-box thermal time). These are resource
estimates, not predictions of when the top flux becomes stationary or solar.

## Local CPU test

WSL currently exposes 16 logical CPUs arranged as eight pairs of sibling
threads, about 19 GiB RAM, and 64 GiB swap. `nproc` prints 1 in this shell
because `OMP_NUM_THREADS=1`; `nproc --all` and CPU affinity show 16 usable
logical CPUs. The old four-rank continuation was restricted to logical CPUs
4–7, which correspond to two physical cores. It was a scheduling choice, not
a four-core machine limit.

An identical `t=5000–5004` restart, with slices and checkpoints disabled,
gave these *simulation* wall times. The four- and eight-rank trials mapped
one rank per core; 12 and 16 mapped by hardware thread. All completed, with
the same final top flux to 12 decimal places of `F_sun`, no solver fatal, and
the same final mass to the printed precision.

| MPI ranks | Wall time [s] | Speedup vs four ranks |
| ---: | ---: | ---: |
| 4 | 40.18 | 1.00 |
| 8 | 31.30 | 1.28 |
| 12 | 29.55 | 1.36 |
| 16 | 22.56 | 1.78 |

These are short, output-light trials; check throughput again during the
full-output campaign. Each trial report and manifest is under
`/mnt/d/output/rad/wor74_entropy_calibration_20261105/rank_benchmark_np*_t5000_5004/`.

## Prepared campaign

`tst/nrfld_validation/prepare_solar_entropy_calibration.py` writes a
hash-checked `plan.json` and executable `run_all.sh`. The current long-run
plan is in
`/mnt/d/output/rad/wor74_entropy_calibration_20261105/calibration_t5000_5500/`.
Its script runs the common baseline and three entropy variants sequentially
on 16 ranks, then calls `solar_convection_entropy_sweep.py` to generate a
comparison HTML. Every case also receives its own report HTML. Checkpoints
are saved every 10 code-time units. The plan itself does not start the long
calculation; start it with:

```bash
bash /mnt/d/output/rad/wor74_entropy_calibration_20261105/calibration_t5000_5500/run_all.sh
```

The short two-case `t=5000–5008` smoke plan is under
`/mnt/d/output/rad/wor74_entropy_calibration_20261105/smoke_t5000_5008/`.
Both 16-rank cases completed, with 113/113 reported Newton solves converged
per case, zero fatals, and their own reports. The comparison HTML is in its
`comparison/` directory. Raising `s_in` by `1e6` increased the late bottom
hydro flux by `0.322 F_sun` and the differential stored-energy rate by
`0.295 F_sun`, while the late top-flux change was only
`-0.0000018 F_sun`. This checks the boundary response and illustrates the
thermal lag; it is not a steady calibration. The short-run analyzer now skips
the time-averaged 3D profile only when fewer than two volume snapshots exist.
