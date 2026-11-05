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
