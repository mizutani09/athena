# FLD performance measurements (WOR-20)

See the [2026-10-22 pilot results](wor20_performance_20261022.md) for actual
measurements, reproducibility commands and limitations of the first experiment.

Run from any directory with an absolute path to the runner. A fresh
`performance_run_<timestamp>_JST/` directory is created in the launch directory.
Source copies, builds and MPI runs are isolated; cases and repeats run sequentially.
The source copies include local uncommitted changes. Normal Athena runs do not
enable the new timers unless `<performance> enabled=true` is supplied.

```bash
python3 /home/kosuke/simulation/athena/tst/nrfld_validation/performance.py \
  --cases hd hd_fld --ranks 1 2 4 --compare-timer \
  --hdf5-path /usr/lib/x86_64-linux-gnu/hdf5/openmpi
```

`--hdf5-path` is machine-specific; omit it if the compiler already finds HDF5.
Requirements: Python numpy/h5py/matplotlib, C++11, Open MPI (`--bind-to` options),
and the parallel HDF5 C library. The MESA SDK's serial HDF5 is not suitable for
this MPI build. Builds use MPI, no OpenMP, double precision HDF5 and `-O3`.
Use `--build-jobs` to control compilation concurrency (1–8, default 4).

Default pilot: 32³ cells, 16³ MeshBlocks, 5 warm-up + 20 measured steps,
3 repeats, ranks 1/2/4. These short runs establish correctness of measurement;
they are not necessarily long enough for publication-quality timings.
Increase `--steps` until even the fastest run has a measured window of at least
10 seconds; keep the same steps across compared cases. `short_window` flags
this condition without confusing statistical quality with numerical failure.

Add any of `hd_gravity hd_fld_gravity mhd mhd_fld mhd_gravity mhd_fld_gravity`
to `--cases`. Each configuration has its own executable. Nonzero uniform B is
divergence-free; smooth gas density and radiation fields vary in all 3 directions.
Coupling and radiation force are enabled. Periodic MG gravity uses G=0.01 and
threshold 1e-10. FLD uses Newton threshold 1e-8, linear threshold 1e-10 and aborts
on failure. Requested dt=1e-4 is a user timestep upper bound: the runner rejects
a run if the actual CFL-limited dt is different. This is a controlled benchmark,
not a validated physical model of a star.

## Scaling and build reuse

```bash
# Longer strong scaling, retaining total grid and block size:
python3 /home/kosuke/simulation/athena/tst/nrfld_validation/performance.py \
  --build-root /absolute/path/to/previous/performance_run_... \
  --cases hd hd_fld --ranks 1 2 4 --resolution 64 --block 16 --steps 200

# Weak scaling: keep resolution^3 cells/rank, grow the periodic domain:
python3 /home/kosuke/simulation/athena/tst/nrfld_validation/performance.py \
  --build-root /absolute/path/to/previous/performance_run_... \
  --cases hd hd_fld --mode weak --ranks 1 2 4 --resolution 32 --block 16
```

Build reuse checks the binary hash and retains original build/source provenance.
It deliberately does **not** rebuild recent source edits. Omit `--build-root` to
test new code. Keep the original run directory available when reusing builds.
Change `--block` at fixed total resolution for a block-size comparison; the
runner requires evenly distributed blocks, with at least one block per rank.

The current WSL environment reports 4 cores / 8 logical CPUs. Use ranks 1/2/4
with default core binding for the main comparison. For a separate SMT experiment,
use `--ranks 4 8 --bind-to hwthread`; do not label it 8 physical cores.
`hardware.json` and MPI binding messages in each log record this distinction.
An application sharing the host can affect results even when the guest is idle.

## What is measured

`performance_rank<N>.json` records local elapsed time, exclusive main-loop phases,
and the cycle, dt, active global cells, Newton count/residual/reason for each step.
One MPI barrier starts the measurement window after warm-up. No barriers or
reductions are added between phases. Initial setup and final snapshots lie
outside the timed window; launcher/end-to-end time is recorded separately.

- `explicit_integrator`: HD or MHD stage task list, including radiation transport,
  coupled Riemann work and gravity source work where enabled.
- `gravity_solve`: Poisson solves, accumulated over all explicit stages.
- `fld_prepare`, `fld_solve`, `fld_sync`, `fld_finish`: opacity preparation,
  Newton solve, hydro primitive synchronization, final opacity/diagnostic work.
- `amr`, `output`, `other`: remaining framework costs. The pilot is uniform-grid.

Implicit FLD fraction uses rank-mean implicit time divided by rank-mean window
time. It is **not the total radiation cost**: shared explicit work remains in
`explicit_integrator`. Task-level attribution, linear MG/retry counters and
the split of radiation from shared flux/source kernels remain follow-up work.
The timer itself currently supports coarse main-loop phases only.

Overall throughput uses maximum-rank elapsed time and active leaf-cell updates.
Never sum the separate rank maxima to construct a stacked wall-time budget.
The stacked plot uses rank means; small unassigned gaps at loop boundaries are
recorded separately. Scaling efficiency is relative to the smallest measured
rank count per case, and the baseline rank count is written to CSV.

## Validation and artifacts

The runner requires finite output, positive rho/pressure/radiation energy,
the expected final time and measurement cycles, strict Newton convergence and
nontrivial Newton work. History mass drift is checked at 1e-5 (history output
has limited decimal precision). This does not establish energy-conservation
accuracy or an analytic error bound; those belong to WOR-9.

`--compare-timer` repeats **every** run with timing disabled and records bitwise
equality of final HDF5 field arrays, plus Newton statuses and iteration counts.
The acceptance limit is 32 double-precision epsilons on each field's maximum
difference divided by max(1, max_abs_on, max_abs_off). This admits observed
one-ulp differences in parallel MHD magnetic-field output, but is much stricter
than the nonlinear solver tolerance. Newton statuses/counts must match exactly.
The recorded on/off end-to-end ratio includes launch and I/O and is not an
isolated timer-overhead estimate; short-run ratios are noisy.

- `manifest.json`: source/build hashes, provenance, commands and per-run status.
- `summary.json`, `summary.csv`: metrics, repeat ranges, scaling efficiency.
- `timing_breakdown.png`, `scaling.png`: diagnostic figures.
- Each case/rank/repeat directory: materialized input, log, raw rank timings,
  initial/final HDF5 snapshots, history, optional `timer_off/` control.
- Nonzero exit on any failed run. Failed measurements retain their logs and
  manifest entries; they are excluded from timing figures.

Future stages from [WOR-20](https://linear.app/work-on-fld/issue/WOR-20): detailed
FLD task/MG attribution, robust timer overhead measurement, stiffness/opacity
sweeps, fixed SMR then dynamic AMR, and multi-node scaling. Do not infer those
properties from this uniform, single-node pilot.
