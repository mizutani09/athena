# WOR-20 current-source local pilot, 2026-11-07 JST

The present Athena++ source (`209e22398` in `src/`; benchmark runner from
clean `3952b4ff`) passed a short isolated HD versus HD+FLD smoke benchmark.
The runner's source-snapshot SHA-256 was
`5e9e063afd271373485ed1af2fe7646465c78daeda8843d9940ea9a128d94ddf`.
The HD binary SHA-256 was
`e16ebf564f5a7c24b66c170a7a9d3697c13067b6b4ccdc80d998437f0807c2df`;
the HD+FLD binary SHA-256 was
`9361857837e94ad40a4cc24e8131bde69c611ea0e9e0fae72c06342007bfa624`.
Builds and outputs are under
`/home/kosuke/simulation/test/rad_nr/wor20_pilot_20261107/performance_run_20261107_014707_807568_JST/`.

The case uses 32³ uniform cells, 16³ MeshBlocks, dt=1e-4, two warm-up
steps, five measured steps, one repeat, and 1/2 MPI ranks bound to cores.
The runner's strict numerical checks passed all four runs. The FLD Newton
average was 3.6 iterations per measured step on both rank counts.

| Case | Ranks | Seconds/step | Measured window (s) | Implicit FLD rank-mean share |
| --- | ---: | ---: | ---: | ---: |
| HD | 1 | 0.00743535 | 0.0372 | 0 |
| HD | 2 | 0.00391580 | 0.0196 | 0 |
| HD+FLD | 1 | 0.09301167 | 0.4651 | 0.7814 |
| HD+FLD | 2 | 0.06910231 | 0.3455 | 0.8435 |

All windows are far below the planned 10 s minimum, and the host is
shared. These times and apparent 1-to-2-rank efficiencies are smoke values,
not stable performance estimates. The FLD share covers the implicit solve
and nearby preparation/synchronization as classified by the timer;
radiation work inside the explicit integrator remains there. HD and HD+FLD
also evolve different physics. No timer-off comparison was requested in
this pilot.

Reproduce with the runner at `3952b4ff`:

```bash
python3 tst/nrfld_validation/performance.py \
  --cases hd hd_fld --ranks 1 2 --resolution 32 --block 16 \
  --steps 5 --warmup 2 --repeat 1 --build-jobs 2 --timeout 300 \
  --hdf5-path /usr/lib/x86_64-linux-gnu/hdf5/openmpi \
  --output-root /home/kosuke/simulation/test/rad_nr/wor20_pilot_20261107 \
  --bind-to core
```

The output directory contains `manifest.json`, `summary.json`,
`summary.csv`, `timing_breakdown.png`, `scaling.png`, exact inputs, logs,
isolated builds, and a source snapshot. The summary and figures are also
versioned beside this report in `wor20_pilot_results_20261107/`.

Next: fix an exclusive host or dedicated node, use at least 10-second
windows and three repeats, compare timer on/off, then compare MG and
FGMRES at matched physics and tolerances. WOR-20 remains In Progress.
