# WOR-106: coupled AMR restart consistency

## Cause and fix

The coupled NR-FLD solve updates active hydro conserved values and radiation
energy after the explicit hydro stages. The existing end-of-step boundary
sync sent hydro primitives, but AMR prolongation internally used the coarse
**conserved** buffer, which had not been refreshed after the implicit solve.
It could therefore leave a stale pressure ghost at the coarse/fine face.
Restart initialization rebuilt that buffer from current active conserved
values, changing the next explicit hydro flux. Radiation ghosts were also
stale until the next boundary exchange. In the original static Test C probe,
the immediate restart ghost differences were as large as 0.183 in pressure
and 0.947 in `E_rad`; the same-time next-step `E_rad` difference was 0.00792.

`SynchronizeFLDBoundariesAfterNR` now exchanges the updated hydro
conserved values and radiation energy before exchanging hydro primitives and
prolongating both fields. Restart initialization uses the same final boundary
sync. Separately, a restart file stores the **next** timestep, because
checkpoint output occurs after `NewTimeStep()`. `Mesh::Initialize()` was
calling `NewTimeStep()` again and could grow that saved timestep a second
time. It now rechecks timestep limits but caps the result by the saved value.
This still permits a more restrictive timestep from changed settings or an
earlier `tlim`.

## End-to-end regression

Build with `python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld
-mpi -hdf5`, then run:

```sh
python3 tst/nrfld_validation/wor106_coupled_amr_restart.py \
  --binary bin/athena --output /tmp/wor106_restart_regression
```

The script starts coupled, variable-limiter Test C with density ratio 1.15,
opacity 0.004, and transverse amplitude 0.3, runs five cycles, restarts a
cycle-2 checkpoint, checks all nonzero Newton solves, and compares cycle-5
AMR topology, time, and HDF active fields. The limiter is fixed within each
Newton solve. Both the legacy and shared-face paths are covered.

| Path | AMR | Ranks before→after restart | Maximum field difference |
| --- | --- | ---: | ---: |
| Legacy | Static | 4→4 | 8.53e-14 |
| Legacy | Adaptive | 4→4 | 2.84e-14 |
| Shared face | Static | 4→4 | 5.68e-14 |
| Shared face | Adaptive | 4→4 | 2.84e-14 |
| Shared face | Adaptive | 1→4 | 2.84e-14 |

Each pair has identical cycle-5 time `0.006229967897188556`. The adaptive
tests actually derefine after cycle 2. Outputs and logs are in
`/tmp/wor106_restart_regression_20261106c`. A previous intermediate trial
that refreshed only hydro conserved ghosts reduced the same-time field
difference to `4.38e-8`; synchronizing radiation as well reduced it to
roundoff. A hydro-primitive-only restart sync did not fix the discrepancy.
