# WOR-124: rotating and magnetized table-EOS collapse pilot

This experimental branch starts at `protostellar-firstcore-v1`
(`6e658312bb504a90b2e385604193213c2bb07702`), whose solver source tree
was used for the validated nonrotating, nonmagnetic first-core run. The
existing worktrees and `main` are not modified.

## Scope of the code change

The frozen source has the general-EOS MHD energy bug documented in WOR-112:
its FLD exchange read `E_total-E_kinetic` as gas energy and wrote back
`E_gas+E_kinetic`, dropping `B²/2`. We copied the minimal `MagneticEnergyAt`
logic from [WOR-112 commit c79e78a8](https://github.com/mizutani09/athena/commit/c79e78a8e92cbd6eaefe7bed460fd5511b11aed0)
into `src/fld/fld.cpp`. The problem generator's table-EOS+B guard remains
closed unless `problem/enable_tabular_mhd_pilot=true` is explicitly set.
This is an experimental ideal-MHD calculation, not a non-ideal MHD model.

The physical initial sphere matches the prior table run: 1 Msun, 3000 au,
14 K. Optional initial fields are solid-body `Omega=1.48e-13 s^-1` and
uniform vertical `B0=129 microG`, values used in [Mayer et al. (2025),
Sec. 6.2](https://arxiv.org/html/2510.12620v4). The resulting rotation
energy ratio is beta=0.004973. Our EOS, opacity, Cartesian mesh, and ideal
MHD differ from Mayer's non-ideal RMHD calculation.

## Local 2-rank, one-cycle gate

All cases use the same 32³ root, adaptive `numlevel=2`, CFL 0.3,
table EOS, Semenov table opacity, and original FGMRES 40/300 with MG2.
The input is `input_table_mhd_pilot.in` with only the explicit pilot key
added to the frozen input. Command/environment, table hashes, full logs,
resource logs, restart files, HDF5, and per-case JSON live under
`/home/kosuke/simulation/test/collapse_rotmhd_pilot_v2_20261116/`.

| Case | One-cycle result | Wall s | Initial dt, code units | Max FGMRES iterations | Max true relative residual | Bmax G | divBmax |
|---|---|---:|---:|---:|---:|---:|---:|
| Neither | PASS | 68.8 | 31.1148 | 207 | 5.73e-8 | 0 | 0 |
| Rotation | PASS | 83.1 | 31.1148 | 236 | 4.18e-8 | 0 | 0 |
| Magnetic field | PASS | 71.9 | 10.5395 | 118 | 1.91e-8 | 1.310e-4 | 2.96e-18 |
| Both | PASS | 127.2 | 10.5395 | 186 | 1.77e-8 | 1.311e-4 | 3.24e-18 |

The output history has identical gas mass (to the printed precision) for all
four cases. In the magnetic-only and combined HDF outputs, all 61,440 active
cells have positive gas energy. The maximum relative closure error in
`E_total=E_gas+E_kinetic+E_magnetic` is `2.56e-6` and `2.57e-6`, respectively,
at HDF single precision. Without subtracting magnetic energy, the median
error is about 17.3 times the local gas energy because the dilute ambient is
magnetically dominated. The local binary SHA256 was
`bf9c68b7fac5381bb3c36e380affaf1d9070a8bc72cfe412a822358694ec6f8b`.
These one-cycle wall times are a *pilot*, not full-collapse throughput;
the magnetic field already shortens the initial timestep by 2.95×.

## Current gate

The combined case at cycle 1 reaches a second-cycle FGMRES true relative
residual of about `5.95e-6` after the original 300 iterations and aborts.
This occurs in both continuous c0→c2 and c1-checkpoint→c2 runs, so it is
not a restart-specific discrepancy. From the same c1 restart, the 80/600
MG3 settings previously used for long collapse reach c2: two active Newton
calls converge, three FGMRES solves have at most 295 iterations and true
relative residual `1.996e-8`; the one-cycle continuation took 183 s on the
local 2-rank host. This establishes a short restart gate, not yet sustained
collapse throughput. **No long magnetized collapse is validated by this
short gate.**

`collapse_rotmhd_pilot_20261116.py` repeats the local gate;
`collapse_rotmhd_energy_audit_20261116.py` checks HDF energy closure;
`xd_mtest_fourway.sbatch` is the next short, matched-condition XD job using
80/600 MG3 for all four variants.
