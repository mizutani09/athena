# WOR-124 second-core preparation, full-domain AMR gate

## State at 2026-11-17 19:40 JST

The original worktree has unrelated uncommitted changes. This experiment uses
the isolated branch `exp/wor124-second-core-amr-gate-20261117`, based on the
completed cutout audit commit `9db57282d1b83f612702bba8a53b7b3e9c1ea40f`.
No main-branch merge or source-code change is part of this gate.

The previous central cutout is not a valid second-core production setup:
its fluid, MHD, and FLD boundaries remained periodic, and the crop's gravity
excluded external mass. At cycle 2250, matched-cell radiation energy across
the ±750 au crop exceeded the full-domain value by 160.5%. The center itself
remained close, which supports a diagnostic use only.

## XD job progress

- Rotation-only high-AMR job `341013` was RUNNING on M-large-a, two nodes and
  128 MPI ranks. Its latest completed segment was cycle 4380, with 5146
  blocks, `rho_max=2.10745e-11 g cm^-3`, and minimum Jeans resolution 27.65
  cells. Its 24-hour limit ends at about 2026-11-17 22:12 JST. This job was
  left untouched.
- Full-domain five-cycle jobs `341626` (`mesh/numlevel=11` control) and
  `341627` (`mesh/numlevel=12` trial) were submitted to M-test-a from the
  same cycle-2250 weak-field plus rotation restart. Both use the previously
  validated XD binary, 64 MPI ranks, one node, a 110 GB memory request, and
  the same CFL=0.3 and NR-FLD settings. At submission, both were pending for
  `QOSMaxJobsPerUserLimit`. The exact job script is
  [`wor124_secondcore_amr_gate.sbatch`](wor124_secondcore_amr_gate.sbatch).
- The checkpoint had 4572 blocks, `rho_max=1.01188e-10 g cm^-3`,
  `Tgas_max=338.34 K`, and a 16.13-cell minimum Jeans resolution. The gate
  requires an intact cycle-2255 restart, no opacity-domain event, positive
  density and temperature, and at least 16 Jeans cells. Analyze the
  per-cycle iteration history, true residual, conservation, elapsed time,
  memory, AMR level, and HDF fields before longer continuation.

## Local table feasibility audit

The current material-only Tomida-family EOS table spans `1e-21` to
`1e-4 g cm^-3`, with temperatures about 5 to 100,000 K. The current
Semenov-based opacity table spans `1e-21` to `1e-7 g cm^-3` and about 5
to 9900 K; its source opacity itself is only valid down to `2e-18 g cm^-3`.
The initial second-core scale is much smaller than the current finest
cell, so neither table extension nor spatial refinement alone is sufficient.

The existing EOS generator produced a finite, positive, monotone *trial*
table through `1e-2 g cm^-3` with the same H/He model, at 107 density and
256 temperature grid points. Its common inverse-energy axis contracted from
`log10 x2=[8.41578,13.29615]` to `[8.41578,13.25155]`. At 30 overlapping
sample coordinates, the largest relative interpolation changes were
`4.67e-4` in `P/e`, `3.64e-4` in temperature, `1.46e-4` in entropy, and
`7.64e-3` in `nabla_ad`. The trial table and audit are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/secondcore_amr_gate_20261117/`.
This is only a generator audit. It is not a validated physical second-core
EOS/opacity package; the high-density gas opacity and smooth restart
transition still need independent work.

Tomida et al. (2013), Appendix 2, combined Semenov dust with Ferguson
low-temperature gas and Opacity Project gas tables, and explicitly noted
that the dense protostellar region lies outside the available opacity
coverage. They extrapolated the nearest high-density values there while
describing the uncertainty. Their Appendix 1 also warns that their ideal
chemical-equilibrium EOS becomes invalid above roughly `0.1 g cm^-3`.
Our `1e-2 g cm^-3` trial stays below that stated EOS limit, but a
source-backed gas-opacity bridge and sensitivity brackets remain necessary.
