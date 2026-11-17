# WOR-124 rotation-only long-run review (XD job 341013)

Date: 2026-11-18 JST

## Outcome

XD M-large-a job `341013` reached the 24-hour wall limit.  This is not a
solver failure: its final MPI segment completed cycle 4390 and wrote the
opacity diagnostics and HDF output before Slurm terminated it.  The automatic
`final.rst` was interrupted during output and is truncated: it has
1,006,263,566 bytes instead of the expected 1,013,946,254 bytes and is
correctly rejected by the restart-layout guard.  The ordinary numbered
checkpoint `rotation_floor_amr11_2node.00444.rst` is complete at the same
cycle and time (`1987.5665222692214`), has 5146 MeshBlocks, and has SHA-256
`34268ab26806ec364816098e3ace7f7683bb967d7d527e64ac8dfbd1e305a992`.

The explicit first-core density candidate was not reached.  The final maximum
density is `2.11970065e-11 g cm^-3`, maximum gas temperature is 252.46 K,
and final/minimum-history Jeans resolutions are 27.61/27.03 cells, safely
above the 16-cell stop guard.  The run used 48 node hours.  Maximum reported
Slurm-step RSS was 40.03 GB.

## Solver and table audit

All 362 segment logs, including the wall-limited final segment, were parsed.

- All 3620 active Newton solves converged in at most four iterations.  The
  largest final maximum residual was `8.5296e-6`, below the configured
  `1e-5` criterion.
- 10,875 FGMRES solves used 123,321 iterations in total and at most 15 in one
  solve.  Their largest independently reported true relative residual was
  `1.99998485e-8`, below the strict `2e-8` target.  There were no solves above
  the target, no stagnation accepts, and no breakdowns.
- All 362 opacity diagnostics report zero out-of-domain, clamped, negative,
  zero, and nonfinite results.
- There are 8072 slow-MG warnings.  They occur after the MG defect has reached
  the very small floor; no FGMRES or Newton failure, residual growth, or
  iteration increase accompanies them.

## Independent final-HDF audit

The cycle-4390 HDF has 2,634,752 active cells.  A fresh table-EOS audit found
zero cells outside the EOS table and zero cells with nonpositive inferred gas
energy.  At the single-precision HDF accuracy, the relative
`Etot = Egas + Ekin` closure discrepancy has median `2.142e-7`, p99
`3.903e-5`, and maximum `9.039e-5`.

From the first unique history row after c770 through c4390, gas mass is
unchanged at printed precision and angular momentum changes by
`+2.598e-4` relative.  The density maximum is at approximately
`(-2.014, -3.479, -0.183) au`.  At about 1 au from that point, the spherical
volume mean is `rho=1.99e-11 g cm^-3`, `T=209 K`, and
`vr/sqrt(P/rho)=-0.013`; at 8.25 au the corresponding values are
`4.68e-12 g cm^-3`, 143 K, and -1.91.  This shows strongly slowed central
inflow but is not by itself a hydrostatic first-core proof.

## Interpretation and continuation

The rotation-only trajectory remains much slower than the weak-field run; it
is neither hung nor spending cycles on failed nonlinear solves.  A linear fit
to log maximum density over the last 100--500 cycles would place the
`1e-10 g cm^-3` crossing roughly 9,000--14,000 additional cycles away.  This
is only a planning extrapolation because rotational oscillations and later
acceleration can change the slope.  A further 24-hour job is therefore not
expected reliably to reach the density candidate.

The first restart probe exposed the truncated `final.rst`; it did not enter
the time-integration loop.  P-debug job `341788` then replayed c4390--4400
from the complete numbered checkpoint with the same source, CFL, AMR cap,
FGMRES and opacity settings.  It completed normally in 7:11 on one node and
64 MPI ranks.  All 10 active Newton solves converged in three iterations;
30 FGMRES solves used 358 iterations in total, at most 12 per solve, with
maximum true relative residual `1.96850e-8`, no stagnation or breakdown, and
no result above `2e-8`.  The c4400 HDF has zero EOS-outside cells, zero
nonpositive gas energies, and maximum single-precision energy-closure error
`9.011e-5`.  The normally completed gate wrote a new full-size `final.rst`
with SHA-256
`a3996f967f9b1d978286fc1fa3e1e733af9d770a76eccd4e21b51d3e55d5ef83`.

The HTML review, figures, complete history, solver summary, final HDF and
analysis script are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261118/final_rotation_job341013/`.
The immutable XD evidence is under
`/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/high_amr_long/rotation_floor_amr11_2node_job_341013/`.
The successful restart evidence is under
`/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/continuation/rotation_c4390to4400_numbered_restart_gate_job_341788/`.
