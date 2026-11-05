# WOR-67 XD2000 campaign, 2026-11-05

## Scope and provenance

The comparison target is the Mach 5 gas-temperature spike at `t = 0.04 s`.
Uniform `N2048`, one-level AMR with base `N1024`, and three-level SMR with
base `N512` all have a finest equivalent of `N2048` (`dx = 2.930 cm`). The
source tree is pinned to `codex/wor67-spike-convergence@9419d224`; its source
snapshot SHA-256 is
`5c2da0dce1e79795c14eb5d5c59b217148c6907f032543fad2eda4f76f38a16a`.
The full runs reuse the common pilot executable with SHA-256
`8dcf371fa64a99dd56c655e7934efc53b549ac58bc4924718b59cdbc3b45728f`.
The separate SMR 64-rank pilot rebuilt a binary with a different final-link
object order, so the full SMR run uses the common executable instead.

XD output root: `/work/mizutaniks/wor67/output`.
XD Slurm log root: `/work/mizutaniks/wor67/slurm`.
The local preparation branch is `codex/wor67-xd-20261105` in the separate
`/home/kosuke/simulation/athena-wor67` worktree.

The Slurm launchers use Intel MPI and parallel HDF5. `run_suite.py` invokes
`mpirun -np 1` for the binary configuration check inside a Slurm allocation;
launching that check directly failed with `PMI2_Job_GetId returned 14` in the
first, superseded pilot (job 335300). This change does not alter the solver.

## 100-cycle sizing runs

These runs ended by the requested `--nlim 100` limit. They establish resource
use and initial numerical behavior, not physical acceptance at `t = 0.04 s`.
All listed runs had strict Newton convergence without relaxed acceptance.

| Layout | Job | Ranks | Directory below output root | Solver time | Time reached | Blocks | Slurm step MaxRSS |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| uniform N2048 | 335302 | 8 | `wor67_hpc_uniform_n2048_pilot100b` | 39.461 s | 1.009101396e-4 s | 512 L0 | 1,883,660 K |
| AMR base N1024 | 335307 | 8 | `wor67_hpc_amr_base1024_pilot100` | 92.102 s | 1.008858037e-4 s | 254 L0, 4 L1 | — |
| AMR base N1024 | 335309 | 32 | `wor67_hpc_amr_base1024_pilot100_r32` | 72.824 s | 1.008858037e-4 s | 254 L0, 4 L1 | 3,061,160 K |
| SMR base N512 | 335308 | 8 | `wor67_hpc_smr_base512_pilot100` | 471.151 s | 1.009099544e-4 s | 221 L1, 70 L2 | 4,561,256 K |
| SMR base N512 | 335310 | 64 | `wor67_hpc_smr_base512_pilot100_r64` | 138.018 s | 1.009099544e-4 s | 221 L1, 70 L2 | 8,953,124 K |

The shock is covered by the finest level in the AMR and SMR pilots. AMR
performed 102 strict Newton solves during 100 coarse cycles because it
subcycles. Slurm step MaxRSS is an allocation-level proxy here, not a verified
per-rank peak. A further SMR 64-rank cadence test, job 335320, reused the
common executable and changed output2/3 to `dt = 0.002 s`, as in the full run.
Its 100 cycles took 132.28 s (step MaxRSS 8,709,356 K), versus 138.018 s
with every-cycle pilot output. The full SMR runtime estimate is roughly
15 hours, subject to late-time solver and timestep changes.

## Full runs and automatic audit

Status checked at 2026-11-05 12:02 JST:

| Layout | Job | Resources | Directory below output root | Status |
| --- | ---: | --- | --- | --- |
| uniform N2048 | 335306 | `M-bulk-a`, 1 node, 8 ranks, 32 GiB, 11 h | `wor67_hpc_uniform_n2048` | running; cycle 3497, time 0.0035595 s, strict NR convergence |
| AMR base N1024 | 335316 | `M-bulk-a`, 1 node, 32 ranks, 64 GiB, 11 h | `wor67_hpc_amr_base1024` | running; cycle 1118, time 0.0011173 s, strict NR convergence |
| SMR base N512 | 335321 | `M-large-a`, 1 node, 64 ranks, 96 GiB, 24 h | `wor67_hpc_smr_base512` | pending: `QOSMaxJobsPerUserLimit` |
| three-layout audit | 335322 | `M-test-a`, 1 rank, 8 GiB, 30 min | `wor67_hpc_comparison_n2048` | pending: `afterok:335306:335316:335321` dependency |

The full runs stop at `t = 0.04 s`; each runner writes `index.html`,
`summary.json`, `manifest.json`, `cases/mach5/input.used`, `run.log`, and
`resources.txt` under its run directory. After all three complete
successfully, job 335322 runs `wor67_spike_audit.py` and writes the combined
comparison to `wor67_hpc_comparison_n2048/index.html` and `summary.json`.
The audit compares peak error, amplitude recovery, aligned L1, shock position,
time phase, solver behavior, and saved diagnostic data using the same
definitions. The `N4096` series is gated on this `N2048` comparison and has
not been submitted.

No conclusion about the full-time spike or AMR/SMR equivalence is available
yet. Pilot outputs must not be treated as final physical measurements.

## 2026-11-05 15:58 JST update: uniform completed, analysis recovered

Uniform simulation job 335306 reached `t = 0.04 s` after 36,722 cycles,
with 36,722 strict Newton convergences and zero relaxed acceptances. Its
Slurm batch step returned exit 1 only because the optional MP4 writer could
not find `ffmpeg` after the solve and most analysis had completed. The saved
final output SHA-256 is
`3be289220f77cd037a843106bcb2c3897f32d2351d2855330c6a0be5d140e268`.
The runner now skips MP4 creation if `ffmpeg` is unavailable and omits the
video element from HTML. Recovery job 335514 ran `--postprocess` over the
existing outputs from a separate analysis worktree at `dad399ee`, without
rerunning the solver; it completed in 15 seconds. The source tree used for
analysis has the same `src` tree as `9419d224`.

The recovered uniform `summary.json` is `passed`: numerical peak 991.437 K,
semi-analytic point peak 1077.101 K, relative peak error 7.9532%, and spike
amplitude recovery 61.3047%. Aligned L1 errors are 0.06860% (rho), 0.09155%
(Tgas), and 0.07589% (Trad). Raw shock position error divided by domain width
is 3.0599%, and previous aligned snapshot change is 0.02837%. All are within
the issue's gates. The result reproduces the prior local uniform N2048 peak.
The summary retains the original `ffmpeg` error as `recovered_analysis_error`.

AMR job 335316 was still running near `t = 0.03275 s` at 15:54 JST; SMR job
335321 was pending on `QOSMaxJobsPerUserLimit`. Recovery jobs 335515 and
335516 will run after those jobs finish. Because the original audit job 335322
had an unsatisfied dependency on the failed Slurm step 335306, it was canceled
and replaced with job 335517, dependent on successful recovery jobs
335514, 335515, and 335516. The physical AMR/SMR comparison remains pending.
