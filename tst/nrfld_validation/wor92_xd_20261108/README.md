# WOR-92 XD CFL=0.3 gate and P-large-a continuation

Date: 2026-11-08 JST. Isolated local worktree: `/home/kosuke/simulation/athena-wor92-xd-restart`, branch `codex/wor92-xd-restart-sync-20261107`. The source baseline is `873a1b4c3bda584b52e1cae2b41e333ddb2f793a`; the four-file restart/gravity change is commit `c586cee59531c5a9a68cbae485a638c2039556d6`. The XD source archive at `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/source_restart_gravity_trial` has matching SHA-256 for all four changed source files. Its `bin/athena` SHA-256 is `41f16d20537cef3615a41d1b2150338878d64bf9f9888dc80925edd6a299d72c`.

The input is `wor92_core_symmetry_cycle_output_gravity_20261107.in`, SHA-256 `95d9fca3409705006947ea26da2f9abd3e36a4907a1c2335cfde805502f8cbb6`. It adds only `gravity/reset_fmg_color_each_solve=true` and `gravity/repeat_initial_fmg_after_restart=true` to the handoff input. Tomida EoS table SHA-256 is `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`; Semenov opacity table SHA-256 is `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`. The initial P-debug runs use 32³ periodic root, 8³ MeshBlocks, AMR `numlevel=6`, CFL=0.3, the WOR-92 conservative/variable-limiter/coupled/regrid environment flags recorded in each provenance file, and a strict FGMRES true-relative-residual target of `2e-8`. The optional inexact cap is `1e-7`; none of the initial P-debug FGMRES solves exceeded the strict target.

## P-debug observations

| Job | MPI ranks / scope | Slurm state | Newton stage-0 | FGMRES solves, max iterations, max true residual | Elapsed / peak step MaxRSS | Numerical result |
| --- | --- | --- | --- | --- | --- | --- |
| 337321 | 32, fresh cycles 0–6 | FAILED (post-run `python3.11` absent on compute node) | 6/6 converged; iteration counts 6,3,3,3,3,7 | 25; 280; `1.9995549955e-8` | 9:10 / 9,096,856 K | PASS: solver completed cycle 6 and wrote valid final and cycle-5 checkpoints; batch verification failed only after solver exit |
| 337346 | 32, cycle-5 checkpoint to cycle 6 | COMPLETED | 1/1 converged; 3 iterations | 3; 223; `1.9665015372e-8` | 1:14 / 8,110,272 K | PASS |
| 337347 | 64, fresh cycle 0–1 | COMPLETED | 1/1 converged; 5 iterations | 5; 279; `1.9859448789e-8` | 1:02 / 10,961,972 K | PASS |
| 337350 | 64, two one-cycle segments with restart | COMPLETED | 2/2 converged; 5,3 iterations | 8; 279; `1.9859448789e-8` | 1:39 / 10,983,244 K peak | PASS: cycle 0→1→2, restart, history and opacity checks |

For job 337321 versus the restart job 337346, both final checkpoints have cycle 6, 960 MeshBlocks and time `87.57472964443517`. Their next time steps are `14.040120925181595` and `14.040120925581583` (difference `3.99988e-10`). Stage-0 Newton iterations differ (7 versus 3); both converge, and all FGMRES true residuals remain below `2e-8`. The FMG restart initial solve also reports a zero-residual stage-1 solve. This is a convergence and trajectory check, not bitwise restart equivalence.

Every initial P-debug numerical run reports `FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0`; no solver fatal error was found. The fresh six-cycle history gives gas mass `2.27300e33` initially and `2.27302e33` at cycle 6, radiation energy `1.68144e42` to `1.67610e42`, `rho_max=5.27320e-18 g/cm³`, `Tgas,max=13.9891 K`, and `jeans_min=60.5484`. These history values have limited printed precision and do not establish a high-accuracy conservation budget. The full logs, histories, provenance and summaries (excluding bulky checkpoint/HDF5 fields) are in `wor92_xd_debug_evidence_20261108.tar.gz`, SHA-256 `155f49c0a200a0b57b5255e69b20b368fe6b35744cd553ed9b3fa4b30be752a7`.

The compute nodes provide Python 3.6.8, so the batch scripts use `read_athena_restart_metadata_py36.py` (the repository reader without the Python 3.7-only future annotation import). The reader was checked on XD against a 960-MeshBlock cycle-5 checkpoint. The two-segment batch smoke verifies the same restart loop used in the long script.

## P-large-a job

Job **337351** was submitted to `P-large-a` on 2026-11-08 JST with 64 ranks, one shared node, 200 G memory and a 24-hour limit. At submission it was `PENDING (QOSMaxJobsPerUserLimit)` because a separate user job occupied the applicable queue allowance. It has no dependency on that job and should start when the allowance clears. The exact submitted script is `wor92_plarge_cap6_gravity.sbatch`, SHA-256 `56abdd6521db23bc29e39d2cf59e0833e0c53d3d017c5c05758554e4ee3cce9a`.

The run advances in 10-cycle segments with checkpoint/restart, CFL=0.3, and upper time bound 10000 code units. It records cycle, simulation time, Jeans minimum, density/temperature maxima, gas mass, radiation energy, level, exit status and reason in `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/job_337351/progress.csv`. It stops on `jeans_min<16`, opacity-domain diagnostic counts, `rho_max≥1e-7 g/cm³`, `Tgas,max≥9900 K`, the time bound, solver failure, or the wall-time reserve. Each segment retains its full log and final restart. The long calculation ended after its second segment failed; the details and validated continuation are below. First-core formation, late AMR/domain behavior, matched-time resolution/CFL comparison, and a precise conservation audit remain open. WOR-92 remains In Progress.

## Long-run failure and validated continuation (2026-11-08 JST)

Slurm job **337351** was `FAILED` (`1:0`) after 5:57; peak step MaxRSS was 11,038,632 K. Segment 0 advanced cycles 0–10 with Newton 10/10 and FGMRES 33/33 below the strict `2e-8` target (max 279 iterations, max true relative residual `1.9859448789e-8`). Segment 1 restarted from the valid cycle-10 checkpoint, advanced through cycle 12, then its cycle-13 FGMRES reached the configured 300 iterations at true relative residual `1.2641885398e-6` and aborted. The last saved checkpoint is cycle 10, `t=142.02505510886945`, `dt=13.030432166917308`, 960 MeshBlocks, SHA-256 `2a5d093fd9a0f84244169b2fdfea0a3c5aa18e04dc2d630d3f87f08c153f1708`. The history also contains time `180.712` after cycle 12; it is not a saved restart state. The original FGMRES limit was insufficient at this state; the underlying cause of slower convergence has not been established.

Two independent `P-debug` runs used the same source binary, input physics and strict `2e-8` residual target, changing only `nrfld/fgmres_max_iterations` from 300 to 600:

| Job | Restart and endpoint | Slurm / elapsed / MaxRSS | Newton | FGMRES | Result |
| --- | --- | --- | --- | --- | --- |
| 337766 | cycle 10 → 14 | COMPLETED / 1:27 / 11,014,832 K | 4/4 converged | 9/9 converged; max 480 iterations, max true residual `1.9975934482e-8` | PASS; the previously failing solve converged at 354 iterations |
| 337771 | cycle 14 → 20 | COMPLETED / 1:07 / 11,505,240 K | 6/6 converged | 14/14 converged; max 139 iterations, max true residual `1.9724120251e-8` | PASS |

Both runs report zero opacity out-of-domain, clamped, negative, zero or nonfinite results. The cycle-20 checkpoint is at `t=268.26820770391913`, `dt=12.350696652560487`, 960 MeshBlocks, SHA-256 `a282642722d65ac4d35601c67af7500aa02bdef6ddd7b60fb9642c559b8e7bf6`. Its history gives `rho_max=5.45461e-18 g/cm³`, `Tgas,max=13.986 K`, `jeans_min=59.5267`. This remains far below the `1e-10 g/cm³` first-core comparison density. The limited history precision does not establish a high-accuracy conservation budget.

A new 64-rank, 24-hour `P-large-a` continuation, **job 337772**, was submitted from the validated cycle-20 checkpoint. Its script `wor92_plarge_cap6_fgmres600_continuation.sbatch` has SHA-256 `e2b569af5c71f0596e84e438f33ea2c883d3c00a212e4056f59737d133d3e2e1`. It retains the strict residual target, uses the 600 iteration cap, and writes restart checkpoints every five cycles. At submission it was `PENDING (QOSMaxJobsPerUserLimit)`; an unrelated P-large-a job was running. Its output will be under `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/job_337772/`.

The source scripts, self-contained [HTML summary](report.html), original long-run evidence (`wor92_xd_long337351_evidence.tar.gz`, SHA-256 `82547d6d8b76e1d7419adc9f3cac9551fb992f99f4142236b0480418da28b0da`), and two P-debug retest logs (`wor92_xd_fgmres600_debug_evidence.tar.gz`, SHA-256 `29fb3bbb6bf2c251c942130551e9ad79add0a52f336314ba116cab4430110f2c`) are grouped in this directory. The earlier Mayer5 benchmark HTML belongs to a separate baseline experiment and remains in its original location.
