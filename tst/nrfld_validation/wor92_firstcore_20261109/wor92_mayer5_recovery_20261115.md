# Mayer ideal / constant opacity: cycle-830 restart recovery (2026-11-15)

User requested interruption and diagnosis after XD job 339331 stopped making
progress. The original 5-node/120-rank M-large-a run reached cycle 830 through
81 successful ten-cycle segments. At 2026-11-14 20:18 JST, segment 82 began
from c830 but emitted only six configuration lines, then no cycle, history,
or status output for about 21 hours. Slurm still reported RUNNING and CPU
accounting increased. It was **canceled at 2026-11-15 17:44 JST on user
request**, without altering any earlier successful output. Its c830 final
checkpoint was copied to `c830_job339331_preserved.rst`, byte-compared with
the original, and both have SHA-256
`8f3570e8acb768aa7d9ef732480e4972e20bcd19a4e79ebd9f82d6500c9f0be7`.
Metadata: cycle 830, code time 2450.3337515825274, 1968 leaf MeshBlocks.
Its latest HST rho_max was 8.51694258e-12 g/cm³; c0-to-c830 mass was constant
to the recorded 15 digits, minimum pressure 1.69137e-10 dyn/cm², and minimum
Jeans resolution 31.713 cells. The old job did not reach first core.

The final log stopped before `Setup complete`, during restart initialization
or the subsequent FLD boundary synchronization. It did not log a Newton,
FGMRES, pressure, or HDF failure. Exact subroutine and whether the cause is
an MPI communication race or a transient XD node/network condition remain
unproven. This is a failure to make numerical progress, not evidence of a
physical solver divergence.

## Controlled restart tests

All tests use the unchanged source/binary SHA-256
`9e7c2faa80832a7cd5d8aa3fd4689aed1648a7f346d4d492c2c0742217020b7f`,
the same preserved c830 checkpoint and all `ATHENA_*` controls as 339331.
They write to separate run directories and never overwrite the original.

| Job | Layout | Requested restart span | Status / result |
| --- | --- | --- | --- |
| 340112 | M-debug, one node/64 rank | 830→831 | COMPLETED 0:0 in 29 s; no initialization hang |
| 340114 | M-debug, one node/120 rank | 830→831 | COMPLETED 0:0 in 35 s; same rank count as failed job, no hang |
| 340115 | M-debug, one node/120 rank | 830→840 | COMPLETED 0:0 in 2:10; 2304 blocks, 15/15 Newton convergence, 34 FGMRES solves, max true relative residual 1.94538e-8; max gas-mass drift 4.44e-16; pressure min 5.93440e-10 dyn/cm² and Jeans min 31.957 cells; Slurm step MaxRSS 53,160,352 K |
| 340118 | M-debug, one node/120 rank | 840→850 | COMPLETED 0:0 in 2:07; 11/11 Newton converged, 30 FGMRES solves/470 iterations, max true relative residual 1.82786e-8; no fatal error, mass drift 4.44e-16, pressure min 5.91688e-10 dyn/cm², Jeans min 42.0193 cells, rho_max 1.00515e-11 g/cm³; Slurm step MaxRSS 40,510,008 K |
| 340113 | M-test-a, two nodes/120 rank | 830→831 | COMPLETED 0:0 in 29 s; cross-node restart startup passed |

The one-node c840 restart from 340115 has SHA-256
`f6d62a65e9cc7c69d237b4ca027339226590199e2b8eb91664141ce04e6909e8`.
It remains on the same c0→c830 physical history and uses the same binary,
input, solver settings, and AMR criteria. Repartitioning the restart changes
MPI layout but does not import an unrelated physical trajectory.

## Guarded continuation

M-large-a job **340116** was submitted from the validated c840 restart,
one node/120 MPI ranks, 110 G requested memory, same binary/input SHA and
same mass/pressure/Jeans gates as the original. Each ten-cycle segment has
an 1800 s Athena budget and an outer `timeout` (runtime + 120 s). A stalled
startup can therefore stop after at most about 32 minutes, leaving the last
completed restart for diagnosis rather than occupying an entire 24-hour job.
The 1-node/120-rank 2304-block test used 53.2 G Slurm step MaxRSS, leaving
headroom under the 110 G request. The M-large-a job was pending for priority
at 17:54 JST, with a tentative Slurm start 19:27 JST. This is a queue
forecast, not a guarantee of launch or first-core completion.

Remote case root:
`/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/`.
The original job 339331 and all diagnostic outputs are retained there.
Small logs, HST, status, and provenance for jobs 340115/340118 were copied to
`/home/kosuke/simulation/test/rad_nr/wor92_xd_20261107/firstcore_integration_evidence/`.

## c1194 stop and controlled replay (20:20 JST)

Job 340116 ran on `cna105` through 35 completed ten-cycle segments, from
c840 to c1190. Segment 35 then advanced from c1190 through c1194, but its
log and HST stopped at 19:33:30 JST, despite another 30 minutes of runtime.
The outer guard returned exit 124 (`FAIL_SEGMENT_TIMEOUT`), and Slurm marked
the batch job FAILED. The complete c1190 restart was preserved in the original
directory, with SHA-256
`cbf2e924b14f4a45c15af6bea988d1aa82b31647ca335ebe23aa6a52ca0ef6f6`.
The c1194 HST row has rho_max `4.269612649083216e-11 g/cm³`, pressure minimum
`5.449430854194125e-10 dyn/cm²`, and Jeans minimum `34.9456` cells. Over the
entire c840–1194 HST, the minimum Jeans value was `31.9913` and the maximum
relative mass change from c0 was `7.852e-15`. The last logged Newton solve
converged, and the three preceding FGMRES solves had true relative residuals
`5.24e-9`, `1.19e-8`, and `1.94e-8`. The log has no FATAL ERROR. This does
not establish which code phase stopped making progress.

Using the unchanged binary and settings, M-debug job 340278 restarted the
same c1190 checkpoint on another node (`cna208`) and reached c1196 with exit
0 in 82 s. The c1191–1194 HST rows agree **exactly in all 156 numeric fields**
with the preceding job. This disproves a deterministic solver failure at
c1195 for that state and binary; an intermittent MPI or node issue remains
possible. The replay had 7/7 converged Newton solves, 18 FGMRES solves with
maximum true relative residual `1.956862e-8`, rho_max
`4.295610660958805e-11 g/cm³`, pressure minimum `5.448676959945764e-10`,
Jeans minimum `34.9108`, and mass change at most `7.852e-15`. Its c1196 final
restart SHA-256 is
`aae6c65334b2b898f92b04451c4ed0c1e90caff3daf91c5ce6bdf3a3fc243bad`.

The c1196 restart seeded M-large-a continuation job 340281 on `cna078` at
20:18 JST. It excludes the previous node `cna105`, preserves separate output,
and keeps the ten-cycle mass/pressure/Jeans/first-core gates. The per-segment
time guard was reduced from 1800 to 900 s after a 6-cycle replay completed
in 82 s. This is a watchdog adjustment, not a numerical correction. The
first-core density gate has not been met yet.
