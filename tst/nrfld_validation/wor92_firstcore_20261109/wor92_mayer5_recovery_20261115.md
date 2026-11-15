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
c840 to c1190. Segment 35 then advanced to the accepted c1194 state, but its
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
`5.24e-9`, `1.19e-8`, and `1.94e-8`. The log has no FATAL ERROR. Source
inspection shows that `cycle=1194` is printed at the start of that step. No
Newton entry follows, so the hang occurred after entering the c1194 step and
before the Newton log; the exact phase remains unidentified.

Using the unchanged binary and settings, M-debug job 340278 restarted the
same c1190 checkpoint on another node (`cna208`) and reached c1196 with exit
0 in 82 s. The c1191–1194 HST rows agree **exactly in all 156 numeric fields**
with the preceding job. This disproves a deterministic solver failure in the
c1194 step for that state and binary; an intermittent MPI or node issue remains
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

An additional M-test-a replay, job 340293, pinned the same c1190 restart and
120 ranks to the original `cna105`. It also reached c1196, exit 0, in 80 s.
The c1191–1196 HST records agree exactly with job 340278 in all 273 numeric
fields. Slurm step MaxRSS was 38,635,204 K. Thus the c1194-step hang is not
deterministically reproduced even on the same node. Both replays support
continuation from a validated checkpoint; neither proves whether the original
stall came from MPI progress, I/O, node load, or another intermittent cause.

## Intermediate 3D physical audit

The c1240 primitive HDF5 from job 340281 was copied to the local evidence
directory as `profiles/c1240.athdf`, SHA-256
`573ecd98c95ef58d52fe75b9c032657aef093c728e657edfc4735cd6b871a61e`.
Independent leaf-cell reduction gives maximum density
`4.9164e-11 g/cm³`, gas temperature at that cell `396.34 K`, and radial
velocity `-0.00674 km/s`. On the positive x-axis, the strongest density
drop from 3–15 au is at `7.5073–7.8735 au`, with inner/outer density
ratio `6.007` and inflow speeds `0.427/1.768 km/s`. The x-axis positive/
negative density difference over 3–15 au is at most `0.140%`. Mass inside
approximately 7.7 au is `0.0242 M☉`. These are strong signs of a supported
first-core structure and an accretion shock, but the prescribed
rho_max≥1e-10 benchmark has not yet been reached. Physical time is
`36.8273 kyr`, or `1.268` times the pressureless uniform-sphere free-fall
time; the comparison with Mayer's roughly 29.09 kyr result requires explicit
discussion of solver and discretization differences rather than an assertion
of agreement.

At the maximum-density cell, an independent calculation with the pgen's ideal
gas constant gives `Tgas=396.340968 K` and the HDF5 radiation energy gives
`Trad=396.340935 K`, a relative difference `8.37e-8`. The hot core gas and
radiation are therefore tightly coupled at this snapshot.
As a separate, approximate check, a finite-difference pressure gradient
from the spherical shell means divided by `rho G M(<r)/r²` is `0.79–1.09`
between about 1 and 6.5 au. Shell averaging and the discrete derivative
limit precision, but the result is consistent with pressure support rather
than a freely collapsing density spike.

The full c0→c1196 HST records from jobs 339330, 339331, 340115, 340116,
and 340278 contain 1321 saved 39-column rows at 1197 unique times. All
124 duplicate-time rows agree exactly, including across the five-node to
one-node restart. No accepted row violates mass drift `1e-8`, pressure
`1e-18 dyn/cm²`, or Jeans `16 cells`. The maximum relative mass change is
`7.8522e-15`, minimum pressure `1.69137e-10 dyn/cm²`, and minimum Jeans
resolution `31.7129 cells`. At Mayer's reported 29.09 kyr benchmark time,
this trajectory has rho_max only about `3.49e-16 g/cm³`, so the onset timing
is substantially later. The earlier ideal-EOS path reaching near 29.14 kyr
used ambient density/pressure ratios `0.01/0.01` and a pre-implicit sink,
whereas this path uses `1/30` density, pressure equilibrium, and a
post-implicit sink; those paths do not isolate a single cause of the timing
difference. The initial c0 maximum cell mass here is `7.2855e-6 M☉`, about
22 times Mayer's AREPO target `3.33e-7 M☉`, another material resolution
difference.

A second independent 3D output at c1340 (SHA-256
`9b75d4f8cee1c2469978bab4dd6eb450a864e9339e19587254e278913d1f2058`)
shows continuous evolution: rho_max `6.59093e-11 g/cm³`, central gas
temperature `453.79 K`, central radial velocity `-0.00723 km/s`, and an
accretion-shock candidate at `7.141–7.507 au` with density ratio `5.67` and
inflow speeds `0.259/1.681 km/s` on the inner/outer sides. The shock has
moved inward from c1240, and the mass inside its midpoint is about
`0.02696 M☉`. Positive/negative x-axis density asymmetry over 3–15 au is
at most `0.337%`.

## Cycle 1416 pressure-floor stop and AMR isolation

The guarded continuation job 340281 advanced c1196→c1416 in 37:59, then
stopped at the first bad history row. At c1415 the minimum pressure was
`5.225038548e-10 dyn/cm²` and minimum Jeans resolution `31.9967` cells.
At c1416 these became the hydro floor `1e-19 dyn/cm²` and `1.4856e-6`
cells; maximum density was `7.9638230e-11 g/cm³`, below the
`1e-10 g/cm³` endpoint. Mass remained `2.936555728050224e33 g`, within
`7.85e-15` relative of its initial value. The c1416 restart SHA-256 is
`ec50846dfdc0ff385e628785d76e1d9032a7b0ab4cd678ee7a24780d7f18e9a1`;
it is not a safe continuation point. Slurm FAILED/exit 2 records the
intentional watchdog stop, not a solver crash.

Independent HDF5 inspection of c1416 (M-debug snapshot job 340317,
`profiles/c1416.athdf`, SHA-256
`921508feeba885b14a1e1c20d2642706a5704a3f7e9e5c818fefa10d0f69aa5c`)
found exactly two pressure-floor cells. They lie at approximately
`(±5.7678,-2.8381,-2.8381) au`, radius `7.0269 au`, on newly created
level-11 blocks at the accretion shock. Their total mass is
`1.42145e-8 M☉`; ambient tracer fractions are approximately `5e-24`,
so the ambient sink does not explain this event. The c1415 HDF5 has
no floor cells. Its level-10 parents at the same locations have
`P≈0.00615 dyn/cm²`, `T≈131.3 K` and positive internal energy.

An independent replay from a c1410 restart, job 340318 on M-debug,
reproduced the c1416 HST row exactly and wrote the good c1415 restart
SHA-256 `15723c62b753add16c9db17ea5a217523029c47fed5315eba52501078c7f81c2`.
The controlled c1415→1416 replay 340325 kept the same binary and 120
ranks but set `problem/N_J=1` for that one step, suppressing new blocks.
It finished with zero blocks created and `Pmin=5.223125215e-10 dyn/cm²`,
`Jeans min=31.9899` cells. This isolates new AMR refinement as necessary
for the observed failure on this step. It does not yet prove whether the
new-cell conservative interpolation itself or a later step involving
those cells is responsible.

A separate source/binary experiment now tests a conservative,
positivity-preserving hydro prolongation limiter. The original binary
and all validated outputs remain untouched. The limiter is enabled only
with `ATHENA_WOR92_ZERO_B_POSITIVE_AMR=1` and only for the unmagnetized
ideal-EOS case. No claim of success is made before the c1415 replay and
subsequent continuation pass the same physical gates.

The isolated XD build completed in job 340329: binary SHA-256
`b64491aa31da79216c99dbfcc6eb2b9cba805e4c385a4a67d092c303a622b5cb`,
modified `mesh_refinement.cpp` SHA-256
`c1d30575e44c3bfa1eadc43ada51ea6a97e83ec0559bce1aa241fbccf363d18a`.
For the same c1415 restart, A/B jobs 340331 and 340332 used this *same*
binary with the limiter on/off. OFF reproduced the original c1416 floor
(`Pmin=1e-19`, Jeans `1.4856278e-6`); ON completed c1420 with
`Pmin=5.2204654e-10 dyn/cm²`, Jeans `32.0109` and unchanged global mass.
Both cases created 14 new blocks. Independent HDF5 audits found zero
pressure-floor cells at c1416 and c1420 with the limiter on. In the
two affected fine parent families the changed density occupies 8 children
each; their summed density differs from the unmodified HDF5 by only
`5.9e-9` and `1.8e-8` relative at float32 output precision. The local
physical pressure changed from the code floor `1e-10` to about
`1.09e6` code units at the two original bad cells. This is a limited
reconstruction change, not a continuation with AMR disabled.

The guarded M-large-a continuation job 340334 started from the good
c1420 restart SHA-256
`d02ae7ad40028c2f1a359a93ab37168a040f78ecb5ac43c1f831346c8eebc64d`.
Its c1430 and c1440 checkpoints passed the full-row pressure, Jeans and
mass gates. At c1440: rho_max `8.30935e-11 g/cm³`, Pmin
`5.20741e-10 dyn/cm²`, Jeans min `40.7285`, and the original gas mass.
The monitored density threshold remains to be checked.

## Final guarded continuation and physical result

Job 340334 completed successfully on `cna053` with Slurm `COMPLETED 0:0`
in 28:47 (1727 s), using one M-large-a node and 120 MPI ranks. All
14 ten-cycle segments exited 0. The final cycle was 1560 at code time
`2459.5802394519724`, physical time `36.87079 kyr`, with
`rho_max=1.000879401423554e-10 g/cm³`. The final HST row has
`Pmin=5.127027032841770e-10 dyn/cm²`, Jeans min `40.6429744` cells,
`Tgas,max=545.261345 K`, and gas mass `2.936555728050223e33 g`.
The final restart SHA-256 is
`10fb044ac5f0ae4101640d440e13a5aef8b28841c4844131cca1b145fc29eba4`.
The final c1560 primitive HDF5 SHA-256 is
`7257c299d3f7e91e9209d86d6b5a36129c184a80dd239b03dea71dbab7db0c53`.
An independent full-volume scan of that HDF5 found **zero** pressure-floor
cells. The restart and all earlier checkpoints remain on XD; local
analysis copies are in the `ideal_mayer5_postsync_c0_20261110` evidence
directory.

The validated c0→c1560 history deliberately excludes the failed original
c1416 row and joins the good c1415 checkpoint to the fixed-binary
c1416→c1420 replay and c1421→c1560 continuation. It contains 1527 unique
saved times and 157 identical duplicate rows. Across this path the maximum
gas-mass relative drift is `7.8522e-15`, the minimum pressure
`1.69137e-10 dyn/cm²`, and the minimum Jeans resolution `31.7129` cells.
All three gates pass. The final job logged 156/156 converged Newton solves,
420 FGMRES solves with 6418 iterations, maximum FGMRES true relative
residual `1.98614e-8`, no fatal lines, and maximum Slurm step RSS
`45,854,652 KiB` (about 43.7 GiB). These solver totals apply to the
final c1420→c1560 job; earlier jobs have separate provenance.

The independent c1560 leaf-cell profile gives central maximum-density
temperature `545.26 K` and radial velocity `-0.00772 km/s`. The strongest
positive-x density drop from 3–15 au lies at `6.775–7.141 au`, density
ratio `7.586`, with inner/outer infall speeds `0.488/2.169 km/s` and
about `0.02993 M☉` inside the shock midpoint. Positive/negative x-axis
density asymmetry over 3–15 au is `0.192%`. These support a first-core
structure rather than a lone density spike. The numerical gates and density
threshold pass, but this trajectory reaches the threshold at `36.87 kyr`,
about `1.267` times Mayer's reported `29.09 kyr`; the shock radius is
close to Mayer's 6–7 au while the profile read-off ratios differ by radius.
The initial ambient and spatial refinement differ, so the timing gap is
not assigned to a single solver defect.

The standalone local HTML report at
`/home/kosuke/simulation/test/rad_nr/wor92_xd_20261107/firstcore_integration_evidence/ideal_mayer5_postsync_c0_20261110/mayer5_validated_report.html`
contains the history, radial profiles, shock structure, A/B correction,
Mayer approximate read-off, and solver audit. The report's `PASS` means
the numerical mass/pressure/Jeans/density/solver gates passed; it does not
assert precision agreement with Mayer's physical trajectory.
