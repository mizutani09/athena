# Mayer §5 ideal-EOS constant-opacity collapse: consistent c0 path

Date: 2026-11-10 JST. Isolated branch: `codex/wor92-mayer5-ideal-c0-20261110`.
This is a new cycle-0 path; it never reads a checkpoint from the earlier ideal
or table-EOS calculations. The M-large-a job 339331 depends on the successful
M-test-a preflight 339330 and reads only that preflight's cycle-10 restart.

## Initial conditions and paper match

The [Mayer et al. 2025 §5.1](https://arxiv.org/html/2510.12620#S5) ideal
reference uses an initially uniform, nonrotating and unmagnetized 1 Msun,
3000 au, 14 K sphere, ideal gas with mu=2.381 and gamma=5/3, and constant
kappa_P=kappa_R=1 cm²/g, in a 12000 au cube with periodic hydro and
nonperiodic gravity. Its cited [precursor §3.1](https://arxiv.org/html/2410.07016#S3)
states that the ambient density is rho_core/30 and initially pressure-balanced
with the core. The input therefore initializes ambient gas at 420 K and
uses a sharp sphere boundary (`transition_width=0`). The discrete sphere mass
is 1.0008577 Msun; the initial core density is 5.25158548e-18 g/cm³.

The paper does not specify initial ambient *radiation* temperature. We set it
to 14 K as an explicit assumption, separately from the 420 K gas, because
ambient radiation in LTE at 420 K caused a radiation/gas pressure ratio of
about 3.06e4 and immediate first-step Newton failure (control 339297).
The background gas temperature is reset after the implicit solve, matching the
paper's fixed ambient temperature prescription. The Cartesian Jeans AMR is
not the paper's AREPO mass-refinement algorithm; this is a matched physical
initial condition, not an identical discretization.

Final input: `wor92_mayer5_reference_postreset_c0_20261110.in`, SHA-256
`382ab6df9ca41769a96475c52548fa008f486e3a43b5693118b378d991fa5ebb`.
Source base commit `582acd5999c1a5681ab1bc1d1af292676642ed40`; the
isolated source also includes background-radiation initialization, a narrowly
bounded Newton halo roundoff clamp, and the post-reset boundary sync described
below. Final binary SHA-256:
`9e7c2faa80832a7cd5d8aa3fd4689aed1648a7f346d4d492c2c0742217020b7f`.
Its main.cpp SHA-256 is
`4e4320d57b41c718db4428463e63f7c977a146900f49d4eb359f493bd84cd84b`.
The other source hashes and compiler configuration are in the XD
`build_provenance.txt` for M-debug build 339328 (COMPLETED 0:0, 1:44).

## Post-reset conservation defect and correction

The original halo-roundoff preflight 339313 solved all 10 cycles, but gas mass
decreased 2.6454%. The decrease began on the first step *after* the accepted
background temperature reset. Three controls identified the cause:

| XD job | Change from original | Result at cycle 2 |
| --- | --- | --- |
| 339313 | AMR, post-implicit sink | gas mass −0.4538% |
| 339318 | suppress derefinement | gas mass −0.4538%; canceled after diagnosis |
| 339319 | fixed grid, same sink | gas mass −0.9135% |
| 339320 | fixed grid, no Riemann radiation pressure | gas mass −0.917%; hence that term was not the cause |
| 339327 | fixed grid, disable background sink | mass unchanged to displayed 15 digits; Jeans<16 because this is a fixed-grid control |

The existing `SynchronizeFLDBoundariesAfterNR` ran before
`Mesh::UserWorkInLoop`, where the post-implicit sink changes accepted gas
pressure and radiation energy. The next Godunov step could therefore use
stale interblock/periodic primitive ghosts. The correction calls that same
sync once after the sink, only when `background_sink_after_implicit=true`.
No hydro flux, Riemann radiation-pressure term, physical initial condition, or
AMR criterion was changed for the adopted run. A fixed-grid repeat 339329
completed 0:0 in 8 s, with gas mass 2.962995592972076e33 to
2.962995592972074e33 g over two cycles (15-digit history precision).

## Full AMR preflight and long submission

M-test-a job **339330**, one node/64 MPI ranks, completed 0:0 in 5:21.
It reached cycle 10 with 120 MeshBlocks (896 created, 840 destroyed), so AMR
regridding was exercised. The input and binary hashes above were checked by
the job script. Of 17 Newton status entries, 12 were converged physical
solves and five were zero-dt initialization. The 40 FGMRES solves used 4019
iterations total, maximum 241 per solve; largest reported true relative
residual was 2.35577e-8 (the configured inexact cap is 1e-7). Largest final
Newton L2 and max residuals were 3.71940e-9 and 3.75748e-8. The accepted
state had minimum pressure 2.37067e-10 dyn/cm² and minimum Jeans resolution
42.0794 cells. Initial/final gas mass was 2.936555728050246e33 /
2.936555728050223e33 g; maximum relative drift in the 15-digit history was
7.88e-15. The job logged 194 bounded negative halo roundoff events, with no
accepted active-cell negativity or fatal error. Slurm step MaxRSS was
18,922,080 K (as reported by sacct, not a per-node reconstruction). The final
restart SHA-256 is
`fd8e9655f18622e778d454fab0fc0a290417bc64c081eede683584258cbc57c8`.

The preflight script rejects any accepted history row with gas mass drift
>1e-8 relative to c0, pressure ≤1e-18 dyn/cm², or Jeans resolution <16.
The long script applies the same checks to each ten-cycle segment, reads the
c0 mass from the preflight history, and stops at the first saved snapshot
with rho_max≥1e-10 g/cm³ or wall budget. The 5-node/120-rank layout respects
the 120-block c10 seed and leaves memory headroom for later refinement.
M-large-a job **339331** was submitted with `afterok:339330` and the same
binary/input hashes. At submission it was pending for `QOSMaxJobsPerUserLimit`
while a separate large job on the same account was active; its eventual physics result
is not yet known. The previously rejected runs 339296/339298 were canceled
before execution and are not part of this path.

XD directory: `/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_postsync_c0_20261110/`.
Local preflight evidence: `/home/kosuke/simulation/test/rad_nr/wor92_xd_20261107/firstcore_integration_evidence/ideal_mayer5_postsync_c0_20261110/job_339330/`.
