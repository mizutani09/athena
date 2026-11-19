# WOR-20 publication-source local performance baseline (2026-11-20)

## Scope and provenance

This measurement uses the solver `src/` tree
`c4ab200383c7317a2feccc50aa9578d5cf71478d`, identical to the validated
table-EOS first-core source and the `protostellar-firstcore-v1` collaborator
release. The performance branch is
`exp/wor20-publication-baseline-20261120`. The HD+FLD binary reused by every
matched solver-policy run has SHA-256
`f7bb58cdf162d9369e68479e19e5b33fca4459710ac4ed37f6662591b1ae1ba8`.

The controlled benchmark is uniform 32³ HD+FLD with 16³ MeshBlocks, MPI
1/2/4 ranks, ten warm-up and 100 measured steps, and three repeats. The three
policies differ only in the linear solve and required transfer pair:

1. standalone MG, rediscretized hierarchy, trilinear prolongation;
2. FGMRES with one rediscretized MG V-cycle per preconditioner application;
3. FGMRES with a uniform Galerkin hierarchy and constant variational
   prolongation.

Every measured window exceeds ten seconds. The timer wrapper records each
rank's peak RSS. A separate timer-on/off pilot found bitwise-identical final
fields and Newton status/iteration histories for all 1/2/4-rank cases.

## Results

All 27 matched runs passed. Every run used 200 outer Newton corrections,
strictly met the configured final Newton norms, retained positive finite
states, and had zero mass drift at history precision.

| Linear policy | ranks | median seconds/step | range | strong efficiency | FGMRES iterations/run | max iterations/solve | median max-rank RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MG / rediscretized | 1 | 0.267717 | 0.267234–0.268375 | 1.000 | 0 | 0 | 127.5 MiB |
| MG / rediscretized | 2 | 0.162374 | 0.161810–0.165491 | 0.824 | 0 | 0 | 81.5 MiB |
| MG / rediscretized | 4 | 0.122465 | 0.115446–0.556537 | 0.547 | 0 | 0 | 60.0 MiB |
| FGMRES / rediscretized | 1 | 0.366754 | 0.346933–0.369512 | 1.000 | 1475 | 7 | 137.9 MiB |
| FGMRES / rediscretized | 2 | 0.206105 | 0.205975–0.208767 | 0.890 | 1475 | 7 | 87.1 MiB |
| FGMRES / rediscretized | 4 | 0.143812 | 0.133745–0.419197 | 0.638 | 1475 | 7 | 62.2 MiB |
| FGMRES / Galerkin | 1 | 0.404633 | 0.360631–0.617119 | 1.000 | 1929 | 9 | 137.7 MiB |
| FGMRES / Galerkin | 2 | 0.252944 | 0.251636–0.253564 | 0.800 | 1929 | 9 | 87.2 MiB |
| FGMRES / Galerkin | 4 | 0.165150 | 0.149177–0.174492 | 0.613 | 1929 | 9 | 62.3 MiB |

At the median, FGMRES/rediscretized costs 37.0%, 26.9%, and 17.4% more per
step than standalone MG at 1, 2, and 4 ranks. Its rank peak RSS increases by
8.1%, 6.8%, and 3.6%. Galerkin increases the identical-run FGMRES iteration
count from 1475 to 1929 (+30.8%) and is slower than the rediscretized
preconditioner at all three rank counts. It therefore has no performance
benefit for this uniform benchmark.

The policies give numerically equivalent final states. Against standalone MG,
the maximum scaled field difference is `6.069e-16`; the largest relative L2
difference is `1.025e-15`. Output times are bitwise identical. Thus the timing
difference is not caused by a looser or changed accepted solution.

## HD baseline and weak scaling

The pure-HD control was extended to 2,500 measured steps so every strong-
scaling window exceeded ten seconds. Its median seconds/step are
`0.012622 / 0.007096 / 0.005131` at 1/2/4 ranks, corresponding to strong
efficiencies `1.000 / 0.889 / 0.615`. The standalone-MG HD+FLD medians are
21.2, 22.9, and 23.9 times the corresponding HD cost. Internal phase timers
attribute about 87–90% of HD+FLD time to the implicit FLD phases; shared
explicit radiation work remains in the explicit timer.

For weak scaling with 16³ MeshBlocks, pure HD gives median seconds/step
`0.012603 / 0.015228 / 0.030883` and efficiencies
`1.000 / 0.828 / 0.408`. The current FGMRES/rediscretized HD+FLD path gives:

| MeshBlock | ranks | median seconds/step | weak efficiency | FGMRES iterations/run | median max-rank RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16³ | 1 | 0.334669 | 1.000 | 1475 | 137.8 MiB |
| 16³ | 2 | 0.428524 | 0.781 | 1475 | 141.0 MiB |
| 16³ | 4 | 0.615955 | 0.543 | 1475 | 151.7 MiB |
| 32³ | 1 | 0.307343 | 1.000 | 1475 | 110.4 MiB |
| 32³ | 2 | 0.375333 | 0.819 | 1475 | 111.4 MiB |
| 32³ | 4 | 0.536924 | 0.572 | 1475 | 119.0 MiB |

The FGMRES call, total-iteration, and maximum-iteration counts are identical
at every rank and both block sizes. The weak-scaling loss is therefore the
cost of the same Krylov/MG work, not convergence degradation. Increasing the
MeshBlock from 16³ to 32³ reduces time and memory and raises the 4-rank
efficiency from 54.3% to 57.2%, but does not reach the exploratory 70–80%
target. Block-management overhead is a contributor, not the sole cause.

## Opt-in FGMRES internal timing

An off-by-default `performance/linear_detail` diagnostic measures four
additive parts of each FGMRES solve on the rank with the largest solve wall
time: the MG preconditioner, operator matvec, FGMRES global reductions, and
vector/other work. It is deliberately not enabled by ordinary inputs. Each
record is rejected by the analysis if its parts do not sum to its total.

The diagnostic was run for ten warm-up plus 100 measured steps, three repeats,
and 1/2/4 ranks. All nine runs and all timer-off controls passed. Every
timer-on/off pair had bitwise-identical final arrays and identical Newton
status/iteration histories. The median end-to-end on/off ratios were
`0.990 / 0.977 / 0.995`; no diagnostic overhead is resolved above host noise.
Every run performed 232 linear solves, while every measured window performed
200 Newton corrections.

| ranks | MG preconditioner | operator matvec | FGMRES reductions | vector/other |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 29.7% | 62.5% | 0.02% | 7.8% |
| 2 | 39.3% | 49.5% | 5.3% | 5.9% |
| 4 | 40.8% | 40.8% | 12.6% | 5.0% |

The matvec is the largest single-rank cost. With more local ranks, MG
preconditioning becomes comparable and the explicit Krylov-reduction share
grows from effectively zero to 12.6%. This directly supports the earlier
inference that synchronization/communication, rather than increased Krylov
iteration count, limits this small-grid scaling. The detailed suite happened
to run during a slower host interval (especially at four ranks), so its
absolute scaling rate does not replace the matched baseline above; only the
within-solve fractions are used for attribution.

## HD/MHD/FLD/gravity cost matrix

Eight current-source physics configurations were built independently. The
non-FLD cases use 1,200 measured steps, except that the separate 2,500-step
HD suite supplies the HD row. FLD cases use 100 measured steps. The selected
1/4-rank aggregates below all contain three repeats and windows of at least
ten seconds; all 48 runs passed.

| Case | 1-rank s/step | 4-rank s/step | strong efficiency | implicit FLD fraction (1/4 rank) | max-rank RSS (1/4 rank) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HD | 0.012622 | 0.005131 | 0.615 | — | 51.8 / 38.6 MiB |
| HD + gravity | 0.024314 | 0.018410 | 0.330 | — | 56.4 / 39.8 MiB |
| MHD | 0.029375 | 0.013326 | 0.551 | — | 66.9 / 43.1 MiB |
| MHD + gravity | 0.045845 | 0.034442 | 0.333 | — | 71.6 / 44.8 MiB |
| HD + FLD | 0.330868 | 0.147860 | 0.559 | 0.901 / 0.907 | 137.8 / 62.3 MiB |
| HD + FLD + gravity | 0.358807 | 0.155196 | 0.578 | 0.865 / 0.826 | 142.3 / 63.6 MiB |
| MHD + FLD | 0.355096 | 0.146392 | 0.606 | 0.851 / 0.862 | 153.2 / 67.0 MiB |
| MHD + FLD + gravity | 0.371405 | 0.157198 | 0.591 | 0.820 / 0.787 | 157.8 / 68.5 MiB |

Implicit FLD dominates every FLD-enabled case. Adding ideal MHD or
self-gravity changes the total cost much less than enabling FLD. Conversely,
gravity-only cases scale poorly at this small fixed grid because their
global/coarse work is a large fraction of the calculation. The inexpensive
smooth ideal-MHD control does not predict the conditioning difficulty of a
strong-field collapse state.

## Conservative static-AMR operator cost

The performance harness was extended with a separate `test_c` problem mode.
Static AMR is deliberately allowed only in this mode, so it selects the
validated conservative common-face operator by default. An attempted generic
performance-pgen AMR pilot stopped at the existing unvalidated-problem guard
and was excluded; neither the guard nor the legacy nonconservative escape was
used. The selected Test C binary SHA-256 is
`bb6fe320ccc2fc4de5077f244588918bb1b0c8853c79553b02fe5238cf97010c`.

The matched Test C control has a 32³ root grid with 8³ MeshBlocks. The AMR
case replaces the eight centered root blocks with their 64 level-1 children,
giving 120 MeshBlocks and 61,440 leaf cells versus 64 MeshBlocks and 32,768
cells in the uniform case. All selected groups have three repeats and every
measurement window exceeds ten seconds.

| layout | ranks | median seconds/step | strong efficiency | maximum FGMRES iterations/solve | median max-rank RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 1 | 0.207448 | 1.000 | 3 | 221.0 MiB |
| uniform | 2 | 0.146582 | 0.708 | 3 | 132.7 MiB |
| uniform | 4 | 0.100195 | 0.518 | 3 | 90.8 MiB |
| centered level-1 AMR | 1 | 1.967438 | 1.000 | 9 | 580.6 MiB |
| centered level-1 AMR | 2 | 1.192895 | 0.825 | 9 | 314.7 MiB |
| centered level-1 AMR | 4 | 0.822838 | 0.598 | 9 | 184.1 MiB |

The AMR/uniform time ratios are 9.48, 8.14, and 8.21 at 1/2/4 ranks. Dividing
by the 1.875 leaf-cell ratio still leaves factors 5.06, 4.34, and 4.38. This
is not a pure interface-overhead measurement: the conservative AMR system
requires nine rather than three FGMRES iterations, and each iteration also
contains coarse/fine transfers, shared-face assembly, and a larger hierarchy.
It does show directly that extrapolating uniform-cell throughput to a
collapse AMR calculation would be misleading.

The AMR FGMRES fractions at 1/2/4 ranks are respectively:

| ranks | MG preconditioner | operator matvec | FGMRES reductions | vector/other |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.3% | 60.5% | 0.01% | 4.1% |
| 2 | 46.3% | 49.2% | 1.3% | 3.2% |
| 4 | 44.5% | 47.2% | 5.3% | 2.9% |

All nine selected AMR timer-on/off pairs have bitwise-identical final arrays
and identical Newton status/iteration histories. All selected uniform and AMR
runs have one converged Newton iteration per measured step, zero mass drift at
history precision, positive finite states, and rank-invariant maximum FGMRES
iteration counts. The smaller reduction fraction than in the 32³ smooth
coupled benchmark is consistent with more useful AMR work per rank; it is not
evidence for multi-node scaling.

The harness now also handles both valid Athena output endings: one termination
snapshot, or a scheduled final snapshot followed by a duplicate-time
termination snapshot. It requires an initial and at least one final output and
always validates the first and last arrays. The previous exact-two assumption
was exposed only by the 120-step output cadence; the affected pilot suites are
excluded.

## XD conservative-AMR scaling pilot

The multi-node experiment uses a fixed 128³ root grid, centered level-1 static
AMR, 8³ MeshBlocks, 7,680 leaf MeshBlocks, 3,932,160 active leaf cells, and 32
MPI ranks per node. It reuses the same validated Test C path and detailed
FGMRES timer. This is a strong-scaling test; the numerical problem and ten
measured steps remain fixed.

The XD checkout is clean at commit
`6ff44924b64d955be27fd240269ca4cbd488b982`; binary SHA-256 is
`f42869088d82a3294b35679380a06bac4140163bf0cdbc94df47f4779d38aca6`.
The fixed input and separately deployed rank-resource wrapper have SHA-256
`cfe44d6a597f16df7dc84f70bb4b42eb14074c4d678aac6eba668b5403c89b8c`
and `d30afe06b6b70e72686051cc3479b82dcf43fc5468a2e0ead497877b38776d3c`.

The staged XD jobs are `343669`, `343671`, and `343685` (M-test-a, 1/2/4
nodes and 32 MPI ranks/node).  They completed with exit `0:0` in 2:16, 0:57,
and 0:31.  All 32/64/128 rank-resource and performance records are present.
Every run reports the same ten measured steps; all 12 Newton solves converged,
the 12 FGMRES solves used 132 iterations in total and at most 11 per solve,
and no relaxed acceptance, breakdown, warning, or fatal event occurred.  Mass
drift is zero at 16-digit history precision.

| nodes | ranks | s/step | speedup | efficiency | M leaf-cell steps/s | MG prec. | matvec | reductions | vector/other | max-rank RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 10.64899 | 1.000 | 1.000 | 0.369 | 82.275% | 12.328% | 1.030% | 4.368% | 1,241.6 MiB |
| 2 | 64 | 4.05353 | 2.627 | 1.314 | 0.970 | 80.256% | 16.417% | 1.086% | 2.241% | 754.8 MiB |
| 4 | 128 | 1.79601 | 5.929 | 1.482 | 2.189 | 74.581% | 20.124% | 2.658% | 2.637% | 479.2 MiB |

The measured superlinear speedup is real for these three executions but is
not generalized beyond this fixed problem and single repeat per node count.
It is not caused by a convergence or rank-imbalance change: iteration counts
are identical and max/mean measured-rank wall-time ratios are
`1.00011 / 1.00018 / 1.00026`.  Instead, the per-rank active work, Krylov
workspace (`44.24 / 22.12 / 11.06 MB`), and peak RSS all shrink with rank
count.  The sum of rank user CPU time also falls from 4,126.5 to 3,187.1 to
2,968.5 seconds, consistent with a working-set/cache or memory-bandwidth
effect in addition to ordinary parallelism.  More repetitions would be
required for a publication claim about the magnitude of superlinearity.

The independent sums of per-rank peak high-water marks are 38.52, 46.48, and
58.61 GiB.  They are not asserted to be simultaneous node memory.  The
max-rank values in the table are the safer per-process requirement.  The
FGMRES reduction fraction increases to 2.66% at four nodes but remains much
smaller than the MG preconditioner and operator matvec; this problem has not
yet reached a reduction-dominated scaling limit.

The accepted 1/2/4-node `summary.json` SHA-256 values are respectively
`4be4dbcfffefc1d4c2aa1eebdf44885f65e266cad28bdc3acb1e719bf6a40144`,
`114b36c53f59d7dafde5046a73e25949066bf56e9aae925ad98b3231f7d13aae`,
and `c242a5bdfb2f4f0eadffe8946a7a1fb3ac3a230d90e06e3145d3969a13963b2f`.
The final aggregate JSON SHA-256 is
`ada2b4631b05fd7c361fbd500344d9031d49158b0459d8c9333ea9cb7b2abf60`.

Several earlier job attempts are retained but excluded from performance data:
compute-node Python 3.6 could not parse the typed wrapper; a Python 3.11 path
present only on the login node was unavailable on compute nodes; and the first
shared-Python run closed Slurm's inherited PMI2 descriptor before the MPI
payload. The accepted wrapper uses the shared Cray Python and explicitly
preserves inherited descriptors. These were launcher failures before a valid
Athena measurement, not solver, convergence, conservation, or memory failures.

## Interpretation and limits

Standalone MG is the fastest choice for this easy, uniform, well-conditioned
problem. FGMRES remains important for difficult AMR/collapse states where the
standalone iteration may fail or converge poorly; this result rejects only a
blanket claim that switching to FGMRES speeds up every FLD solve. Similarly,
uniform Galerkin is not justified as a default by these data.

The local machine is WSL on four physical cores/eight logical CPUs. Several
long sequential runs show whole-kernel slowdowns without any change in Newton
or FGMRES work (notably one MG and one rediscretized-FGMRES 4-rank repeat).
Those values are retained in the ranges rather than discarded. Medians are
more representative.  The separate XD 1/2/4-node result is the multi-node
engineering baseline for the fixed conservative-AMR case, but its single
repeat per node count and one problem size do not establish broad publication
scaling. Rank peak RSS values are per-process high-water marks; summing them
is not a simultaneous node RSS measurement.

Raw suites:

- MG/rediscretized:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_013020_727610_JST`
- FGMRES/rediscretized:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_013455_501800_JST`
- FGMRES/Galerkin:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_013953_970398_JST`
- Standalone comparison HTML/JSON/PNG:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/solver_comparison_20261120`
- Long-window HD strong scaling:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_014822_050118_JST`
- HD weak scaling (16³ blocks):
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_015150_170344_JST`
- FGMRES weak scaling (16³ / 32³ blocks):
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_020010_064813_JST` and
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_020916_515324_JST`
- Physics matrix non-FLD / FLD suites:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_022718_790019_JST` and
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_023853_067722_JST`
- Physics matrix HTML/JSON/PNG:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/physics_matrix_20261120`
- Opt-in FGMRES detailed timing (JSON/CSV/PNG and timer-off controls):
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/fgmres_detail_long_20261120/performance_run_20261120_030203_224241_JST`
- Conservative Test C uniform long windows (1 rank and 2/4 ranks):
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/uniform_testc_long_20261120/performance_run_20261120_034418_279450_JST` and
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/uniform_testc_p24_long_v3_20261120/performance_run_20261120_034904_563900_JST`
- Conservative Test C AMR long windows (1/2 ranks and extended 4 rank):
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/amr_testc_long_20261120/performance_run_20261120_033637_961591_JST` and
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/amr_testc_p4_long_20261120/performance_run_20261120_034212_554844_JST`
- Conservative AMR comparison HTML/JSON/CSV/PNG:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/amr_comparison_v2_20261120`
- XD 1/2/4-node raw copies and final HTML/JSON/CSV/PNG:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/xd_scaling_20261120`

`wor20_compare_solvers.py` validates matched options and suite PASS status,
recomputes FGMRES counts and final-field differences, and regenerates the
standalone report. `wor20_physics_matrix.py` selects only long-window
aggregates, recomputes phase medians, and regenerates the eight-case report.
`wor20_amr_performance.py` rejects short or failed selected windows, recomputes
FGMRES counts and leaf-cell ratios, checks available timer controls, and
regenerates the conservative AMR comparison report.
