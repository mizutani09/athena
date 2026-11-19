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
more representative, but these are local engineering measurements, not final
multi-node publication scaling. Rank peak RSS values are per-process high-water
marks; summing them is not a simultaneous node RSS measurement.

Raw suites:

- MG/rediscretized:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_013020_727610_JST`
- FGMRES/rediscretized:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_013455_501800_JST`
- FGMRES/Galerkin:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/performance_run_20261120_013953_970398_JST`
- Standalone comparison HTML/JSON/PNG:
  `/home/kosuke/simulation/test/rad_nr/wor20_publication_baseline_20261120/solver_comparison_20261120`

`wor20_compare_solvers.py` validates matched options and suite PASS status,
recomputes FGMRES counts and final-field differences, and regenerates the
standalone report.
