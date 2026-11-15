# WOR-114 collapse source candidate update (2026-11-16)

This independent branch starts from the historical Draft PR #3 head
`6ad84f78e57acc2f9e657970e329e2d814df8a6a`. It imports only three
source files from the verified WOR-92/118 table-EOS first-core run commit
`ff60f8f07842acb00e7753932bcfe30867e3a66e`:

| File | Reason for restoring the run source |
| --- | --- |
| `src/fld/calculate_radiation_face_states.cpp` | Use the shared block-face radiation closure state validated in WOR-118. |
| `src/task_list/time_integrator.cpp` | Exchange radiation after the implicit mixed-frame source so the next stage sees the updated state. |
| `src/newton_raphson/Newton_Raphson_driver.cpp` | Accept a finite near-root trial that meets both requested Newton norms, while retaining strict decrease for unconverged trials. |

After these imports, `src/`, `configure.py`, `Makefile.in`, and
`inputs/radiation/athinput.protostellar_core_fld_mayer5` are byte-identical to
the verified run source. The `src/` tree object is
`c4ab200383c7317a2feccc50aa9578d5cf71478d`. The branch retains the
Draft PR's WOR-95 regression driver and WOR-106 restart diagnostics; its
documentation and tests may therefore differ from the run commit. No solar
source edits, default-policy change, mainline merge, or release tag is included.

## Verification available on this exact solver source

The table-EOS/variable-opacity AMR c0→first-core run on the same `ff60f8f0`
source reached cycle 1080, peak density `1.044e-10 g cm^-3`, and peak
temperature `434.524 K`. The source/input/table/output provenance and limits
are in WOR-92/118 and the WOR-115 application audit. This statement is about
the same source tree and production input, not a rerun of the full trajectory
from this documentation branch.

The isolated WOR-95 regression worktree
`codex/wor95-table-source-regression-20261116` starts at `ff60f8f0` and
changes only validation tooling. Its
`tst/nrfld_validation/wor95_table_source_20261116/summary.json` and README
record the raw-output location, hashes, commands, and pass/fail rules:

| Test | Result on exact source |
| --- | --- |
| Gaussian diffusion, FGMRES/MG | Both PASS, eight samples each; time-extrapolated spatial orders 1.9851/2.0048 and fixed-grid temporal orders 0.9890/0.9945. The eight final profiles and metrics agree exactly between the two solvers. |
| Jeans-only AMR, 2 MPI ranks | N_J=4 stays at 8 blocks; N_J=32 creates 56 blocks and finishes at 64 leaves. Both one-step Newton/FGMRES and mass/true-residual checks PASS. |
| Local gas–radiation coupling | Low/high gas-energy cases PASS, 20/20 and 1,800/1,800 Newton steps, respectively, against a fixed radiation bath. Both use one MPI rank and have zero FGMRES iterations; they check the local nonlinear root, not distributed Krylov behavior or closed energy conservation. |
| Closed periodic box, WOR-64 | Five local cases PASS on this exact candidate commit `01256e87`: Newton targets 1e-6/1e-8/1e-10 at CFL 0.2 plus CFL 0.1/0.4 at target 1e-8. All reached code time 1; all Newton/FGMRES calls converged, maximum mass and total-energy relative drift across cases were 8.66e-15 and 8.77e-15. |

The closed-box raw run is under
`/home/kosuke/simulation/test/rad_nr/wor114_collapse_candidate_20261116/closed_box/`.
Its manifest records clean Git commit `01256e87fb1d83f99fff21411e301d460ec5ba3c`,
source SHA-256 `07f023b3399effde56eede6bb963a027405194f528e044f9bc836dc11812d668`,
binary SHA-256 `286740f1d31a3bcea7a5ee243c66df33fe840a603edc060cc4c601d2a7b6c0aa`,
input hash, run logs, HST, and plotted conservation histories. The isolated
build plus five runs took 118.7 s wall time; individual solver runs were
0.54–0.67 s and peaked at 33,480–34,116 KiB RSS. This one-block serial
box tests the coupled local/conservation path; AMR and MPI are covered by
different tests.

The old Draft PR #3's 2026-11-09 table AMR guard and one-step uniform result
belong to its earlier `8e0ee6aa` source and are historical. The final paper
source has still not been selected: WOR-114 must resolve the full FLD/solar
claim scope, exact figure provenance, and any remaining policy comparisons.
These local regressions do not certify every publication claim.
