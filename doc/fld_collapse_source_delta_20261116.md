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
| Optically thin streaming, WOR-59 | Four 2-rank cases PASS through 2 ns: all Newton and FGMRES calls converged, requested dt held except for the final shortened step, and final HDF5 snapshots agree. The thin, fine-step case has front-center lag −2.870916 cm and 10–90% width 9.258959 cm, matching the older WOR-59 128-cell values to the quoted 0.001 cm. |
| Coupled AMR restart, WOR-106 | Six direct/checkpoint pairs PASS on this candidate source: legacy/shared-face, static/adaptive, 4→4 and adaptive 1→4 ranks. All final times and topologies agree exactly, even with 0 allowed ULP; largest active-field difference is 1.137e-13 against 1e-10. All recorded Newton/FGMRES calls converge. |
| Mixed-frame fast matrix | 11/11 local runs and report PASS on clean commit `3f4e29e4`: Mach 2/5 at N=64, static/moving pulse at N=32, periodic box, and local Jv in both implicit/explicit modes. All solver calls strictly converge without relaxed acceptance. This fast matrix measures a low-resolution sanity gate, not the publication accuracy of either shock. |

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

The streaming run and its local HTML report are under
`/home/kosuke/simulation/test/rad_nr/wor114_collapse_candidate_20261116/streaming/`.
The manifest records the same clean candidate commit `01256e87`, binary
SHA-256 `b38c313e970e764b7ded1576dd522f7fb58ba43eeda953bda926d3253293a6b0`,
and build-source SHA-256 `f13aa5ba2ff8c449a30ad3b3d834b1c09f142f9f6eb011eebfdca7ccfcc0523b`.
The four cases used `chi_R=1e-4,1e-7 cm^-1` and `dt=dx/(2c),dx/(20c)`:

| chi_R [cm^-1] | dt divisor | Newton steps | FGMRES calls | fitted speed/c | x50−(x0+ct) [cm] | 10–90% width [cm] | wall [s] | peak RSS [KiB] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 2 | 154 | 202 | 0.906396 | −8.548100 | 22.568720 | 94.25 | 218256 |
| 1e-4 | 20 | 1535 | 1937 | 0.965783 | −3.086468 | 13.917801 | 886.30 | 218160 |
| 1e-7 | 2 | 154 | 230 | 0.910307 | −8.429497 | 20.503655 | 121.23 | 218288 |
| 1e-7 | 20 | 1535 | 2743 | 0.970132 | −2.870916 | 9.258959 | 1302.52 | 218244 |

The suite's PASS checks solver convergence, output integrity, and dt; the
front lag and width are reported separately. They reflect the finite-step,
finite-grid implicit FLD transport documented in WOR-59, not a conservation
failure. The local output includes profile CSVs, 21-frame MP4, PNGs, and a
standalone HTML report. The entire isolated build and four runs took 42.2 min.

The six restart pairs are recorded in
[`restart_summary.json`](../tst/nrfld_validation/wor114_candidate_20261116/restart_summary.json)
and in raw logs/HDF5 under
`/home/kosuke/simulation/test/rad_nr/wor114_collapse_candidate_20261116/restart_amr/`.
The binary SHA-256 is
`fb8a2c7855e167920ea31e33a817e9d9df09c118b24433c5c8692e7bf6968297`;
the JSON summary records the base and per-case input hashes, direct/restarted
Newton and FGMRES counts, maximum FGMRES true-relative residual, final times,
and active-field differences. Configure and isolated make took about 110 s;
the six-pair suite took 27.85 s with launcher-reported peak RSS 123,176 KiB.
The older Draft PR #3 had two 1-ULP strict-time failures and passed with a
2-ULP allowance. This exact source passed 0 ULP for these six cases; that
observation does not prove strict bitwise restart timing for other inputs.

To reproduce the AMR restart cases, build an isolated copy of this source with
`python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld -mpi -hdf5
-h5double --cflag=-O3`, then run
`python3 tst/nrfld_validation/wor106_coupled_amr_restart.py --binary
/path/to/build/bin/athena --input inputs/radiation/athinput.nrfld_diff_amr_interface
--output /path/to/new/output --time-ulps 0` from the repository root.

The mixed-frame fast matrix is under
`/home/kosuke/simulation/test/rad_nr/wor114_collapse_candidate_20261116/mixed_frame_fast/`.
Its standalone `index.html`, summary JSON, input/log/HDF5, resource files,
and two isolated build logs retain the exact evidence. A
[compact result JSON](../tst/nrfld_validation/wor114_candidate_20261116/mixed_fast_summary.json)
is versioned with this branch. Mach 2 raw L1 errors
relative to the bundled semi-analytic profile are 0.629/0.629% in density
(implicit/explicit), 0.696/0.696% in gas temperature, and 0.278/0.278% in
radiation temperature. Mach 5 raw L1 is about 7.22% density, 6.55–6.56% gas
temperature and 6.50–6.52% radiation temperature; after aligning the
density jump for shape comparison, the corresponding density L1 is about
1.37–1.38%. The raw offset is retained, not hidden by that alignment.

The moving-pulse case has 1,778 strict-converged cycles per mode. Relative L1
to the shifted static pulse is 0.003765/0.003761 for density and
0.018004/0.018006 for radiation energy (implicit/explicit); these are
finite-resolution differences, not zero. The periodic-box total-energy drift
is `−7.87e-15` with implicit drift and `−6.36e-8` with explicit drift;
mass drift is `−7.40e-16`/`−2.27e-8`. The local fixed-coefficient gas and
radiation Jv finite-difference relative errors fall by approximately 10×
when epsilon falls from `1e-4` to `1e-5` to `1e-6`; both modes pass. This
Jv check does not substitute for a global coarse/fine boundary Jv test.

The fast matrix used 2 MPI ranks for shock/pulse and 1 rank for box/Jv.
Its shock and pulse binary SHA-256 values are
`2e6cd07967310c06678cf07dd5872a2fb2d7fdff73cade4e0adaf823d0b301c0`
and `286740f1d31a3bcea7a5ee243c66df33fe840a603edc060cc4c601d2a7b6c0aa`.
The numerical claims above are bounded by N=64/32, the bundled reference
profiles, and the exact input hashes in the summary; high-resolution Mach 5
temperature-spike qualification remains separate.

The old Draft PR #3's 2026-11-09 table AMR guard and one-step uniform result
belong to its earlier `8e0ee6aa` source and are historical. The final paper
source has still not been selected: WOR-114 must resolve the full FLD/solar
claim scope, exact figure provenance, and any remaining policy comparisons.
These local regressions do not certify every publication claim.
