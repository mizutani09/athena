# WOR-114 candidate reproducibility status

Candidate parent: `8e0ee6aa2261e5cb4f4b53ca94752c4caa549b80` (WOR-92 first-core branch). The integration branch adds the WOR-95 regression driver and historical artifacts plus the WOR-106 diagnostic time-tolerance option; it does not change `src/`. The PR head, not this document, is the exact candidate source SHA. No publication tag or final paper source has been declared.

The moving WOR-92 branch advanced to `3518882f` during this work. Its new `54657730` Newton line-search change is absent from this source and these tests. Draft PR #3 uses the fixed review base `codex/wor114-fixed-base-8e0ee6aa`; results cannot be transferred to the moving branch without rerunning the numerical gates.

## Local tests, 2026-11-09

Environment: g++ 15.2.0, Open MPI 5.0.10, HDF5 1.14.6, Python 3.14.4. Each build used an isolated source snapshot and `-O3`. The first three runs used source commit `e8968801`; the full convergence and table build used `5acf8573`. These commits have identical `src/` trees. Output/logs and manifests are held locally under `simulation/test/rad_nr/wor114/`, outside the public repository.

| Test | Configuration and result | Build SHA-256 | Qualification |
|---|---|---|---|
| FLD diffusion smoke | MPI 1 and 2, HDF5, 16³, 92 steps; both PASS, relative L1 `0.00425233` < `0.01`; final x-slice fields and time bitwise equal | `f1e9e08aba1cafb759b9883c56116f275d09f9fecbea740542023fca18269804` | Short Gaussian check only |
| Coupled FLD | MPI 1, 20 steps, FGMRES; 20/20 Newton and 20/20 FGMRES converged; maximum log gas-energy error `4.993e-6`, radiation drift `0` | `e90dcde5863025495a9cdf380b4ec186b58248005f1d3eacbe429fbc6279fdf0` | Fixed-radiation bath, not a closed-energy test |
| WOR-95 FGMRES diffusion convergence | MPI build, N=16/32/64; `passed=true`; spatial order `1.9702, 2.0007`, time-extrapolated spatial order `1.9851, 2.0048`, fixed-grid temporal order `0.9890, 0.9945` | `f1e9e08aba1cafb759b9883c56116f275d09f9fecbea740542023fca18269804` | Recomputed on this source; old archived plots remain historical |
| WOR-106 coupled AMR restart | MPI 4→4 and 1→4, static/adaptive, legacy/conservative; strict exact-time test FAILS on two static cases by 1 binary64 ULP (`1.735e-18`). Explicit `--time-ulps 2` passes all six; max field difference `2.274e-13`, adaptive topology matches. | `dea850feec6724a67a64e0fc4245cf090f14f40e8c19f5d78de86b0eec0cdb04` | 2-ULP qualification only; strict equality is not claimed |
| Table EOS/opacity collapse | MPI 2, general EOS + HLLD-FLD, uniform 32³, CFL 0.02, one step: build and run finish on cycle limit, Newton `converged` in 4 iterations. Two FGMRES calls reach 300 iterations and are accepted through `precision_floor_accept=1`, with reported relative residuals `3.38e-8` and `3.39e-8` against `1e-8`. | `4ead5307e13a6291920114819250c74bac5d4f22e165ecea1c844c54c6af64cf` | Startup smoke only, no first-core or physical accuracy claim |
| Table EOS AMR guard | Same table input, 32³, two levels: program prints FATAL unsupported multilevel conservative-default guard before evolution. The MPI launcher returned zero, so the log must be checked for FATAL. | Same table binary | **Unsupported**, not PASS |

The table smoke used the existing pilot bundle's `athinput.protostellar_core_fld_tabular`, SHA-256 `98ae3a305b6c95b200488187c224a02bb767dc14add87af7760db9ce3221deb2`. The EOS file SHA-256 is `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`; opacity file SHA-256 is `6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`. The bundle's `SHA256SUMS` check passed. The pilot uses Semenov Planck **extinction** as an absorption proxy; this scientific approximation needs review before a table-EOS collapse claim.

## Publication candidate and gaps

The integration branch is a **review candidate** for the fork's FLD implementation, with reproducible local diffusion, coupling, limited restart and table startup checks. The paper-ready source can be fixed only after the following gaps are resolved or explicitly removed from the paper's claims:

- Select and revalidate one WOR-24/25 versus WOR-92 centered-face/default and bounded-FGMRES policy with same-state 1/2-rank Jv, face flux and true-residual comparisons. The selected branch currently retains WOR-92's opt-in centered-face trial, not WOR-24's validated Test C default.
- Reconcile the table-EOS AMR guard and the first-core source/checkpoint/input chain; ideal-gas Mayer5 and long table-EOS collapse were not rerun on this SHA. The short uniform run used bounded precision-floor acceptance.
- Rerun mixed-frame, Mach 2/5, streaming, closed-box energy, AMR regrid and HLLD/MHD gates on the selected numerical policy. WOR-105's prior Mach 5 results belong to its original source and do not establish this branch's science output.
- Integrate and separately verify solar HD restart/boundary and MHD magnetic-energy changes before any solar result is tied to the candidate SHA. WOR-70/111 short pilots do not establish thermal stationarity or the solar flux claim.
- Map every proposed paper figure to its exact binary/source/input/table SHA and output. Revised physics/solver defaults may require reruns; coordinate with WOR-87/88/90 and the application gates WOR-115/116.

Future PrincetonUniversity/athena patch splitting, license/compatibility review, non-FLD upstream regression and upstream proposal are tracked in WOR-117. This branch must remain a Draft PR until author review; no mainline merge or release tag follows from the tests above.
