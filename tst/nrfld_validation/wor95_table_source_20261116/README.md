# WOR-95 regression on the actual table-EOS collapse source (2026-11-16)

The source of the verified WOR-92/118 c0→first-core run is
`ff60f8f07842acb00e7753932bcfe30867e3a66e`. This isolated worktree
starts at that commit. Only the validation driver extension from
`codex/wor95-validation-20261107` and the new one-cycle Jeans harness were
added; no solver or problem-generator source was changed. The isolated
diffusion build source is recorded by its SHA-256 in [summary.json](summary.json).
Raw logs, exact generated inputs, manifests, HST, and binaries are under
`/home/kosuke/simulation/test/rad_nr/wor95_table_source_20261116/`.

## Results

| Test | Result | Evidence |
| --- | --- | --- |
| Gaussian FLD, FGMRES | PASS; time-extrapolated spatial order 1.9851/2.0048, fixed-grid temporal order 0.9890/0.9945 | 8 samples, 2,641 active Newton solves, 5,280 FGMRES solves, max 6 iterations and true relative residual 9.9851e-11 against 1e-10; 216.1 s including build |
| Gaussian FLD, MG | PASS; exactly the same 8 final profiles and error/order metrics as FGMRES | Same isolated binary; 2,641 active Newton solves; 200.8 s |
| Jeans-only AMR, N_J=4 | PASS; root 32³ with 16³ blocks stays at 8 MeshBlocks/level 1 | CFL 0.02, 2 MPI ranks, one cycle, dt 4.1446358997 code, 1/1 Newton, 2 FGMRES solves, max true residual 1.0480e-8 ≤2e-8, mass relative change +4.15e-14, MaxRSS 122,260 KiB, 7.5 s |
| Jeans-only AMR, N_J=32 | PASS; 56 blocks created, 64 leaves/level 2 | Same binary and root, dt 2.0723179499 code, 1/1 Newton, 2 FGMRES solves, max true residual 7.6452e-9 ≤2e-8, mass relative change −4.51e-14, MaxRSS 746,652 KiB, 51.7 s |
| Local gas–radiation coupling, low/high gas energy | PASS; low 20/20 and high 1,800/1,800 Newton steps | One 4³ MeshBlock, 1 MPI rank, fixed radiation bath; maximum log gas-energy errors 4.993e-6 and 2.202e-3 against 5e-3; zero reported radiation drift. Both FGMRES logs have zero iteration/zero residual because the test isolates the local coupling root; this is not an FGMRES stress test. |

The unrefined root Jeans length is 15.2334 cells. It exceeds 4 and falls
below 32, explaining the two AMR outcomes. The refined minimum is 30.4667
cells because the test intentionally limits maximum AMR level to 2. These
outcomes and convergence orders agree with the prior WOR-95 run on source
`209e22398`, now measured on the actual table-collapse source as well.

The Jeans harness reproduces the conservative prototype flags from the
verified XD c0→first-core script, including shared coefficients, variable
limiter, coupled operator, regrid, centered face, and positive NR ghost. It
uses the ideal-gas Mayer5 input to isolate AMR triggering from table
interpolation; the full table-EOS AMR trajectory itself is the verified
WOR-92/118 run. The one-cycle regression uses CFL 0.02 and a strict Newton
target 2e-8, matching the original WOR-95 Jeans test. The full table run
uses CFL 0.3 and bounded-inexact linear acceptance; the one-cycle test is
not a substitute for it.

The coupling runs use the same source snapshot but a separately configured
`nrfld_couple` binary (SHA-256 `d0488bcf738645343374af664014f681328698aa03e30a46edacd017033f955b`).
The low case used an isolated build (126.2 s wall time including compilation;
0.466 s solver run); the high case reused that binary (1.17 s solver run).
These are fixed-radiation-bath checks, so their zero radiation drift is not
a closed energy-conservation measurement. The single MeshBlock limits this
coupling test to one MPI rank; the Jeans tests above cover two-rank AMR.

## Controlled failed attempts

The first Jeans preflight omitted the production conservative-prototype
environment flags and correctly hit the unsupported-configuration guard.
MPI returned zero despite Athena printing FATAL, so the harness now also
checks the log and required HST. A second exploratory run used CFL 0.3 and
Newton target 1e-8 instead of the original WOR-95 CFL 0.02; Newton stopped
at final max norm 1.322e-7 after 8 iterations. It is a different, more
demanding setup and is **not** counted as a failure of the matched regression.
Its log is retained. No repeated long run was launched.

## Reproduction

From this branch, with an output directory that does not yet exist:

```bash
python3 tst/nrfld_validation/diffusion_convergence.py --max-cores 2 \
  --linear-solver fgmres --output-root /path/to/output --run-name diffusion_fgmres
python3 tst/nrfld_validation/diffusion_convergence.py --max-cores 2 \
  --binary /path/to/output/diffusion_fgmres/build_source/bin/athena \
  --linear-solver mg --output-root /path/to/output --run-name diffusion_mg
python3 tst/nrfld_validation/wor95_table_source_jeans.py \
  --output-root /path/to/output/jeans
python3 tst/nrfld_validation/run_suite.py --case couple_low --max-cores 2 \
  --mpi-ranks 1 --couple-linear-solver fgmres --no-plots \
  --output-root /path/to/output --run-name coupled_low
python3 tst/nrfld_validation/run_suite.py --case couple_high --max-cores 2 \
  --mpi-ranks 1 --couple-linear-solver fgmres --no-build \
  --binary /path/to/output/coupled_low/couple_low/build_source/bin/athena \
  --no-plots --output-root /path/to/output --run-name coupled_high
```

The [compact JSON](summary.json) contains binary/input/HST SHA-256,
iterations, residuals, mass, runtime, memory, and each PASS/FAIL decision.
Publication-wide source selection and solar/other solver regressions still
belong to WOR-114. WOR-95 remains open until that final integrated source
is fixed and compared with this exact-run source; no additional long
collapse is implied by that gate.
