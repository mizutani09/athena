# NRFLD validation runner

This runner automates the diffusion and matter-radiation coupling tests and
organizes radiative-shock results.  It records reproducibility metadata and
keeps each test's plotting implementation in a separate file.

Run both initial cases with at most four build jobs or MPI ranks:

```bash
python tst/nrfld_validation/run_suite.py --max-cores 4
```

Run one case:

```bash
python tst/nrfld_validation/run_suite.py --case diffusion --max-cores 4
python tst/nrfld_validation/run_suite.py --case couple --max-cores 4
```

Results are written under `~/simulation/test/rad_nr/runs/<timestamp>/` by
default.  Use `--output-root` or `--run-name` to change this location.  Every
case is built in its own `build_source/` snapshot, so `configure.py`,
`Makefile`, `src/defs.hpp`, `obj/`, and `bin/` in the parent repository are not
modified. The snapshot includes uncommitted source changes. Each case contains
its pgen-specific executable, `input.used`, build/config/run logs,
`manifest.json`, raw output, and an optional figure. `summary.json` collects the
final metrics. Plot implementations are kept separately in
`plots/plot_diff.py`, `plots/plot_couple.py`, and
`plots/plot_radiative_shock.py`.

`--max-cores` is a hard upper bound for both `make -j` and MPI ranks.  The
actual MPI count is also limited by the number of MeshBlocks.  Cases run
sequentially because they use different problem generators.

Useful development options:

```bash
python tst/nrfld_validation/run_suite.py --help
python tst/nrfld_validation/run_suite.py --case diffusion --no-plots
python tst/nrfld_validation/run_suite.py --case diffusion --no-build \
  --binary /path/to/athena
```

`--no-build` checks the selected executable with `athena -c` and rejects a pgen
mismatch before running it. If `--binary` is omitted, `bin/athena` is used as
the explicit legacy default. Use one calculation case per `--no-build`
invocation. The manifest records input, binary and external-file SHA-256
digests, effective settings, compiler/MPI versions, a limited environment
allowlist, timeout/exit records, and separate analysis/calculation provenance.
Missing references and timeouts still leave a final manifest.

## Radiative shocks downloaded from another machine

Radiative shocks are plot-only by default. The suite does not configure,
compile, or launch Athena++ unless `--run-radiative-shock` is explicitly given.
Point it at a downloaded run directory containing `output/*uov_x*.athdf`:

```bash
python tst/nrfld_validation/run_suite.py \
  --case radiative_shock_mach2 \
  --radiative-shock-mach2-dir ~/simulation/test/rad_nr/test_20260811/radiative_shock/mach2 \
  --radiative-shock-mach2-analytic ~/simulation/test/rad_nr/test_20260723/test_radiative_shock_mach2/mach2_semianalytic_bvp.csv \
  --max-cores 4
```

Mach 5 uses the corresponding `--radiative-shock-mach5-dir` and
`--radiative-shock-mach5-analytic` options. The large HDF5 results remain in
their downloaded location. Small reproducibility files (`athena`, `athinput*`,
and `problem_parameters.txt`) are copied into the validation case directory.

To deliberately run the expensive calculation locally, add the explicit flag:

```bash
python tst/nrfld_validation/run_suite.py \
  --case radiative_shock_mach2 --run-radiative-shock --max-cores 4
```

For a short smoke test, also pass `--radiative-shock-nlim 2`. Without that
option, the time and cycle limits in the selected input file are retained.

The shock plotter can also be used independently:

```bash
python tst/nrfld_validation/plots/plot_radiative_shock.py \
  --case mach2 --run-dir ~/simulation/test/rad_nr/test_20260811/radiative_shock/mach2 \
  --analytic ~/simulation/test/rad_nr/test_20260723/test_radiative_shock_mach2/mach2_semianalytic_bvp.csv
```

Mach 2 and Mach 5 use shared loading and AMR-profile merging logic. The Mach 5
plot enables a temperature inset by default. Use `--no-inset` or `--inset` to
override this behavior.
