# test_long3 failure investigation (2026-09-16)

Base commit: dbf2e3e2. Downloaded logs/input/results were inspected in
`test_long3/`; original user data and the parent build were preserved.
No downloaded restart was opened. Validation used `/tmp/athena_long3_validation`.

Job 314399 really used 512 MPI ranks. It completed 406 solves before hitting
the default 100-iteration Newton cap. Failed gas max=5.95306743e-3,
gas L2=1.45338567e-6, total-energy max=6.04940803e-10. The gas max/L2
ratio is essentially 4096=sqrt(256 cubed), indicating one-cell localization,
not bulk NaN/dt collapse. Job 314406 (112 ranks) instead reached its planned
wall limit normally at cycle 56, t=0.3801984901. Its subsequent known shutdown
double-free message was excluded.

## Corrections

- General-EOS thermal exchange differentiates T at the current Newton gas
  energy, rather than retaining dT/deg from step start. Ideal EOS is unchanged.
- Failure-only diagnostics identify the maximal gas-residual cell and its
  position, gas/radiation state, source, floor and old/current temperature
  derivatives. Rejected-iterate defects are reevaluated after restoration.
- Newton summaries/fatal messages include cycle/time/stage/dt; the existing
  summary prefix and residual semantics are retained.
- Makefile.in tracks headers and the generated configuration Makefile for
  all objects. This conservative, compiler-independent rule also works for
  old obj/ trees with no dependency files. The default goal stays `all`.

The old build defect was reproduced: after changing only Newton_Raphson.hpp,
`make -n` said nothing to build. After the correction/configuration update,
the build scheduled main.cpp, Newton_Raphson.cpp, nr_task_list.cpp and the
other callers, and rebuilt all 196 objects without deleting the old object
tree. dbf2e3e2 changed return-structure layout and virtual API; an older caller
object can be ABI-incompatible. Whether remote binaries were built without
cleaning remains unconfirmed without the user's build commands/artifacts.

## Tests and limitations

The nonideal-table regression differentiates the production gas residual
after changing its Newton energy, covering both cooling and heating.
Old-Jacobian relative errors are 7.6–18.2%; corrected errors are <1.6e-10.
The formal regression runner passed. It additionally checks make's header
prerequisites for main.o and the default `all` goal.

Solar settings retained EOS/opacity/profile, CFL 0.3 and physical domain.
Only grid/cycle/output/path overrides were used; output/restarts were disabled
except the deliberately small restart test.

| Case | Result |
|---|---|
| Baseline 32 cubed / 1 rank | t=3, 56 cycles; 448 Newton iterations total |
| Corrected 32 cubed / 1 rank | t=3, 56 cycles; 414 Newton iterations total |
| Baseline 64 cubed / 1 rank | t=3, 111 cycles; 878 iterations, final max <=9.09789e-7 |
| Corrected 64 cubed / 8 ranks | t=3, 111 cycles; 869 iterations, final max <=9.90233e-7 |
| Deliberately capped 32 cubed | Expected max_iterations/fatal; maximal-cell diagnostic printed after rollback |
| Corrected 32-cubed restart | Generated at cycle 2, resumed to cycle 4 with the same binary |

Local MPI waits eventually progressed; no MPI-side correction was made.
Some exploratory comparison runs were stopped to control local resource use;
they are not counted as numerical failures or completed validation runs.
Different MPI counts/host load prevent a clean before/after throughput claim.

Both baseline smaller grids reach t=3, so the particular 256-cubed failure has
not been reproduced or proved resolved. The confirmed Jacobian/build defects
are fixed, but they must not be presented as a certified explanation of that
specific failed cell. A fresh, coherent build and a matching 256-cubed,
512-rank test through >=500 cycles or t=3 are needed before unrestricted
production continuation. Neither tolerance nor iteration cap was relaxed.
Old remote-restart compatibility is not established by the small restart test.

Detailed transient history/profile/flux statistics and generated figure names
are in `test_long3/output/quicklook_plots/test_long3_diagnosis.md`.
