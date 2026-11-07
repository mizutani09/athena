# WOR-92: clean isolated integration of the table local root and inexact gate

The Tomida-table problem generator, Semenov opacity callback, mixed-frame general-EOS local gas root, bounded bracket expansion, and trial true-residual cap were applied as one [candidate source patch](wor92_clean_localroot_results_20261107/candidate.patch) to a **new isolated Git worktree** at base commit `9c584cc107a4303476f3d36dc180b57d0c1dfc27`. The source was compiled from a fresh copy of `configure.py`, `Makefile.in`, and `src` with the table EoS, MPI, HDF5, and NR-FLD settings below. This corrects the earlier clean-integration gap in which the local-root source compiled but the committed ideal-gas problem generator could not initialize the Tomida input.

The fresh binary SHA-256 is `3cda116485bccaac0340be6b47bb1e546f2cacef5582a242505aa4be59435571`. Its CFL=0.3, root-32³, small-AMR, two-rank, 30-cycle run with exact local gas solve, precision floor **disabled**, and true-residual cap `1e-7` completed all 30 Newton solves. Every FGMRES line, Newton-converged line, and cycle line was identical to the previous copied-build trial with the same gate. The 30-cycle history file was byte-identical. The three-cycle same-binary control also matched the previous trial's first three cycles. The bracket change therefore does not alter these ordinary early-collapse cases. Prior synthetic mixed-frame stress testing is documented in [bracket trial](wor92_local_bracket_trial_20261107.md).

The fresh source does not include the earlier copied-build rank face/energy ledger instrumentation. Equality of the numerical trajectory supports transfer of that run's early-state results; it is not a separate conservation measurement. The [source patch SHA-256](wor92_clean_localroot_results_20261107/summary.json) is `e401925ac405c0b83f763a5d5bdd3f82d1b48142afe3c945db9928da8d6a288a`. The [summary JSON](wor92_clean_localroot_results_20261107/summary.json) and [build logs, input, full run logs and patch](wor92_clean_localroot_results_20261107/evidence.tar.gz) provide hashes. Regenerate the comparison with `python3 tst/nrfld_validation/wor92_clean_localroot_20261107.py` from the report worktree.

Rebuild from a fresh worktree at the listed base commit, applying the archived patch with `git apply`, then copy `configure.py`, `Makefile.in`, and `src` into a new isolated build directory. From that build directory:

```bash
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -b -hdf5 -mpi --cflag=-O3
make -j4
```

The 30-cycle run command from `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle` was:

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7 \
  timeout 300s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_clean_localroot_trial/bin/athena \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_inexact_cfl030_30cycles.used \
  -d /path/to/fresh/output
```

The source remains uncommitted on the dedicated trial worktree `codex/wor92-local-root-clean-trial-20261107`; the patch is archived in the report. This is a reproducible **pilot candidate**, not a verified full-collapse production fix. The [inexact-Newton audit](wor92_inexact_newton_20261107.md) records failure under tighter nonlinear tolerance, CFL=0.15, the later opacity-domain guard, and the remaining conservation/physics checks. WOR-92 remains **In Progress**. No XD2000 access or merge occurred.
