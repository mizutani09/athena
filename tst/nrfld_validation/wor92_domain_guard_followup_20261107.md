# WOR-92 continuation: local gas root, bounded Newton, opacity domain, and AMR ghost positivity

## Adopted local pilot components

The clean Tomida table problem generator, Semenov opacity callback, mixed-frame general-EOS exact local gas root, bounded bracket expansion, and bounded true-residual inexact-Newton gate are committed on the **dedicated trial branch** `codex/wor92-domain-guard-trial-20261107` as `6da81d20`. This is the exact [candidate patch](wor92_clean_localroot_results_20261107/candidate.patch) previously rebuilt and tested for 30 early CFL=0.3 cycles: all 30 Newton solves converged, the earlier copied-build trial's full numerical trace and history matched, and ten-cycle state arrays were bitwise identical [clean integration audit](wor92_clean_localroot_20261107.md). The gate is opt-in through `ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7`, uses independently recomputed true FGMRES residual, and leaves the requested strict target `2e-8` unchanged. This is a validated **local pilot**, not a claim that the full collapse meets the physics and convergence criteria.

For trial opacity lookups, `fld/opacity_table_domain_policy=clamp` now has a separately opt-in active-cell accepted-state check (`fld/opacity_accepted_state_guard=true`, commit `ff632195`). It checks gas temperature and density against the actual table coordinate bounds after the implicit step and rejects any accepted state outside the table. It also reports the global minimum accepted active temperature and aggregate clamp counters. Both settings must be enabled together. This distinguishes a transient Newton lookup outside the table from a physical accepted state outside the model domain. The check is before AMR regridding, so a post-regrid validity audit remains necessary.

At the saved cycle-74 checkpoint, CFL=0.3 with the clamp and guard reaches cycle 82. All nine Newton solves converge, with 560 table-outside trial lookups clamped and no accepted active cell below the 5 K table minimum (minimum 11.493790 K through cycle 81). A same-binary guard-off control has identical cycle, Newton and FGMRES lines and a byte-identical nine-row history file. Thus the guard itself does not perturb the numerical trajectory. Clamping is currently an exploratory way to let trial evaluations proceed; the number of clamped calls and their effect on accepted trajectories still need a science accuracy assessment.

## Next stop exposed at cycle 87

Extending the same CFL=0.3 run beyond the opacity stop reaches cycle 87, then fails **before the first Newton iteration** when the initial NR-FLD coefficient calculation sees negative radiation energy. A diagnostic-only location patch identifies `Er=-0.0730309` at `(k,j,i)=(1,2,2)` on `gid=7` and `Er=-0.0730431` at `(9,1,2)` on `gid=67`. Both are **ghost cells on level-3 blocks facing level-2 neighbors**; neither is an accepted active cell. The generic three-dimensional coarse-to-fine prolongation limits each directional slope with minmod, but the sum of three bounded directional contributions can still exceed the parent value. A positive coarse mean therefore does not guarantee positive individual fine ghosts. The failed original and ghost-limited runs have identical cycle, Newton and FGMRES lines, and byte-identical history through cycle 86.

Two isolated, opt-in boundary trials were made; their source patch is archived and remains **uncommitted**:

| Trial | Method | CFL=0.3 outcome from cycle 74 |
| --- | --- | --- |
| `ATHENA_WOR92_POSITIVE_NR_GHOST=1` | Floor only negative coarse-to-fine NR radiation ghosts at `TINY_NUMBER` | 26/26 Newton-converged steps through cycle 99; 48 ghost values floored in cycles 87–88; cycle-100 limit reached. |
| `ATHENA_WOR92_POSITIVE_NR_GHOST=2` | Scale all eight children about their mean until the minimum is positive, preserving that coarse-group mean | 26/26 Newton-converged steps through cycle 99; limiter activated in cycles 87–88; cycle-100 limit reached. |

Both runs have 576 clamped opacity lookups, no negative/nonfinite opacity result, and accepted-state table-domain checks pass. Their 27-row histories agree byte-for-byte before cycle 87, then differ after the ghost correction. At the last history row (`t=2266.63` code units), the printed `rho_max=3.80798e-14 g/cm³`, `Tgas,max=20.5938 K`, gas mass, and gas/radiation energy columns are the same to the history file's limited printed precision. The angular-momentum column differs by about `0.119%` between the two boundary treatments; this is a **material numerical sensitivity**, and the rank face/energy ledger has not yet been run for either. The mean-preserving method has the stronger local conservation property, but its full AMR face flux and state effects need testing before it can be adopted. Neither ghost trial is committed or pushed as solver source.

A CFL=0.15 control from the same checkpoint, with the clamp and accepted-state guard but **no** ghost limiter, reached cycle 110 and code time `2202.91`, beyond the CFL=0.3 failure time `2156.43`. All 37 Newton solves converged. This indicates the first negative ghost is sensitive to the evolved spatial profile and timestep; it is not a matched-time CFL convergence test because the two trajectories and cycle counts differ.

## Status and next checks

The direct early strict-FGMRES precision barrier is analyzed separately in [roundoff mechanism](wor92_roundoff_mechanism_20261107.md). The adopted pilot's inexact gate permits Newton to converge despite that inner barrier, with an explicit cap. The new opacity guard passes the local continuation, and a positive AMR ghost reconstruction can remove the next stop. These are distinct numerical issues. The Semenov/Optab opacity-definition difference remains a separate physical-model issue.

WOR-92 remains **In Progress**. The next verification is a rank-by-rank AMR face and total-energy ledger across cycles 86–100 with the mean-preserving limiter, plus higher-precision state comparisons and a post-regrid domain check. An invalid Newton trial should be backtracked or rejected without silently accepting an out-of-domain state; a full `dt` reduction/retry mechanism also needs validation. Continue from a saved state to much higher density and compare matched-time CFL and mesh resolutions. The last density above is still about 2,600 times below `1e-10 g/cm³`, so it is not first-core validation. The trial opacity extrapolation/clamp policy needs an accepted-state physical budget before long runs.

## Reproduction

The [summary JSON](wor92_domain_guard_followup_results_20261107/summary.json), [run CSV](wor92_domain_guard_followup_results_20261107/runs.csv), [accepted-temperature CSV](wor92_domain_guard_followup_results_20261107/accepted_temperatures.csv), [figure](wor92_domain_guard_followup_results_20261107/continuation.png), [logs, exact input, history and compressed ghost patch](wor92_domain_guard_followup_results_20261107/evidence.tar.gz), and [audit generator](wor92_domain_guard_followup_20261107.py) contain the evidence. Regenerate the artifacts with:

```bash
python3 tst/nrfld_validation/wor92_domain_guard_followup_20261107.py
```

Source commits: `6da81d20` and `ff632195`, based on `6ac9288f`. Final isolated trial binary SHA-256: `e7854e44e35a3ed85f9ab1963a801c1b09a6736db7356c6ff53dd2c0519217da`. The saved cycle-74 checkpoint SHA-256 is `3336de5537d7d0f5b8a1f9b7941727d202a980dac81bfdd088611edaee215240`; the guard input SHA-256 is `7cd5ee8a6a3fe749624985b75dbafd000f34e713b9b2f723a191689bbef09cd0`. The checkpoint comes from the earlier [restart audit](wor92_inexact_newton_20261107.md); no source checkout or output was overwritten.

From `/home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle`, use a fresh `-d` directory:

```bash
env ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1 \
  ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1 \
  ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1 \
  ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1 \
  ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1 \
  ATHENA_NRFLD_EXPERIMENTAL_REGRID=1 \
  ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0 \
  ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7 \
  ATHENA_WOR92_POSITIVE_NR_GHOST=2 \
  timeout 150s mpirun -np 2 \
  /home/kosuke/simulation/test/rad_nr/wor92_operator_audit_20261107/build_table_domain_guard_trial/bin/athena \
  -r /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/table_floor32_restart30_to80_2rank/cfl030_flux_precision16_local.00002.rst \
  -i /home/kosuke/simulation/test/rad_nr/wor92_wor105_retest_20261107/input_table_domain_guard_cfl030_restart.used \
  -d /path/to/new/output time/nlim=100 time/tlim=10000.0
```

For the guard-off control, omit the ghost environment flag, set `fld/opacity_accepted_state_guard=false`, and use `time/nlim=82`. For the CFL control, omit the ghost environment flag and set `time/cfl_number=0.15 time/nlim=110`. The ghost source patch in the archive applies after commit `ff632195`; it is a diagnostic/trial patch. Do not set `ATHENA_WOR92_FIXED_ROSSELAND` in a table-opacity run. No XD2000 access or merge occurred.
