# WOR-92 local CFL and solver diagnosis, 2026-11-07 JST

## Source and provenance

The current Athena++ checkout is `209e22398`. The saved Dropbox bundle
`/AstroResearch/Protostellar_Collapse_Tabular_20261107` passed all 17
`SHA256SUMS` checks after download. Its README identifies a `c1d6ba6c...`
checkout with local source changes, but carries data and inputs only.
Current `209e22398` has an ideal-gas-only `protostellar_core_fld` pgen and
does not implement the input's `fgmres_flux_form_matvec` or
`fgmres_precision_floor_factor` parameters. Thus the old CFL=0.02/0.05
pilot and WOR-93 job 335063 cannot be reproduced exactly from this source.

All new runs used independent worktree
`/home/kosuke/simulation/athena_wor92_diagnosis`, branch
`codex/wor92-local-diagnosis-20261107`, and isolated builds under
`/home/kosuke/simulation/test/rad_nr/wor92_20261107/`. An **uncommitted
experimental patch** there ports the table/opacity pgen and adds opt-in
linear backward-error diagnostics, a precision-floor acceptance trial,
flux-form matvec trial, mixed-frame local gas solve and late Jacobian checks.
The complete patch is `source_trial.patch`, SHA-256
`216b2e464c20c0200969e29228b40f0517c901bee7a9033731024403bae7bacb`.
These changes are not accepted as a fix: CFL=0.3 AMR still fails.

Final experimental table binary SHA-256:
`7954e9cda7b8d9bfc243d27c1a24f0e81b9c1d61cb0dae2b973b76b78d978ece`;
matched ideal binary:
`0415ca77807aa32ca7e338457dbf19caa252ac41b6b584aa894e4dd483dca36e`.
Table SHA-256: EOS
`742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`,
opacity
`6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`.

## Configuration comparison

| Item | WOR-93 actual ideal job | Saved table pilot | Current controlled trial |
| --- | --- | --- | --- |
| EOS / opacity | ideal gamma=5/3; constant 1 cm²/g | Tomida H/He table; Semenov extinction | same table files as pilot |
| CFL | 0.3 | 0.02/0.05 passed old source; 0.3 failed | 0.02, 0.05, 0.3 |
| FGMRES target | 2e-8 | 1e-8 | 2e-8 |
| Flux-form matvec | enabled | disabled | opt-in trial; absent in current HEAD |
| Precision floor | factor 16 | factor 4 | 0 or 16; absent in current HEAD |
| Exact local gas solve | enabled for ideal gas | absent from saved input; table branch blocked by mixed frame | opt-in table mixed-frame trial |
| Mixed frame | enabled | enabled | on/off controls |
| Newton / update cap | strict 1e-5 / cap off | 1e-5 / cap off | strict 1e-5, relaxed bounds equal strict / cap off |
| Mesh | production AMR, observed max level 10 | 32³ root, 8³ blocks, two levels | 32³ root, 16³ blocks, two levels; uniform control |

The saved ideal input has flux form off and precision floor zero; WOR-93
reported runtime overrides. The current source ignores those unimplemented
input keys. Full machine-readable comparison is `configuration_comparison.csv`.

## Controlled outcomes

All table AMR runs used two local MPI ranks, 32³ root, 16³ blocks, one
refinement level, `N_J=32`, tangent correction prolongation and the same
Tomida/Semenov tables. The 0.3 cases had an actual first dt of
31.114819909 code units. The matrix below records the last failed linear
solve when the run did not reach its requested cycle count.

| Case | Reached cycle | Last true `||b-Ax||/||b||` | Local-stencil η | Result |
| --- | ---: | ---: | ---: | --- |
| table 0.02, current strict FGMRES | 0 | 5.72e-8 | unavailable | stopped at target 2e-8 |
| table 0.02, stronger preconditioner (restart 20, 2 MG cycles) | 0 | 5.72e-8 | 3.62e-16 | stopped |
| table 0.02, precision factor 16 | 3 | 4.81e-8 | 5.58e-16 | Newton strict; 8 floor acceptances |
| table 0.05, flux form + floor + local gas | 1 | 0.618 | 2.36e-6 | stopped in cycle 2 |
| table 0.3, strict, original matvec | 0 | 4.41e-7 | 1.76e-15 | stopped |
| table 0.3, strict, flux form | 0 | 1.45e-7 | 5.70e-17 | stopped |
| table 0.3, flux form + floor | 1 | 0.881 | 9.68e-6 | first Newton converged; next linear solve stopped |
| table 0.3, plus mixed-frame exact local gas | 1 | 0.881 | 9.69e-6 | same later stop |
| table 0.3, mixed frame off + local gas | 1 | 0.499 | 5.00e-9 | later stop remains |
| table 0.3, restart 20 + 2 MG cycles + local gas | 1 | 5.73e-6 | 9.98e-14 | improved, still above target |
| table 0.3, uniform grid + local gas | 3 | 1.46e-7 | 7.83e-17 | Newton strict; 10 floor acceptances |
| ideal gas 0.3, matched small AMR | 2 | 0.492 | 6.25e-6 | first two cycles strict, third stopped |

The componentwise quantity is
`max_i |b_i-(Ax)_i| / (|b_i|+Σ_j |a_ij x_j|)` for the local seven-point
stencil. AMR ghost corrections are interpolated, so it is not an exact
assembled-global-matrix backward error. A small value supports roundoff
stagnation in that local representation. It does not prove small forward
solution error. The first-step 0.3 stall has η near machine precision;
the later 0.881 stall has η about 1e-5 and cannot be accepted by the
precision floor. Simply increasing the iteration limit is unsupported.

## Coupling and Jacobian checks

In the original table branch, `exact_local_gas_solve` is gated by
`!mixed_active`; both saved inputs enable mixed-frame transport, and the
saved table input does not set this option. The experimental extension
includes the frozen mixed drift in the scalar gas residual. At CFL=0.3 it
reduced first-cycle Newton iterations from 7 to 4 and the final gas max
residual from about `7.9e-6` to `7.6e-13`. The logged local scalar residual
over gas scale was at most about `6.4e-13` in solved cells. Some line-search
trials reported zero exact cells because the current bracket did not contain
a root; this implementation still needs a robust bracket and conservation
audit. The next-cycle linear failure was unchanged.

The opt-in finite-difference Jacobian check in the failing table run sampled
3,375 interior cells of MeshBlock 0 at cycle 1. Gas L2 relative discrepancy
scaled `1.187e-4 → 1.187e-5 → 1.190e-6` as epsilon decreased by tenfold;
radiation L2 discrepancy was `6.16e-11 → 6.03e-10 → 6.13e-9`, consistent
with a roundoff floor. FGMRES matvec additivity/homogeneity errors at the
failing cycle were `2.63e-15` and `5.00e-15`. This supports local
derivative and operator linearity at that sample; it does not check
coarse/fine interface rows or preconditioner quality.

Semenov's published Planck mean is extinction while the Optab quantity is
absorption. Their shared-range difference and the inaccessible 14 K Optab
state remain a separate physical-definition problem. Neither the table
sidecars nor these logs show a table-domain exception as the direct stop.
The EOS sidecar reports positive, finite, temperature-monotone P/e and
conservation checks; an independent thermodynamic reference is still needed.

## Reproduce and next experiment

```bash
cd /home/kosuke/simulation/test/rad_nr/wor92_20261107/bundle
sha256sum -c SHA256SUMS
cd ..
python3 make_inputs.py
git -C /home/kosuke/simulation/athena worktree add --detach \
  "$PWD/reproduction_source" 209e22398
git -C reproduction_source apply "$PWD/source_trial.patch"
mkdir -p build_source
cp -a reproduction_source/{configure.py,Makefile.in,src} build_source/
cd build_source
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -b -hdf5 -mpi --cflag=-O3
make -j2
cd ../bundle
mkdir -p ../reproduction_late_jacobian_output
mpirun -np 2 ../build_source/bin/athena \
  -i ../cfl030_late_jacobian/input.used \
  -d ../reproduction_late_jacobian_output
cd ..
python3 analyze.py
```

The command above reproduces the selected
late-Jacobian case; `make_inputs.py` and each `input.used` preserve the
other controls. `configure.log`, `build_late_diagnostic.log`,
`configure_ideal.log`, and `build_ideal.log` retain build output.
Each case has `input.used`, `run.log`, and output. `summary.json`,
`experiments.csv`, `configuration_comparison.csv`,
`residual_controls.png`, `source_trial.patch` and this report are the
reproduction set. The JSON, CSV and figure are also versioned beside this
report in `wor92_results_20261107/`. Early trial binaries were replaced during iterative
diagnostics; their exact source variants are described by the log stage and
final patch, but their binary SHA-256 is unavailable.

Next: recover the full WOR-93/WOR-92 source snapshots and runtime inputs;
then measure the assembled AMR-interface `Jv`/true residual and MG
preconditioner action at the failed cycle, including level/rank localization.
Repeat 0.3 with the same source and mesh for ideal/table physics only after
that audit. Check EOS P/e/derivatives against an independent reference and
resolve Planck absorption separately. No XD2000 run or long collapse is
justified by these local results. WOR-92 remains open.
