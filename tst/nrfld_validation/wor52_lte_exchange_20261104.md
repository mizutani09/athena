# WOR-52: local LTE exchange and energy-balance validation (2026-11-04)

## Changes

- With `fld/cut_diff=true`, Marshak face fluxes are now disabled along with
  interior diffusion. Previously the nominally local coupled RSG test retained
  an outer-boundary sink and stalled at a total-energy residual of `3.63e-8`
  for CFL 0.01.
- For a coupled table EOS with no diffusion, the local update solves the gas
  equation using `Er = Er_old + (egas_old - egas)`. This preserves the small
  radiation state when gas energy is much larger, and enforces the local
  total-energy constraint. The existing fixed-Er and full-FLD paths remain
  on their bracketed gas solve.
- Set `ATHENA_NRFLD_ENERGY_LEDGER=1` to print each committed implicit stage's
  gas change, radiation change, diffusion change, physical Marshak contribution,
  and balance. It uses the gas state before `UpdateHydroVariables` overwrites it.

## Independent single-cell reference

Run:

```text
cd tst/regression
python3 run_tests.py nrmgfld/local_lte_exchange --hide_make
```

The regression calls the production `NRFLD::AddDifference`, and independently
bisects the scalar EOS equation in Python. It covers 18 synthetic nonideal-table
cases (including a temperature-slope knot) and 22 material-only Tomida table
cases (including a table knot and the WOR-52 stiff-cell parameters). It checks
fixed-Er and coupled modes, positivity, table coordinates, and gas/radiation
balance with prescribed external input and boundary loss. Maximum relative gas
root error was `1.192e-15`; maximum normalized source/flux balance error was
`2.220e-16`. A separate cell-face check found zero defect when diffusion is
cut, and a vanishing defect when an active Marshak flux is balanced by the
radiation transient. Both face checks use a `1e-14` limit.

For the reported NATA failure cell, `rho=124.966319114`,
`egas_old=453.487553374`, `Er=7.14792218147`, and
`dt*c*sigma_P=6.91260968506e12`, the double-precision root neighbors are
`5.684e-14` apart. With `a_r=7.5657e-20`, the reference tool gives a best raw
residual of `-0.432` (normalized `9.49e-4`), while one ULP of either radiation
energy corresponds to `0.00614` in the multiplied source. The backward-error
criterion is therefore necessary even when the state is converged to a ULP.

## RSG production configuration and runtime checks

Build and input:

```text
scripts/configure_rsg.sh
inputs/rsg/athinput.st35gm03n07_fld
```

The input uses the material-only EOS and gray opacity, six-face Marshak vacuum,
`exact_local_gas_solve=true`, `backward_error_convergence=true`, Newton target
`1e-8`, FGMRES relative target `1e-4`, and production CFL `0.3`. The one-step
series had actual first-step sizes `0.221332548`, `0.663997645`, `2.213325482`,
and `6.639976445` for CFL `0.01`, `0.03`, `0.1`, and `0.3` respectively.

| Mode | Four-CFL one-step result | Worst final Newton max | Worst implicit energy balance |
| --- | --- | ---: | ---: |
| fixed Er, no diffusion | all strict converged | `2.49e-9` | fixed radiation bath, not conserved |
| coupled, no diffusion | all strict converged | `5.52e-13` | `7.58e-18` absolute relative |
| coupled, full FLD | all strict converged | `7.63e-9` | `2.40e-11` absolute relative |

The full-FLD energy ledger includes a nonzero outward Marshak term; at CFL 0.3
it is `-7.903e6` in code-energy units, and the interior diffusion net is
`-3.55e-11`. The coupled no-diff ledger reports exactly zero Marshak term.
The ledger compares states immediately before and after the implicit solve;
external hydro heating/gravity/drag have already contributed to its old state.
It measures the exchange/diffusion stage, not the entire open-boundary hydro
step.

At a common code time `t=2`, the full-FLD and coupled no-diff CFL series also
strictly converged (10, 4, 1, and 1 steps). The one-step series above is needed
to exercise the actual CFL 0.3; at `t=2` both CFL 0.1 and 0.3 are capped by
the final-time limit. The maximum absolute relative implicit energy-balance
errors across all `t=2` stages were `9.06e-13` with full FLD and `8.97e-18`
without diffusion. The full-FLD final field differences relative to CFL 0.01
are recorded by variable in the JSON summary; the largest ratios are in the
velocity field, whose small normalization makes them less useful as a
convergence measure.

Reproduce the logged series with `tools/rsg/validate_lte_cfl.py` using
`--mode full|coupled_local|fixed_local`, `--one-step`, and `--energy-ledger`
for coupled modes. The script records exact commands, inputs, binary hash,
strict-solver status, field differences, and energy-ledger tolerances in
`summary.json`.

Related NR-FLD regressions passed: `general_eos_exchange_jacobian`,
`adiabatic_gradient` (table and ideal EOS), `coupled_residual_norms`, and
`coupling_mode`.
