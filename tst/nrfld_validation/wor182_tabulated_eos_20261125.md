# WOR-182 rotating FLD star with a material tabulated EOS

## Scope

This branch adds a fresh-map path for the accepted q=0.25 RLOF-synchronous,
rotating-atmosphere, 4wz, full mixed-frame FLD star using Athena++'s general
EOS interface.  It is intentionally separate from `wor-182-rotating-full-fld`;
an ideal-EOS restart must not be reinterpreted with a table EOS.

## EOS choice

Use `data/eos/rsg_tomida2013_material_eos.tab` (SHA-256
`7051d1c5222f3ca7dddbbba1c48d473cb39bc5e76badb869b082625473a71576`).
Its metadata declares `material_only=true` and excludes photon pressure,
energy, and entropy.  Do not combine `rsg_nata_eos.tab` with FLD: that table
contains photon terms and would double count radiation.

The Tomida table assumes X=0.70, Y=0.28 and omits metals.  It is therefore a
numerical integration candidate, not yet an exact MESA-composition EOS.

## Implementation

Branch: `wor-182-tabulated-eos`

- `scf_rsg.cpp` solves
  `P_material(rho,e)+a*T(rho,e)^4/3=P_SCF` for every stellar cell average.
- Conserved gas energy and the FLD gas cache use the table internal energy.
- Opacity temperature and output temperature use `TempFromRhoEg`.
- Mach diagnostics use `AsqFromRhoP`.
- The accepted atmosphere-matched profile, rho=1.7e-11 g cm^-3, and a
  corotating numerical atmosphere are materialized in the dedicated input.
- The atmosphere keeps the accepted endpoint pressure 22.24714707 dyn cm^-2.
  The material table maps this to about 1.08e4 K, preserving mechanical rather
  than exact 1e4-K matching for the first A/B gate.

Configure:

```text
--prob=scf_rsg --coord=cartesian --eos=general/eos_table --flux=hllc \
--nscalars=2 --grav=mg --nghost=4 -nrmgfld -hdf5 -mpi
```

## XD gates

The launcher is `analysis/wor182/xd_q025_corot_tabulated_eos_gate.sh`.

### Fresh map, job 347859

- 3 M-test-a nodes, 168 MPI ranks
- `nlim=0`
- completed normally in 69.31 s
- all LTE pressure splits succeeded; no NaN, table-domain, or fatal error
- final restart size: 1.680 GB

### One step, job 347860

- 3 M-test-a nodes, 168 MPI ranks
- fresh map, CFL=0.1, dt=1e-5
- completed normally in 37.66 s
- Newton: 3 iterations, final L2 2.48e-10, final max 7.64e-9
- each FGMRES solve: 2 iterations; relative residual <=1.46e-7
- mass drift after the step is approximately -2.3e-13
- interior radial Mach remains 3.60e-6; surface-taper maximum changes only
  from 1.24035e-3 to 1.24040e-3
- full-force relative maximum decreases from 0.442 to 0.139 during the first
  adjustment step

### Ten steps, job 347862

- fresh map, CFL=0.1, 10 steps to t=1e-4
- all Newton solves converged: 3 iterations on step 1, then 2 iterations
- final Newton max residual stayed below 9.84e-11 after step 1
- mass drift remained approximately -2.3e-13
- interior/surface radial Mach at t=1e-4: 3.60e-6 / 1.24085e-3
- full-force relative maximum: 0.1393

Job 347863 extends the fresh-map gate to 100 steps and t=1e-3.

## Open acceptance items

1. Complete 10/100-step startup gates, then raise CFL 0.1 -> 0.2 -> 0.3.
2. Compare L3/L4 and ideal/table force, Mach, Newton, and FGMRES histories.
3. Add explicit runtime diagnostics for distance to EOS table boundaries.
4. Replace cell-average closure with a derived, table-thermodynamic profile if
   the first A/B shows appreciable thermal projection noise.
5. Resolve or narrowly qualify the legacy general-EOS + xorder>2 warning;
   WOR-120's density-cliff fallback is present, but this integrated combination
   still needs a longer gate.
