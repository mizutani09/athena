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

The fresh-map launcher is
`analysis/wor182/xd_q025_corot_tabulated_eos_gate.sh`; continuations use
`analysis/wor182/xd_q025_corot_tabulated_eos_restart_gate.sh`.

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

### One hundred steps, job 347863

- fresh map, CFL=0.1, 100 steps to t=1e-3
- completed normally in 209.87 s
- all Newton solves converged; final max residual 4.06e-11
- relative drift: mass -4.68e-13, angular momentum +1.76e-10,
  gas+radiation energy +1.55e-9
- interior/surface radial Mach: 4.70e-6 / 1.245e-3
- full-force relative maximum: 0.144

Commit `dfcb9836c` moves the table-temperature-floor fatal report outside the
OpenMP SIMD row, allowing the production Intel SIMD build without weakening
the failure contract.  Commit `83ea88750` limits the 2019 legacy general-EOS
high-order warning to the old PPM/4 paths; 4wz has EOS-aware point conversion
and an explicit density/pressure quality fallback.  Intel SIMD build job
347865 completed, and its 10-step numerical gate 347868 reproduced the
non-SIMD residual history in 79.18 s versus 101 s for job 347862.

### Startup release, job 347871

- Intel SIMD build, fresh map, CFL=0.1
- completed 110 steps to t=1e-2 in 209.81 s
- after the t=1e-3 cap, dt increased through the normal Athena factor-of-two
  growth limiter; every solve converged in 3--5 Newton iterations
- final Newton max residual: 5.68e-9
- relative drift from t=0: mass -1.40e-12, angular momentum +1.76e-9,
  gas+radiation energy +4.09e-9
- interior/surface radial Mach: 4.67e-5 / 1.300e-3
- full-force relative maximum: 0.174

### CFL=0.2 nonlinear-floor diagnosis and gate

The first continuations from t=0.01 (jobs 347886 and 347887) reached t=0.1,
but the final `dt=0.04332` solve was accepted only at the relaxed tolerance:
L2=3.96e-7 and max=1.72e-4.  Tightening FGMRES from 1e-4 to 1e-5 did not
change that residual, ruling out linear-solve accuracy as the cause.

The verbosity-2 reproduction, job 347889, localized the plateau to a
total-energy row next to a radiation cell already clipped to `TINY_NUMBER`.
The linear solve repeatedly requested a negative correction in that floor
cell, while positivity clipping applied zero; neighbouring rows therefore
used a correction that could never be realized.  Commit `9559f3ccb` treats
this lower bound as a Newton active-set constraint (`delta_Er=0`) and uses a
projected residual for the constrained radiation equation.

Job 347891 repeated exactly the CFL=0.2 path after this change:

- completed normally from t=0.01 to 0.1 in 64.35 s
- all three timesteps converged normally; there was no relaxed acceptance
- the previously stalled final timestep converged in 7 Newton iterations
- final L2/max residual: 2.53e-12 / 1.48e-9
- FGMRES retained its production 1e-4 relative threshold

### CFL=0.3 and 4wz AMR ghost admissibility

The first full CFL=0.3 timestep, `dt=0.1216300565`, converged in 7 Newton
iterations.  Before the following step, jobs 347892 and 347895 exposed a
separate explicit-stage issue: multidimensional coarse/fine interpolation
made finite negative radiation-energy ghost values near -9.3e-10, although
the owning active cells remained positive.  Commit `dc0f01759` bounds direct
FLD prolongation, and commit `667fb5a2c` enforces ghost-stencil admissibility
at the 4wz radiation reconstruction consumer while retaining fatal checks for
all negative active cells and non-finite ghosts.

Job 347897 then completed from t=0.1 to 0.3:

- 3 M-test-a nodes, 168 MPI ranks, 47.89 s wall time
- `dt=0.1216300565` and `dt=0.0783699435` both converged in 7 Newton iterations
- final max residuals: 2.62e-9 and 1.27e-9
- no negative-radiation, relaxed-Newton, or table-domain warning
- the production CFL=0.3 gate is therefore passed through t=0.3

Job 347898 continues this accepted CFL=0.3 state to t=3.0 as the next
medium-duration gate.

## Open acceptance items

1. Complete the t=3 CFL=0.3 gate, then extend to a material fraction of a
   dynamical time with periodic restarts.
2. Compare L3/L4 and ideal/table force, Mach, Newton, and FGMRES histories.
3. Add explicit runtime diagnostics for distance to EOS table boundaries.
4. Replace cell-average closure with a derived, table-thermodynamic profile if
   the first A/B shows appreciable thermal projection noise.
5. Add a focused AMR regression for positivity-constrained FLD ghost
   prolongation and the Newton radiation active set.
