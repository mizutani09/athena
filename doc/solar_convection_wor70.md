# WOR-70: non-magnetic local solar convection with NR-FLD

Status: commissioning and long-run test, 2026-11-04. This note separates
solver throughput, hydrodynamic stability, thermal relaxation, and physical
agreement with the Sun. The present run is a **1.536 Mm cubic pilot**, not a
Rempel-sized solar surface box.

## Physical target and reference

Rempel (2014), [ApJ **789**, 132](https://arxiv.org/abs/1405.6814), §II.2,
uses horizontal periodic boundaries and a top open for outflows but closed for
inflows. Its HD2 lower boundary prescribes upflow entropy and controls the
mean pressure while damping pressure fluctuations; the incoming entropy is
chosen to yield approximately the solar flux. The small reference domain is
6.144 × 6.144 × 3.072 Mm. The paper begins most magnetic runs from a thermally
relaxed **non-magnetic** convection state. The `789, 22` citation in the local
untracked `athinput.nrfld_solar_convection` is a bibliographic typo; the paper
is 789, 132.

The tracked `simple_convection_lhllc_fld` generator includes an HD2-like
bottom, a top outflow mode, tabulated general EOS, tabulated Planck/Rosseland
opacity, and gray FLD. The supplied production input uses a 3.072-code-unit
cube with `leng_unit=5e7 cm`, i.e. 1.536 Mm on a side. At 64³ the spacing is
24 km. This is useful for checking the coupled solver and early flow, but is
too small to establish Rempel-sized granulation statistics. Rempel's radiative
transfer uses short characteristics; gray FLD makes a different photospheric
approximation. Agreement with a flux imposed by boundary control is therefore
not independent validation. We must also compare temperature/velocity
stratification, radiative cooling, optical-depth position, and resolution.

## Reproducible solver commissioning

- Git source base: `209e2239` (`origin/nr_mgrad_dev`), isolated branch
  `codex/wor70-solar-convection`; configured with
  `--prob=simple_convection_lhllc_fld --eos=general/eos_table --flux=lhllc
  -nrmgfld -hdf5 -mpi --cxx=g++`. Magnetic fields are **off**.
- Input: `inputs/radiation/athinput.simple_convection_lhllc_fld_general_hpc_256`,
  with explicit `nrfld/linear_solver=mg|fgmres`, rediscretized MG hierarchy,
  and `fgmres_threshold=1e-8`. The three external EOS/opacity/profile files
  are linked in the output root's `data/`; their SHA-256 hashes are recorded
  in `tst/nrfld_validation/total_energy_residual_validation.md` and by the
  new campaign runner.
- Local CPU availability: 16 logical CPUs, eight physical cores, no cgroup CPU
  quota. `OMP_NUM_THREADS=1` causes GNU `nproc` to print one; MPI was tested
  at 12 ranks using `mpirun --use-hwthread-cpus --bind-to hwthread --map-by
  hwthread`. The WSL memory limit is about 19 GiB with 64 GiB swap.
- Output: `/mnt/d/output/rad/wor70_solar_20261104/`.

| 3D case | solver | result | wall | slow MG warnings | physical time reached |
|---|---|---|---:|---:|---:|
| 32³, 1 rank, CFL 0.3, 10 cycles | FGMRES | completed | 4.04 s | 0 | 0.527 code |
| 32³, 1 rank, CFL 0.3, 10 cycles | MG | completed | 6.01 s | 0 | 0.527 code |
| 64³, 12 ranks, CFL 0.3, 30 cycles | FGMRES | completed | 13.94 s | 0 | 0.807 code |
| 64³, 12 ranks, CFL 0.3, 30 cycles | MG | completed | 24.53 s | 0 | 0.807 code |
| 64³, 12 ranks, CFL 0.5 | FGMRES | stopped near cycle 15 | — | 0 | 0.666 code |
| 64³, 12 ranks, CFL 0.3, to `t=10` | FGMRES | completed, 369 cycles | 4:33 | 0 | 10 code = 3.73 min |

All 369 nonzero-step Newton solves in the `t=10` run reported `converged`,
with 1,026 FGMRES calls; no relaxed acceptance, divergent V-cycle, or fatal
error was reported. On the common 64³ 30-cycle pair, the stored history's
final top total flux differs by only 2.2e-6 relative between solvers. The
short comparison supports FGMRES throughput on **this** setup; it is not a
speed ratio against an old binary. Historical MG runs under
`/mnt/d/output/rad/test_20260903/conv/` do exhibit frequent slow-convergence
and divergent-V-cycle messages, but their inputs and code versions differ.

A second controlled pair restarted the **same** developed FGMRES checkpoint
near `t=60` and ran 24 cycles to `t=60.4` at CFL 0.3 on four MPI ranks confined
to CPUs 12–15. MG completed in 48.34 s and FGMRES in 66.58 s. Both reported
24/24 converged Newton steps, zero slow-MG warnings and zero fatals; the final
stored top flux differs by only `4.2e-6` relative. Thus the early FGMRES speed
advantage does **not** persist for every state or rank count. MG uses its
`threshold=0` automatic V-cycle stopping rule, while FGMRES uses a relative
linear residual target of `1e-8`; both passed the same nonlinear acceptance
criterion. The comparison establishes successful solver operation, not a
universal FGMRES speedup. The 12-rank long run was isolated on CPUs 0–11
while this pair used CPUs 12–15; shared memory bandwidth may still affect
absolute timings.

The CFL 0.5 run reached converged Newton solves immediately before radiation
energy became negative in the stage update. This does not isolate the source
of negativity or show a universal 3D CFL ceiling. The current RK2 code allows
`cfl_limit=1`; CFL 0.3 is the empirically safe commissioning value here.
From the developed `t≈60` checkpoint on four ranks, FGMRES also completed
18 cycles at CFL 0.4 and 15 cycles at CFL 0.5 to the same `t=60.4`, with every
Newton solve converged and no fatal error. These are short restart tests, not
long-run stability evidence. The different initial and developed outcomes
point to state dependence; they do not justify raising the production CFL yet.

At `t=10`, the last-quarter mean top flux is only 0.322 times the solar
reference `6.3e10 erg cm^-2 s^-1`, compared with 0.416 in the preceding
quarter. The mean vertical RMS speed fell from 1.21 to 1.05 km/s. Each
quarter covers less than 0.05 vertical crossing times. These are **transient**
measurements and do not establish solar agreement or statistical stationarity.
The `t=10` xz slice is still dominated by horizontally coherent vertical
structure; it is not evidence of developed granulation. The first snapshot
has a solar-size top flux by construction, which then declines, so its initial
agreement must not be presented as a validation result. At fixed initial
optical-depth-one height `z≈2.0` (not a recomputed optical-depth surface),
the slice-mean gas temperature falls from about 5,520 K in the first fifth
of output frames to about 3,956 K in the last fifth. This large transient
cooling motivates an energy-budget and initial-relaxation review.

The imposed lower **radiation** flux is `9.81981111e6 erg cm^-2 s^-1`,
only `1.56e-4` of the solar reference. In this HD2 setup, incoming gas
entropy is expected to carry the remaining energy; a low lower radiative
flux is therefore intentional. The lower **hydrodynamic** flux has so far
oscillated through several solar fluxes and must be averaged after the flow
relaxes. Between `t=10.1` and `54.3`, the change in stored gas + radiation +
gravitational energy is `8.22e29 erg`; trapezoidal integration of the sampled
bottom-minus-top boundary flux gives `8.31e29 erg` (about 1.0% difference).
This sampled ledger does not establish an exact discrete conservation identity,
but does not indicate a large missing energy term over this interval.

The new `plot_simple_convection_tau.py` reconstructs the Rosseland `tau=1`
surface from xz-slice density and gas temperature using the **same** rho–T
opacity table. Its initial mean height is `z=2.004303` code, agreeing with
the generator's printed initial profile (`2.004311`) to `8e-6` code. The
initial slice-mean gas temperature at that surface is about 6072 K; it falls
to 4585 K by `t=10`, then rises to about 5782 K by `t≈71.5` (one snapshot,
horizontal slice standard deviation 167 K). The `tau=1` height moves from
`z≈2.00` to `1.80` and then `2.11`; comparing temperature at a fixed height
would obscure this motion. This is a **single xz slice**, not a 3D surface
average, and the run is still thermally evolving. Near agreement with solar
effective temperature at one time is not validation of a steady atmosphere.

## How to repeat and extend

The runner generates an input copy, links external tables, records Git/binary/
table hashes and the full command, runs MPI, then writes `summary.json`, a
large-font case `index.html`, history and slice plots, and an output-root
`index.html` linking the cases. The plotting scripts require Python with
`numpy`, `scipy`, `h5py`, and `matplotlib`. The runner rejects an existing nonempty
case directory. Build the binary first as above, then:

```bash
python3 tst/nrfld_validation/run_solar_convection.py \
  --data-dir /mnt/d/output/rad/test_20260903/conv/data \
  --output-root /mnt/d/output/rad/wor70_solar_20261104 \
  --case fgmres_cfl03_64_np12_new --solver fgmres \
  --cfl 0.3 --nx 64 --ranks 12 --tlim 10 --xy-slice-z 2.0
```

For a restart, pass `--restart /path/to/previous.final.rst`, choose a new
`--case`, and increase `--tlim`. If the parent `.hst` lacks its column header,
also pass `--history-header-from /path/to/original/header-bearing.hst`. A
short solver-only run can use `--no-slices`. To evaluate stationarity across
restart boundaries, pass each earlier history with repeated `--history-prepend`;
duplicate restart timestamps are resolved in favor of the later segment.
The run's own solver log still determines the nonlinear-solver status in its
HTML, so the late stationarity windows should lie within the current segment.
Use `--plan` to inspect the command. The history report's candidate flag
requires two late windows each spanning a vertical crossing time, flux near
the solar value and stable between windows, stable vertical velocity RMS,
small mass drift, and clean solver status. This is a screening rule only;
solar comparison requires profile and resolution diagnostics.
The supplied base input samples the xy plane at `z=1.536`; the initial
optical-depth-one layer is near `z=2.0`, hence the explicit slice override
above when photospheric structure is the target.

## Next scientific gates

1. Continue the 64³ CFL 0.3 run far beyond `t=10`; assess whether the top flux
   and vertical RMS settle. If not, inspect the initial stratification and
   HD2/upper radiation boundary energy budget before increasing resolution.
2. Determine which of the untracked `nrfld_solar_convection` files is intended
   for the Rempel-sized box. Review its many safety caps and flux controls,
   establish ownership, and bring a tested version under Git before production.
3. Run a non-magnetic 6.144 × 6.144 × 3.072 Mm pilot with at least two grid
   spacings. Compare time-averaged optical-depth-aligned temperature/density,
   vertical velocity, radiative/enthalpy/kinetic fluxes, granule scale and
   surface brightness distribution with published definitions. Keep gray FLD
   limitations explicit near optical depth unity.
