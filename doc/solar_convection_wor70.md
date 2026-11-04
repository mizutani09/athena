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

The new `plot_simple_convection_tau.py` reconstructs Rosseland `tau=1` and
`tau=2/3` surfaces from xz-slice density and gas temperature using the **same** rho–T
opacity table. Its initial mean height is `z=2.004303` code, agreeing with
the generator's printed initial profile (`2.004311`) to `8e-6` code. The
initial slice-mean gas temperature is 6072 K at `tau=1` and 5754 K at
`tau=2/3`. The latter is the appropriate gray-atmosphere comparison with
solar `Teff≈5772 K`; gray Eddington gives about 6103 K at `tau=1`. By `t=10`,
the respective temperatures fall to 4585 K and 4345 K, while the `tau=1`
height moves from `z≈2.00` to `1.80`. At `t≈76`, the `tau=2/3` slice mean is
about 5443 K and `tau=1` height is `z≈2.28`, illustrating continued evolution.
This is a **single xz slice**, not a 3D surface average. Even a momentary
temperature match at one optical depth does not establish a steady atmosphere.

The 12-rank CFL 0.3 continuation reached `t=100` at cycle 5,425 after
1:16:29 wall time for the `t=10`→`100` segment. All 5,056 nonzero-step Newton
solves converged; there were 17,729 FGMRES calls, **zero** slow-MG warnings,
zero relaxed accepts and zero fatals. The final two 25-code-time windows have
mean top total fluxes `0.737` and `0.786 F_sun`, and mean vertical RMS speeds
`2.12` and `2.27 km/s`. Each covers only 0.77–0.82 full-box crossing times;
the maximum mass drift is 1.56%. At `t=100`, the xz-slice mean
`T(tau=2/3)=5467 K`, with 211 K spatial standard deviation. The state is
still evolving and does **not** pass the stationarity screening. From
`t=10.1`→`100`, sampled boundary-flux integration and stored-energy change
agree within about 0.41% relative. The full case HTML includes 180 finite xz
movie frames and the optical-depth plot.

The new bottom mass-flux diagnostic separates positive upflow, negative
downflow, and net transport. In the small-box `t=50`→`75` and `75`→`100`
windows, the absolute mean net flux is about 0.04% and 0.25% of the combined
gross up/down flow. The RMS net fraction is 6.8% and 8.9%, exposing residual
instantaneous oscillations. The boundary supports nearly balanced
counterflows on average there. Near-zero mean net mass flux alone does not
establish thermal stationarity.

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
Pass `--movie` on a plotted run to include an inline xz MP4 in its HTML.
Use `--plan` to inspect the command. The history report's candidate flag
requires two late windows each spanning a vertical crossing time, flux near
the solar value and stable between windows, stable vertical velocity RMS,
small mass drift, and clean solver status. This is a screening rule only;
solar comparison requires profile and resolution diagnostics.
The supplied base input samples the xy plane at `z=1.536`; the initial
optical-depth-one layer is near `z=2.0`, hence the explicit slice override
above when photospheric structure is the target.

## Rempel-sized non-magnetic pilot

The user confirmed that the August–September `simple_convection_lhllc_fld`
series is the one to continue. A separate input,
`inputs/radiation/athinput.simple_convection_lhllc_fld_general_6Mm_box`,
keeps this tracked problem generator and the same photosphere while extending
the box to 6.144 × 6.144 × 3.072 Mm. The first proposed grid is
128 × 128 × 64, giving 48 km **cubic cells**. NR-FLD multigrid rejects
non-cubic cells; the runner now checks this before launching Athena++.
The initial `tau=1` height is `z≈2.0` code, leaving 0.536 Mm above the
photosphere, whereas Rempel's small-domain setup places the top about 0.700 Mm
above it ([§II.2](https://arxiv.org/html/1405.6814v1)). Thus the box size
matches but the photospheric placement differs by about 0.164 Mm. The
current 48 km cells are also coarser than the paper's 32 km and finer grids.
The pgen's optional time-dependent mass-balance adjustment of `P_BND` extends
Rempel's fixed-pressure HD2 prescription; this difference must be tested
when interpreting coherent bottom-boundary motion.

The old EOS table ends at `log10(rho/[g cm^-3])=-4.5` and cannot invert the
deeper hydrostatic profile. The original NATA rho–u source covers through
`-2.0`; converting it with the same tool and extending the selected density
range to `-3.0` produces a table whose entire overlap with the old table is
**bit-for-bit identical**. This is a table-range extension, not an EOS model
change. The resulting initial profile spans `z=-3.072` to `+3.072` code;
its maximum hydrostatic residual is `2.95e-5`, EOS density margin is 1.223 dex,
and `tau=1` lies at `z=2.00002`. Its bottom density is `5.98e-5 g cm^-3`
and bottom radiative flux is `3.54e5 erg cm^-2 s^-1`. The EOS/profile hashes
are in the pilot's manifest.

To regenerate the external files in a fresh output location:

```bash
python3 scripts/convert_nata_eos_table.py \
  /home/kosuke/simulation/nata_eos/runs/athena_solar_fine_safe/nata_eos_rho_u_table.dat \
  /path/to/AthenaNataEosTable_solar_deep.tab \
  --logrho-min -10.5 --logrho-max -3.0 \
  --logx2-min 11.2 --logx2-max 14.5 --dlogx2 0.02
python3 scripts/build_solar_initial_profile.py \
  --eos-table /path/to/AthenaNataEosTable_solar_deep.tab \
  --opacity-table /path/to/opacity_table_rhoT_3p0_6p3.h5 \
  --output-dir /path/to/profile_dir \
  --zmin -3.072 --zmax 3.072 --nz 2049
```

Place/symlink these three files by their input names in a single `--data-dir`.
The 64 × 64 × 32, one-rank, two-cycle preflight completed in 7.65 s with
2/2 nonzero-step Newton solves converged and no warning/fatal. It establishes
input and boundary compatibility only; it is not a solar validation run.
The 128 × 128 × 64, 12-rank, CFL 0.3 pilot then reached `t=1` in 25 cycles
and 78.82 s: 25/25 Newton converged, no slow-MG warning or fatal, with a
measured sum of rank RSS around 4.3 GiB. This is comfortably below the WSL
19 GiB memory limit. The coarsest MG grid reports that it is not reduced to a
single cell, so this topology may leave some solver efficiency available.
The early `tau=2/3` slice temperature changes from about 5704 K at `t=0`
to 5290 K at `t=1`, showing the initial thermal transient in the larger box.
The pilot does not establish granulation statistics or steady solar agreement.
During the first few code-time units of the deeper box, the bottom flow is
almost entirely one-way at a time and reverses sign. Its hydro energy flux
therefore swings by tens to hundreds of solar flux units. This is a strong
global startup transient, so early bottom flux values must not be interpreted
as a convective luminosity. A longer run is needed to determine whether the
coherent motion decays.
The larger-box input now writes sparse full 3D primitive and user-variable
volumes every 10 code-time units. `plot_simple_convection_surface3d.py`
reconstructs the `tau=2/3` height and gas temperature over the **entire**
horizontal surface, rather than one xz slice, and plots its temperature and
vertical velocity. It also records horizontally averaged full-3D profiles of
gas temperature, density, vertical RMS speed, and radiative cooling in
`profiles3d.json`. When at least two snapshots fall in the last quarter of
the recorded interval, `plot_simple_convection_late_profiles3d.py` writes
their temporal mean and snapshot standard deviation as
`profiles3d_late_mean.json` and a four-panel plot. This is a descriptive
window and is not labeled stationary until the separate history screen passes.
`plot_simple_convection_flux3d.py` reconstructs interior vertical profiles
of radiation, convective enthalpy, mean-flow enthalpy, kinetic and gravitational
flux from full volumes. Its FLD radiation component uses cell-centered
gradients and the code's Levermore-Pomraning limiter; it is an **approximate
interior** profile. The history's face fluxes, not the profile end cells,
are authoritative for the top/bottom energy ledger. The convective enthalpy
term subtracts transport by the horizontally mean mass flow. A strongly
varying sum of these profiles is a sign of thermal transience, not a solar
flux agreement test.
A one-cycle 128 × 128 × 64 smoke run produced paired full
volumes; at `t=0` the 3D mean `T(tau=2/3)=5703.993 K` versus `5704.008 K` in
the xz slice, and the mean heights agree within `6e-6` code. This checks the
full-volume assembly and optical-depth interpolation on the smooth initial
state. It does not test a convectively structured surface. The pre-fix 128 ×
128 × 64 full-box run reached `t=10` in 241 cycles; every nonzero-step Newton
solve converged, with no slow-MG warning or fatal. At `t=10`, the full 3D
`tau=2/3` mean temperature was 4061 K, its horizontal standard deviation
only 0.20 K, and the vertical velocity there was almost uniformly downward
at 2.53 km/s. The top flux in the last quarter was 0.364 solar and the
bottom flow was still coherent. This is an early transient, not granulation.
The simulation itself finished normally, but its wrapper session ended before
postprocessing; the plots, movie, JSON and HTML were recovered from the saved
HDF5/restart files, and the manifest records that recovery.

## HD2 lower-boundary initialization defect (WOR-71)

The previously implicit `PBND` was initialized from the center of the first
cell, while the HD2 formula treats it as the **boundary-face pressure**.
Both dynamic pressure means were also initialized from that first cell,
instead of the first and second cells separately. For the 48 km initial
profile, a centered estimate of the first-cell pressure-force residual is
about 51% of gravity on the first step and 26% after the means update. With
the two-cell log-linear face pressure and means it is about 0.9%. This is an
estimate, not the finite-volume Riemann flux residual.

Setting the temperature perturbation amplitude to zero in the old 96 km
coarse-grid run leaves the `t≈0.56` bottom net mass flux, mass drift, and
bottom hydro energy flux unchanged to the displayed precision. The large
startup wave therefore is not primarily seeded by the temperature noise.
With the corrected initialization at the same resolution, EOS, opacity,
profile, CFL, rank count and CPU placement, the `t≈0.5` bottom net mass flux
changes from -27.31 to +0.93 code units, mass drift from -1.34% to +0.028%,
and bottom hydro flux from -976 to +33.8 solar units. The `t=0`→`2` net-mass-
flux RMS falls from 17.18 to 0.71 code units. The corrected `t=10` run
completes 120/120 Newton steps with zero slow-MG warning and fatal; its
`tau=2/3` temperature is 4737 K versus 4165 K with the old boundary. The
top flux at `t≈1` stays close to 0.71 solar in both runs, so the boundary
fix does not remove the separate initial photospheric cooling transient.
The 1D builder labels the missing radiative flux as a future convective
contribution: at optical depth about 3, the imported profile has only
`F_rad≈0.34 F_sun` and an inferred convective fraction near 0.66. The actual
3D initial state has zero velocity, hence zero convective enthalpy transport.
The top initially emits nearly `F_sun`; cooling until convection develops is
therefore expected even with a correct lower pressure boundary. A thermal
relaxation run, not a static snapshot, is required for solar comparison.
The pre-fix long run remains a solver durability test but is **not** the
physical baseline for solar validation. The matched 48 km comparison also
confirms the boundary correction: at `t≈0.5`, bottom net mass flux changes
from `-10.5947` to `+0.3144` code, mass drift from `-0.7664%` to `+0.0128%`,
and bottom hydro flux from `-380.95` to `+11.38 F_sun`. Over `t=0`→`2`,
net-flux RMS falls from `7.318` to `0.1574` code and maximum absolute mass
drift from `1.3225%` to `0.0266%`. Both full-box `t=10` runs completed all
Newton solves (old 241/241, fixed 240/240), with zero slow-MG warnings or
fatals. The **full 3D** `tau=2/3` surface at `t=10` changes from mean
temperature `4061` to `4465 K` and vertical-velocity RMS from `2.533` to
`0.033 km/s`; the old nearly uniform downflow has largely disappeared.
The late top flux remains only about `0.36 F_sun` in both, so this startup
fix does not establish thermal relaxation. The matched comparison, with
figures and links to both case reports, is under
`rempel_box_pilot/hd2_comparison_48km/index.html`.

A corrected 96 km large-box run was restarted from its `t=10` checkpoint.
The first history sample at `t=10.155` has `PBNDScale=0.998040`, following
`0.998202` at `t=10`; bottom net mass flux changes smoothly from `0.502`
to `0.397` code. This verifies that the saved boundary state continues
without a fresh startup spike. WOR-71 is complete; subsequent thermal and
solar-physics questions remain in WOR-70.

The corrected 96 km large box reached `t=100` (37.27 physical minutes) at
cycle 1309. All 1190 Newton solves in the `t=10`→`100` segment converged;
the log has 3820 FGMRES calls, zero slow-MG warnings, and zero fatals. Its
late two 25-code-time windows have mean top fluxes `0.543` and `0.585 F_sun`,
vertical RMS speeds `1.323` and `1.422 km/s`, and only `0.239` and `0.257`
full-height crossing times. Maximum mass drift in the final window is
`0.28%`; mean net bottom mass flux is about `0.11%` of gross up/down flow.
At `t=100`, the full 3D `tau=2/3` surface has mean temperature `5028 K`,
horizontal standard deviation `76 K`, and vertical-velocity RMS
`1.98 km/s`. Warm broad upflows and narrow cooler downflow lanes are visible
in the surface maps: the early uniform state has developed a granulation-like
pattern. The flux and temperature are still below solar, and the averaging
windows are too short for stationarity. The case HTML includes the maps,
movie, 3D profiles and a `t≈80`→`100` three-snapshot profile average.
The next corrected continuation targets `t=300` to enlarge the statistical
window; this is a relaxation test, not a claim that 96 km resolves solar
granulation.
The pre-fix 64³ continuation was intentionally stopped at `t≈153` after
writing a `t=150` restart, to free six physical cores for the corrected
baseline. Between `t=100` and the stop, all 3,303 reported Newton solves
converged, with 11,973 FGMRES calls and zero slow-MG warnings or fatal errors.
Its HTML explicitly says "intentionally stopped" and keeps the partial
history/movie. A fresh corrected 64³, 12-rank, CFL 0.3 run to `t=100` is now
running, alongside the corrected 48 km large-box comparison.

The large-box pilot uses the generated files under
`/mnt/d/output/rad/wor70_solar_20261104/profile_3Mm_candidate/input_data`.
For a fresh run after building the same non-magnetic binary:

```bash
python3 tst/nrfld_validation/run_solar_convection.py \
  --base-input inputs/radiation/athinput.simple_convection_lhllc_fld_general_6Mm_box \
  --data-dir /mnt/d/output/rad/wor70_solar_20261104/profile_3Mm_candidate/input_data \
  --output-root /mnt/d/output/rad/wor70_solar_20261104/rempel_box_pilot \
  --case fgmres_128x128x64_new --solver fgmres --cfl 0.3 \
  --nx1 128 --nx2 128 --nx3 64 --ranks 12 --tlim 10 \
  --xy-slice-z 2.0 --movie
```

MPI placement can be varied explicitly with `--mpi-map-by core
--mpi-bind-to core` for one rank per physical core, or left at the
`hwthread` defaults. Compare wall time on an identical checkpoint before
selecting a faster placement; the current 12-rank and four-rank runs use
six and two physical cores respectively.

## Next scientific gates

1. Continue the **corrected** 64³ CFL 0.3 run and assess whether the top flux
   and vertical RMS settle. If not, inspect the initial stratification and
   HD2/upper radiation boundary energy budget before increasing resolution.
2. Keep the confirmed, tracked `simple_convection_lhllc_fld` series as the
   baseline. Review the untracked `nrfld_solar_convection` files separately
   before using their safety caps or flux controls in a comparison.
3. Extend the non-magnetic 6.144 × 6.144 × 3.072 Mm pilot with at least two grid
   spacings. Compare time-averaged optical-depth-aligned temperature/density,
   vertical velocity, radiative/enthalpy/kinetic fluxes, granule scale and
   surface brightness distribution with published definitions. Keep gray FLD
   limitations explicit near optical depth unity.
