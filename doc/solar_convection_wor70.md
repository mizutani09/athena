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
short solver-only run can use `--no-slices`. Pass each earlier history with
repeated `--history-prepend` to show a continuous history plot; duplicate
restart timestamps are resolved in favor of the later segment. The report
evaluates stationarity and the energy ledger from the **current run segment
only**, so the late windows cannot silently mix two solver/source settings.
The run's own solver log determines the nonlinear-solver status in its HTML.
Pass `--movie` on a plotted run to include an inline xz MP4 in its HTML.
Use `--plan` to inspect the command. The history report's candidate flag
requires two late windows each spanning a vertical crossing time, flux near
the solar value and stable between windows, stable vertical velocity RMS,
small mass drift, and clean solver status. In each late window, both the
least-squares trend and the endpoint change rate of stored
gas+radiation+gravitational energy must be below `0.1 F_sun` in magnitude.
Oscillatory phase makes the endpoint rate sensitive to window placement;
an endpoint-only failure means the averaging interval is insufficient to
establish stationarity, not necessarily that secular heating is present.
An apparently solar top flux
while the box is heating or cooling rapidly cannot pass. This is a screening
rule only; solar comparison requires profile and resolution diagnostics.
Each case HTML plots the current-segment stored-energy change alongside
integrated sampled boundary supply, with their running residual separately.
The plot is diagnostic because history fluxes are not exact RK-stage
integrals.
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
late two approximately 22-code-time windows have mean top fluxes `0.551`
and `0.585 F_sun`, vertical RMS speeds `1.341` and `1.440 km/s`, and only
`0.219` and `0.234`
full-height crossing times. Maximum mass drift in the final window is
`0.28%`; mean net bottom mass flux is about `0.11%` of gross up/down flow.
At `t=100`, the full 3D `tau=2/3` surface has mean temperature `5028 K`,
horizontal standard deviation `76 K`, and vertical-velocity RMS
`1.98 km/s`. Warm broad upflows and narrow cooler downflow lanes are visible
in the surface maps: the early uniform state has developed a granulation-like
pattern. The flux and temperature are still below solar, and the averaging
windows are too short for stationarity. The case HTML includes the maps,
movie, 3D profiles and a `t≈80`→`100` three-snapshot profile average.
The late full-3D temperature and density plots now overlay the imported
solar 1D starting stratification as a visual reference; it is not an
independent observation or a stationarity criterion.
An additional Rosseland-optical-depth-aligned full-3D profile removes the
height shift when comparing the photosphere. In the **legacy gravity-source**
`t=260`→`300` five-snapshot average, `T(tau=2/3)=5145 K` versus `5770 K`
in the imported 1D solar starting profile; the discrepancy is concentrated
near the surface, while the deeper `tau≈10^3` temperatures are close. This
transient difference is not a final physical validation result. The plot and
data are `simple_convection_optical_profiles3d_late_mean.png` and
`optical_profiles3d_late_mean.json` in the case report.
For orientation, a flux of `0.585 F_sun` implies a blackbody effective
temperature about `5048 K` if the same solar reference is used, close to the
instantaneous `tau=2/3` mean of `5028 K`. This agreement is a gray-atmosphere
consistency check, not evidence of a relaxed solar solution.
The HTML now also reports a **sampled** energy ledger: stored gas+radiation+
gravitational energy change minus time-integrated recorded boundary hydro,
radiation, and potential-energy fluxes. In the coarse box, the residual is
equivalent to `+0.103 F_sun` over the current `t=10`→`100` segment and
`+0.0466 F_sun` over its last quarter. It persists at about `+0.08 F_sun`
in the subsequent `t=100`→`189`
segment. This is not yet proof of a code conservation error because the
history samples accepted-step fluxes, not their exact RK-stage integrals.
The cadence and discrete source/flux audit is tracked separately as WOR-72;
until then, this residual limits precision claims for solar flux comparison.
The corrected 96 km box then completed `t=300` (111.8 physical minutes) at
cycle 4254. In the `t=100`→`300` continuation, all 2,946 reported Newton
solves converged, with 11,711 FGMRES calls, maximum 31 iterations per call,
and zero slow-MG warnings or fatal errors. In the current-segment late
windows `t≈200`→`250` and `250`→`300`, mean top flux rises from `0.722`
to `0.789 F_sun`, vertical RMS from `2.00` to `2.26 km/s`, and neither
window spans a full-height crossing time (`0.726` and `0.818`). Bottom
hydro flux means are `-2.39` and `-3.88 F_sun`; maximum mass drift in the
last window is `1.06%`. The full-3D `tau=2/3` mean temperature changes from
about `5251 K` in a `t=200` snapshot to `4924 K` at `t=300`. Thus the
stationarity screen fails despite robust implicit convergence. The sampled
energy-ledger residual in the current-segment last quarter is
`+0.116 F_sun`; its status remains diagnostic until the stage-flux/source
audit in WOR-72 is complete. The endpoint stored-energy change rate in that
window is `-2.92 F_sun`, but the all-sample secular trend is only about
`-0.043 F_sun`. The discrepancy reflects an oscillatory storage signal;
both measures are therefore required by the conservative screening rule,
and the other stationarity criteria above also fail.
This is a relaxation test, not a claim that 96 km resolves solar granulation.
A `t=100`→`150` continuation with the HD2 bottom mass-balance feedback
disabled completed with 750/750 Newton solves converged, 2,587 FGMRES calls,
maximum 32 iterations per call, and no slow-MG warnings or fatal errors.
Over the matched `t=100.2`→`150` window, feedback **on/off** gives mean top
flux `0.623/0.615 F_sun`, vertical RMS `1.58/1.48 km/s`, maximum mass drift
`0.359/0.689%`, and bottom net mass-flux RMS `0.765/1.044` code. The
feedback improves mass control in this interval and changes top flux only
about 1.2%; it is not a sufficient explanation for the large thermal
transience. The paired plot and JSON are in
`rempel_box_pilot/mass_control_comparison_64x64x32_t100_150/`.
The 48 km, 128×128×64, 12-rank continuation with the **legacy** gravity-work
source was intentionally stopped after a complete `t=40` checkpoint. Its
`t=10`→`40` segment had 754/754 Newton solves converged, 2,180 FGMRES
calls, maximum 52 iterations per call, and no slow-MG warning or fatal.
The old-source partial report and plots remain in
`rempel_box_pilot/fixed_hd2_128x128x64_np12_cfl03_t100/`, explicitly
marked as stopped. The corrected face-flux gravity source is continuing
from that exact checkpoint to `t=100` on the same 12 MPI ranks under
`rempel_box_pilot/face_gravity_hd2_128x128x64_np12_cfl03_t100/`.
This source switch at `t=40` is recorded in both manifests; its early
state still contains the legacy source history, so a later corrected-source
relaxation window is required for physical acceptance.
The pre-fix 64³ continuation was intentionally stopped at `t≈153` after
writing a `t=150` restart, to free six physical cores for the corrected
baseline. Between `t=100` and the stop, all 3,303 reported Newton solves
converged, with 11,973 FGMRES calls and zero slow-MG warnings or fatal errors.
Its HTML explicitly says "intentionally stopped" and keeps the partial
history/movie. The corrected 64³, 12-rank, CFL 0.3 run completed `t=100`.
All `5359/5359` nonzero-step Newton solves converged, with `18,489` FGMRES
calls, zero slow-MG warnings and zero fatals. Its two late 25-code-time
windows have mean top fluxes `0.724` and `0.810 F_sun`, vertical RMS speeds
`2.185` and `2.397 km/s`, and only `0.792` and `0.869` vertical crossing
times. The last window reaches `2.07%` maximum mass drift. At `t=100`, the
xz-slice mean `T(tau=2/3)` is `5466 K` with `205 K` horizontal standard
deviation. The old pre-fix `t=100` case had a late flux of `0.786 F_sun`
and `1.56%` maximum mass drift; correcting startup does not by itself
resolve late flux or mass drift. The small box is a long solver and boundary
test, not a Rempel-size solar validation.
From the same corrected `t=100` checkpoint, a `t=100`→`102` benchmark on
CPUs 0–11 measured `220.78 s` for six MPI ranks bound one per physical core
and `151.55 s` for 12 MPI ranks on the corresponding hyperthreads. Both
completed 121 cycles with 481 FGMRES calls, zero warnings and zero fatal
errors; their final recorded top fluxes differ by `0.039%`, reflecting a
small decomposition dependence. The 12-rank placement is `31%` shorter in
wall time for this small-box checkpoint while the other two physical cores
run the coarse large-box continuation. This result should not be generalized to larger
meshes without a separate benchmark.
An identical small-box `t=100`→`102` restart with 20-fold finer history
cadence produced the same 121 cycles and 481 FGMRES calls. Over the common
`t=100.1`→`102` interval, the sampled energy-ledger residual was
approximately `-0.0010 F_sun` with ordinary cadence and `-0.0024 F_sun`
with fine cadence. This short small-box interval has a much smaller residual
than the coarse large-box windows, so it does not settle WOR-72.

For WOR-72, matched corrected large-box startups give sampled `t=0`→`10`
ledger residuals of `+0.187 F_sun` at 96 km and `+0.0286 F_sun` at 48 km.
The old hydro gravity-work source uses cell-centered `rho*v_z`, while the
continuity update uses Riemann face mass fluxes. The corrected
`problem/gravity_work_face_flux=true` source instead uses those same mass
fluxes and face/cell gravitational-potential differences. In a matched
96 km **developed-state** restart, `t=100.065`→`110`, four MPI ranks,
16-digit history and every-step cadence, the sampled ledger residual fell
from `+0.07591 F_sun` with the old source to `+0.000615 F_sun` with the
face-flux source. In the second half of that interval the corrected residual
was only `+0.00000215 F_sun`, versus `+0.1152 F_sun` with the old source.
Both variants completed 153/153 Newton solves, had no slow-MG warnings or
fatal errors, and took about 210 seconds. Over `t=100.1`→`110`, mean top
flux was `0.593` versus `0.575 F_sun`; vertical RMS was `1.550` versus
`1.556 km/s`. The roughly 3% top-flux difference over a short interval
means old-source long-run solar flux estimates must be treated as provisional.
At a later corrected-source `t=850` checkpoint, otherwise identical four-rank
restarts to `t=855` changed only the HST cadence from `0.1` to `0.001` code
time. Both completed 74/74 Newton solves. At every common recorded time the
time, stored energy, and selected boundary fluxes matched exactly. Over the
shared sampled interval `t=850.137443`–`855`, trapezoidal sampled-ledger
residuals were `+0.003191` and `+0.001167 F_sun` equivalent respectively;
left/right rectangle residuals span about `-0.195`/`+0.201 F_sun` at coarse
cadence and `-0.124`/`+0.126 F_sun` at every-step cadence. Sample timing and
integration method therefore matter substantially in this oscillating
boundary state. The remaining every-step residual is not a measured RK-stage
conservation error. See
`rempel_box_pilot/energy_ledger_cadence_t850_855/index.html` and
`tst/nrfld_validation/solar_convection_ledger_cadence.py`.
A third restart with HST cadence `0.2` shows a consistent ordering when all
three cases are restricted to their *same* sampled interval
`t=850.206253`–`855`: the trapezoidal residual is `+0.010562`, `+0.002591`,
and `+0.000538 F_sun` for cadence `0.2`, `0.1`, and every step (25, 49, and
71 history samples). This is stronger evidence that much of the observed
ledger mismatch is sampling error. HST cadence `0.05` records every step as
well, so it provides no additional refinement level. The three-way report is
`rempel_box_pilot/energy_ledger_cadence_3way_t850_855/index.html`.
An independent four-CFL check restarts the same `t=850` state with every-step,
16-digit history and CFL numbers `0.3`, `0.15`, `0.075`, and `0.0375`. All
four complete `t=855`; their 74, 146, 290, and 579 Newton solves converge
without slow-MG warnings or fatal errors. Comparing each run over the same
`t=850.2`–`855` physical interval gives sampled ledger residuals of
`+0.000363`, `+0.000803`, `+0.000654`, and `+0.000442 F_sun` equivalent.
The single-interval result is phase-sensitive. A scan of 19 overlapping
intervals beginning at `t=850.2`–`852.0` and all ending at `855` gives
residual RMS `0.003909`, `0.001042`, `0.000725`, and `0.000472 F_sun`, and
maximum absolute values `0.006683`, `0.002353`, `0.001092`, and
`0.000574 F_sun`. The decreasing envelope favors temporal sampling or
time-discretization error over a fixed large conservation defect in this
developed-state window. It does **not** prove exact discrete conservation:
the analysis linearly interpolates endpoints and integrates sampled
accepted-step fluxes, rather than accumulating actual RK-stage boundary
fluxes and sources. Results, plots, and case links are in
`rempel_box_pilot/energy_ledger_cfl_refinement_t850_855/index.html`; the
reproducer is `tst/nrfld_validation/solar_convection_ledger_timestep.py`.
The same `t=850` restart was also run once with two additional history
columns, `FadvRadBottom` and `FadvRadTop`, from the Godunov radiation-energy
flux. This closes a previously omitted boundary term in the sampled ledger.
Over the identical 71-sample `t=850.206253`–`855` interval, the residual is
`+0.00053833 F_sun` without that term and `+0.00045738 F_sun` with it. The
net advective-radiation boundary supply is only `+0.00008095 F_sun` on this
window (mean bottom/top `+0.00007901`/`+0.00000125 F_sun`). Thus the term
is real but does not account for the remaining mismatch. The run used the
newer limiter-closure binary; both residuals are evaluated on that *same*
trajectory, avoiding a cross-binary attribution. The diagnostics append to
the pgen history without changing the solver, and the reader accepts an old
header with these two new columns. The comparison and reproducer are
`rempel_box_pilot/advective_radiation_flux_budget_t850_855/index.html` and
`tst/nrfld_validation/solar_convection_advective_radiation_audit.py`.
On this rapidly oscillating boundary, comparing each step's stored-energy
change rate against left-point, trapezoidal, or right-point sampled supply
gives RMS mismatches `0.3871`, `0.01856`, and `0.3839 F_sun` respectively.
This explains why one-sided integration of the HST samples was misleading;
it still does not identify the exact flux evaluations inside the update.
The input uses `time/integrator=rk2` (Heun/SSPRK2). Its explicit update weights
the stage-1 and stage-2 flux divergences by one half each; the HST face-flux
arrays retain only the last stage. Thus an end-sample right rectangle omits
the stage-1 flux, while neighboring HST samples provide a good but not exact
trapezoidal proxy. The code then performs **one** full-timestep implicit FLD
solve after both explicit stages. An exact audit must record both actual RK
boundary fluxes and the implicit diffusion boundary flux/source exchange,
and compare against the conserved-state changes before and after that solve.
The existing per-cell `fld/stage_diagnostics` switch is restricted to an
ideal-gas fixed-limiter mesh and explicitly rejects this general-EOS solar
case, so a separate scalar audit is needed.
The corrected source is now the default for this problem; set
`gravity_work_face_flux=false` to reproduce the legacy calculation. The
matched plot and JSON are in
`rempel_box_pilot/gravity_work_comparison_64x64x32_np4_t100_110/`.
The 96 km corrected-source continuation from the legacy `t=300` checkpoint
completed to `t=500` with 3,052/3,052 Newton solves converged, 12,530
FGMRES calls (maximum 38 iterations), and no slow-MG warning or fatal error.
The sampled `t=300`–`500` energy-ledger residual was `-0.0070 F_sun`,
supporting the corrected gravitational work over a long developed-state run.
The `t=400`–`450` and `450`–`500` windows span 1.12 and 1.19 vertical
crossing times, respectively, but top flux rises `1.285`→`1.362 F_sun`,
maximum mass drift reaches `1.84%`, and stored-energy secular trends are
`+1.41` and `+0.65 F_sun`. Thus solver robustness and improved energy
accounting do **not** establish statistical stationarity. The case HTML is
`rempel_box_pilot/face_gravity_hd2_64x64x32_np4_cfl03_t500/index.html`.
At the final full-volume snapshot the optical-depth-`2/3` surface has mean
temperature `4892 K`, horizontal standard deviation `98 K`, and vertical
velocity RMS `7.65 km/s`; five snapshots from `t≈460`–`500` give mean
`T(tau=2/3)=5297 K`, versus `5769 K` for the imported initial 1D profile.
These are transient coarse-grid measurements, not a solar match.
The previously reported `t=300` coarse run and the 48 km `t=10`→`40`
continuation used the legacy source. Their solver-convergence evidence is
valid, but their long-time solar-flux estimates need a corrected-source
continuation before physical acceptance. The corrected-source 48 km,
128×128×64, 12-rank continuation from `t=40` to `100` completed 1,739/1,739
Newton solves, maximum 32 FGMRES iterations, and zero slow-MG warnings or
fatals. In its final `t=85.1`–`100` window, mean top flux is
`0.524 F_sun`, maximum mass drift `0.188%`, and stored-energy trend and
endpoint rate are `+2.66` and `+2.30 F_sun`. This window spans only 0.15
vertical crossing times, so the run is transient. At `t=100`, the full-3D
`tau=2/3` surface has mean temperature `4990 K` and vertical-velocity RMS
`1.72 km/s`; the two-snapshot `t=90`–`100` optical-depth mean gives
`4981 K`, versus `5769 K` in the imported initial 1D profile. The case
HTML is `rempel_box_pilot/face_gravity_hd2_128x128x64_np12_cfl03_t100/index.html`.
The source switch is explicit in both corrected-run manifests and reports.

The large-box lower hydro flux, total-mass drift, and pressure controller
show a coherent oscillation, investigated separately in WOR-73. A matched
50-code-time spectral diagnostic gives the legacy `t=100`–`150` lower-flux
standard deviation `27.5 F_sun` and dominant period `12.47` code time
(`4.65 min`); legacy `t=200`–`250` has `56.3 F_sun` at `12.50` code time.
The corrected-source `t=350`–`400` window has `136 F_sun` at `12.52`
code time. The legacy mass-feedback-OFF control at `t=100`–`150` has
`37.4 F_sun` and broad power around 12–17 code time, so feedback is not
established as the sole cause. These windows are at different evolutionary
stages and cannot by themselves assign growth to the gravity-source change.
The full-volume fraction of vertical-velocity variance in the horizontally
mean mode grows from `0.294` (four legacy dumps at `t=100`–`150`) to `0.471`
(five legacy dumps at `t=200`–`250`) and `0.710` (five corrected-source dumps
at `t=350`–`400`). This identifies a box-scale coherent breathing motion,
not merely local granular fluctuations; the feedback-OFF control has no
full-volume dumps for the same metric.
The plot, exact windows, and JSON are in
`rempel_box_pilot/acoustic_mode_diagnostic/`. Compare boundary damping on
the same corrected-source restart before changing the physical setup.
Rempel (2014), §II.2, introduced HD2 pressure-fluctuation damping after
finding over-excited standing pressure waves with symmetric fluctuations in
deeper domains. Our ghost-pressure formula and default `Cdmp=0.95` match its
stated HD2 equations, but its observation does not establish the cause of
the growing mode in this FLD/Athena++ run.
Here `Cdmp` acts only on the pressure departure from the horizontal mean;
it cannot directly damp the dominant horizontally uniform mode identified
above. Same-checkpoint tests should therefore prioritize mean-pressure
mass feedback OFF and a faster feedback relaxation time, with a stronger
`Cdmp` as a control for the nonuniform component.
The imported initial 1D profile gives an acoustic one-way travel time
`integral dz/sqrt(Gamma1 P/rho) = 4.92 min` across the 3.072 Mm layer;
the observed `4.65 min` period is close to this timescale. An acoustic
overtone is plausible, but this estimate does not identify an eigenmode or
prove which boundary excites it. The controller's `MassRate` matches the
finite-difference derivative of total-mass drift over `t=400`–`450`
(correlation approximately 1, amplitude ratio 0.99–1.00), so a gross
measurement error in that feedback input is not apparent.

A same-checkpoint `t=400`–`450` test with mass feedback disabled completed
732/732 Newton solves, no warnings or fatals. Compared with the standard
feedback time constant of 10 code units, lower hydro-flux standard deviation
rose from `158` to `254 F_sun`, mass-drift standard deviation from `1.03%`
to `2.18%`, and maximum mass drift from `1.84%` to `6.35%`. Both have a
dominant period near `12.5` code units. Mean top flux changed from `1.286`
to `0.742 F_sun`; disabling feedback is not an acceptable physical solution.
The matched history and spectral HTML reports are
`rempel_box_pilot/mass_feedback_off_comparison_t400_450/` and
`rempel_box_pilot/mass_feedback_diagnostic/`.
With feedback enabled but the response time shortened from 10 to 2 code
units, another same-checkpoint `t=400`–`450` test completed 737/737 Newton
solves without warnings or fatals. Relative to the standard control, lower
hydro-flux standard deviation fell `158.2`→`37.6 F_sun`, mass-drift standard
deviation `1.03%`→`0.23%`, and maximum mass drift `1.84%`→`0.54%`.
The dominant period moved from `12.51` to `9.97` code units. Mean top flux
changed `1.29`→`0.88 F_sun`, while the stored-energy secular trend decreased
`+1.47`→`+0.45 F_sun`. The latter is still above the stationarity screen.
The global mass-weighted vertical-velocity standard deviation fell from
`0.392` to `0.251 km/s`, so the mean mode itself is weakened, though sparse
full-volume dumps still show a substantial coherent component.
The matched plot is in
`rempel_box_pilot/mass_feedback_tau2_comparison_t400_450/`; the three-case
spectral report is in `rempel_box_pilot/mass_feedback_diagnostic/`.
Further controls from the *same* `t=400` checkpoint completed without
solver failures: a fresh `tau=10` restart, `tau=1`, and `Cdmp=0.5` at
`tau=10`. Their `t=400`–`450` bottom hydro-flux standard deviations are
`158.1`, `39.8`, and `160.8 F_sun`, versus `37.6 F_sun` for `tau=2`;
mass-drift standard deviations are `1.031%`, `0.170%`, `1.046%`, versus
`0.226%`. The fresh `tau=10` values agree with the continuous baseline
(`158.2 F_sun`, `1.03%`), excluding a large restart artifact. Stronger
`Cdmp` barely changes the box-wide mode, as expected because it damps
horizontal pressure fluctuations rather than their mean. `tau=1` controls
mass slightly more tightly but does not improve the lower-flux oscillation
over `tau=2`; the latter remains the long-run baseline. The four-case
spectral report is
`rempel_box_pilot/mass_feedback_parameter_sweep_t400_450/index.html`.
The improved control continued from `t=450` to `650` with 2,884/2,884
Newton solves converged, maximum 30 FGMRES iterations, and no warnings or
fatals. Lower hydro-flux standard deviation in 25-code windows fell from
`44.1 F_sun` at `t=400`–`425` to `7.85` and `8.86 F_sun` at `t=600`–`625`
and `625`–`650`; mass-drift standard deviation fell `0.245%`→`0.076%`.
The two late 50-code windows have mean top flux `0.624` and
`0.617 F_sun`, but span only 0.56 and 0.51 vertical crossing times.
Their stored-energy regression trends are `+0.120` and `+0.037 F_sun`,
while endpoint rates are `+0.484` and `-0.188 F_sun`. The five-snapshot
late `tau=2/3` temperature mean is `5109 K` versus `5769 K` in the initial
1D solar profile. The case report is
`rempel_box_pilot/face_gravity_hd2_mass_tau2_64x64x32_np4_t450_650/index.html`.
Thus the boundary mode is decaying, but neither solar flux nor statistical
stationarity has been established. The continuation to `t=850` completed
2,849/2,849 Newton solves, maximum 32 FGMRES iterations, and no slow-MG
warnings or fatals. Its `t=750`–`800` and `800`–`850` windows have mean
upper flux `0.6108` and `0.6096 F_sun`; each spans only about 0.49 vertical
crossing times. Their stored-energy regression slopes are `-0.257` and
`-0.251 F_sun`, with endpoint rates about `-0.183` and `-0.144 F_sun`.
Over `t=750`–`850` as a single window, the trend is `-0.098 F_sun` but
the endpoint rate is `-0.146 F_sun`, failing the strict thermal screen.
The `t=850` full-3D optical-depth-`2/3` surface has mean temperature
`5091 K`. The case HTML is
`rempel_box_pilot/face_gravity_hd2_mass_tau2_64x64x32_np4_t650_850/index.html`.
The lower hydro-flux standard deviation remains near `8 F_sun` in the last
25-code windows, while mass-drift standard deviation stays near `0.07%`.
The box-wide oscillation is greatly reduced, but not eliminated. Its
coherent phase can make a longer window's energy slope deceptively small;
for example `t=650`–`850` gives slope `-0.025 F_sun` and endpoint rate
`-0.098 F_sun`, while the last two 50-code windows show substantial cooling.
There is no claim of statistical stationarity or solar agreement.
Matched `t=800`–`850` continuations from the same checkpoint varied the HD2
mass-rate correction gain from `0` to `10`, `20`, and `40`, with the pressure
relaxation time fixed at `2.0`. All four finished
714/714 Newton solves without slow-MG warnings or fatals. The lower hydro-flux
standard deviations are `10.92`, `7.90`, `6.44`, and `4.95 F_sun`;
mass-drift standard deviations decrease from `0.0822%` to `0.0721%`,
`0.0607%`, and `0.0433%`. The dominant period remains
about `16.6` code time in every case. The mean top flux is nearly unchanged
(`0.612` to `0.608 F_sun`). Gain `40` best damps this coherent boundary
oscillation among the four measured gains, but does not fix the large thermal
flux deficit. The
comparison is `rempel_box_pilot/mass_rate_gain_sweep_t800_850/index.html`.
The first gain-`80` run accidentally used the default pressure relaxation
time `10.0` and cannot be compared as a gain-only control. Its output is
retained under `invalid_controls/` with suffix `_invalid_relax10`.
The corrected gain-`80`, relaxation-`2.0` run completed 715/715 Newton
solves without warnings/fatals, but its lower hydro-flux standard deviation
was **`185.19 F_sun`** and mass-drift standard deviation `0.6611%`,
versus `4.95 F_sun` and `0.0433%` for gain `40`. Its maximum mass drift
reached `1.87%`. The mean top flux stayed near `0.608 F_sun` over this
short window. The controller is evidently overdriven at gain `80` on this
checkpoint, despite successful implicit solves. The five-case HTML now
includes both the full amplitude and a zoomed view of gains `0`–`40`.
An intermediate gain `60` also completed 714/714 Newton solves without
warnings/fatals, but its lower hydro-flux standard deviation was
`92.65 F_sun` and its mass-drift standard deviation `0.1181%`.
The upper mean flux stayed `0.609 F_sun`. Thus gains `60` and `80` are
clearly worse than `40` on this checkpoint. Gain `50` also completed
714/714 Newton solves without warnings/fatals; it gave lower hydro-flux
standard deviation `6.34 F_sun` and mass-drift standard deviation
`0.0376%`. Compared with gain `40`, it trades a slightly smaller mass
variation for a larger energy-flux oscillation. In the measured seven-point
sweep, gain `40` minimizes the lower hydro-flux oscillation; gains `50`–`60`
bracket the sharp deterioration. No claim of an optimal gain across other
states or resolutions follows from this one checkpoint.
Matched long continuations with gains `10` and `40` and the same relaxation
time `2.0` reached `t=1000`, with all 2,129 and 2,844 Newton steps converged
respectively and no warnings/fatals. Over `t=851`–`1000`, lower hydro-flux
standard deviations were `8.05` and `3.74 F_sun`, mass-drift standard
deviations `0.07285%` and `0.03355%`, while mean upper flux was `0.6096` and
`0.6054 F_sun`. The dominant period remained about `16.55` code time.
For gain `10`, the last two 37.5-code windows had upper flux `0.6040` and
`0.6121 F_sun` and stored-energy endpoint rates `-0.4555` and
`-0.3956 F_sun`. Each spans only about `0.37` crossing times, and the late
regression slope `+0.0513 F_sun` is misleadingly small against the endpoint
change. Its sampled late ledger residual is only `-0.00044 F_sun` equivalent.
The last two 50-code windows of the gain-`40` case had upper flux `0.6044`
and `0.6033 F_sun` and stored-energy endpoint rates `+0.0363` and
`+0.0081 F_sun`; each was only `0.47` or `0.46` vertical crossing times.
Its sampled late energy-ledger residual was `+0.0000051 F_sun` equivalent.
Its `t=1000` full-3D `tau=2/3` temperature was `5128 K`, versus `5247 K`
with gain `10`. The energy-budget appearance is better, but neither the
window duration nor the solar-flux criterion passes. The comparison HTMLs
are `rempel_box_pilot/mass_rate40_vs10_t851_1000/index.html` and
`rempel_box_pilot/mass_rate40_vs10_history_t851_1000/index.html`.
The gain-`40` `t=800`–`850` HST samples (499 common times) match the shorter
case bit-for-bit despite different output settings, confirming a controlled
continuation. A further `t=1000`–`1200` run tests persistence of the
lower-amplitude state.
An interim `t=450`–`500` diagnostic has least-squares stored-energy trend
`+0.016 F_sun` but endpoint storage rate `+0.678 F_sun` (sampled ledger
residual `-0.0006 F_sun`). The small regression slope alone would therefore
give false confidence; neither measure can yet establish thermal stationarity.
Over `t=500`–`600`, a 100-code window, the corresponding values are
`+0.026` and `-0.014 F_sun`. The dependence on window length is why the
next continuation retains the conservative screening thresholds and uses
longer windows.
For WOR-74, the profile-derived HD2 inflow entropy is
`1.581097454457282e9 erg g^-1 K^-1` at the 96 km bottom-cell center;
the 48 km first cell uses `1.581460760857411e9`. Same-checkpoint
`t=650`–`750` sensitivity pilots at 96 km use the baseline plus and minus
`5e6 erg g^-1 K^-1`, with `s_in_from_profile=false` and `tau=2`.
Both pilots completed with all reported Newton solves converged and no
slow-MG warning or fatal. Over their common `t=650.11`–`750` interval,
the first 25-code-time lower hydro-flux differences are `+1.872` and
`-1.310 F_sun`; the last 25-code-time differences are `+1.858` and
`-0.838 F_sun`. The corresponding late upper-flux differences are only
`-0.00193` and `+0.00117 F_sun` relative to baseline `0.61679 F_sun`.
The upper response is too small and has the opposite sign to a simple
steady calibration inference. These pilots measure transient sensitivity,
not a calibrated solar flux. Their comparison is in
`rempel_box_pilot/inflow_entropy_sensitivity_t650_750/index.html`.
A corrected-source
96 km `t=40`–`100` continuation from the corresponding old-source restart
completed to make the early 48 km resolution comparison less confounded by
the gravity-source switch. It converged in 830/830 Newton solves with no
slow-MG warnings or fatals. Over the matched `t=41`–`100` window, mean top
flux is `0.538 F_sun` at 96 km versus `0.489 F_sun` at 48 km; maximum
mass drift is `0.279%` versus `0.188%`. At `t=100`, the 96 km full-volume
`tau=2/3` surface has mean temperature `5026 K` and vertical-velocity RMS
`1.97 km/s`, versus `4990 K` and `1.72 km/s` at 48 km. This is a transient
resolution comparison, not a convergence or solar-agreement claim. The
matched plot and metrics are in
`rempel_box_pilot/resolution_comparison_face_gravity_t41_100/index.html`.
The initial profile's column internal energy, `integral rho*specific_u dz`,
divided by `F_sun` gives a whole-box thermal time of about `5.24 d`
(`2.0e4` code-time units). The top 1 Mm gives about `0.44 h`
(`70` code units), and the top 0.5 Mm about `0.026 h` (`4` code units).
These energy-over-flux times are scale estimates, not
measured response times. They explain why a bottom-entropy perturbation can
change the lower hydro flux promptly but leave the top nearly unchanged for
tens of code units. WOR-75 tracks a discrete initial-equilibrium audit and
ways to shorten the thermal transient without changing the target physics.
At the measured `t=650`–`850` 96 km four-rank cost of 3,285 wall seconds
for 200 code-time units, a linear extrapolation to one whole-box thermal
time (about 20,000 code units) is around 90 wall hours under similar load.
This is an order-of-magnitude planning figure, not a convergence-time
prediction; multiple thermal times or higher resolution would cost more.
The initial 1D model expects nearly all of `F_sun` to be carried by
convection at z-code `-3` and `0` (radiative fractions about `6e-6` and
`1e-4`), while the 3D pgen starts with zero velocity and therefore zero
resolved convective flux. In the 96 km startup, top flux drops from about
`1.0 F_sun` at `t=0` to a `t=5`–`10` mean of `0.47 F_sun` with the
legacy gravity source. A fresh corrected-source `t=0`–`10` run converged
120/120 Newton solves, with no slow-MG warnings or fatals; its mean top
flux at `t=5`–`10` is `0.441 F_sun`. A matched control with zero initial
temperature noise also gives `0.441 F_sun`, so the imposed 0.1% temperature
perturbation is not the cause of the early top-flux drop. Both corrected
cases begin at `0.997 F_sun` at the top boundary.

A 1D diagnostic sampling the imported profile at the actual 96 km cell
centers, then applying the NR face-opacity and FLD limiter formulas, gives
`0.723 F_sun` through the uppermost *interior* face at `z=2.88`, versus
`0.821 F_sun` from the continuous-profile diagnostic at the same height.
The approximate centered hydrostatic residual reaches `5.6%` at the upper
face. The initial top-boundary flux of `0.997 F_sun` therefore exceeds
both the sampled interior radiative flux and the initial resolved convective
flux (zero), providing a concrete early-cooling mechanism. This sampled
estimate excludes ghost faces, temperature noise, and exact C++ EOS/opacity
interpolation; the boundary and stage-by-stage residuals still need an exact
audit before assigning the entire flux deficit to discretization.
The model split, thermal times, startup histories, and sampled interior
residuals are plotted in
`rempel_box_pilot/initial_flux_and_thermal_time_diagnostic/index.html`.
The sampled-grid comparison at vertical cell counts 32, 64, and 128 gives
uppermost interior radiative fluxes `0.723`, `0.795`, and `0.814 F_sun`
for the old profile, approaching the continuous model value `0.821 F_sun`.
The corresponding maximum centered HSE estimates fall `5.62%`, `1.48%`,
and `0.376%`. These are resamplings of one 1D profile, not additional 3D
simulations.

The profile builder itself used `R=3F/(cE)` when deriving its temperature
gradient. That is the diffusion-limit approximation, whereas both the
generator's diagnostic and the NR solver use the variable Levermore–Pomraning
limiter. At the top, where `F/(cE)=0.5`, `R=1.5` gives only about
`0.82 F_sun`. The builder now solves `lambda(R)*R=F/(cE)` algebraically;
for this top value the matching `R` is 2. A newly generated profile has
continuous-model upper radiative flux `0.997 F_sun` and sampled 96 km
uppermost interior flux `0.892 F_sun`. With the *same existing solver
binary* and corrected gravity source, both profiles completed 120/120
Newton solves without warnings or fatals through `t=10`. The new profile
raises the `t=5`–`10` upper flux only from `0.441` to `0.460 F_sun`; the
initial `tau=2/3` temperature changes from `5769` to `5869 K`. The
limiter-consistent profile is a real fix to the initial 1D construction,
but does not remove the early 3D thermal transient. Its matched comparison
is `rempel_box_pilot/profile_closure_startup_comparison/index.html`.
The HD2 lower-boundary initialization derives its pressure and entropy from
each profile: `PBND=227.23` and `s_in=1.58110e9` for the old profile,
versus `215.82` and `1.58675e9` for the new one. Thus these runs compare
the *complete* profile-based initial conditions, not a perturbation with
inflow entropy held fixed. A separate fixed-entropy control is needed before
attributing longer-time differences solely to the radiation gradient.
Both old/new profile-based 96 km cases continued from `t=0` to `40`, with
478/478 Newton solves converged in each, no slow-MG warnings or fatals.
The new profile gives `t=30`–`40` mean top flux `0.468 F_sun` versus
`0.451 F_sun` for the old one, and `t=40` full-3D `tau=2/3` surface
temperature `4912 K` versus `4857 K`. Both remain strongly transient;
the comparison is
`rempel_box_pilot/profile_closure_t40_comparison/index.html`.
An additional new-profile run fixes `s_in` to the old profile's
`1.581097454e9` instead of deriving it from the new profile. It also
completed 477/477 Newton solves without warnings/fatals. Its `t=30`–`40`
top-flux advantage over the old profile is `+0.01696 F_sun`, essentially
the same as the profile-derived-entropy run's `+0.01711 F_sun`; its `t=40`
`tau=2/3` surface temperature is `4911 K`. This supports the initial
thermal structure, rather than the `+5.65e6` inflow-entropy change, as the
cause of the short-time difference. The initial lower pressure still differs
because it is derived from each profile; the controlled comparison is
`rempel_box_pilot/profile_closure_fixed_sin_t40_comparison/index.html`.

The matched *old-profile* zero-temperature-noise control is more decisive
about convective transport. Through `t=5`–`10`, seeded and zero-noise cases
have essentially identical top flux (`0.4417 F_sun`). By `t=30`–`40`,
the 0.1%-seeded case has `0.4512 F_sun` but the zero-noise case only
`0.2305 F_sun`. At `t=40`, the 3D `tau=2/3` surface temperature is
`4857 K` versus `4027 K`; vertical RMS speed on that surface is `1.345`
versus `0.429 km/s`. Both solver runs completed (478 and 477 converged
Newton solves) with zero slow-MG warnings/fatals. The nonzero speed in
the zero-noise case includes coherent vertical motion and should not all
be called convection. Nevertheless the large flux and surface-temperature
split strongly supports the lack of initially resolved convective transport
as a major startup-cooling mechanism. See
`rempel_box_pilot/initial_perturbation_comparison_t0_40/index.html`.
A third otherwise matched case with 1% initial temperature perturbations
completed 483/483 Newton solves without warnings/fatals. Its `t=5`–`10`
upper flux is still `0.4416 F_sun`, virtually identical to 0 and 0.1%
perturbations, so even tenfold larger noise does not remove the immediate
cooling. At `t=30`–`40`, however, the mean top flux increases from
`0.2305` (no noise) and `0.4512` (0.1%) to `0.5013 F_sun` (1%). The
`t=40` 3D `tau=2/3` surface temperatures are `4027`, `4857`, and
`4931 K`, respectively. Stronger seeding speeds the emergence of
convective transport but leaves a large solar-flux deficit and does not
solve the long thermal relaxation. The three-case plots are in
`rempel_box_pilot/initial_noise_sweep_t0_40/index.html`.
Full-3D horizontal-mean temperature profiles at `t=0,10,20,30,40`
localize this cooling to the upper layers. At `t=40`, the mean temperature
change over the upper 1 Mm is `-2530 K` without an initial perturbation,
`-1014 K` with 0.1%, `-846 K` with 1%, and `-1031 K` for the
FLD-consistent profile with the old inflow entropy. These are differences
from each case's own initial 3D state, not an energy budget. The plots and
height-resolved arrays are in
`rempel_box_pilot/thermal_stratification_t0_40/index.html`.
At 48 km (128×128×64), both old- and new-profile `t=0`–`10` pilots finished
240/240 Newton solves without slow-MG warnings or fatals. The sampled
uppermost interior FLD flux improves from `0.795` to `0.969 F_sun`, but the
`t=5`–`10` mean top flux improves only from `0.38243` to `0.39260 F_sun`
(the `t=10` values are `0.34661` and `0.35218`). The impact is smaller than
at 96 km, so the fixed 1D radiative gradient alone does not resolve the
initial cooling. See
`rempel_box_pilot/profile_closure_48km_t10_comparison/index.html`.
The matched 48 km **full-3D** runs confirm the optical-depth-surface result.
At `tau=2/3`, the initial horizontal-mean temperature is `5704 K` for the
old profile and `5798 K` for the FLD-consistent one. At `t=10` they have
fallen to `4447` and `4461 K`, shrinking their initial `94 K` difference
to only `14 K`. Both completed 240/240 Newton steps with no warnings or
fatals. This is a strong indication that changing the 1D radiative gradient
alone does not prevent the early 3D cooling. The comparison is
`rempel_box_pilot/profile_closure_48km_3d_t10_comparison/index.html`.
At 96 km the old and new profile cases with the *same* HD2 inflow entropy
continued to `t=100`. Over `t=41`–`100`, their mean top fluxes were
`0.54043` and `0.55992 F_sun`; over `t=85`–`100`, they were `0.57746` and
`0.60201 F_sun`. The final full-3D `tau=2/3` temperatures were `5031` and
`5072 K`. All 829 and 839 continuation Newton steps converged respectively,
without slow-MG warnings or fatals. The sampled new-case `t≈40`–`100`
energy-ledger residual was `0.00042 F_sun` equivalent. The benefit is still
small, and the lower PBND remains profile-derived even with entropy fixed;
neither case has a steady solar flux. See
`rempel_box_pilot/profile_closure_fixed_sin_t41_100_comparison/index.html`.
The lower radiation ghost target helper also had a local copy of the
limiter with `6+2R+R^2` in the denominator; the shared FLD limiter uses
`6+3R+R^2`. The code now calls the shared limiter. This correction is
negligible at the current deep lower boundary where `R` is tiny. It was
rebuilt as binary SHA-256 `4b616a13...` and compared against the legacy
`f0bee32b...` binary at 96 km through `t=10`, holding the input and tables
fixed. Both finished 120/120 Newton solves without warnings/fatals. The
mean top flux was `0.54335700139 F_sun` in both; the largest relative HST
differences were `4.9e-16` in top flux and `7.3e-14` in lower hydro flux.
This correction is not the cause of the startup cooling. See
`rempel_box_pilot/ghost_limiter_fix_t0_10_comparison/index.html`.

The earlier one-rank startup to `t=1` gave sampled residuals `+0.0801`
and `-0.0721 F_sun` for the old and new sources, so a one-second diagnostic
was insufficient to establish the improvement. The remaining early startup
discrepancy and the exact RK-stage radiation/boundary flux audit remain in
WOR-72. Use `--history-precision 16` to avoid default six-significant-digit
history rounding in short ledger probes.

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

## Long-run continuity and entropy response

The gain-40 mass-feedback baseline continued from `t=800` to `1200` with
5699/5699 Newton solves converged, no slow-MG warnings, and no fatal errors.
The `t=1000`–`1200` execution session received SIGTERM at `t≈1191`, not an
out-of-memory failure; the last numbered restart was used to finish `t=1200`.
The joined history and late-window checks are in
`rempel_box_pilot/mass_rate40_combined_t800_1200/index.html`. The last two
approximately 100-code-time windows contain only `0.900` and `0.898`
vertical crossing times. Their top fluxes average `0.606` and `0.612 F_sun`.
The stationarity screen fails the crossing-duration and solar-flux checks;
the solver's successful continuation is not a thermal-equilibrium claim.

A same-checkpoint `t=1000`–`1200` inflow entropy sensitivity sweep uses
`s_in=1.581097454457282e9 ± 5e6 erg g^-1 K^-1`, with identical code,
grid, CFL, HD2 feedback, and tables. The plus/minus runs completed
2885/2885 and 2816/2816 converged Newton solves, respectively, without
slow-MG warnings or fatals. Compared with the joined baseline, the top
flux response over the final 25 code-time units is `+0.00890` and
`-0.00182 F_sun`; over the final half of the common interval it is
`+0.00237` and `-0.00362 F_sun`. The bottom hydro-flux response over the
final 25 units is much larger and asymmetric (`+0.257` and `-0.687 F_sun`).
The system has not relaxed enough to infer a steady entropy calibration
from these numbers. Plots and exact windows are in
`rempel_box_pilot/inflow_entropy_sensitivity_t1000_1200/index.html`.

The checkpoint restart itself is deterministic: two independent restarts
from the `t=1190.065` checkpoint give identical history values. However,
the first post-checkpoint step differs from the uninterrupted path by
`-1.40e-6` relative in total-plus-gravity energy and by
`-0.096 F_sun` in instantaneous lower hydro flux. The checkpoint's
energy and HD2 pressure scale match the original at its saved time.
WOR-76 tracks this restart-to-continuous difference. A zero-step restart's
`MassDrift=-1` is a diagnostic initialization artifact: its static
`hd2_current_mass` is populated in `UserWorkInLoop` at the first step,
where the drift returns to a normal value. Do not interpret the zero-step
sample as physical mass loss.

The same baseline continued to `t=1300` with 1423/1423 further converged
Newton steps, no slow-MG warnings, and no fatals. The four-segment
`t=800`–`1300` result has 7122/7122 converged solves. The two final
125-code-time windows span `1.123` and `1.108` vertical crossing times.
Their top fluxes are `0.6112` and `0.6148 F_sun`; their bottom hydro
fluxes are `0.7377` and `0.5873 F_sun`. Both energy endpoint rates
(`+0.0788` and `-0.0230 F_sun`), both fitted energy trends (`+0.0301`
and `-0.0150 F_sun`), mass drift (below `0.059%`), top-flux change,
and vertical RMS stability meet the screening thresholds. Only the
requirement that both top fluxes lie within 10% of the solar flux fails.
The final full-3D `tau=2/3` temperature is `5152 K`. See
`rempel_box_pilot/mass_rate40_combined_t800_1300/index.html`. The whole-box
thermal time estimate is much longer than this run, so the near-flat top
flux is not proof of final thermal equilibrium.

The lower hydro-flux oscillation amplitude has decreased over five roughly
100-code-time windows from `4.557`, `3.562`, `2.830`, `2.299` to
`1.914 F_sun` standard deviation across `t=800`–`1300`. The dominant
period in these windows is about `6.2` minutes, and the horizontal-mean
vertical-velocity variance fraction falls from `0.308` to `0.188`.
This establishes damping under gain-40 mass feedback at 96 km, but does
not identify the acoustic mode's cause. The comparison is in
`rempel_box_pilot/gain40_late_oscillation_t800_1300/index.html`.

For WOR-76, an experimental post-NR FLD ghost synchronization did not
reduce the continuous-versus-restarted first-step difference and was
reverted. At the checkpoint time, every full-3D active-cell primitive
value in the 32 blocks matches exactly between the original output and a
zero-step restart. The five user-output variables differ at only 13
locations, with maximum relative difference `3.35e-11` over the full
array. The source of the first-step divergence remains unresolved.
Ghost-zone output localized a real HD2 ordering defect: the lower
hydro boundary ghosts were built before `UserWorkInLoop` updated the
pressure means and PBND feedback state. A zero-step restart rebuilt them
from the new state. The old lower ghost density differed by up to
`0.1103` code units (`1.53e-4` relative to the maximum density).
Reapplying the physical lower boundary after that HD2 update reduces
the ghost comparison to float32 output roundoff and keeps a `t=1300`–`1320`
smoke run converged for 288/288 Newton solves. The first post-restart
total-energy difference is reduced only from `-1.399e-6` to `-1.304e-6`
relative, so WOR-76 remains open. Combining the HD2 fix with a post-step
FLD ghost synchronization did not further reduce the difference; that
additional generic synchronization was also reverted.

## Next scientific gates

1. Continue the corrected Rempel-sized box past `t=300`, compare the HD2
   mass-balance-off control, and audit its boundary/gravity energy budget.
   Solver convergence alone does not establish thermal stationarity.
2. Keep the confirmed, tracked `simple_convection_lhllc_fld` series as the
   baseline. Review the untracked `nrfld_solar_convection` files separately
   before using their safety caps or flux controls in a comparison.
3. Extend the non-magnetic 6.144 × 6.144 × 3.072 Mm pilot with at least two grid
   spacings. Compare time-averaged optical-depth-aligned temperature/density,
   vertical velocity, radiative/enthalpy/kinetic fluxes, granule scale and
   surface brightness distribution with published definitions. Keep gray FLD
   limitations explicit near optical depth unity.
4. Once WOR-73's boundary mode is controlled and long-window thermal trends
   are measured, use WOR-74 to determine whether the imported lower upflow
   entropy needs calibration for the gray-FLD solar-flux target. Change
   `s_in_from_profile=false` and an explicit `s_in` only in matched restarts;
   do not tune a transient top-flux sample.
