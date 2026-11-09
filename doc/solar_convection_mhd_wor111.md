# WOR-111: magnetic solar-convection FLD pilot

## Scientific question

The 96 km nonmagnetic local box reached an outgoing flux of about
`0.63 F_sun` at code time 24000. Can magnetic fields account for that
deficit, and does the MHD hyperbolic step remain compatible with the
implicit gray FLD step?

Rempel (2014, [ApJ 789, 132](https://arxiv.org/abs/1405.6814)) studied a
6.144 × 6.144 × 3.072 Mm box with zero net vertical flux. Its OSb lower
magnetic boundary uses symmetric components with the HD2 flow condition;
the top magnetic condition can be vertical or potential. Most dynamo
experiments started from relaxed `B=0` convection with a `10^-3 G` seed.
Magnetic power spectra required an 8 km or finer grid in that study.

Our 96 km pilot uses the same box size and an OSb-like symmetric lower
field/vertical upper field, but a coherent 50 G sinusoidal vertical seed.
That seed is intentionally strong enough to exercise MHD+FLD at local
resolution. It is **not** a small-scale dynamo experiment and cannot test
Rempel's 8 km spectral results. The gray FLD photosphere also differs from
Rempel's short-characteristics radiative transfer.

Rempel (2020, [ApJ 894, 140](https://arxiv.org/abs/2004.01795)) found a
quiet-Sun `mean |Bz|=69 G` case about 0.6% brighter than the nonmagnetic
reference at fixed setup. This suggests magnetism alone is unlikely to
explain a roughly 37% `F_sun` shortfall here; the differing radiation and
thermal structures prevent direct transfer of the 0.6% number.

## Code and experiment

- Branch: `codex/wor111-solar-mhd-fld-20261109`.
- Build: `python3 configure.py --prob=simple_convection_lhllc_fld
  --coord=cartesian --eos=general/eos_table --flux=hlld_fld -b -nrmgfld
  -mpi -hdf5 --cxx=g++ && make -j8`.
- Initial data: the **same** six primitive fields exported from the
  nonmagnetic full-volume `t=24000` dump using
  `tst/nrfld_validation/export_solar_convection_snapshot_seed.py`.
  The seed is 64 × 64 × 32 with 96 km cubic cells. This is a new run at
  `t=0`, not a bitwise continuation of the old hydro restart; HD2's
  pressure/mass feedback internal state is reinitialized. Comparing the
  two MHD cases isolates the seed field over the pilot interval.
- Input: `inputs/radiation/athinput.simple_convection_mhd_fld_96km_pilot`.
  The zero-field MHD control and `magnetic_seed_gauss=50`,
  `magnetic_seed_mode=zero_net_vertical` cases share EOS, opacity,
  radiation, boundaries, seed and CFL 0.3. The seed has
  `Bz = 50 sin(2πx/Lx) G`, independent of height, so the initial discrete
  divergence is exactly zero. The initial horizontal mean `Bz` is zero.
- Output: `/mnt/d/output/rad/wor111_mhd_fld_20261109/`.

To recreate the shared initial state, use the three source paths recorded
in `data/t24000_seed.json` under that output root:

```bash
python3 tst/nrfld_validation/export_solar_convection_snapshot_seed.py \
  --prim /path/to/t24000.prim_full.athdf \
  --uov /path/to/t24000.uov_full.athdf \
  --input /path/to/athinput.generated \
  --output /mnt/d/output/rad/wor111_mhd_fld_20261109/data/t24000_seed.bin
```

Start the two runs with the frozen MHD binary `athena.mhd.diode` and
`problem/magnetic_seed_gauss=0` or `50`, then restart each `t=2` final
checkpoint through `t=10`. The input contains a full conserved dump
(`output9`) to verify the magnetic-energy invariant. The HD comparison
uses `--flux=lhllc` without `-b`; `--flux=lhllc_fld` is not a valid
`configure.py` argument. Binary checksums are in the output root.

The problem generator now initializes face-centered field and includes
magnetic energy in conserved hydro energy. Its HD2 lower and upper
hydrodynamic ghost construction subtracts magnetic energy when recovering
the gas internal energy and restores it to the total energy. The lower
magnetic ghost field is symmetric; the upper tangential field is
antisymmetric (`magnetic_top_mode=vertical`). MHD builds write total
magnetic energy, a normalized discrete divergence, and the mean unsigned
top `Bz` to the history file.

The general-EOS FLD active-cell exchange also has to subtract `B²/2` when
loading gas energy from the conserved hydro state and restore `B²/2`
after its implicit update. An earlier pilot omitted those terms and failed
near `t=10.8`; the corrected calculation reached `t=12`. See WOR-112.

For the `rempel_outflow` upper boundary, setting the ghost normal velocity
to zero for downward active-cell flow did **not** prevent the HLLD flux
from bringing mass inward. At `t=2`, the zero-field MHD run had
`TopMdotOut=-0.00492` and `FhydTop=-0.0371 F_sun`. Reflecting that
velocity, `v_ghost=|v_active|`, instead of setting it to zero removes
the spurious inflow. With the repaired boundary, the B=0 MHD top flux at
`t=2` is `0.62956 F_sun`, close to the independent HD LHLLC-FLD control
with the same revised boundary at `0.63061 F_sun`. The old runs are retained
as `*_old_top_bc`; WOR-113
tracks this fix.

In the MHD build, the old history names `FhydBottom` and `FhydTop` refer
to the **total MHD energy flux** from `phydro->flux`, including magnetic
transport. They are not a pure hydrodynamic flux. A separate exact
Poynting/enthalpy decomposition remains needed for the long-run energy
budget; do not add a second Poynting estimate to `FtotTop`.

## Local interpretation

The corrected two-code-time pair passed all 28 Newton solves per case,
and its `t=2` top fluxes were `0.62956` (B=0) and `0.62953 F_sun` (50 G).
The 50 G run's maximum normalized `div B` was `3.76e-15` at that time.

Both corrected cases continued from their `t=2` MHD checkpoints to
`t=10`, each with 139 converged Newton solves and no fatal errors. Their
mean outgoing fluxes over the **same** `t=8–10` window were `0.63326`
and `0.63366 F_sun`, respectively: a transient `+0.00040 F_sun` for 50 G.
At `t=10`, its reconstructed `tau=1` surface has
`mean |Bz|=32.05 G` and mean `Pmag/Pgas=0.00310`; the maximum normalized
`div B` through the pilot was `1.75e-14`. The full-dump energy invariant
`E_total=E_gas+E_kin+B²/2` holds to `1.12e-7` relative to gas energy
(the HDF5 field precision is 32 bit). The 50 G continuation to `t=12`
also crosses the old failure time without solver failure or energy drift.

The report generator `scripts/plot_solar_mhd_comparison.py` produces a
large-text `index.html`, diagnostic plot and `summary.json`. The report
records the achieved end time, solve status, flux difference, `div B` and
`Bz` on the reconstructed Rosseland `tau=1` surface. An interval much
shorter than a vertical crossing time or thermal relaxation time must
remain classified as a coupled-solver pilot.
Regenerate the report from the repository root with:

```bash
python3 scripts/plot_solar_mhd_comparison.py \
  /mnt/d/output/rad/wor111_mhd_fld_20261109 \
  --input inputs/radiation/athinput.simple_convection_mhd_fld_96km_pilot \
  --opacity-table /mnt/d/output/rad/wor111_mhd_fld_20261109/data/opacity_table_rhoT_3p0_6p3.h5
```

For this box, `t=10` is 224 seconds (3.73 minutes), about 0.09 of the
observed vertical crossing time near `t=24000` (approximately 115 code
time). The whole-box thermal energy divided by `F_sun` is of order
`2e4` code time. A `t=10` flux difference is therefore a transient
response only.

## Next physics test

For a genuine magnetic flux-impact measurement, repeat the zero-field and
weak random-field starts from *the same thermally relaxed high-resolution
state*, use Rempel-like OSb/V or OSb/potential magnetic boundaries, and
run long enough for magnetic saturation and then several independent
statistical windows. Measure `mean |Bz|` at `tau=1`, magnetic/kinetic
energy spectra, outgoing bolometric flux, top and bottom total energy
budgets, and a converged resolution series. At 96 km there should be no
claim of small-scale dynamo growth or of a stationary solar flux effect.
