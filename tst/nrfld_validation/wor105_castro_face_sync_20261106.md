# WOR-105: CASTRO AMR radiation-flux comparison and local diffusion tests

## What CASTRO actually does

Zhang et al. (2011), Section IV, defines the implicit radiation flux as
`F_R = -d grad(E_r)` and stores coarse and fine fluxes at refinement borders.
The register records `delta F_R = <F_R,fine> - F_R,coarse` (their Eq. 47 and
AMR outline). The difference is added as a source to the **next implicit
diffusion update**. This is their deferred sync for subcycled AMR; the paper
explicitly notes that the register must survive checkpoint/restart and
regridding. It does not prescribe a coarse/fine series-resistance formula.

The current CASTRO source forms face `d = c lambda / kavg(kappa_L,kappa_R,dx)`.
Its `kavg` option 2 uses arithmetic averaging in thin material, harmonic
averaging in thick material, and the Howell–Greenough cap at a transition.
Athena++ `RadFLD::FaceOpacity` already has this same formula. CASTRO's
`levelFlux` and `levelFluxReg` then form and register face fluxes.

References:

- [Zhang et al. 2011, Sections III.2 and IV](https://arxiv.org/html/1105.2466v2)
- [CASTRO face-opacity averaging, source commit 371ba40](https://github.com/AMReX-Astro/Castro/blob/371ba40de0ef71963fa8f185d45c7d54e84ec041/Source/radiation/rad_util.H#L264-L295)
- [CASTRO face coefficient](https://github.com/AMReX-Astro/Castro/blob/371ba40de0ef71963fa8f185d45c7d54e84ec041/Source/radiation/RadSolve.cpp#L331-L405)
- [CASTRO implicit flux/register update](https://github.com/AMReX-Astro/Castro/blob/371ba40de0ef71963fa8f185d45c7d54e84ec041/Source/radiation/SGRadSolver.cpp#L191-L244)
- [Commerçon et al. 2011, RAMSES FLD, Section 3.7](https://arxiv.org/html/1102.1216)

Athena++ advances the AMR hierarchy at one global time step and solves the
composite implicit residual. WOR-102's existing shared-face path is the
same-time spatial counterpart of CASTRO's fine-face flux correction: the
fine face sends `D` and `D*E_f`, the coarse face uses their area averages,
and its residual contains the opposite of the sum of fine fluxes. It does
not need a deferred flux register. This branch adds explicit source comments
for that relationship and keeps the selected solver method unchanged.
The independent RAMSES FLD description is closer to this time integration:
it solves diffusion once per coarse step across the whole AMR hierarchy,
uses the coarse/fine center distance of `3h/2` for each fine-face gradient,
and explicitly sums the fine-face fluxes into the coarse-face energy balance
(Section 3.7, configurations 2 and 3). Thus the shared-flux construction is
also supported by a published composite-AMR FLD discretization; the
particular opacity average still needs its own accuracy test.

## Experimental resistance alternative

The user also proposed the physical resistance of the coarse and fine
half-paths. This is **not** CASTRO's documented averaging rule. For fixed
`lambda=1/3`, coarse width `H=2h`, and piecewise constant extinction
`sigma_c, sigma_f`, a one-dimensional interface has

`F = -c (E_f-E_c) / [3 h (sigma_c + sigma_f/2)]`.

With WOR-102's two-point fine ghost `(E_f+2E_c)/3`, this is implemented by
`sigma_face=(2 sigma_c+sigma_f)/3` on fine cells bordering a coarser block.
The coarse side still receives the fine-face moments, so conservation is
unchanged. This trial is restricted to radiation-only, fixed-limiter,
Cartesian Test C with shared coefficients and requires
`ATHENA_NRFLD_EXPERIMENTAL_SERIES_RESISTANCE=1`. The standard CASTRO-like
face opacity remains selected without that flag. Thin-limit flux capping is
not established for the resistance branch, so it is not a production option.

## 1D-like analytical mode

`wor105_diffusion_mode.py` initializes exact cell averages of
`E_r=1+0.1 cos(2 pi x)` at constant density and opacity, with `lambda=1/3`.
For one backward-Euler step, the continuum cell-average reference has the
same cosine multiplied by `1/(1+D(2 pi)^2 dt)`, `D=c/(3 sigma)`. The test
uses periodic x1, no transverse variation, AMR in `0.25<=x1<=0.75`, and
`kappa=4`, corresponding to optical depth at least 250 per fine cell in
this uniform-density N32 case. It measures the AMR leaf values against the
analytic cell averages and records the global implicit diffusion ledger.

| Base N | Mode | Time | Mean absolute error | Max error | Global diffusion ledger |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | Existing | 0.001 | 8.885e-7 | 1.436e-5 | +4.784e-8 |
| 32 | Shared fine-face | 0.001 | 4.917e-7 | 3.099e-6 | -9.15e-21 |
| 32 | Series resistance | 0.001 | 4.917e-7 | 3.099e-6 | -9.15e-21 |
| 64 | Existing | 0.000605 | 1.248e-7 | 3.219e-6 | +6.824e-9 |
| 64 | Shared fine-face | 0.000605 | 7.544e-8 | 7.153e-7 | +6.72e-20 |

N32 four-rank results matched serial field errors and kept the shared-face
global diffusion ledger at `-9.32e-21`. At each of the two x1 coarse/fine
planes, the old operator's relative face mismatch was `0.4882%`; the shared
operator's was at most `1.91e-15` in this four-rank run. The series branch
matches the shared
branch exactly for constant opacity. The N64 mean errors approach the
float32 VTK output precision, so the N32 comparison is more informative.
All solves converged strictly and kept positive radiation energy.

## 1D-like density jump

The existing Test C one-step profile comparison was rerun with no transverse
variation, `kappa=4`, and dense-slab ratios `0.1` and `10`. The uniform fine
numerical solution is the reference. Values below are mean absolute errors
in the **one-step change** after projecting that reference onto AMR leaves.

| Density ratio | Base N | Existing | Shared fine-face | Series resistance |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 32 | 1.661e-6 | 2.366e-6 | **1.371e-6** |
| 0.1 | 64 | **9.406e-7** | 1.383e-6 | 1.506e-6 |
| 10 | 32 | 3.297e-7 | 3.912e-7 | **2.757e-7** |
| 10 | 64 | 3.651e-7 | 3.716e-7 | **3.111e-7** |

The resistance trial improves three of these four local comparisons relative
to the existing stencil, but degrades the dilute-slab N64 case. Its interface
diffusion ledger remains at `~1e-21`; the shared-face branch also conserves.
A four-rank N32 dense-slab run reproduced the serial errors and ledger.
These are one-step, float32-output comparisons against a numerical fine-grid
reference, not a complete two-material analytic benchmark. In particular,
the exact extinction presented to the fine ghost at a material jump should
be audited before attributing the N64 difference solely to the averaging
formula.

## Decision

Keep the CASTRO-like `kavg` shared fine-face method as the main conservative
candidate. Retain series resistance behind its experimental flag; the current
results do not justify replacing the selected face opacity. The analytic
uniform test supports the accuracy and conservation of the current candidate
in the thick diffusion limit. A longer two-material test with an exact
piecewise-diffusion reference and double-precision output is the next
accuracy gate. The Newton flux limiter remains frozen throughout each solve.

Reproduce the local uniform-opacity checks after building with
`python3 configure.py --prob=nrfld_diff_amr_interface -nrmgfld -mpi -hdf5`
and `make -j8`:

```sh
python3 tst/nrfld_validation/wor105_diffusion_mode.py \
  --binary bin/athena --output /tmp/wor105_diffusion_check \
  --resolutions 32 64 --opacity 4 --cfl 2.0 --series-resistance
```

The density-jump comparison uses the extended
`wor102_profile_comparison.py` with `--fixed-limiter`,
`--series-resistance`, `--skip-two-point`, `--opacity 4`,
`--transverse-amplitude 0`, and `--contrast 0.1` or `10`.

Local output roots: `/tmp/wor105_diffusion_mode_cfl2`,
`/tmp/wor105_diffusion_mode_final`,
`/tmp/wor105_series_thick_dilute`,
`/tmp/wor105_series_thick_dense`, and
`/tmp/wor105_series_dense_mpi4`.
