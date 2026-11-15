# WOR-122: initial discrete FLD face-flux audit (2026-11-15)

This follows `wor122_rsg_surface_thermal_rebalance_20261115.md`. The
question is why the mapped 1-D LTE/FLD RSG profile cools at 813–817 R☉
and warms nearby during the first 0.255 day. The comparison uses the same
Tomida material EOS, WOR-55 opacity, trial LTE/FLD exterior, PPM (`xorder=3`),
uniform Cartesian grids, and first positive time step of 0.1 code units.
The physical time unit is 21,999.9657 s. Large run files remain in `$WORK`.

## Direct operator diagnostic

The opt-in `nrfld/initial_diffusion_audit` reads the radiation state and
six face coefficients when `CalculateCoefficientsOnce` constructs the **first
positive-step, pre-Newton** discrete operator. For an interior face, the
outward flux from a cell is `Dface*(Er_cell-Er_neighbor)/dx`; at a Marshak
face it uses `MarshakFlux` itself. The cell diffusion source is minus the sum
of its six outward fluxes divided by `dx`, exactly matching the residual in
`CalculateCoefficients`. It logs volume-integrated radiation energy,
diffusion source, absolute diffusion source, approximate radial flux, local
thermal coupling, and physical-boundary loss in configurable radius bins.
The audit is read only, off by default, and logs once per MeshBlock. Its
time-zero meaning is *after the explicit first-stage update and before the
first implicit FLD solve*, rather than the untouched HDF initial condition.
The input uses 1-R☉ bins from 750 to 900 R☉; the printed radius is in code
units, equal to R☉ here.

A local 64³ whole-box audit included all 262,144 cells. Its summed diffusion
power was `−142509.85015633283` code units and its six-face Marshak loss was
`+142509.85015633283`; their sum was zero to the reported precision. This
checks the sign, face accounting, and block-interface conservation of the
diagnostic. The 64³ underresolved surface is a code check, not a candidate
physical model.

XD M-debug jobs **340105** (256³, 44 s) and **340109** (128³, 10 s) both
completed with zero opacity-domain excursions and converged Newton solves.
They wrote compact results to `initial_diffusion{256,128}.json` in the
`wor122_surface_force_20261115` directory. Job 340104 stopped before
initialization because the new input keys had not been added to the remote
input file; no physics step ran in it. The corrected job is 340105. All
XD data and scripts are under `$WORK/wor122/diffaudit{128,256}_t01/` and
`$WORK/wor122/diffaudit{128,256}_t01.sh`.

## Resolution dependence

The table compares the approximate spherical luminosity from the actual
Cartesian face coefficients, `4πr_mid²` times the shell-mean radial flux,
with a volume-weighted shell average of the saved 1-D profile's radiation
flux. The 1-D column agrees with a direct Levermore–Pomraning reconstruction
from its `Er`, density, and Rosseland opacity to 0.4% in 800–817 R☉ and
to about 2% in 817–830 R☉. The Cartesian estimate is a cell-centered
projection of face fluxes, so it is **not** a conservative spherical
surface integral; the diffusion power below is the exact discrete residual.

| Shell [R☉] | 1-D `Lrad/Lheat` | 128³ discrete | 256³ discrete | 128³ diffusion/`Lheat` | 256³ diffusion/`Lheat` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 750–800 | 0.01165 | 0.01228 | 0.01160 | — | — |
| 800–813 | 0.00938 | 0.9362 | 0.1359 | −1.8097 | −0.5018 |
| 813–817 | 0.23352 | 1.9717 | 0.8453 | −1.1852 | −0.9179 |
| 817–830 | 0.99835 | 2.1364 | 1.2275 | +1.5779 | +0.5513 |
| 830–850 | 0.99996 | 1.0000 | 0.9265 | +0.6500 | −0.0443 |

The 800–813 R☉ discrete luminosity is about 100 times the imported 1-D
flux at 128³ and 14.5 times at 256³. In 813–817 R☉ it is 8.4 and 3.6
times the 1-D value. The transition is 3.35 R☉ wide, versus cell widths
15.79 and 7.89 R☉. The discrete flux transition spreads **inward** over
many cells and puts strong cooling beneath the intended radiative join.
The full radial curves and per-bin diffusion powers are in
`doc/figures/wor122_initial_diffusion_resolution_20261115.png`. The 64³
one-step audit further gives 8.62 `Lheat` at 800–813 R☉, consistent with
the strong improvement as the grid is refined. Neither 128³ nor 256³
resolves the actual layer. These results support an initialization and
resolution problem, not evidence of a defective diffusion solver.

The imported profile is also not a complete thermal steady state for the
initially almost static 3-D gas: its radiative luminosity is about
`0.014 Lheat` at 775 R☉ and about `Lheat` at 840 R☉, with no matching
initial convective enthalpy flux. Thus the 1-D radiation flux itself has
a large outward increase across the surface layer. Improving spatial
resolution can localize its divergence but cannot remove the need for
convective transport or a controlled thermal relaxation. The initial
pre-Newton diffusion rate should not be extrapolated as a first-step
explicit energy change because gas–radiation coupling is stiff and the
implicit state evolves during the step. Indeed, 830–850 R☉ has a small
negative initial diffusion power at 256³ yet is warmer at code time 1.

## Next model gate

Before another long RSG run, construct an initial model whose 3-D
**discrete** gas force, FLD face flux, and energy flow are jointly acceptable.
The surface radiation transition needs several cells or a physically
justified broadened layer re-solved for the same FLD closure and HSE; simply
averaging the current profile into cells already failed the short gate.
Then measure the total and convective luminosities during a short
relaxation and extend only if the exterior cooling and force drift remain
small. A uniform grid that gives four cells across 3.35 R☉ would need
roughly 2,400 cells per axis over this box, so targeted surface refinement
or a validated broadening is the practical route. No long run was started.

Reproduce the compact numbers with `wor122_initial_diffusion_audit.py` and
the figure with `wor122_plot_initial_diffusion.py`. The logs and HDF files
are on XD; the local 64³ check is in
`/home/kosuke/simulation/test/wor122_initial_diff64_full/`.
