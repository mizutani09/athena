# WOR-122: first RSG static-refinement gate (2026-11-14)

## Scope and setup

This is a local short test, not an approved RSG production run. The isolated
`wor-122-rsg-smr` branch merges the WOR-105 conservative common-face FLD
operator into the RSG source. The explicit `problem/static_refinement_trial`
flag enables that operator only for a Cartesian, coupled, one-level *static*
mesh with second-order reconstruction, user boundaries, Tomida material EOS,
and the WOR-55 opacity table. Fourth-order radiation force and core gravity
quadrature remain outside this multilevel gate.

The smoke mesh has a 64³ root grid and one level-1 patch in the positive-x,
negative-y, negative-z exterior quadrant: 63 root blocks plus 8 fine blocks.
It uses the LTE/FLD RSG profile, a fixed Freytag potential, six
Marshak-vacuum faces, and a central luminosity source of 229929.466 code
units. Root and fine cell widths are 31.58 and 15.79 Rsun. The measured
continuum radiative-luminosity transition is only 3.35 Rsun, so this patch
does **not** resolve the transition and does not cover the entire surface.

### Interface failure isolated before the accepted run

With tangential coarse/fine face reconstruction enabled, the first implicit
step produced negative pre-energy ghost Er (−4.76×10⁻⁵) on a fine block near
radius 831 Rsun. Disabling that reconstruction made the same step converge.
The RSG opt-in defaults it off; Test C and Mach 5 defaults are unchanged.
The second-order gate also disables both fourth-order initial cell-average
corrections (`correct_ic` and `correct_ic_fld_radiation`). The original
corrections substantially changed surface cells and are now rejected for
the multilevel trial. The input uses the rediscretized coarse operator;
the Galerkin hierarchy does not support this mesh topology.

## Short-run measurements

All runs below used two MPI ranks unless identified otherwise. History was
written at 16-digit precision; HDF primitive fields use single precision.
The time unit is 21999.97 s, so code time 1 is 0.255 days. The multi-step
runs capped the timestep at 0.1 code units.

| Gate | Uniform | One-patch SMR |
| --- | ---: | ---: |
| 0.1 code time: Newton | converged | converged |
| 0.1: opacity clamps | 0 | 0 |
| 0.1: relative mass change | −4.26×10⁻¹⁰ | −4.38×10⁻¹⁰ |
| 0.1: Δ(Ehydro+Erad)−Δt(Lheat−ΣLrad) | −13.583 | −13.666 |
| 1.0 code time, dt≤0.1: Lrad/Lheat | 3.6269 | 3.5612 |
| 1.0, dt≤0.1: maximum Mach | 0.03335309 | 0.03335243 |
| 1.0, dt≤0.1: minimum saved Er | 6.40×10⁻⁶ | 6.13×10⁻⁶ |
| 1.0, dt≤0.1: relative mass change | −4.26×10⁻¹⁰ | −4.38×10⁻¹⁰ |
| 1.0, dt≤0.1: integrated energy-budget residual | −1228.114 | −1227.897 |
| 1.0, dt≤0.1: elapsed wall time | 55 s | 95 s |

The integrated budget uses the end-of-step face luminosity at each history
time and sums each time interval; its absolute residual also contains the
shared hydro/gravity/time-discretization terms. The *difference* of the two
residuals at time 1 is 0.217 code energy units, or 9.5×10⁻⁷ of the heat
injected over that interval. At time 0.5 their residuals differ by only
0.00865. Thus no large coarse/fine-specific energy defect is detected.

The serial and two-rank one-step SMR HDF fields are bitwise identical;
history values differ by at most 6.2×10⁻¹⁰ relative. Every implicit step in
the dt≤0.1 runs converged and ended with positive saved Er; opacity clamps
were zero. These checks do not prove that *every internal ghost state* was
positive at all iterations, nor do they establish long-term stability.

The one-step time-1 runs are strongly time-step dependent: the uniform
one-step result had Lrad/Lheat=4.93 and Mach=0.01848, versus 3.63 and
0.03335 after ten 0.1-sized steps. A CFL-allowed single large step cannot
serve as a reliable stability comparison here.

## Early-acceleration controls

Matched uniform 64³ runs to code time 0.5 (five steps at dt=0.1) show:

| Control | Maximum Mach | Boundary Lrad [code] |
| --- | ---: | ---: |
| Baseline | 0.015112417 | 826519.578 |
| Initial velocity perturbation = 0 | 0.015112404 | 826519.578 |
| Central heating = 0 | 0.015112417 | 826519.578 |
| FLD diffusion disabled | 0.014971516 | 142507.728 |

The initial acceleration is insensitive to the tiny velocity seed and to
center heating over this interval. Removing diffusion prevents the gas
thermal-energy loss but lowers maximum Mach by only about 0.9%. This points
to an initial discrete force imbalance as the leading short-time driver;
it does not identify which of gas pressure, gravity, and radiation force is
responsible. The production RSG generator rejects disabling radiation force,
so that control was not interpreted as a valid run.

The first 0.1-code-time velocity increment is already direction-dependent.
In the core (30–166 Rsun) the rms acceleration magnitude is 0.00216 code
units in cells with `max(|x|,|y|,|z|)/r>0.9`, versus 0.000642 in the
diagonal sector `<0.7`. The same axis/diagonal rms ratio is about 2.4 in
the 750–830 Rsun layer. The groups contain different radial samples, so
this is an anisotropy indicator rather than a matched-radius convergence
measurement.

### Uniform fourth-order comparison

With the intended fourth-order initial cell-average corrections, the
uniform 64³ run has the same mapped initial gas mass as the second-order
run to machine precision. At code time 0.5, changing `xorder` from 2 to 4
reduces core rms gas speed from 8.51×10⁻⁴ to 2.00×10⁻⁴ code units. Adding
both core gravity quadrature and fourth-order radiation force lowers it
further to 6.70×10⁻⁵. However, the fourth-order runs increase the boundary
luminosity from 3.59 to 9.38 times Lheat and maximum Mach from 0.0151 to
0.0377. Crucially, they also cause **384 out-of-domain opacity lookups to
be clamped**, whereas the second-order run has zero. A fourth-order run
without the cell-average corrections has 2016 such lookups. These are
short-run diagnostics, not evidence that the fourth-order option is a
physical improvement. The unclamped saved snapshots do not exclude
intermediate reconstructed or solver states outside the table.

Changing the opacity policy to `error` locates the first excursions at
active cells in the dilute box exterior, for example
`(x,y,z)=(-457.9,-994.7,15.8) Rsun`, `rho=1.23×10⁻¹³` code units and
`T=247 K`. The table minimum is 316 K. Thus the 384 clamps are low-T
states before the first accepted step, not evidence that the table is too
narrow at the stellar photosphere. Increasing `hydro/pfloor` tenfold is
not a valid quick fix: the unchanged initial profile is then materially
below the requested pressure floor and the problem generator correctly
rejects it. A bounded fourth-order primitive/energy reconstruction or a
consistent exterior profile is needed before using this option.

## Status and next gates

Conservative static refinement now passes local short numerical gates, but
the patch reduces boundary luminosity by only 1.8% at time 1 and leaves the
maximum Mach unchanged. The profile's surface radiative transition remains
unresolved, and the early boundary luminosity exceeds the heating luminosity
by more than a factor of three. No stable long-duration RSG model is claimed.

Next: compare second- and fourth-order uniform force/gravity discretizations
for the initial acceleration and locate the fourth-order opacity excursions;
then design a radiation-carrying *discrete 3-D*
surface model or a finer, surface-covering mesh. Before any large XD run,
validate a resolved surface patch, temporal convergence, and the first-day
mass/energy/Mach evolution. No XD job was submitted during this gate.

Raw local outputs: `/home/kosuke/simulation/test/wor122_rsg_smr`.
The AMR-aware summary script is `wor122_rsg_smr_diagnostics.py`.
