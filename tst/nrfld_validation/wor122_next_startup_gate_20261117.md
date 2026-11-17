# WOR-14: next RSG startup gate after the force/transport audit

## Candidate screen

The production equations retain radiation momentum coupling. The existing
1D structure is gas-plus-radiation-force HSE but assumes almost the entire
luminosity is transported by convection across much of the envelope. Its
3D initial velocity field supplies nearly zero correlated enthalpy flux.
The 256³ supported run at t=52 (13.24 days) still requires preparation
heating of about 0.95 Lstar while the 800–830 Rsun convective covariance
is 0.00267 Lstar. The surface thermal support therefore cannot yet be
tapered. These are measurements, not a prediction of the eventual state.

The following single-parameter candidates have already failed a gate:

| Change | Relevant result | Decision |
| --- | --- | --- |
| Increase global velocity perturbation 1e-4→1e-2 | At 128³ t=8, convective covariance 0.002189 vs 0.002187 Lstar; the seed vanishes at the surface (jobs 340236/340237) | No transport gain |
| Add 1% surface velocity perturbation | At 128³ t=8, covariance 0.002209 vs 0.002187 Lstar (jobs 340269/340270) | No transport gain |
| Redirect core heating to supported shells conservatively | At 128³ t=8, surface temperature ratio 0.952 and radiative output 1.371 Lstar (jobs 341276/341330) | Worsens thermal balance |
| Lower Lstar to 0.3 at fixed radius with consistent Teff | Cold outer atmosphere; Mach 0.503 by t=3 (~0.76 day), job 341338 stopped | Unstable short gate |
| Restore 3487-K nominal Teff only | 700–800 Rsun radiative fraction and ~3.01-Msun mapped mass almost unchanged | No deep-transport gain; LTE/FLD exterior unvalidated |
| Replace WOR-55 opacity by local OPAL values at unchanged state | 700-Rsun Lrad would rise only to ~0.137 Lstar in a fixed-gradient estimate; 800–817-Rsun opacity cliff persists | Not a self-consistent HSE candidate |

Resolution alone is also expensive. The 10–90% radiative transition in
the imported profile spans 3.35 Rsun, versus 7.89 Rsun per cell at 256³.
An idealized whole-star refined spherical shell 30 Rsun thick around
830 Rsun would require approximately 270 million cells at 8× finer
spacing (0.987 Rsun; 3.4 cells across the transition). At 16× finer
spacing it would require approximately 2.16 billion cells (0.493 Rsun;
6.8 cells across the transition), before real block/refinement overhead.
These are geometric lower estimates, not resource allocations. A partial
one-quadrant patch already failed to cover the whole photosphere, so no
large refinement job is warranted before improving the initial model.

## Next model construction

The next useful change is a **new 1D structure or staged preparation**,
not another variant of the present narrow thermal source. A candidate
must specify how the flux `Lstar - Lrad` is carried, including the EOS
enthalpy contrast and a velocity scale, and how any associated turbulent
pressure changes mechanical balance. A static HSE with a prescribed
convective gradient but no flux closure is not sufficient. The candidate
must retain gas+radiation-force HSE for the physics calculation.

[Goldberg, Jiang & Bildsten (2022)](https://arxiv.org/html/2110.03261v2)
provide a relevant proof of concept: their Athena++ RSG *envelope* models
include radiation pressure, start with a low-luminosity radiative state,
drive convection through a hot inner boundary or fixed inner luminosity,
and relax over years. Their geometry has a 300–400-Rsun inner boundary,
not a full Cartesian core; their boundary condition cannot simply be
copied into this star-in-a-box calculation. It motivates explicitly
modeling the preparation stage and measuring its mass/energy budget.

Before submitting another 3D job, require all of the following for the
new candidate:

1. ~3-Msun mapped envelope and the intended potential, radius, luminosity,
   EOS, and WOR-55 opacity recorded with checksums.
2. Gas plus radiation force balance in 1D and on the intended 3D mesh;
   include the velocity/turbulent-stress contribution if it is seeded.
3. LTE/FLD-compatible exterior and a measured finite-volume FLD face
   luminosity/divergence, especially 800–830 Rsun.
4. A feasible convective enthalpy flux across the required shells, with
   its assumptions and Mach scale explicit; identify any shell where it
   fails rather than hiding the deficit in a mass-shoot parameter.
5. A short 128³ or 256³ gate with measured Mach location, shell velocity,
   luminosity, temperature, mass, energy ledger, and solver status.
6. A source-taper gate before any year-scale production run.

At this first screen no new XD job was submitted: no candidate passed the
pre-3D transport/exterior criteria. The immediate implementation work is
to formulate a flux-closed 1D envelope (or a documented staged precursor)
and test it locally. Subsequent 3D jobs should be limited to candidates
that pass the cheap checks above.

## Local transport timescale and one limited continuation

Using the optimistic efficient-MLT relations in Goldberg et al. (2022),
Eqs. (2)–(3), with `alpha=3`, `nu=8`, the tabulated material EOS for
`c_P` and the isobaric density response, and the *initial* flux deficit,
gives the following indicative scales. The gas pressure height uses the
initial radiation-reduced gravity; this is not a self-consistent turbulent
HSE. Near the photosphere radiative leakage makes the efficient-MLT
assumption poor, so the figures are diagnostic scales, not predictions.

| Radius [Rsun] | MLT speed [km/s] | Mach | `H/v` [days] | `3H/v` [days] | implied `P_turb/P_gas` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 700 | 10.52 | 0.45 | 78.1 | 234.4 | 0.33 |
| 750 | 9.66 | 0.51 | 69.6 | 208.8 | 0.37 |
| 800 | 6.50 | 0.53 | 53.6 | 160.9 | 0.33 |
| 810 | 4.97 | 0.60 | 34.0 | 102.0 | 0.41 |
| 813 | 5.24 | 0.68 | 27.1 | 81.3 | 0.54 |
| 817 | 5.29 | 0.72 | 19.5 | 58.6 | 0.76 |

The completed 256³ state at 13.24 days is shorter than even a local
`H/v` at 813 Rsun. This does not establish that the preparation will
work; it means the absence of developed convection at that checkpoint
is not by itself a decisive failure. The large implied turbulent pressure
also reinforces the need for a new mechanically consistent mean state.
Reproduction: `wor122_mlt_transport_scale.py`,
`wor122_mlt_transport_scale_20261117.json`, and
`doc/figures/wor122_mlt_turnover_20261117.png`.

One bounded continuation was therefore submitted to XD as **341450**,
M-bulk-a, 64 MPI ranks, 110 GB, 11 h 30 min walltime. The script
`wor122_prep256_to110_20261117.sh` resumes the valid t=52 restart,
first to t=80 (~20.4 days), then to t=110 (~28.0 days). The second stage
runs only if t=80 has maximum Mach <0.5, mass drift <1e-4, six-face
`Lrad/Lheat` between 0.5 and 1.5, and normal solver/opacity termination.
All output, including ~3-GB restarts, remains under `$WORK/wor122/`.
The script SHA-256 is
`859f670d482c09f3c9c6891caf9de801cae0fe03e6ea6554c85287cfcc3fb8c0`.
No other RSG job was submitted. A successful t=110 check still requires
shell-temperature, surface-Mach, convection-flux, and source-power
analysis before considering any taper or production extension.
