# WOR-14: local flux-closure audit of the RSG initial profile

## Decision

Do not launch another 256³ continuation from the current 1D profile or from
the one-parameter variants below. The current model has a large deep-envelope
convective-flux deficit, while all of the variants retain an unresolved
surface FLD luminosity transition. The variants are local feasibility tests,
not initialized 3D convective solutions. No XD job was submitted for this
screen.

## Why the present profile cools when preparation heating is removed

The preceding paired XD experiment reached `t=130` in both branches. With
the artificial shell thermal source disabled, six-face luminosity fell from
1.042 to 0.831 times the target between `t=120` and `t=130`, and the gas at
813–817 R☉ cooled by approximately 19% from `t=110`. The source-held control
stayed near 0.95 of the target luminosity. The unsupported 800–830 R☉ shell
carried only about 0.013 stellar luminosities in a shell-averaged convective
enthalpy proxy. See `wor122_prep256_release_result_20261118.md` for the
matched-run analysis and its limits.

The 1D generator `tools/rsg/make_gray_rsg_hse.py` imposes a temperature
gradient no greater than `nabla_ad + Gaussian superadiabatic excess`, then
shoots the excess peak to the requested mapped mass. It includes radiative
force in gas-pressure hydrostatic balance, but it does not solve a convective
flux-closure equation or include turbulent pressure. A prescribed gradient
can therefore produce the correct mass and approximate force balance while
being unable to carry the prescribed luminosity once 3D convection is
allowed to evolve without artificial thermal support.

Using the current EOS and opacity, the diagnostic estimates the most
optimistic efficient-mixing-length convective capacity (mixing length
`alpha=3`, velocity coefficient `nu=8`, with no radiative leakage from
eddies). The actual 1D temperature gradient is measured over a ±5 R☉
window. The table compares this capacity with the missing luminosity
`Lstar-Lrad` in the reference profile. A capacity below one means that even
this optimistic local estimate cannot carry the missing flux at the
profile's gradient.

| Radius [R☉] | `Lrad/Lstar` | Current `∇` | Local flux-closed `∇` | Capacity / missing flux |
|---:|---:|---:|---:|---:|
| 400 | 0.175 | 0.399 | 0.429 | 0.0081 |
| 500 | 0.024 | 0.336 | 0.398 | 0.0000 |
| 600 | 0.010 | 0.190 | 0.254 | 0.0025 |
| 700 | 0.014 | 0.464 | 0.613 | 0.153 |

The zero at 500 R☉ means the fitted gradient is slightly below the
EOS-consistent adiabatic gradient. The same diagnostic repeats the fit with
±3 and ±10 R☉ windows; these are retained in the JSON so gradient noise can
be checked. A locally flux-closed efficient-MLT estimate implies
`P_turb/P_gas≈0.086, 0.106, 0.347` at 500, 600, 700 R☉, respectively.
Thus turbulent support can be material and a fully consistent new profile
needs to solve force and flux balance together. This is an estimate, not a
measurement of turbulent pressure in the XD run.

## Controlled gradient-width scan

Only the Gaussian superadiabatic width in the existing 1D generator was
changed from `0.1 Rstar` to `0.2`, `0.3`, or `0.4 Rstar`; each profile was
re-shot to about 3 M☉ and given the same LTE FLD exterior, with exterior
join end at 900 R☉. The EOS, opacity, radiation force, luminosity, radius,
effective temperature, and grid were held fixed. The broader profiles have
different central entropy and temperature: this is a structural test, not
a small perturbation of a dynamically relaxed 3D state.

| Gaussian width [`Rstar`] | Capacity at 400 R☉ | Capacity at 500 R☉ | Capacity at 600 R☉ | FLD 10–90% width [R☉] |
|---:|---:|---:|---:|---:|
| 0.1 (reference) | 0.008 | 0 | 0.003 | 3.35 |
| 0.2 | 0.001 | 0.073 | 1.91 | 4.00 |
| 0.3 | 0.456 | 1.89 | 5.98 | 4.32 |
| 0.4 | 5.63 | 5.97 | 12.73 | 4.52 |

The 0.4 profile has enough *estimated local capacity* over the sampled
400–800 R☉ range, but capacity falls below one again near 813–817 R☉.
Efficient MLT itself is questionable there because the optical depth and
eddy leakage matter. More importantly, the 10–90% FLD luminosity transition
is only 3.35–4.52 R☉ in all four profiles, compared with 7.89 R☉ for a
256³ Cartesian cell. Merely widening this Gaussian cannot pass the
surface-resolution gate. The strongest variant still lacks initial 3D
velocity–enthalpy correlations to carry its estimated convective flux.

## Interpretation and next gate

The paired run isolates rapid surface cooling after release. This 1D audit
adds an independent, structural reason why the present prepared state is
unlikely to sustain `Lstar`: most deep-envelope luminosity is assigned to
convection, but the imposed gradient has too little optimistic convective
capacity over hundreds of solar radii. It does **not** prove this is the
only cause of the XD transient; the under-resolved surface transition and
the absence of initial 3D correlations remain separate problems.

Next, construct a flux-closed 1D precursor with a transport prescription
that includes eddy radiative leakage near the photosphere, and include its
turbulent pressure in the force equation. Give the mass, effective
temperature, and surface transition enough independent boundary parameters
to be shot simultaneously. Audit the resulting *discrete* FLD flux and
gas+radiation force balance after Cartesian mapping. Only then use a short
128³/256³ taper and source-free test; require persistent surface temperature
and boundary luminosity, as well as a physically meaningful interior Mach
diagnostic, before a long run. The target surface width must be reconsidered
against the actual 3D mesh and radiation scheme; broadening a profile in
postprocessing alone is not a physical closure.

## Reproduction and data

The scripts are `wor122_flux_closure_gap.py` and
`wor122_plot_flux_closure_scan.py`. The reference diagnostic is
`wor122_flux_closure_gap_20261118.json`; the trial profiles, metadata,
individual diagnostics, and `scan_summary.json` are in
`wor122_flux_closure_scan_20261118/`. Figures are
`doc/figures/wor122_flux_closure_gap_20261118.png` and
`doc/figures/wor122_flux_closure_scan_20261118.png`.

The local scalar closure holds density, temperature, EOS thermodynamic
derivatives, and opacity fixed at each radius and solves for the gradient;
it is **not** an integrated stellar model, a discrete 3D FLD operator
calculation, or a prediction of an actual turbulent flux. The scan uses a
5 R☉ gradient fit for the comparison, and the tiny-capacity points depend
on the local adiabatic-gradient difference. The closure solver checks that
its radiative plus convective flux agrees with the specified total to
relative tolerance `1e-8`.
