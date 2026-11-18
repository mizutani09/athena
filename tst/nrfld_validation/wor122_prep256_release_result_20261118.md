# WOR-14: source-free 256³ RSG release gate to t=130

## Matched experiment

XD job **341970** completed normally (2026-11-18 11:07–13:58 JST,
02:51:26 elapsed). It ran two serial 64-rank continuations from the
matched t=120 snapshots: one kept the spherical-shell preparation source
(`prep_thermal_tau=0.1`), while the other disabled it entirely
(`prep_thermal_tau=-1`). Each reached t=130 (33.10 days). Both had zero
opacity table domain/clamp/nonfinite errors, empty stderr, and valid
restart/HDF output under `$WORK/wor122/prep256_{hold,off}_t130/`.
The first analysis submission, **342091** on M-debug, completed normally
in 02:01. Its compact data were copied locally under
`tst/nrfld_validation/wor122_prep_thermal_20261118/`.

| Diagnostic | Source held | Source off |
|---|---:|---:|
| t=120 six-face Lrad/Lheat | 0.9570 | 1.0423 |
| t=130 six-face Lrad/Lheat | 0.9544 | 0.8306 |
| t=130 global history maximum Mach | 0.5336 | 0.5425 |
| t=130 maximum Mach location | r≈1108 R☉, density floor | r≈1108 R☉, density floor |
| t=130 800–860 R☉ maximum Mach proxy | 0.215 | 0.243 |
| t=120→130 gas energy change [code] | +0.158 million | −1.898 million |
| t=120→130 preparation energy change [code] | +2.173 million | 0 |
| t=120→130 800–813 R☉ Tgas ratio | 0.999 | 0.905 |
| t=120→130 813–817 R☉ Tgas ratio | 0.998 | 0.893 |
| t=120→130 817–830 R☉ Tgas ratio | 1.001 | 0.924 |
| t=130 800–830 R☉ convective enthalpy proxy/Lstar | 0.0023 | 0.0127 |

After the t=110→120 taper, the source-free continuation radiated briefly
above Lheat, then its boundary luminosity fell steadily from 1.015 at
t≈121 to 0.831 by t=130. The held control stayed near 0.956. In the
unsupported run, the gas-temperature drop at 813–817 R☉ accumulated
to approximately 19% relative to t=110 (`0.9102 × 0.8926 ≈ 0.812`),
and cooling spread to neighboring shells. By t=130 the mean radial
speed in the 800–813 R☉ shell was +0.239 km/s versus −0.006 in the
held case; in 817–830 R☉ it was −0.166 versus +0.013 km/s. The thermal
change is now inducing a local circulation, although the stellar-surface
Mach remains modest at this checkpoint. The shell-averaged convective
enthalpy proxy rose to 0.0127 Lstar but is far too small to replace the
roughly stellar-luminosity preparation source. The proxy is a shell-volume
covariance, not an exact spherical-face flux.

The central flow is not causing the *difference* between these two cases.
The 0–166 R☉ mean radial speed is 0.142 km/s in both, and the 300–700 R☉
axis/diagonal RMS speeds agree between branches. The global Mach above
0.5 still occurs near r=1108 R☉ in the density-floor exterior, rather
than representing a stellar outflow. Outer-shell grid anisotropy persists,
but the immediate source-free failure is a surface thermal deficit.

The affected 813–817 R☉ shell is only 4 R☉ wide, versus ~7.9 R☉ per
cell at 256³. Its exact amplitude is resolution sensitive. The paired
experiment establishes that this *mapped profile* cannot be released
after a short taper and remain thermally steady. It does not establish
that all slower tapers or better resolved, flux-closed profiles must fail.
An additional long continuation of this same unsupported state would
mostly measure a known cooling transient and is not prioritized.

Figure: `doc/figures/wor122_prep256_release_gate_20261118.png`.
Reproduce with
`python3 tst/nrfld_validation/wor122_plot_release_gate_20261118.py`.
Analysis script: `tst/nrfld_validation/wor122_prep256_release_analysis_20261118.sh`.

## Next model gate

Construct a new 1D/3D precursor that accounts for the convective
luminosity deficit in the envelope and the associated turbulent pressure
while retaining radiation force in the final physics calculation.
Before another 256³ run, require a discrete FLD radiative-flux audit,
EOS-consistent enthalpy flux closure, gas+radiation mechanical balance,
and a resolved photospheric transition, then pass a short 128³/256³
source-taper test. A longer taper of the present unsupported state is a
possible *diagnostic* only after estimating whether its convective flux
can grow from ~0.013 to ~1 Lstar on the corresponding time scale.
