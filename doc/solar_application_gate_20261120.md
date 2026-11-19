# Solar local-convection application gate (WOR-116, 2026-11-20)

## Decision

Include the solar local box as a **limited numerical application**. The
defensible claim is that the implementation supports stable, non-magnetic 3D
convection and radiative energy transport with grey FLD, including a
quasi-steady near-solar-flux state in a controlled 96 km calculation. Do not
claim that the calculation quantitatively reproduces the observed solar
photosphere, that the grey boundary is unique, or that the emitted flux is
grid converged.

This decision closes the publication gate without making the unfinished
48/24 km and non-grey studies prerequisites for the FLD-method paper.

## Fixed evidence

The new gate report reads the numerical JSON files rather than transcribing
plot values. Its selected late windows are:

| Case | Window | Top radiation | Top material + gravity | Bottom/input | Storage | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 96 km, closed lower boundary, two-cell top-face reconstruction | 1000 | 0.9926 Fsun | 0.00225 Fsun | imposed 1 Fsun | +0.00353 Fsun | quasi-steady near-solar flux |
| 96 km, closed lower boundary, three-cell reconstruction | 1000 | 0.9941 | 0.00308 | imposed 1 | +0.00198 | reconstruction-order control agrees |
| 96 km, open HD2, lower inflow entropy | 2000 | 0.9773 | 0.00250 | 0.9791 | approximately 0 | quasi-steady lower bracket |
| 96 km, open HD2, higher inflow entropy | 2000 | 1.0142 | 0.00441 | 1.0194 | +0.00086 | quasi-steady upper bracket |
| 48 km, refined closed-boundary branch | 1000 | 0.9583 | 0.0600 | imposed 1 | -0.0185 | total closure is good; radiation differs |
| 48 km, open HD2 at t=70000 | final 500 | 0.8699 | 0.0282 | 0.8913 | -0.00259 | stable but not converged to 96 km flux |

For the two 96 km HD2 branches, all 77,863 and 78,247 recorded Newton solves
converged, there were no fatal messages, final mass drift was about 0.363%,
and the 2000-code-time energy closure errors were -0.00072 and -0.00016
Fsun. Their top fluxes bracket one solar flux without opacity scaling or
volumetric heating. They start from a warm, previously relaxed 3D state and
cover less than the estimated whole-box thermal time, so they demonstrate a
strong quasi-steady state, not uniqueness of the asymptotic solution.

The earlier t=20700--22700 paper-diagnostic window independently established
the 3D flow morphology: top/bottom flux 0.6291/0.6355 Fsun, all 30,017 Newton
solves converged, mass drift 0.0286%, tau=2/3 gas temperature 5182 K,
vertical-velocity RMS 1.23 km/s, upflow area 49.5%, and a dominant horizontal
velocity wavelength near 1.54 Mm. At box centre the diagnosed convective
enthalpy and kinetic-energy fluxes were +0.680 and -0.083 Fsun. These values
support sustained convection but are not a solar calibration.

## What fixed the original flux deficit

The original top radiation ghost copied the Rosseland `kappa*rho` of the
uppermost active cell. Because `kappa*rho` falls steeply through the surface,
that copy overestimated the optical depth of the upper half-cell and added a
grid-dependent resistance to escaping radiation. Reconstructing and
integrating `log(kappa*rho)` across the half-cell removed most of this
resistance. A Marshak coefficient close to the independent grey formal
transfer moment, `F/(cE) approximately 0.56`, supplied a smaller secondary
correction. The same heat input then emerged at the top instead of remaining
as storage.

The correction is supported by matched restart comparisons, long persistence,
two- versus three-cell reconstruction, top optical-depth audits, and open-HD2
inflow-entropy brackets. It remains application-boundary logic: it must not be
presented as a general non-grey atmosphere boundary, and should remain opt-in
until its integration and regression scope are reviewed.

The exact XD calculation provenance is a detached checkout at base commit
`1924314e611eca8e89e451860032e4363dd0f9e3` plus an uncommitted four-file
diff (`src/fld/fld.cpp`, `fld.hpp`, `NRFLD.cpp`, and
`simple_convection_lhllc_fld.cpp`). The selected executable SHA-256 is
`203a72c932d944ac510faa8ab5621b91cb438c45cbd087366dd67662e4d9b232`, the
pgen SHA-256 is
`cf045b866babbefd0bbcb46a9b06e60b68831ffa748a674ee34a4d36ab3f4350`, and
the four-file diff SHA-256 is
`57e4dfdfd580c8c4329c784959ef84cdbccded37ef27d01a92feb134053fe70f`.
The archived two-cell top-face patch has SHA-256
`dca932ded6c463f171fc0e03e98381439c8d72f506c70f96916c4abd743f80d8`.
These hashes make the evidence auditable, but an uncommitted calculation
checkout is not a releasable source baseline. Source adoption therefore stays
outside this application-gate decision.

## Publication figures and captions

Recommended compact application figure:

1. a temperature/vertical-velocity x-z slice with the Rosseland tau=1 surface,
   paired with an x-y vertical-velocity or temperature slice;
2. a time history of top radiation, bottom/input energy, and storage for the
   late 96 km state, with the two HD2 entropy branches as a small inset;
3. if space permits, the vertical radiative/convective flux decomposition.

Caption boundary: “Non-magnetic local convection evolved with tabulated EOS,
physical Rosseland opacity, and grey FLD. The calculation demonstrates stable
3D radiative-convective transport and a controlled near-solar luminosity at
96 km. It is not a non-grey spectral validation; the 48 km HD2 flux remains
different and resolution convergence is not claimed.”

The 100--10000 nm Optab/LTE post-processing result (0.646 Fsun for the older
0.630-Fsun FLD snapshot) may be cited only as a fixed-snapshot radiative
diagnostic. It does not validate the grey-FLD evolution equation. The 397 nm
formation layer lies outside that box, and vertical coarsening changed the
498 nm flux by 26%, so detailed solar spectrum, colour, and limb-darkening
claims remain outside this application.

## Reproduction and remaining work

Generate the dedicated HTML/JSON/PNG with:

```bash
python3 tst/nrfld_validation/wor116_solar_application_gate.py \
  --results-root /path/to/wor91_fld_solar_20261116 \
  --hd2-48-audit /path/to/hd2_nata48_audit_342923.json \
  --output-dir /path/to/wor116_solar_gate \
  --portable-provenance
```

The portable mode retains every input SHA-256 while replacing workstation
absolute paths with stable `external/...` labels suitable for a shared report.

The current standalone report is under
`/home/kosuke/simulation/test/rad_nr/wor116_solar_gate_20261120/report`.
The 24 km XD run was still active when this gate was fixed and is deliberately
excluded. Its outcome belongs to the stronger future resolution claim, not to
the limited application decision. No WOR-91 source change was merged into the
collapse/publication candidate by this work.
