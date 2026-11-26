# WOR-182 one-temperature LTE radiation EOS

## Purpose

This branch adds the optically thick comparison closure

`E_total = E_material(rho,T) + a_rad T^4`,

`P_total = P_material(rho,T) + a_rad T^4/3`.

It is deliberately distinct from full FLD: photons are perfectly trapped,
share the material temperature, and have no diffusion equation.  It provides
the middle member of the controlled CE comparison: material-only HD,
one-temperature LTE-radiation HD, and full mixed-frame FLD.

## Implementation and table qualification

`src/eos/general/eos_table.cpp` contains a runtime reference implementation
with safeguarded inversion.  Enabling it in an NR-FLD build is a fatal error
because that would count the photon energy twice.  A root64 one-step test took
117 s with the runtime inversion, versus 5.8 s for an ordinary table, so the
reference path is used to generate and validate a production table rather
than in a production Riemann solve.

`src/pgen/eos_test.cpp` generated
`data/eos/rsg_tomida2013_lte_radiation_eos.data` from the material-only Tomida
table.  The binary table has 256 density by 1024 energy samples, seven fields,
and SHA-256
`1327f9849d31d014ff480a77bdf34bf0fa007431bc460c800583adc8fc09f2a5`.
On a 256 by 256 grid at least 0.5 density dex and 0.1 energy dex inside the
table edges, maximum energy and pressure round-trip errors were `9.31e-4` and
`3.81e-3`.  Sound speed squared and `nabla_ad` remained positive.  The same
root64 one-step test took 5.35 s and 621 MiB locally, restoring ordinary
table-lookup performance.

The non-FLD SCF mapper now obtains internal energy through
`EgasFromRhoP(rho,P)` instead of using an ideal-gamma conversion.  Thus the
same source total pressure is mapped consistently through either the material
or combined table.  HD comparisons disable core-luminosity deposition because
neither HD closure transports that luminosity.

## XD short gates

All gates used root64, central L4 plus surface L2 AMR, RK4+4wz, CFL 0.2,
one node/56 MPI ranks, and target time 0.1.  Each run saved final restart,
complete run configuration, source commit, and SHA-256 provenance.

| Job | Rotation | EOS | Wall | Interior Mach | Surface Mach | Mass at Mach > 0.1 |
|---|---|---|---:|---:|---:|---:|
| 348349 | RLOF-corotating | material only | 17.78 s | 3.60e-4 | 1.79e-3 | 0 |
| 348350 | RLOF-corotating | LTE radiation, first floor gate | 17.87 s | 3.88e-4 | 4.20e-3 | 0 |
| 348351 | nonrotating | material only | 18.34 s | 3.56e-4 | 1.09e-3 | 0 |
| 348352 | nonrotating | LTE radiation, first floor gate | 18.67 s | 3.85e-4 | 1.09e-3 | 0 |

The first combined-table gates remained finite and mechanically quiet, but
their numerical density floor exactly touched the table's `1e-12 g cm^-3`
lower edge.  They are diagnostic passes, not restart candidates.  Production
combined-table gates use `3.2e-12 g cm^-3`, still 5.3 times below the
`1.7e-11 g cm^-3` atmosphere, to enforce the required 0.5-dex lower guard.
Jobs 348361 and 348362 repeated the rotating and nonrotating gates with that
floor.  They reproduced the Mach results to roundoff, and both ended with a
lower-density margin of `+0.505 dex` and a lower-energy margin of
`+0.187 dex`.  Their wall times were 17.70 and 19.38 s.  These are the
accepted short-gate configurations for extension.

The accepted restarts were extended continuously from time 0.1 to time 1 at
CFL 0.2.  Jobs 348370--348373 all completed in 49--52 s of solver wall time.
The rotating material/LTE models ended at interior/surface Mach
`0.00165/0.01103` and `0.00188/0.01174`; the nonrotating material/LTE models
ended at `0.00163/0.01071` and `0.00185/0.01070`.  No model contained stellar
mass above radial Mach 0.1.  Both LTE models retained lower density and energy
table margins of `+0.505` and `+0.187 dex`.  Jobs 348388--348391 continue these
same restart chains to time 10 at CFL 0.3.

The two rotating time-10 gates completed in 278 and 273 s on one node.
Material-only job 348388 ended at interior/surface Mach
`0.00709/0.10275`; LTE-radiation job 348389 ended at
`0.00704/0.11191`.  Their masses above radial Mach 0.1 were only
`4.20e27` and `6.93e27 g` (`2.1e-6` and `3.5e-6 Msun`), and the maximum at
density above the surface threshold remained about 0.006.  Segment mass,
angular-momentum, and mean-radius drifts were respectively
`-1.21e-9/+1.90e-6/+4.75e-6` for material and
`+5.43e-8/-2.29e-5/+5.20e-6` for LTE radiation.  The LTE lower density and
energy margins remained fixed at `+0.505/+0.187 dex`.  Thus the trapped
photon terms slightly increase the negligible-mass outer maximum but do not
destabilize the resolved envelope through time 10.

## Acceptance path to CE

Short gates are followed by time 1, time 10, and then undamped long relaxation
only after Mach, mass drift, and all four EOS margins remain acceptable.
The accepted final restart is not merely a checkpoint: its run configuration,
executable/input/profile/EOS hashes, source commit, and history are retained
beside it.

The legacy `$ATHENA/src/pgen/ce_main.cpp` uses different problem-specific
restart arrays, code units, and EOS support, so its executable cannot safely
read an `scf_rsg` binary restart as a cross-problem checkpoint.  The handoff
therefore has two artifacts: the native restart for continuing `scf_rsg`, and
a full-volume `cons` HDF5 snapshot on the native AMR mesh for CE import.
`analysis/wor182/xd_ce_handoff_export.sh` creates the latter without advancing
the state and records the density, energy, length, velocity, time, and momentum
units plus hashes.  This permits a CE-side conservative importer and explicit
unit conversion rather than an implicit, format-dependent restart read.
