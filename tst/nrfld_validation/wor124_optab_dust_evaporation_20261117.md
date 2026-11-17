# WOR-124: OPTAB dust scope and dynamic evaporation gate

Date: 2026-11-17 (JST)

## Result

The current OPTAB checkout (`2d95b7c1a944e15d80605afee783c22eed441ae1`)
is a gas-opacity generator.  Its README describes ideal-gas opacity tables and
its selectable sources cover atomic/molecular lines, continuum processes,
CIA, and Thomson/Rayleigh scattering.  No dust/grain/condensate equilibrium,
optical constants, size distribution, or Mie/DHS source exists in the local or
XD checkout.  Dust therefore cannot be enabled by an OPTAB input switch.

Adding dust at generation time is possible in principle, but is a separate
physics implementation: determine condensate abundances and gas depletion,
compute wavelength-dependent grain absorption/scattering for an explicit
composition and size/shape distribution, add those coefficients to the gas
spectrum, and only then recompute Planck and Rosseland means.  In particular,
Rosseland means are harmonic and cannot in general be made self-consistent by
adding two already averaged tables.  The committed Semenov-to-OPTAB bridge is
therefore a pragmatic, auditable bridge, not a replacement for a spectral
dust implementation.

## Dynamic evaporation experiment

`wor124_optab_evaporation_gate.in` is a periodic, uniform, table-EOS thermal
exchange test.  At `rho=1e-10 g cm^-3`, gas starts at 900 K and radiation at
7000 K.  It crosses the Semenov final composite-dust evaporation point
(1230 K at this density) and all three trial transition ends (1430, 1530, and
1630 K).  The run uses the Newton--FGMRES path, exact local gas solve, a strict
linear target of `2e-8`, and the already supported bounded true-residual
inexact cap `ATHENA_WOR92_INEXACT_TRUE_CAP=1e-6`.

The test pgen initially contained a validation-only defect: it passed primitive
pressure to `TempFromRhoEg` as if pressure were internal energy.  This kept the
opacity lookup on the low-temperature branch and made all transition widths
spuriously identical.  The fix converts pressure to internal energy through
the configured EOS before asking for temperature, matching
`protostellar_core_fld.cpp`.  This does not change the production collapse
pgen.

Without the bounded inexact gate, the corrected lookup reaches the known
finite-precision FGMRES floor at the transition: 300 iterations leave true
relative residuals `1.36e-7`, `1.43e-7`, and `1.86e-7` for the representative
300, 200, and 400 K tables, above the requested `2e-8`.  With the `1e-6` cap,
all outer Newton stages converge in at most two iterations and no relaxed
Newton acceptance is used.

| transition end offset | cycles | final Tgas (K) | final Trad (K) | max total-energy relative drift | max true FGMRES residual | max RSS | wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 K | 71 | 1842.0353 | 4193.5303 | 9.20e-10 | 4.83e-7 | 43.9 MiB | 14.06 s |
| 300 K | 71 | 1840.0226 | 4252.3944 | 9.20e-10 | 3.87e-7 | 43.9 MiB | 14.37 s |
| 400 K | 71 | 1837.6912 | 4317.3274 | 9.47e-10 | 3.67e-7 | 43.8 MiB | 14.64 s |

Relative to the 300 K bridge, the maximum common-time gas-temperature
difference is 44.75 K for 200 K and 33.73 K for 400 K.  Thus the corrected
test actually resolves the modeling choice; the bridge width changes the
thermal relaxation but does not trigger a nonlinear failure or conservation
loss in this controlled gate.  All three runs report zero out-of-domain,
clamped, negative, zero, and non-finite opacity results.

A two-block 300 K run through 1754 K was repeated with one and two MPI ranks.
Both have 35 cycles, 69 linear solves, 62 solves above the strict target, and a
maximum true residual `3.77349e-7`.  The complete history differs by at most
`7.82e-14`; final gas and radiation temperatures are identical at printed
binary64 precision.  This establishes rank reproducibility for this reduced
gate, not a new physical qualification of the transition prescription.

## Provenance and limits

- Athena source base before this change: `fa9552b0f20fb3c35f71e77f33dd8dda8168441d`.
- Local debug binary SHA-256: `34b2779c3278b6639437bcaa867510eee72cfa4da62c451cdc053e3a7a30a550`.
- Input SHA-256: `2216b53917c5db87c0cd6182278418d30a000814a85d5efbcb4cfe4e3221530b`.
- Tomida-family EOS SHA-256: `742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43`.
- Hybrid opacity SHA-256 values: end-200
  `ea336a24c6a4a6a46fec33bedac43fff160992a26fffdd7ed6e2feb9490b878e`,
  end-300 `3a7048546bf140454b7bd8d7d108cafa04e45080f92d1cfadd7af49888a7ed5c`,
  and end-400 `a242eb7d2ffe78cc4ae1d6c59aef05f749ddbb1b513412b31ce49c38cb909276`.
- Local evidence root:
  `/home/kosuke/simulation/test/collapse_rotmhd_20261117/optab_evaporation_gate_20261117`.

The high fraction of bounded inexact linear accepts is expected from the
known double-precision correction floor but means this experiment is an outer
Newton/conservation gate, not evidence that every inner solve reaches
`2e-8`.  A fully self-consistent low-temperature OPTAB product remains future
work and would need a documented condensation and grain-opacity pipeline.
