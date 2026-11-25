# WOR-183 prescribed-companion smoke gate

This branch adds an optional circular point-mass companion to `scf_rsg` without
replacing the accepted SCF mapping, tabulated EOS, 4wz hydrodynamics,
self-gravity, or mixed-frame FLD implementation.  The default remains
`companion_enabled=false`, so the isolated-star baseline is unchanged.

The first implementation deliberately reuses only the relevant mechanisms
from `ce_main.cpp` and `nrfld_conv_star_ce`:

- compact spline softening;
- a prescribed circular position;
- the indirect acceleration required by primary-centred coordinates;
- a smooth gravity ramp; and
- acceleration/orbital timestep limits.

Momentum and gas-energy forcing use the gradient and face differences of the
same total external potential.  Six history fields record the companion
position, ramp, separation, maximum tidal acceleration, and gas power.
`check_scf_companion.cpp` verifies the ramp, orbit, cancellation of direct and
indirect acceleration at the primary origin, and the finite-difference
potential/analytic-force agreement.

The initial smoke test keeps the companion outside the current `+/-1200 R_sun`
box.  This still includes the donor, its Roche-facing surface, and the tidal
field, and allows a restart from the accepted isolated-star state.  It does
not resolve the companion, its accretion flow, or orbital backreaction.  A
full contact/CE calculation therefore still requires an expanded domain,
local companion refinement, and a live particle orbit with gas-force and
energy/angular-momentum ledgers.

The q=0.25 RLOF parameters come from
`analysis/wor177/corecut7p00_q025_corot_source_manifest.json`:

- separation: `1.2939385270211592e14 cm`;
- synchronous angular velocity: `4.0100850320255668e-8 rad s^-1`;
- smoke softening support: `3.0e12 cm`.

`xd_q025_binary_smoke.sh` ramps the companion after the t=10 isolated-star
restart and advances only a few CFL=0.3 steps.  Passing this gate establishes
source-term and solver compatibility, not binary stability.

## XD validation, 2026-11-25/26

The mass used by the companion source is intentionally
`q*(gas_mass_g+core_mass_g)`.  Here the grid gas is 14.49 Msun and the
excised core particle is 6.52 Msun, so their sum is the manifest's 21.00-Msun
donor and q=0.25 gives a 5.25-Msun companion.  The history `mapped_mass_g`
contains only grid gas and must not be used by itself to set the companion
mass.

Job 348001 first advanced the accepted isolated-star `t=10` restart to
`t=10.25` on 168 ranks/3 M-test-a nodes.  The smooth ramp reached 0.15625,
exactly its analytic value, and the recorded separation agreed with the input
to about 1e-11 relatively.  All three timesteps converged in 6--7 Newton
iterations; 19 FGMRES solves used at most 19 iterations.  There was no NaN,
failure, backtracking, relaxed acceptance, or Krylov breakdown.

Job 348002 repeated the restart through the complete ramp and stopped at
`t=11.5`.  It completed in 99.8 s with 13 normal Newton solves (at most 7
iterations) and 79 FGMRES solves (mean 14.81, maximum 20).  Against the
companion-free L2 job 347973 at the same time:

- interior radial Mach was 0.00751 versus 0.00735;
- `rho >= rho_surface` radial Mach was 0.00697 versus 0.00679;
- taper radial Mach was 0.2414 versus 0.2421;
- no stellar mass exceeded radial Mach 0.3; and
- all four EOS-table margins remained positive.

Job 348003 continued the fully enabled companion from `t=11.5` to 13.  It
completed in 114.94 s with 13 normal Newton solves (at most 7 iterations) and
81 FGMRES solves (mean 14.57, maximum 19).  Against companion-free L2 job
347985 at `t=13`, interior and resolved-surface Mach increased by only 5.5%
and 6.2%, to 0.00805 and 0.00748; taper Mach changed by 0.4%, to 0.2734.
No stellar material exceeded Mach 0.3, and the EOS margins remained positive.

This is an intentionally short numerical gate.  One code time is 0.2546 d,
whereas the donor dynamical time is about 450 code times and the orbit is
7122 code times (1813.5 d).  The tested `t=10` to 13 interval is only 0.0067
donor dynamical times.  It proves that the prescribed companion is compatible
with tabulated EOS, 4wz, mixed-frame FLD, AMR, and FGMRES; it does not prove a
tidally relaxed RLOF initial condition.  A physical onset calculation should
ramp or relax the tide over a substantial fraction of a dynamical time, or
construct a non-axisymmetric binary equilibrium, before measuring mass
transfer.
