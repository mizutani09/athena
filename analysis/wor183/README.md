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
