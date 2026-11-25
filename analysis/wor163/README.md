# Nonrotating aggressive-core FLD baseline

This branch adds a radiation-enabled path to the accepted WOR-177
`corecut7p00` equilibrium without modifying the HD result.  At every SCF
quadrature point, the total pressure is split by solving

`P_total = rho R T / mu + a_rad T^4 / 3`.

Athena stores the resulting gas internal energy and LTE radiation energy in
separate variables.  The cell-average initialization repeats the closure from
the averaged density and total pressure so that the discrete gas and radiation
temperatures agree initially.  A 100 K external radiation floor prevents
negative reconstruction at the numerical atmosphere boundary.  Planck
coupling is tapered only across the artificial surface taper; Rosseland
transport remains active.

The first gate deliberately injects no luminosity.  It tests mechanical
support, LTE exchange, FLD diffusion, the Marshak boundary, and self-gravity
before WOR-164 introduces a physically justified radiative luminosity.  It
uses FGMRES and a `1e-5` code-time startup cap through `t=1e-3`; a coarse-grid
test then grows the timestep normally and reaches `t=1e-2` in 110 cycles.
An initial full-L3 run exhausted Newton backtracking at `dt=1.6e-4`, but a
clean restart from cycle 100 converged through that step and reached
`dt=1.28e-3`. Diagnostics localized the later failure to optically thin
atmosphere cells at physical and L3/L4 boundaries, not the stellar interior:
a Newton correction larger than the local radiation energy was clipped to the
positivity floor and spoiled the global line-search direction. Restart tests
therefore use `max_update_fraction=1.0`. The same L3 checkpoint then reached
`t=1e-2` in 109 cycles without fixing the timestep; `0.5` was too restrictive,
while `10.0` and an uncapped update failed at the larger step. This is a
short solver gate, not yet a long-term stellar-stability result.

The current `4wz` reconstruction has an explicit FLD guard.  This baseline
therefore uses the supported second-order hydro path.  Combining 4WZ hydro
with FLD remains a separate validation task.
