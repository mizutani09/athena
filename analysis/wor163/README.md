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
At full L3 resolution, doubling beyond `dt=8e-5` exhausted Newton
backtracking at `dt=1.6e-4`, so the production startup gate conservatively
holds `dt=1e-5` through `t=1e-2`.

## Berta 4wz hydrodynamics with FLD

This branch removes the former FLD guard after separating the numerical
roles cleanly. Berta WENO-Z supplies the hydrodynamic point states, while the
existing fourth-order explicit FLD face-average-to-point construction supplies
the radiation state used by HLLC and the explicit radiation force. The
implicit diffusion, Newton solve, and FGMRES/MG preconditioner retain their
established order.

On smooth single-block radiation transport, the 32-cubed and 64-cubed L1
errors are `6.47306e-8` and `4.17637e-9`, an observed order of 3.95.
Fixed-size multi-block tests show order reduction at block interfaces, so this
is a local explicit-operator check rather than a claim that the complete FLD
scheme is fourth order.

For memory control, 4wz does not allocate the legacy PPM face-state and scalar
scratch arrays. The two refinement tracers use conservative PLM; density,
momentum, and gas energy retain 4wz. A two-rank, 120-MeshBlock AMR smoke test
completed one RK4 step with FGMRES in 2--4 iterations per Newton iteration.
AMR requires `coarse_operator=rediscretized`; the Galerkin hierarchy is
uniform-topology only.
