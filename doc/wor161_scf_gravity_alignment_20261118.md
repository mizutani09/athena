# WOR-161: Athena++ gravity, core, atmosphere, and boundary alignment

## Result

The WOR-160 mapper now initializes a gravity-enabled standalone Athena++ HD
problem.  Gas and core accounting is explicit:

```text
rho_total = rho_star + rho_atmosphere
rho_star --Athena++ MG Poisson--> Phi_self
(M_core,h) --analytic compact spline--> Phi_core
momentum source = -rho_total grad(Phi_self + Phi_core)
```

The source mask removes the analytic atmosphere from the Poisson right-hand
side.  No effective spline-core density is added, so the 5.70 Msun core is not
counted twice.  The source gas mass remains 15.30 Msun.

## Sign, units, and softening convention

Both potentials are negative and approach zero at infinity.  Acceleration is
`-grad(Phi)`.  The copied C++ kernel in `scf_core_softening.hpp` is algebraically
the WOR-153 BSCF-rsg kernel at commit `fe3f380`:

- `h=3.541497158802513e11 cm` is the compact-support radius;
- the field is finite at the origin;
- it is exactly Newtonian for `r>=h`;
- it is not a Plummer epsilon (`epsilon_equivalent=h/2.8`).

At the finest spacing `dx=7.47334e10 cm`, `h/dx=4.74`, or 9.48 cells across
the support diameter.  Unit tests independently check the origin, continuity,
Newtonian exterior, and `dPhi/dr=g`.

Athena++ uses `G_code=3.23035112e-7`.  Gas self-gravity uses the existing
face-potential momentum and mass-flux energy source.  The external core uses
the same centered potential gradient for its momentum kick and a midpoint
kinetic-energy kick.  This leaves internal energy unchanged when hydro fluxes
are absent.  Athena++'s existing self-gravity energy source is documented in
the code as not exactly conservative; WOR-161 does not silently claim
otherwise.

## Poisson and hydrodynamic boundaries

The gas solve uses full multigrid, an explicitly fixed origin, no dipole, and
isolated multipole boundaries through l=4 on every face.  Gravity ghost cells
are filled so the same centered gradient is available at block and SMR
interfaces.

The atmosphere is nonrotating and isothermal at 1e4 K.  Outside the reference
equatorial radius its density follows the analytic total-mass monopole:

```text
rho(r) = rho_ref exp[-(Phi_mono(r)-Phi_mono(R_ref))/c_s^2].
```

It is capped at `rho_ref` inward and at the independent hydro floors outward.
The six hydro boundaries extrapolate this hydrostatic state and apply a diode
to the normal velocity.  The recommended `rho_ref=1e-14 g cm^-3` atmosphere
contains `2.55e27 g`, 8.39e-8 of the stellar gas mass.

## Quantitative audit

The residual is evaluated from exactly the cell-centered fields used by the
run:

```text
a_res = -grad(P)/rho - grad(Phi_self) - grad(Phi_core)
        + Omega^2 (x,y,0).
```

The reported L2 norm is stellar-mass weighted.  The gas-potential norm is also
stellar-mass weighted against the imported BSCF gas potential.

| mesh / atmosphere | Phi_self L2 | force L2 | force max |
|---|---:|---:|---:|
| root32 + L5, rho_atm=1e-14 | 7.301e-4 | 1.856e-1 | 0.787 |
| root64 + L5, rho_atm=1e-14 | 5.829e-4 | 6.205e-2 | 0.787 |
| root64 + L5, rho_atm=1e-15 | 5.830e-4 | 6.205e-2 | 0.970 |
| root64 + L5, rho_atm=1e-13 | 5.820e-4 | 6.205e-2 | 0.519 |

The atmosphere scan changes the force L2 value by less than 6e-9 relative.
The maximum changes because the 99.9% stellar-fraction cut selects a different
last surface cell.  It is therefore retained as a localization diagnostic, not
used as the acceptance statistic.

On the root64 equatorial slice, the force residual median, p90, and p99 are
`1.05e-3`, `1.05e-2`, and `1.03e-1`.  The figure shows that the large values
are confined primarily to the surface, with weaker square features at SMR
interfaces.  Work and torque maps show the remaining symmetry error is
centered on the finest core and refinement transitions rather than a global
sign error.

The mapped self-gravity energy differs from the BSCF value by `1.55e-4`; the
core interaction energy differs by `2.84e-4`.  These independent closures
would not both pass if the core effective density had also entered Poisson.

A root32 one-step run (`dt=1e-3` code time = 22.0 s) gives zero stellar-mass
drift at stored precision, `Delta J/J=1.64e-9`, `Delta E_HD/E_HD=-1.39e-6`,
and a total HD+self+core mechanical-energy change of `-3.10e-4`.  This is a
source/boundary smoke test, not the relaxation measurement assigned to
WOR-162.

## Surface-resolution limitation and next decision

The central softening is adequately resolved, but the surface is not.  At the
equator:

| density [g cm^-3] | pressure scale height [Rsun] | cells on 34.375 Rsun root dx |
|---:|---:|---:|
| 1e-8 | 68.3 | 1.99 |
| 1e-9 | 22.6 | 0.66 |
| 1e-10 | 5.47 | 0.16 |
| 1e-11 | 1.16 | 0.034 |

This explains why increasing central refinement alone cannot remove the
surface maximum.  A globally finer root grid is expensive and is already
known from WOR-160 to be needed independently for angular momentum.  The
quantified candidates for WOR-162 are block-level surface AMR driven by
`H_P/dx` or density gradient, and a well-balanced/hydrostatic reconstruction.
Neither is enabled in WOR-161 because their effect must be compared against
the recorded baseline rather than assumed beneficial.

## Reproduction

```sh
python3 configure.py --prob=scf_rsg --coord=cartesian --eos=adiabatic \
  --flux=hllc --nscalars=2 --grav=mg --cxx=g++
make -j4
python3 tests/scf_mapping/test_mapping.py --athena bin/athena
wor161_tabs=$(mktemp -d /tmp/wor161-tabs.XXXXXX)
bin/athena -i inputs/rsg/athinput.scf_rsg_map -d "$wor161_tabs" \
  time/nlim=0 output2/dt=-1 output3/dt=1
python3 analysis/wor161/analyze_gravity.py --tab-dir "$wor161_tabs"
```

The input is `inputs/rsg/athinput.scf_rsg_map`.  Machine-readable metrics,
history files, the equatorial field table, and plots are in
`analysis/wor161/`.
