"""Adapt an isolated athena_st snapshot for the matching Rad-EOS shock reference.

The upstream repository is never edited. Each replacement checks its exact
source context so that a changed upstream implementation fails visibly.
"""

from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one source pattern in {path}: {old[:60]!r}; found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def adapt_reference_source(source: Path) -> None:
    """Match FLD cgs units and initialize Rad-EOS energy from its EOS."""
    eos = source/"src/eos/general/rad.cpp"
    replace_once(eos, '''  Real Rgas = 8.314462618e7; // gas constant in cgs units
  Real mu = pin->GetReal("hydro", "mu_mmw");
  Real Msun = 1.989e33; // solar mass in g
  Real Gcnst = 6.6725985e-8; // gravitational constant in cgs units
  Real MUnit = Msun*pin->GetReal("problem","Mstar_dim");
  Real GUnit = Gcnst;
  Real a_r_unit = std::pow(Rgas/mu, 4)*MUnit/std::pow(GUnit*MUnit, 3);
  a_r_sim_ = a_r_dim_/a_r_unit;
''', '''  // Use the same density, energy, and temperature units as nrfld_shock_tube.
  const Real Rgas = 8.3144621e7;
  const Real mu = pin->GetReal("hydro", "mu");
  const Real rho_unit = pin->GetReal("hydro", "rho_unit");
  const Real egas_unit = pin->GetReal("hydro", "egas_unit");
  const Real T_unit = egas_unit/rho_unit*mu/Rgas;
  a_r_sim_ = a_r_dim_*std::pow(T_unit, 4)/egas_unit;
''')
    problem = source/"src/pgen/shock_tube_hydro.cpp"
    replace_once(problem, "Rgas = 8.31451e+7; // erg/(mol*K)",
                 "Rgas = 8.3144621e+7; // erg/(mol*K), matching NR-FLD")
    replace_once(problem, '''  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real dx1 = pcoord->dx1f(4);''',
                 '''  Real gamma = pin->GetReal("hydro", "gamma");
  Real dx1 = pcoord->dx1f(4);''')
    replace_once(problem, "igm1*p0_L + 0.5*rho0_L*v0_L*v0_L",
                 "peos->EgasFromRhoP(rho0_L, p0_L) + 0.5*rho0_L*v0_L*v0_L")
    replace_once(problem, "igm1*p0_R + 0.5*rho0_R*v0_R*v0_R",
                 "peos->EgasFromRhoP(rho0_R, p0_R) + 0.5*rho0_R*v0_R*v0_R")
    replace_once(problem, '''void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gamma = peos->GetGamma();''',
                 '''void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  (void)pin;''')
