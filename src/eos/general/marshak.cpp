//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file marshak.cpp
//! \brief Implements a Marshak-wave specific EOS in the general EOS framework
//======================================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>

// Athena++ headers
#include "../../athena.hpp"
#include "../../parameter_input.hpp"
#include "../eos.hpp"

namespace {
Real beta_marshak = 0.1;
Real mu_marshak = 1.0;
constexpr Real kRadiationConst = 7.5657e-15;  // erg cm^-3 K^-4
constexpr Real kGasConst = 8.31451e+7;         // erg mol^-1 K^-1
Real temp_unit = 1.0;
Real a_rad_code = 1.0;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure for hydro bookkeeping.
//!        For the Marshak-wave test, hydro is expected to be fixed, so an ideal-gas-like
//!        pressure closure is sufficient here.
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.0) * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  return pres / (gamma_ - 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  return gamma_ * pres / std::max(rho, TINY_NUMBER);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::TempFromRhoEg(Real rho, Real egas)
//! \brief Return code temperature satisfying a_r T^4 = beta * e_g
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  Real egas_safe = std::max(egas, TINY_NUMBER);
  return std::pow(beta_marshak * egas_safe / a_rad_code, 0.25);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas)
//! \brief Return d ln(T) / d ln(egas) at constant density
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  // TempFromRhoEg() is floored at TINY_NUMBER, so its derivative must vanish
  // in the same low-energy regime for a consistent Newton linearization.
  if (egas <= TINY_NUMBER) {
    return 0.0;
  }
  return 0.25;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  beta_marshak = pin->GetOrAddReal("problem", "beta", beta_marshak);
  mu_marshak = pin->GetOrAddReal("hydro", "mu", mu_marshak);

  // Match the FLD unit convention used in the NR-FLD problem generators.
  rho_unit_ = pin->GetOrAddReal("hydro", "rho_unit", rho_unit_);
  egas_unit_ = pin->GetOrAddReal("hydro", "egas_unit", egas_unit_);
  inv_rho_unit_ = 1.0 / rho_unit_;
  inv_egas_unit_ = 1.0 / egas_unit_;
  vsqr_unit_ = egas_unit_ / rho_unit_;
  inv_vsqr_unit_ = 1.0 / vsqr_unit_;

  temp_unit = egas_unit_ / rho_unit_ * mu_marshak / kGasConst;
  a_rad_code = kRadiationConst / (egas_unit_ / std::pow(temp_unit, 4));
  return;
}
