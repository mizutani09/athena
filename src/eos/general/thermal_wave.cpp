//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_wave.cpp
//! \brief Implements a constant-volumetric-heat-capacity EOS for the NR-FLD thermal wave.
//========================================================================================

// C++ headers
#include <algorithm>

// Athena++ headers
#include "../../athena.hpp"
#include "../../parameter_input.hpp"
#include "../eos.hpp"

namespace {
constexpr Real kGasConst = 8.31451e+7;  // erg mol^-1 K^-1
Real rho_cv_phys = 0.05;                // erg cm^-3 K^-1
Real mu_thermal_wave = 1.0;
Real temp_unit = 1.0;
Real rho_cv_code = 1.0;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure for hydro bookkeeping.
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.0) * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return gas internal energy density.
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  return pres / (gamma_ - 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared.
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  return gamma_ * pres / std::max(rho, TINY_NUMBER);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::TempFromRhoEg(Real rho, Real egas)
//! \brief Return code temperature from e_gas = rho_cv * T.
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  Real egas_safe = std::max(egas, TINY_NUMBER);
  return egas_safe / rho_cv_code;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas)
//! \brief Return d ln(T) / d ln(egas) at constant density.
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  if (egas <= TINY_NUMBER) {
    return 0.0;
  }
  return 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput *pin)
//! \brief Initialize constants for the thermal-wave EOS.
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  rho_cv_phys = pin->GetOrAddReal("problem", "rho_cv", rho_cv_phys);
  mu_thermal_wave = pin->GetOrAddReal("hydro", "mu", mu_thermal_wave);

  rho_unit_ = pin->GetOrAddReal("hydro", "rho_unit", rho_unit_);
  egas_unit_ = pin->GetOrAddReal("hydro", "egas_unit", egas_unit_);
  inv_rho_unit_ = 1.0 / rho_unit_;
  inv_egas_unit_ = 1.0 / egas_unit_;
  vsqr_unit_ = egas_unit_ / rho_unit_;
  inv_vsqr_unit_ = 1.0 / vsqr_unit_;

  temp_unit = pin->GetOrAddReal("hydro", "T_unit", -1.0);
  if (temp_unit <= 0.0) {
    temp_unit = egas_unit_ / rho_unit_ * mu_thermal_wave / kGasConst;
  }
  rho_cv_code = rho_cv_phys * temp_unit / egas_unit_;
  return;
}
