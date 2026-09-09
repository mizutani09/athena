//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ideal.cpp
//! \brief implements ideal EOS in general EOS framework, mostly for debuging
//======================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.) * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  return pres / (gamma_ - 1.);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  return gamma_ * pres / rho;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::TempFromRhoEg(Real rho, Real egas)
//! \brief Return gas temperature in the general EOS ideal-gas debug mode
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.) * egas / rho;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas)
//! \brief Return d ln(T) / d ln(egas) at constant density
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  return 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres)
//! \brief Return (d ln T / d ln P)_s for a calorically ideal gas
Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres) {
  (void)rho;
  (void)pres;
  return (gamma_ - 1.0)/gamma_;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}
