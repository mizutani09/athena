//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file noop.cpp
//! \brief Implements no-op versions of the general eos functions

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <limits>
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../eos.hpp"

Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::PresFromRhoEg" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::EgasFromRhoP" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::AsqFromRhoP" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::TempFromRhoEg" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::DlnTDlnEgasFromRhoEg" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::NablaAdFromRhoP" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return std::numeric_limits<Real>::quiet_NaN();
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}
