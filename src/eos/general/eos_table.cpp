//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file eos_table.cpp
//! \brief implements functions in class EquationOfState for an EOS lookup table
//======================================================================================

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/interp_table.hpp"
#include "../eos.hpp"

namespace {
Real dens_pow = -1.0;
constexpr int kPresOverEgas = 0;
constexpr int kEgasOverPres = 1;
constexpr int kAsqRhoOverPres = 2;
constexpr int kTemperature = 3;
constexpr int kDlnTDlnEgas = 4;

inline void GetEosCoordinates(EosTable *ptable, int kOut, Real var, Real rho,
                              Real &x2, Real &x1) {
  x1 = std::log10(rho * ptable->rhoUnit);
  x2 = std::log10(var * ptable->EosRatios(kOut) * ptable->eUnit) + dens_pow * x1;
}

//----------------------------------------------------------------------------------------
//! \fn Real GetEosData(EosTable *ptable, int kOut, Real var, Real rho)
//! \brief Gets interpolated data from EOS table assuming 'var' has dimensions
//!        of energy per volume.
inline Real GetEosData(EosTable *ptable, int kOut, Real var, Real rho) {
  Real x1, x2;
  GetEosCoordinates(ptable, kOut, var, rho, x2, x1);
  return std::pow((Real)10, ptable->table.interpolate(kOut, x2, x1));
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return interpolated gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  return GetEosData(ptable, kPresOverEgas, egas, rho) * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return interpolated internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  return GetEosData(ptable, kEgasOverPres, pres, rho) * pres;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return interpolated adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  return GetEosData(ptable, kAsqRhoOverPres, pres, rho) * pres / rho;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::TempFromRhoEg(Real rho, Real egas)
//! \brief Return interpolated gas temperature
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  Real x1, x2;
  GetEosCoordinates(ptable, kTemperature, egas, rho, x2, x1);
  Real temperature_phys = std::pow((Real)10, ptable->table.interpolate(kTemperature, x2, x1));
  if (temperature_phys <= TINY_NUMBER) {
    // std::stringstream msg;
    // msg << "### FATAL ERROR in EquationOfState::TempFromRhoEg" << std::endl
    //     << "Interpolated temperature is non-positive: T = " << temperature_phys << std::endl;
    // ATHENA_ERROR(msg);
    temperature_phys = TINY_NUMBER;
  }
  // std::cout << "rho = " << rho << ", egas = " << egas << ", temperature_phys = " << temperature_phys << " K" << std::endl;
  return temperature_phys/ptable->tUnit;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas)
//! \brief Return interpolated d ln(T) / d ln(egas) at constant density
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  if (ptable->nVar <= kDlnTDlnEgas) {
    // Fallback for legacy 4-field tables that do not store d ln(T) / d ln(egas).
    const Real eps = 1.0e-3;
    Real eg_lo = std::max(egas*(1.0 - eps), TINY_NUMBER);
    Real eg_hi = std::max(egas*(1.0 + eps), eg_lo*(1.0 + eps));
    Real t_lo = TempFromRhoEg(rho, eg_lo);
    Real t_hi = TempFromRhoEg(rho, eg_hi);
    if (t_lo <= TINY_NUMBER || t_hi <= TINY_NUMBER) return 0.0;
    return (std::log(t_hi) - std::log(t_lo))/(std::log(eg_hi) - std::log(eg_lo));
  }
  Real x1, x2;
  GetEosCoordinates(ptable, kDlnTDlnEgas, egas, rho, x2, x1);
  return ptable->table.interpolate(kDlnTDlnEgas, x2, x1);
}

//----------------------------------------------------------------------------------------
//! void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput* pin) {
  dens_pow = pin->GetOrAddReal("hydro", "dens_pow", dens_pow);
  return;
}
