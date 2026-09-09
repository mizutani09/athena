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
#include <limits>
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
bool clamp_to_table = false;
constexpr int kPresOverEgas = 0;
constexpr int kEgasOverPres = 1;
constexpr int kAsqRhoOverPres = 2;
constexpr int kTemperature = 3;
constexpr int kEntropyRhoEg = 4;
constexpr int kEntropyRhoP = 5;
constexpr int kNablaAdRhoEg = 6;
Real derivative_log_step = 1.0e-3;

inline void GetEosCoordinates(EosTable *ptable, int kOut, Real var, Real rho,
                              Real &x2, Real &x1) {
  x1 = std::log10(rho * ptable->rhoUnit);
  x2 = std::log10(var * ptable->EosRatios(kOut) * ptable->eUnit) + dens_pow * x1;
  if (clamp_to_table) {
    x1 = std::min(std::max(x1, ptable->logRhoMin), ptable->logRhoMax);
    x2 = std::min(std::max(x2, ptable->logEgasMin), ptable->logEgasMax);
  }
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

inline Real GetEntropyData(EosTable *ptable, const int kOut, const Real var,
                           const Real rho) {
  Real x1, x2;
  GetEosCoordinates(ptable, kOut, var, rho, x2, x1);
  return std::pow(10.0, ptable->table.interpolate(kOut, x2, x1));
}

// Differentiate a log10(table field) with respect to one of the table's
// logarithmic coordinates.  At an in-domain edge this becomes a one-sided
// derivative, avoiding the half-slope produced by clamped central samples.
inline Real LogTablePartial(EosTable *ptable, const int kOut, const Real x2,
                            const Real x1, const bool along_x2) {
  const Real center = along_x2 ? x2 : x1;
  const Real lower_bound = along_x2 ? ptable->logEgasMin : ptable->logRhoMin;
  const Real upper_bound = along_x2 ? ptable->logEgasMax : ptable->logRhoMax;
  const Real step = derivative_log_step/std::log(10.0);
  Real lower = center - step;
  Real upper = center + step;
  if (center >= lower_bound && center <= upper_bound) {
    lower = std::max(lower, lower_bound);
    upper = std::min(upper, upper_bound);
  }
  if (!(upper > lower)) return std::numeric_limits<Real>::quiet_NaN();
  const Real f_lower = along_x2
      ? ptable->table.interpolate(kOut, lower, x1)
      : ptable->table.interpolate(kOut, x2, lower);
  const Real f_upper = along_x2
      ? ptable->table.interpolate(kOut, upper, x1)
      : ptable->table.interpolate(kOut, x2, upper);
  return (f_upper - f_lower)/(upper - lower);
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
//! \brief Numerically evaluate d ln(T) / d ln(egas) at constant density
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  // Differentiate the interpolated temperature itself instead of requiring a
  // derivative field in the table.  A symmetric perturbation in ln(egas)
  // gives the logarithmic derivative directly and remains well scaled across
  // the many decades covered by typical stellar EOS tables.
  const Real eg_center = std::max(egas, TINY_NUMBER);
  const Real eg_lo = eg_center*std::exp(-derivative_log_step);
  const Real eg_hi = eg_center*std::exp( derivative_log_step);
  const Real t_lo = TempFromRhoEg(rho, eg_lo);
  const Real t_hi = TempFromRhoEg(rho, eg_hi);
  if (!std::isfinite(t_lo) || !std::isfinite(t_hi)
      || t_lo <= TINY_NUMBER || t_hi <= TINY_NUMBER) return 0.0;
  return (std::log(t_hi) - std::log(t_lo))/(2.0*derivative_log_step);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres)
//! \brief Return (d ln T / d ln P)_s from the tabulated thermodynamic closure
Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres) {
  if (ptable == nullptr || !std::isfinite(rho) || !std::isfinite(pres)
      || rho <= 0.0 || pres <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }

  const Real egas = EgasFromRhoP(rho, pres);
  if (!std::isfinite(egas) || egas <= 0.0)
    return std::numeric_limits<Real>::quiet_NaN();

  // A seventh forward (rho,e_spec) field, when present, is the source EOS's
  // direct nabla_ad=(d ln T/d ln P)_s and is preferred to reconstructed
  // derivatives.  Like all Athena EOS fields it stores log10 of a positive
  // value; its ratio only shifts the log-energy coordinate.
  if (ptable->nVar > kNablaAdRhoEg) {
    const Real nabla_ad = GetEosData(ptable, kNablaAdRhoEg, egas, rho);
    return (std::isfinite(nabla_ad) && nabla_ad > 0.0)
        ? nabla_ad : std::numeric_limits<Real>::quiet_NaN();
  }

  // The indirect identity below needs the standard pressure, Gamma1, and
  // temperature fields.  Older three-field tables have no temperature and
  // therefore cannot provide a thermodynamic temperature gradient.
  if (ptable->nVar <= kTemperature)
    return std::numeric_limits<Real>::quiet_NaN();

  // For the four/six-field schemas, combine the source EOS's tabulated
  // Gamma1=(d ln P/d ln rho)_s with derivatives of the interpolated P(rho,u)
  // and T(rho,u).  At constant T,
  //   chi_rho = P_1 - P_2 T_1/T_2,  chi_T = P_2/T_2,
  // and the exact thermodynamic identities
  //   Gamma1 = chi_rho + chi_T (Gamma3-1),
  //   nabla_ad = (Gamma3-1)/Gamma1
  // give the result.  This avoids treating independently interpolated entropy
  // as an exact potential.  Constants from cgs/code units and field ratios
  // vanish under differentiation.
  Real p_x2, p_x1, t_x2, t_x1;
  Real x2, x1;
  GetEosCoordinates(ptable, kPresOverEgas, egas, rho, x2, x1);
  p_x2 = LogTablePartial(ptable, kPresOverEgas, x2, x1, true) + 1.0;
  p_x1 = LogTablePartial(ptable, kPresOverEgas, x2, x1, false) - dens_pow;
  GetEosCoordinates(ptable, kTemperature, egas, rho, x2, x1);
  t_x2 = LogTablePartial(ptable, kTemperature, x2, x1, true);
  t_x1 = LogTablePartial(ptable, kTemperature, x2, x1, false);
  const Real gamma1 = AsqFromRhoP(rho, pres)*rho/pres;
  if (!std::isfinite(gamma1) || gamma1 <= 0.0
      || !std::isfinite(t_x2) || std::abs(t_x2) <= TINY_NUMBER)
    return std::numeric_limits<Real>::quiet_NaN();
  const Real chi_rho = p_x1 - p_x2*t_x1/t_x2;
  const Real chi_t = p_x2/t_x2;
  const Real nabla_ad = (gamma1 - chi_rho)/(gamma1*chi_t);
  return (std::isfinite(nabla_ad) && nabla_ad > 0.0)
      ? nabla_ad : std::numeric_limits<Real>::quiet_NaN();
}

//----------------------------------------------------------------------------------------
//! \fn bool EquationOfState::HasEntropyTable() const
//! \brief Return whether both NATA entropy representations are present.
bool EquationOfState::HasEntropyTable() const {
  return ptable != nullptr && ptable->nVar > kEntropyRhoP;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EntropyFromRhoEg(Real rho, Real egas)
//! \brief Return specific entropy from the forward (rho, e_spec) field.
Real EquationOfState::EntropyFromRhoEg(Real rho, Real egas) {
  if (!HasEntropyTable()) return std::numeric_limits<Real>::quiet_NaN();
  return GetEntropyData(ptable, kEntropyRhoEg, egas, rho);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EntropyFromRhoP(Real rho, Real pres)
//! \brief Return specific entropy from the inverse (rho, P/rho) field.
Real EquationOfState::EntropyFromRhoP(Real rho, Real pres) {
  if (!HasEntropyTable()) return std::numeric_limits<Real>::quiet_NaN();
  return GetEntropyData(ptable, kEntropyRhoP, pres, rho);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEntropy(Real rho, Real entropy,
//!                                                Real pres_guess)
//! \brief Invert the tabulated entropy at fixed density in log(P/rho).
Real EquationOfState::PresFromRhoEntropy(Real rho, Real entropy,
                                          Real pres_guess) {
  if (!HasEntropyTable() || !std::isfinite(rho) || !std::isfinite(entropy)
      || rho <= 0.0 || entropy <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }

  const Real rho_safe = std::min(std::max(rho, density_floor_),
      std::pow(10.0, ptable->logRhoMax)/std::max(ptable->rhoUnit, TINY_NUMBER));
  Real x1 = std::log10(std::max(rho_safe*ptable->rhoUnit, TINY_NUMBER));
  x1 = std::min(std::max(x1, ptable->logRhoMin), ptable->logRhoMax);
  const Real x2_lo = ptable->logEgasMin;
  const Real x2_hi = ptable->logEgasMax;
  const Real target = std::log(std::max(entropy, TINY_NUMBER));
  const int nscan = 128;

  auto pressure_from_x2 = [&](const Real x2) {
    const Real logvar_phys = x2 - dens_pow*x1
        - std::log10(std::max(ptable->EosRatios(kEntropyRhoP)
                              * ptable->eUnit, TINY_NUMBER));
    return std::pow(10.0, logvar_phys);
  };
  auto residual = [&](const Real x2) {
    const Real s = GetEntropyData(ptable, kEntropyRhoP,
                                  pressure_from_x2(x2), rho_safe);
    return (std::isfinite(s) && s > 0.0)
        ? std::log(s) - target : std::numeric_limits<Real>::quiet_NaN();
  };

  const Real pguess = std::max(pres_guess, TINY_NUMBER);
  const Real x2_guess = std::log10(pguess*ptable->EosRatios(kEntropyRhoP)
                                   *ptable->eUnit)
                        + dens_pow*x1;
  const Real guess = std::min(std::max(x2_guess, x2_lo), x2_hi);
  Real best_x = 0.5*(x2_lo + x2_hi);
  Real best_abs = std::numeric_limits<Real>::infinity();
  Real best_a = 0.0;
  Real best_b = 0.0;
  Real best_fa = 0.0;
  Real best_distance = std::numeric_limits<Real>::infinity();
  bool have_bracket = false;
  Real left = x2_lo;
  Real fleft = residual(left);
  for (int n = 0; n <= nscan; ++n) {
    const Real x = x2_lo + (x2_hi - x2_lo)*static_cast<Real>(n)/nscan;
    const Real fx = residual(x);
    if (std::isfinite(fx) && std::abs(fx) < best_abs) {
      best_abs = std::abs(fx);
      best_x = x;
    }
    if (n > 0 && std::isfinite(fleft) && std::isfinite(fx)
        && (fleft == 0.0 || fx == 0.0 || fleft*fx < 0.0)) {
      const Real distance = std::abs(0.5*(left + x) - guess);
      if (distance < best_distance) {
        best_a = left;
        best_b = x;
        best_fa = fleft;
        best_distance = distance;
        have_bracket = true;
      }
    }
    left = x;
    fleft = fx;
  }

  if (have_bracket) {
    Real a = best_a;
    Real b = best_b;
    Real fa = best_fa;
    for (int iter = 0; iter < 60; ++iter) {
      const Real mid = 0.5*(a + b);
      const Real fm = residual(mid);
      if (!std::isfinite(fm)) break;
      if (std::abs(fm) < 1.0e-10 || (b - a) < 1.0e-11)
        return pressure_from_x2(mid);
      if (fa*fm <= 0.0) {
        b = mid;
      } else {
        a = mid;
        fa = fm;
      }
    }
    return pressure_from_x2(0.5*(a + b));
  }
  if (best_abs < 1.0e-6) return pressure_from_x2(best_x);
  return std::numeric_limits<Real>::quiet_NaN();
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::RhoFromPEntropy(Real pres, Real entropy,
//!                                            Real rho_guess)
//! \brief Invert the tabulated entropy at fixed pressure in log density.
Real EquationOfState::RhoFromPEntropy(Real pres, Real entropy, Real rho_guess) {
  if (!HasEntropyTable() || !std::isfinite(pres) || !std::isfinite(entropy)
      || pres <= 0.0 || entropy <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }

  const Real rho_table_lo =
      std::pow(10.0, ptable->logRhoMin)/std::max(ptable->rhoUnit, TINY_NUMBER);
  const Real rho_table_hi =
      std::pow(10.0, ptable->logRhoMax)/std::max(ptable->rhoUnit, TINY_NUMBER);
  const Real rho_lo = std::max(density_floor_, rho_table_lo);
  const Real rho_hi = std::max(rho_lo, rho_table_hi);
  const Real log_lo = std::log(rho_lo);
  const Real log_hi = std::log(rho_hi);
  const Real target = std::log(std::max(entropy, TINY_NUMBER));
  const int nscan = 128;

  auto residual = [&](const Real logrho) {
    const Real rho = std::exp(logrho);
    const Real s = EntropyFromRhoP(rho, pres);
    return (std::isfinite(s) && s > 0.0)
        ? std::log(s) - target : std::numeric_limits<Real>::quiet_NaN();
  };

  Real best_log = 0.5*(log_lo + log_hi);
  Real best_abs = std::numeric_limits<Real>::infinity();
  Real best_a = 0.0;
  Real best_b = 0.0;
  Real best_fa = 0.0;
  Real best_bracket_distance = std::numeric_limits<Real>::infinity();
  bool have_bracket = false;
  const Real guess_log = std::min(std::max(
      std::log(std::max(rho_guess, rho_lo)), log_lo), log_hi);
  Real left = log_lo;
  Real fleft = residual(left);
  for (int n = 0; n <= nscan; ++n) {
    const Real x = log_lo + (log_hi - log_lo)*static_cast<Real>(n)/nscan;
    const Real fx = residual(x);
    if (std::isfinite(fx) && std::abs(fx) < best_abs) {
      best_abs = std::abs(fx);
      best_log = x;
    }
    if (n > 0 && std::isfinite(fleft) && std::isfinite(fx)
        && (fleft == 0.0 || fx == 0.0 || fleft*fx < 0.0)) {
      const Real distance = std::abs(0.5*(left + x) - guess_log);
      if (distance < best_bracket_distance) {
        best_a = left;
        best_b = x;
        best_fa = fleft;
        best_bracket_distance = distance;
        have_bracket = true;
      }
    }
    left = x;
    fleft = fx;
  }

  if (have_bracket) {
    Real a = best_a;
    Real b = best_b;
    Real fa = best_fa;
    for (int iter = 0; iter < 60; ++iter) {
      const Real mid = 0.5*(a + b);
      const Real fm = residual(mid);
      if (!std::isfinite(fm)) break;
      if (std::abs(fm) < 1.0e-10 || (b - a) < 1.0e-11) return std::exp(mid);
      if (fa*fm <= 0.0) {
        b = mid;
      } else {
        a = mid;
        fa = fm;
      }
    }
    return std::exp(0.5*(a + b));
  }

  // A nearest tabulated point is only acceptable for roundoff-level misses.
  // Otherwise the requested (P,s) pair is outside the supplied EOS domain.
  if (best_abs < 1.0e-6) return std::exp(best_log);
  return std::numeric_limits<Real>::quiet_NaN();
}

//----------------------------------------------------------------------------------------
//! void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput* pin) {
  dens_pow = pin->GetOrAddReal("hydro", "dens_pow", dens_pow);
  clamp_to_table = pin->GetOrAddBoolean("hydro", "eos_table_clamp", false);
  derivative_log_step =
      pin->GetOrAddReal("hydro", "eos_derivative_log_step", derivative_log_step);
  if (!std::isfinite(derivative_log_step) || derivative_log_step <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
        << "hydro/eos_derivative_log_step must be finite and positive." << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}
