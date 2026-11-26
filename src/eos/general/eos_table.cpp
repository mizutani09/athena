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
// Optional one-temperature LTE closure.  The table itself contains material
// thermodynamics only; when enabled, the hydro energy and pressure are
//
//   E = E_material(rho,T) + a T^4,
//   P = P_material(rho,T) + a T^4/3.
//
// This is the optically thick, perfectly coupled comparison model.  It must
// not be combined with NR-FLD, which evolves the same radiation energy as an
// independent conserved field.
bool include_lte_radiation = false;
Real radiation_constant_code = 0.0;
int lte_inversion_max_iterations = 12;
constexpr Real kRadiationConstantCgs = 7.5657e-15;
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

inline Real LogTablePartial(EosTable *ptable, int kOut, Real x2, Real x1,
                            bool along_x2);

inline Real MaterialPresFromRhoEg(EosTable *ptable, const Real rho,
                                  const Real egas) {
  return GetEosData(ptable, kPresOverEgas, egas, rho)*egas;
}

inline Real MaterialEgasFromRhoP(EosTable *ptable, const Real rho,
                                 const Real pres) {
  return GetEosData(ptable, kEgasOverPres, pres, rho)*pres;
}

inline Real MaterialAsqFromRhoP(EosTable *ptable, const Real rho,
                                const Real pres) {
  return GetEosData(ptable, kAsqRhoOverPres, pres, rho)*pres/rho;
}

inline Real MaterialTempFromRhoEg(EosTable *ptable, const Real rho,
                                  const Real egas) {
  Real x1, x2;
  GetEosCoordinates(ptable, kTemperature, egas, rho, x2, x1);
  const Real temperature_phys =
      std::pow(10.0, ptable->table.interpolate(kTemperature, x2, x1));
  return temperature_phys/ptable->tUnit;
}

inline bool MaterialEnergyBounds(EosTable *ptable, const Real rho,
                                 Real *lower, Real *upper) {
  if (ptable == nullptr || !std::isfinite(rho) || rho <= 0.0) return false;
  const Real x1 = std::log10(rho*ptable->rhoUnit);
  if (x1 < ptable->logRhoMin || x1 > ptable->logRhoMax) return false;
  const Real offset = dens_pow*x1
      +std::log10(ptable->EosRatios(kTemperature)*ptable->eUnit);
  *lower = std::pow(10.0, ptable->logEgasMin-offset);
  *upper = std::pow(10.0, ptable->logEgasMax-offset);
  return std::isfinite(*lower) && std::isfinite(*upper)
      && *lower > 0.0 && *upper > *lower;
}

inline Real RadiationEnergyFromTemperature(const Real temperature) {
  return radiation_constant_code*std::pow(temperature, 4);
}

inline Real TotalEnergyFromMaterial(EosTable *ptable, const Real rho,
                                    const Real material_energy) {
  const Real temperature = MaterialTempFromRhoEg(ptable, rho, material_energy);
  return material_energy+RadiationEnergyFromTemperature(temperature);
}

inline Real TotalPressureFromMaterial(EosTable *ptable, const Real rho,
                                      const Real material_energy) {
  const Real temperature = MaterialTempFromRhoEg(ptable, rho, material_energy);
  return MaterialPresFromRhoEg(ptable, rho, material_energy)
      +RadiationEnergyFromTemperature(temperature)/3.0;
}

inline Real MaterialDlnTDlnE(EosTable *ptable, const Real rho,
                             const Real material_energy) {
  Real x1, x2;
  GetEosCoordinates(ptable, kTemperature, material_energy, rho, x2, x1);
  return ptable->table.interpolateX2Derivative(kTemperature, x2, x1);
}

inline Real TotalEnergyLogDerivative(EosTable *ptable, const Real rho,
                                     const Real material_energy) {
  const Real temperature = MaterialTempFromRhoEg(ptable, rho, material_energy);
  const Real radiation_energy = RadiationEnergyFromTemperature(temperature);
  return material_energy+4.0*radiation_energy
      *MaterialDlnTDlnE(ptable, rho, material_energy);
}

inline Real TotalPressureLogDerivative(EosTable *ptable, const Real rho,
                                       const Real material_energy) {
  const Real material_pressure =
      MaterialPresFromRhoEg(ptable, rho, material_energy);
  const Real temperature = MaterialTempFromRhoEg(ptable, rho, material_energy);
  const Real radiation_energy = RadiationEnergyFromTemperature(temperature);
  Real x1, x2;
  GetEosCoordinates(ptable, kPresOverEgas, material_energy, rho, x2, x1);
  const Real dlnp_dlnenergy =
      1.0+ptable->table.interpolateX2Derivative(kPresOverEgas, x2, x1);
  return material_pressure*dlnp_dlnenergy
      +(4.0/3.0)*radiation_energy
       *MaterialDlnTDlnE(ptable, rho, material_energy);
}

template <typename TotalQuantity, typename LogDerivative>
Real MaterialEnergyFromTotal(EosTable *ptable, const Real rho,
                             const Real target, const Real initial_guess,
                             TotalQuantity total_quantity,
                             LogDerivative log_derivative) {
  if (!std::isfinite(target) || target <= 0.0)
    return std::numeric_limits<Real>::quiet_NaN();
  Real lower, upper;
  if (!MaterialEnergyBounds(ptable, rho, &lower, &upper))
    return std::numeric_limits<Real>::quiet_NaN();
  const Real lower_value = total_quantity(ptable, rho, lower);
  const Real upper_value = total_quantity(ptable, rho, upper);
  if (!std::isfinite(lower_value) || !std::isfinite(upper_value)
      || target < lower_value || target > upper_value)
    return std::numeric_limits<Real>::quiet_NaN();
  Real log_lower = std::log(lower);
  Real log_upper = std::log(upper);
  const Real bounded_guess = std::isfinite(initial_guess)
      && initial_guess > 0.0
      ? std::min(std::max(initial_guess, lower), upper)
      : std::sqrt(lower*upper);
  Real log_energy = std::log(bounded_guess);
  for (int iteration = 0; iteration < lte_inversion_max_iterations;
       ++iteration) {
    const Real material_energy = std::exp(log_energy);
    const Real value = total_quantity(ptable, rho, material_energy);
    if (!std::isfinite(value)) return std::numeric_limits<Real>::quiet_NaN();
    const Real relative = value/target-1.0;
    // The source table is piecewise bilinear and does not carry more than
    // roughly decimal-table precision.  Driving this inversion to machine
    // epsilon only repeats identical interpolations at Riemann faces.
    if (std::abs(relative) <= 1.0e-10)
      return material_energy;
    if (value < target) log_lower = log_energy;
    else log_upper = log_energy;
    const Real derivative = log_derivative(ptable, rho, material_energy);
    Real candidate = log_energy;
    if (std::isfinite(derivative) && derivative > 0.0) {
      candidate -= std::log(value/target)*value/derivative;
    }
    if (!std::isfinite(candidate) || candidate <= log_lower
        || candidate >= log_upper) {
      candidate = 0.5*(log_lower+log_upper);
    }
    log_energy = candidate;
  }
  return std::exp(0.5*(log_lower+log_upper));
}

inline Real MaterialEnergyFromTotalEnergy(EosTable *ptable, const Real rho,
                                          const Real total_energy) {
  return MaterialEnergyFromTotal(
      ptable, rho, total_energy, total_energy,
      TotalEnergyFromMaterial, TotalEnergyLogDerivative);
}

inline Real MaterialEnergyFromTotalPressure(EosTable *ptable, const Real rho,
                                            const Real total_pressure) {
  const Real initial_guess =
      MaterialEgasFromRhoP(ptable, rho, total_pressure);
  return MaterialEnergyFromTotal(
      ptable, rho, total_pressure, initial_guess,
      TotalPressureFromMaterial, TotalPressureLogDerivative);
}

struct LteRadiationDerivatives {
  Real total_energy;
  Real total_pressure;
  Real temperature;
  Real sound_speed_squared;
  Real dtemperature_drho_adiabatic;
};

inline LteRadiationDerivatives CoupledDerivatives(
    EosTable *ptable, const Real rho, const Real material_energy) {
  const Real material_pressure =
      MaterialPresFromRhoEg(ptable, rho, material_energy);
  const Real temperature =
      MaterialTempFromRhoEg(ptable, rho, material_energy);
  const Real radiation_energy = RadiationEnergyFromTemperature(temperature);

  Real p_x2, p_x1, t_x2, t_x1;
  Real x2, x1;
  GetEosCoordinates(ptable, kPresOverEgas, material_energy, rho, x2, x1);
  p_x2 = LogTablePartial(ptable, kPresOverEgas, x2, x1, true)+1.0;
  p_x1 = LogTablePartial(ptable, kPresOverEgas, x2, x1, false)-dens_pow;
  GetEosCoordinates(ptable, kTemperature, material_energy, rho, x2, x1);
  t_x2 = LogTablePartial(ptable, kTemperature, x2, x1, true);
  t_x1 = LogTablePartial(ptable, kTemperature, x2, x1, false);

  LteRadiationDerivatives result{};
  result.total_energy = material_energy+radiation_energy;
  result.total_pressure = material_pressure+radiation_energy/3.0;
  result.temperature = temperature;
  if (!std::isfinite(t_x2) || std::abs(t_x2) <= TINY_NUMBER
      || !std::isfinite(t_x1) || !std::isfinite(p_x1)
      || !std::isfinite(p_x2) || !(temperature > 0.0)) {
    result.sound_speed_squared = std::numeric_limits<Real>::quiet_NaN();
    result.dtemperature_drho_adiabatic =
        std::numeric_limits<Real>::quiet_NaN();
    return result;
  }

  const Real chi_rho = p_x1-p_x2*t_x1/t_x2;
  const Real chi_temperature = p_x2/t_x2;
  const Real denergy_drho_temperature = material_energy/rho
      *(-t_x1/t_x2-dens_pow);
  const Real denergy_dtemperature_rho =
      (material_energy/t_x2+4.0*radiation_energy)/temperature;
  const Real dpressure_drho_temperature =
      material_pressure/rho*chi_rho;
  const Real dpressure_dtemperature_rho =
      (material_pressure*chi_temperature
       +4.0*radiation_energy/3.0)/temperature;
  result.dtemperature_drho_adiabatic =
      ((result.total_energy+result.total_pressure)/rho
       -denergy_drho_temperature)/denergy_dtemperature_rho;
  result.sound_speed_squared = dpressure_drho_temperature
      +dpressure_dtemperature_rho*result.dtemperature_drho_adiabatic;
  return result;
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
  if (!include_lte_radiation)
    return MaterialPresFromRhoEg(ptable, rho, egas);
  const Real material_energy =
      MaterialEnergyFromTotalEnergy(ptable, rho, egas);
  return TotalPressureFromMaterial(ptable, rho, material_energy);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return interpolated internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  if (!include_lte_radiation)
    return MaterialEgasFromRhoP(ptable, rho, pres);
  const Real material_energy =
      MaterialEnergyFromTotalPressure(ptable, rho, pres);
  return TotalEnergyFromMaterial(ptable, rho, material_energy);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return interpolated adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  if (!include_lte_radiation)
    return MaterialAsqFromRhoP(ptable, rho, pres);
  const Real material_energy =
      MaterialEnergyFromTotalPressure(ptable, rho, pres);
  return CoupledDerivatives(ptable, rho, material_energy).sound_speed_squared;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::TempFromRhoEg(Real rho, Real egas)
//! \brief Return interpolated gas temperature
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  const Real material_energy = include_lte_radiation
      ? MaterialEnergyFromTotalEnergy(ptable, rho, egas) : egas;
  Real temperature_phys =
      MaterialTempFromRhoEg(ptable, rho, material_energy)*ptable->tUnit;
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
//! \fn Real EquationOfState::EgasFromRhoT(Real rho, Real temperature)
//! \brief Invert the table's monotone T(rho,egas) interpolant at fixed density.
Real EquationOfState::EgasFromRhoT(Real rho, Real temperature) {
  if (ptable == nullptr || !std::isfinite(rho) || !std::isfinite(temperature)
      || rho <= 0.0 || temperature <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real x1 = std::log10(rho*ptable->rhoUnit);
  if (x1 < ptable->logRhoMin || x1 > ptable->logRhoMax) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real target = std::log10(temperature*ptable->tUnit);
  Real lower = ptable->logEgasMin;
  Real upper = ptable->logEgasMax;
  const Real t_lower = ptable->table.interpolate(kTemperature, lower, x1);
  const Real t_upper = ptable->table.interpolate(kTemperature, upper, x1);
  if (target < t_lower || target > t_upper) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  for (int iteration=0; iteration<64; ++iteration) {
    const Real middle = 0.5*(lower+upper);
    const Real value = ptable->table.interpolate(kTemperature, middle, x1);
    if (value < target) lower = middle;
    else upper = middle;
  }
  const Real x2 = 0.5*(lower+upper);
  const Real log_egas = x2-dens_pow*x1
      -std::log10(ptable->EosRatios(kTemperature)*ptable->eUnit);
  const Real material_energy = std::pow(10.0, log_egas);
  return include_lte_radiation
      ? material_energy+RadiationEnergyFromTemperature(temperature)
      : material_energy;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas)
//! \brief Evaluate d ln(T) / d ln(egas) for the table interpolant at fixed density
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  // The table stores log10(T) on a uniform log10(egas/rho)-like coordinate.
  // At fixed rho, d x2 == d log10(egas), so the exact derivative of the
  // piecewise-bilinear interpolant is also d ln(T)/d ln(egas).  A finite
  // centered difference can straddle an interpolation knot and average two
  // slopes even though the nonlinear residual uses only the current cell's
  // slope, making the Newton Jacobian inconsistent in stiff coupling zones.
  const Real material_energy = include_lte_radiation
      ? MaterialEnergyFromTotalEnergy(ptable, rho, egas) : egas;
  Real x1, x2;
  GetEosCoordinates(ptable, kTemperature,
                    std::max(material_energy, TINY_NUMBER), rho,
                    x2, x1);
  const Real derivative =
      ptable->table.interpolateX2Derivative(kTemperature, x2, x1);
  if (!std::isfinite(derivative)) return 0.0;
  if (!include_lte_radiation) return derivative;
  const Real temperature =
      MaterialTempFromRhoEg(ptable, rho, material_energy);
  const Real radiation_energy = RadiationEnergyFromTemperature(temperature);
  const Real denominator = material_energy/derivative+4.0*radiation_energy;
  return denominator > 0.0 ? egas/denominator : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres)
//! \brief Return (d ln T / d ln P)_s from the tabulated thermodynamic closure
Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres) {
  if (ptable == nullptr || !std::isfinite(rho) || !std::isfinite(pres)
      || rho <= 0.0 || pres <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }

  const Real total_or_material_energy = EgasFromRhoP(rho, pres);
  if (!std::isfinite(total_or_material_energy)
      || total_or_material_energy <= 0.0)
    return std::numeric_limits<Real>::quiet_NaN();
  const Real egas = include_lte_radiation
      ? MaterialEnergyFromTotalEnergy(ptable, rho, total_or_material_energy)
      : total_or_material_energy;

  if (include_lte_radiation) {
    const LteRadiationDerivatives derivatives =
        CoupledDerivatives(ptable, rho, egas);
    const Real nabla_ad = pres/derivatives.temperature
        *derivatives.dtemperature_drho_adiabatic
        /derivatives.sound_speed_squared;
    return std::isfinite(nabla_ad) && nabla_ad > 0.0
        ? nabla_ad : std::numeric_limits<Real>::quiet_NaN();
  }

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
  const Real material_energy = include_lte_radiation
      ? MaterialEnergyFromTotalEnergy(ptable, rho, egas) : egas;
  const Real material_entropy =
      GetEntropyData(ptable, kEntropyRhoEg, material_energy, rho);
  if (!include_lte_radiation) return material_entropy;
  const Real temperature =
      MaterialTempFromRhoEg(ptable, rho, material_energy)*ptable->tUnit;
  const Real density_cgs = rho*ptable->rhoUnit;
  const Real radiation_entropy = 4.0*kRadiationConstantCgs
      *std::pow(temperature, 3)/(3.0*density_cgs);
  return material_entropy+radiation_entropy;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EntropyFromRhoP(Real rho, Real pres)
//! \brief Return specific entropy from the inverse (rho, P/rho) field.
Real EquationOfState::EntropyFromRhoP(Real rho, Real pres) {
  if (!HasEntropyTable()) return std::numeric_limits<Real>::quiet_NaN();
  if (!include_lte_radiation)
    return GetEntropyData(ptable, kEntropyRhoP, pres, rho);
  const Real total_energy = EgasFromRhoP(rho, pres);
  return EntropyFromRhoEg(rho, total_energy);
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

  if (include_lte_radiation) {
    Real lower, upper;
    if (!MaterialEnergyBounds(ptable, rho, &lower, &upper))
      return std::numeric_limits<Real>::quiet_NaN();
    auto total_entropy = [&](const Real material_energy) {
      const Real total_energy =
          TotalEnergyFromMaterial(ptable, rho, material_energy);
      return EntropyFromRhoEg(rho, total_energy);
    };
    const Real lower_entropy = total_entropy(lower);
    const Real upper_entropy = total_entropy(upper);
    if (!std::isfinite(lower_entropy) || !std::isfinite(upper_entropy)
        || entropy < lower_entropy || entropy > upper_entropy)
      return std::numeric_limits<Real>::quiet_NaN();
    for (int iteration = 0; iteration < 64; ++iteration) {
      const Real middle = std::sqrt(lower*upper);
      if (total_entropy(middle) < entropy) lower = middle;
      else upper = middle;
    }
    return TotalPressureFromMaterial(ptable, rho, std::sqrt(lower*upper));
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

  if (include_lte_radiation) {
    auto residual_total = [&](const Real logrho) {
      const Real value = EntropyFromRhoP(std::exp(logrho), pres);
      return std::isfinite(value) && value > 0.0
          ? std::log(value)-target
          : std::numeric_limits<Real>::quiet_NaN();
    };
    Real lower_log = log_lo;
    Real lower_residual = residual_total(lower_log);
    for (int n = 1; n <= nscan; ++n) {
      const Real upper_log = log_lo+(log_hi-log_lo)*n/nscan;
      const Real upper_residual = residual_total(upper_log);
      if (std::isfinite(lower_residual) && std::isfinite(upper_residual)
          && lower_residual*upper_residual <= 0.0) {
        Real a = lower_log;
        Real b = upper_log;
        Real fa = lower_residual;
        for (int iteration = 0; iteration < 64; ++iteration) {
          const Real middle = 0.5*(a+b);
          const Real fm = residual_total(middle);
          if (!std::isfinite(fm))
            return std::numeric_limits<Real>::quiet_NaN();
          if (fa*fm <= 0.0) b = middle;
          else {
            a = middle;
            fa = fm;
          }
        }
        return std::exp(0.5*(a+b));
      }
      lower_log = upper_log;
      lower_residual = upper_residual;
    }
    return std::numeric_limits<Real>::quiet_NaN();
  }

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
  include_lte_radiation = pin->GetOrAddBoolean(
      "hydro", "eos_include_lte_radiation", false);
  lte_inversion_max_iterations = pin->GetOrAddInteger(
      "hydro", "eos_lte_inversion_max_iterations", 12);
  if (lte_inversion_max_iterations < 4
      || lte_inversion_max_iterations > 128) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
        << "hydro/eos_lte_inversion_max_iterations must lie in [4,128].";
    ATHENA_ERROR(msg);
  }
#if NRMGFLD_ENABLED
  if (include_lte_radiation) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
        << "hydro/eos_include_lte_radiation=true cannot be combined with "
        << "NR-FLD because that would count a*T^4 twice." << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
  radiation_constant_code = kRadiationConstantCgs
      *std::pow(ptable->tUnit, 4)/ptable->eUnit;
  if (!std::isfinite(radiation_constant_code)
      || radiation_constant_code <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
        << "invalid LTE radiation constant in code units: "
        << radiation_constant_code << std::endl;
    ATHENA_ERROR(msg);
  }
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
