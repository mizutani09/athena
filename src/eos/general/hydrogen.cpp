//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file hydrogen.cpp
//! \brief implements functions in class EquationOfState for simple hydrogen EOS
//======================================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <limits>   // std::numeric_limits<float>::epsilon()
#include <sstream>
#include <stdexcept> // std::invalid_argument

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../parameter_input.hpp"
#include "../eos.hpp"

namespace {
const Real float_eps = std::numeric_limits<float>::epsilon();
const Real float_1pe = 1.0 + float_eps;
Real prec = 1e-12;

//----------------------------------------------------------------------------------------
//! \fn Real x_(Real rho, Real T) {
//! \brief compute ionization fraction
Real x_(Real rho, Real T) {
  return 2./(1.+std::sqrt(1.+4.*std::exp(1./T-1.5*std::log(T)+std::log(rho))));
}

//----------------------------------------------------------------------------------------
//! \fn Real P_of_rho_T(Real rho, Real T) {
//! \brief compute gas pressure
Real P_of_rho_T(Real rho, Real T) {
  return rho * T * (1. + x_(rho, T));
}

//----------------------------------------------------------------------------------------
//! \fn Real e_of_rho_T(Real rho, Real T) {
//! \brief compute internal energy density
Real e_of_rho_T(Real rho, Real T) {
  Real x = x_(rho, T);
  return x * rho + 1.5 * rho * T * (1. + x);
}

//----------------------------------------------------------------------------------------
//! \fn Real asq_(Real rho, Real T) {
//! \brief compute adiabatic sound speed squared
Real asq_(Real rho, Real T) {
  Real x = x_(rho, T);
  Real c = 2 * SQR(T) * (2. + x - SQR(x));
  return (1.+x)*T*(1+(2*c + 2*T*(4 + 3*T)*(1 - x)*x)/(3*c + SQR(2 + 3*T)*(1 - x)*x));
}

//----------------------------------------------------------------------------------------
//! \fn Real invert(Real(*f) (Real, Real), Real rho, Real sol, Real T0, Real T1)
//! \brief Invert an EOS function at a given density and return temperature.
//!        Uses Brent–Dekker method for inversion.
Real invert(Real(*f) (Real, Real), Real rho, Real sol, Real Ta, Real Tb) {
  Real fc, fs, Tc, Td, Ts;
  bool mflag = false;
  Real fa = f(rho, Ta) / sol - 1.;
  Real fb = f(rho, Tb) / sol - 1.;

  if (std::abs(fa) < std::abs(fb)) {
    Ts = Ta;
    Ta = Tb;
    Tb = Ts;
    fs = fa;
    fa = fb;
    fb = fs;
  }

  if (fa * fb > 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState inversion"
        << std::endl << "Root not bracketed" << std::endl;
    ATHENA_ERROR(msg);
  }

  Tc = Ta;
  while ((std::abs(Ta - Tb) >= prec) && (std::abs(fb) >= prec)) {
    fc = f(rho, Tc) / sol - 1.0;
    if ((fc != fa) && (fc != fb)) {
      Ts = Ta * fb * fc / ((fa - fb) * (fa - fc))
         + Tb * fa * fc / ((fb - fa) * (fb - fc))
         + Tc * fa * fb / ((fc - fa) * (fc - fb));
    } else {
      Ts = Tb - fb * (Tb - Ta) / (fb - fa);
    }
    if ( ((Ts < .25 * (3 * Ta - Tb)) || (Ts > Tb)) ||
         (mflag && std::abs(Ts - Tb) >= 0.5 * std::abs(Tc - Tb)) ||
         (!mflag && std::abs(Ts - Tb) >= 0.5 * std::abs(Tc - Td)) ||
         (mflag && std::abs(Tc - Tb) < prec) ||
         (!mflag && std::abs(Tc - Td) < prec) ) {
      Ts = .5 * (Ta + Tb);
      mflag = true;
    } else {
      mflag = false;
    }
    fs = f(rho, Ts) / sol - 1.0;
    Td = Tc;
    Tc = Tb;
    if (fa * fs < 0) {
      Tb = Ts;
      fb = fs;
    } else {
      Ta = Ts;
      fa = fs;
    }

    if (std::abs(fa) < std::abs(fb)) {
      Ts = Ta;
      Ta = Tb;
      Tb = Ts;
      fs = fa;
      fa = fb;
      fb = fs;
    }
  }
  return Tb;
}
} // namespace

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  rho *= rho_unit_;
  egas *= egas_unit_;
  Real es = egas / rho;
  Real T = invert(*e_of_rho_T, rho, egas, std::max(es - 1.0, 0.1*es)/3.0,
                  float_1pe*2.0*es/3.0);
  return P_of_rho_T(rho, T) * inv_egas_unit_;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  rho *= rho_unit_;
  pres *= egas_unit_;
  Real ps = pres / rho;
  Real T = invert(*P_of_rho_T, rho, pres, 0.5*ps, float_1pe*ps);
  return e_of_rho_T(rho, T) * inv_egas_unit_;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  rho *= rho_unit_;
  pres *= egas_unit_;
  Real ps = pres / rho;
  Real T = invert(*P_of_rho_T, rho, pres, 0.5*ps, float_1pe*ps);
  return asq_(rho, T) * inv_vsqr_unit_;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::TempFromRhoEg(Real rho, Real egas)
//! \brief Return gas temperature
Real EquationOfState::TempFromRhoEg(Real rho, Real egas) {
  rho *= rho_unit_;
  egas *= egas_unit_;
  Real es = egas / rho;
  return invert(*e_of_rho_T, rho, egas, std::max(es - 1.0, 0.1*es)/3.0,
                float_1pe*2.0*es/3.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas)
//! \brief Return d ln(T) / d ln(egas) at constant density using a local log-derivative
Real EquationOfState::DlnTDlnEgasFromRhoEg(Real rho, Real egas) {
  Real T = TempFromRhoEg(rho, egas);
  Real deps = 1.0e-6*std::max(std::abs(egas), TINY_NUMBER);
  Real Tp = TempFromRhoEg(rho, egas + deps);
  return (std::log(Tp) - std::log(T))/std::log((egas + deps)/egas);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres)
//! \brief Return (d ln T / d ln P)_s for the analytic hydrogen EOS
Real EquationOfState::NablaAdFromRhoP(Real rho, Real pres) {
  if (!std::isfinite(rho) || !std::isfinite(pres)
      || rho <= 0.0 || pres <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real rho_phys = rho*rho_unit_;
  const Real pres_phys = pres*egas_unit_;
  const Real ps = pres_phys/rho_phys;
  const Real temperature = invert(*P_of_rho_T, rho_phys, pres_phys,
                                  0.5*ps, float_1pe*ps);
  constexpr Real log_step = 1.0e-5;
  const Real fac = std::exp(log_step);
  const Real chi_rho = (std::log(P_of_rho_T(rho_phys*fac, temperature))
                       -std::log(P_of_rho_T(rho_phys/fac, temperature)))
                      /(2.0*log_step);
  const Real chi_t = (std::log(P_of_rho_T(rho_phys, temperature*fac))
                     -std::log(P_of_rho_T(rho_phys, temperature/fac)))
                    /(2.0*log_step);
  const Real gamma1 = asq_(rho_phys, temperature)*rho_phys/pres_phys;
  // Gamma1 = chi_rho + chi_T (Gamma3 - 1), while
  // nabla_ad = (Gamma3 - 1)/Gamma1.
  if (!std::isfinite(gamma1) || !std::isfinite(chi_rho)
      || !std::isfinite(chi_t) || gamma1 <= 0.0
      || std::abs(chi_t) <= TINY_NUMBER) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const Real nabla_ad = (gamma1 - chi_rho)/(gamma1*chi_t);
  return (std::isfinite(nabla_ad) && nabla_ad > 0.0)
      ? nabla_ad : std::numeric_limits<Real>::quiet_NaN();
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput* pin) {
  prec = pin->GetOrAddReal("hydro", "InversionPrecision", prec);
  return;
}
