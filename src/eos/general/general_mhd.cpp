//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file general_mhd.cpp
//! \brief implements most but not all of the functions in class
//! EquationOfState for general EOS MHD
//!
//! These functions MUST be implemented in an additional file.
//!
//! Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! Real EquationOfState::AsqFromRhoP(Real rho, Real pres)


// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <limits>
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../field/field.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../eos.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) :
  ptable{pmb->pmy_mesh->peos_table},
  pmy_block_{pmb},
  gamma_{pin->GetOrAddReal("hydro", "gamma", 2.)},
  density_floor_{pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*float_min))},
  scalar_floor_{pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*float_min))},
  rho_unit_{pin->GetOrAddReal("hydro", "eos_rho_unit", 1.0)},
  inv_rho_unit_{1.0/rho_unit_},
  egas_unit_{pin->GetOrAddReal("hydro", "eos_egas_unit", 1.0)},
  inv_egas_unit_{1.0/egas_unit_},
  vsqr_unit_{egas_unit_/rho_unit_},
  inv_vsqr_unit_{1.0/vsqr_unit_}
  {
  if (pin->DoesParameterExist("hydro", "efloor")) {
    energy_floor_ = pin->GetReal("hydro", "efloor");
    pressure_floor_ = energy_floor_*(pin->GetOrAddReal("hydro", "gamma", 2.) - 1.);
    pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", pressure_floor_);
  } else {
    pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*float_min));
    energy_floor_ = pressure_floor_/(pin->GetOrAddReal("hydro", "gamma", 2.) - 1.);
    pin->SetReal("hydro", "efloor", energy_floor_);
  }
  if (EOS_TABLE_ENABLED) {
    if (!ptable) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::EquationOfState" << std::endl
          << "EOS table data uninitialized. Should be initialized by Mesh." << std::endl;
      ATHENA_ERROR(msg);
    }
    temperature_floor_ = pin->GetOrAddReal("hydro", "tfloor", 0.0);
    if (!std::isfinite(temperature_floor_) || temperature_floor_ < 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::EquationOfState" << std::endl
          << "hydro/tfloor must be finite and nonnegative" << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  InitEosConstants(pin);
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//!           const AthenaArray<Real> &prim_old, const FaceField &b,
//!           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il,int iu, int jl,int ju, int kl,int ku) {

  pmy_block_->pfield->CalculateCellCenteredField(b,bcc,pco,il,iu,jl,ju,kl,ku);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        Real& w_d  = prim(IDN,k,j,i);
        Real& w_vx = prim(IVX,k,j,i);
        Real& w_vy = prim(IVY,k,j,i);
        Real& w_vz = prim(IVZ,k,j,i);
        Real& w_p  = prim(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        const Real& bcc1 = bcc(IB1,k,j,i);
        const Real& bcc2 = bcc(IB2,k,j,i);
        const Real& bcc3 = bcc(IB3,k,j,i);

        Real pb = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
        Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));

        // Apply the energy floor before evaluating the table.  A fixed code-
        // unit floor may be smaller than an ULP of ke+pb, in which case the
        // total-energy reconstruction can still leave zero gas energy.
        const Real gas_before_floor = u_e - ke - pb;
        if (!(gas_before_floor > energy_floor_)) {
          Real floor_egas = energy_floor_;
#if EOS_TABLE_ENABLED
          if (temperature_floor_ > 0.0) {
            floor_egas = EgasFromRhoT(u_d, temperature_floor_);
            if (!std::isfinite(floor_egas) || floor_egas <= 0.0) {
              std::stringstream msg;
              msg << "### FATAL ERROR in EquationOfState::ConservedToPrimitive"
                  << std::endl
                  << "hydro/tfloor lies outside the EOS table at rho=" << u_d
                  << ", Tfloor=" << temperature_floor_ << std::endl;
              ATHENA_ERROR(msg);
            }
          }
#endif
          u_e = floor_egas + ke + pb;
        }
#if EOS_TABLE_ENABLED
        if (temperature_floor_ > 0.0) {
          const Real egas = u_e - ke - pb;
          const Real roundoff_guard = 8.0*std::numeric_limits<Real>::epsilon()
                                    *(std::abs(ke) + std::abs(pb) + 1.0);
          const Real temperature = TempFromRhoEg(u_d, egas);
          if (!std::isfinite(egas) || egas <= energy_floor_ + roundoff_guard
              || !std::isfinite(temperature) || temperature < temperature_floor_) {
            const Real floor_egas = EgasFromRhoT(u_d, temperature_floor_);
            if (!std::isfinite(floor_egas) || floor_egas <= 0.0) {
              std::stringstream msg;
              msg << "### FATAL ERROR in EquationOfState::ConservedToPrimitive"
                  << std::endl
                  << "hydro/tfloor lies outside the EOS table at rho=" << u_d
                  << ", Tfloor=" << temperature_floor_ << std::endl;
              ATHENA_ERROR(msg);
            }
            u_e = floor_egas + ke + pb;
          }
        }
#endif
        w_p = PresFromRhoEg(u_d, u_e - ke - pb);
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//!           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bc,
    AthenaArray<Real> &cons, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  // Force outer-loop vectorization
#pragma omp simd
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      //#pragma omp simd
#pragma novector
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        const Real& w_d  = prim(IDN,k,j,i);
        const Real& w_vx = prim(IVX,k,j,i);
        const Real& w_vy = prim(IVY,k,j,i);
        const Real& w_vz = prim(IVZ,k,j,i);
        const Real& w_p  = prim(IPR,k,j,i);

        const Real& bcc1 = bc(IB1,k,j,i);
        const Real& bcc2 = bc(IB2,k,j,i);
        const Real& bcc3 = bc(IB3,k,j,i);

        u_d = w_d;
        u_m1 = w_vx*w_d;
        u_m2 = w_vy*w_d;
        u_m3 = w_vz*w_d;
        // cellwise conversion
        u_e = EgasFromRhoP(u_d, w_p) + 0.5*(w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz))
                                     + (SQR(bcc1) + SQR(bcc2) + SQR(bcc3)));
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
//! \brief returns adiabatic sound speed given vector of primitive variables

Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]) {
  return std::sqrt(AsqFromRhoP(prim[IDN], prim[IPR]));
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::FastMagnetosonicSpeed(const Real prim[], const Real bx)
//! \brief returns fast magnetosonic speed given vector of primitive variables
//! Note the formula for (C_f)^2 is positive definite, so this func never returns a NaN
Real EquationOfState::FastMagnetosonicSpeed(const Real prim[(NWAVE)], const Real bx) {
  Real asq = AsqFromRhoP(prim[IDN], prim[IPR]) * prim[IDN]; // Actually rho*asq
  Real vaxsq = bx*bx;
  Real ct2 = (prim[IBY]*prim[IBY] + prim[IBZ]*prim[IBZ]);
  Real qsq = vaxsq + ct2 + asq;
  Real tmp = vaxsq + ct2 - asq;
  return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/prim[IDN]);
}

//---------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j,
//!                                                 int i)
//! \brief Apply density and pressure floors to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);

  // apply density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveConservedFloors(AthenaArray<Real> &prim,
//!           AthenaArray<Real> &cons, FaceField &b, int k, int j, int i) {
//! \brief Apply pressure (prim) floor and correct energy (cons) (typically after W(U))
void EquationOfState::ApplyPrimitiveConservedFloors(
    AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
    int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);
  Real& w_p  = prim(IPR,k,j,i);

  Real& u_d  = cons(IDN,k,j,i);
  Real& u_e  = cons(IEN,k,j,i);
  const Real& bcc1 = bcc(IB1,k,j,i);
  const Real& bcc2 = bcc(IB2,k,j,i);
  const Real& bcc3 = bcc(IB3,k,j,i);
  // apply (prim) density floor, without changing momentum or energy
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // ensure cons density matches
  u_d = w_d;

  Real pb = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
  Real e_k = 0.5*w_d*(SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)));
  // apply pressure floor, correct total energy
  u_e = (w_p > energy_floor_) ? u_e : energy_floor_ + e_k + pb;
  w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;

  return;
}

Real EquationOfState::GetGamma() {
  std::stringstream msg;
  msg << "GetGamma is not defined for general EOS." << std::endl;
  ATHENA_ERROR(msg);
}
