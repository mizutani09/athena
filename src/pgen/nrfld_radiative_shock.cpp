//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/
//! \file nrfld_radiative_shock.cpp
//! \brief Zhang et al. (2011) section 6.6 non-equilibrium radiative shocks.

// C++ headers
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../fld/fld.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
constexpr Real kRadiationConst = 7.5657e-15;  // erg cm^-3 K^-4
constexpr Real kGasConst = 8.31451e7;         // erg mol^-1 K^-1
constexpr Real kLightSpeed = 2.99792458e10;   // cm s^-1
constexpr Real kNoShockPosition = 1.0e99;

struct ShockState {
  Real rho;
  Real temp;
  Real vel;
  Real pres;
  Real egas;
  Real erad;
};

Real rho_unit, egas_unit, leng_unit, time_unit, vel_unit, temp_unit;
Real mu, kappa_p_code, chi_r_code;
Real x_discont_phys;
Real amr_curvature_threshold, amr_curvature_eps, amr_derefine_threshold;
Real shock_grad_threshold;
bool use_amr_refinement;
ShockState left_state, right_state;

enum UserRealData {
  PREV_PROFILE = 0,
  CYCLE_REL_DIFF = 1,
  NUSER_REAL_DATA = 2
};

enum PrevProfileIndex {
  PREV_RHO = 0,
  PREV_TGAS = 1,
  PREV_TRAD = 2,
  NPREV_PROFILE = 3
};

enum CycleRelDiffIndex {
  REL_RHO = 0,
  REL_TGAS = 1,
  REL_TRAD = 2,
  REL_MAX = 3,
  NREL_DIFF = 4
};

int RefinementCondition(MeshBlock *pmb);
Real LocalGasTemperaturePhys(const MeshBlock *pmb, int k, int j, int i);
Real LocalRadiationTemperaturePhys(const MeshBlock *pmb, int k, int j, int i);
Real HistoryCycleRelMax(MeshBlock *pmb, int iout);
Real HistoryCycleRelRho(MeshBlock *pmb, int iout);
Real HistoryCycleRelTgas(MeshBlock *pmb, int iout);
Real HistoryCycleRelTrad(MeshBlock *pmb, int iout);
Real HistoryShockPosition(MeshBlock *pmb, int iout);
Real HistoryMassFluxResidual(MeshBlock *pmb, int iout);
Real HistoryMomentumFluxResidual(MeshBlock *pmb, int iout);
Real HistoryEnergyFluxResidual(MeshBlock *pmb, int iout);
Real HistoryTgMax(MeshBlock *pmb, int iout);
Real HistoryTrMax(MeshBlock *pmb, int iout);
Real HistoryRhoMin(MeshBlock *pmb, int iout);
Real HistoryRhoMax(MeshBlock *pmb, int iout);
Real HistoryErMax(MeshBlock *pmb, int iout);
Real HistoryRtime(MeshBlock *pmb, int iout);
Real HistoryEall(MeshBlock *pmb, int iout);

ShockState MakeState(Real rho_phys, Real temp_phys, Real vel_phys, Real gamma) {
  ShockState s;
  s.rho = rho_phys/rho_unit;
  s.temp = temp_phys/temp_unit;
  s.vel = vel_phys/vel_unit;
  s.pres = (rho_phys*kGasConst*temp_phys/mu)/egas_unit;
  s.egas = s.pres/(gamma - 1.0);
  s.erad = (kRadiationConst*std::pow(temp_phys, 4))/egas_unit;
  return s;
}

Real LocalGasTemperaturePhys(const MeshBlock *pmb, int k, int j, int i) {
  const Real rho = std::max(pmb->phydro->w(IDN, k, j, i), TINY_NUMBER);
  return pmb->phydro->w(IPR, k, j, i)/rho*temp_unit;
}

Real LocalRadiationTemperaturePhys(const MeshBlock *pmb, int k, int j, int i) {
  const Real erad_phys = std::max(pmb->prfld2->u_rad(k, j, i)*egas_unit, TINY_NUMBER);
  return std::pow(erad_phys/kRadiationConst, 0.25);
}

Real NormalizedSecondDerivative(Real um, Real u0, Real up) {
  const Real numerator = std::abs(up - 2.0*u0 + um);
  const Real denominator = std::abs(up - u0) + std::abs(u0 - um)
      + amr_curvature_eps*(std::abs(up) + 2.0*std::abs(u0) + std::abs(um));
  return numerator/std::max(denominator, TINY_NUMBER);
}

void ConstantInverseLengthOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
                                  AthenaArray<Real> &prim) {
  (void)u_fld;
  (void)prim;
  FLD2 *prfld = pmb->prfld2;
  int kl = pmb->ks;
  int ku = pmb->ke;
  int jl = pmb->js;
  int ju = pmb->je;
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        prfld->sigma_p(k, j, i) = kappa_p_code;
        prfld->sigma_r(k, j, i) = std::max(chi_r_code, TINY_NUMBER);
      }
    }
  }
}

void SetNRFixedX1(AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas, bool inner,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  const ShockState &s = inner ? left_state : right_state;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int n = 1; n <= ngh; ++n) {
        const int i = inner ? is - n : ie + n;
        u_rad(k, j, i) = s.erad;
        u_gas(k, j, i) = s.egas;
      }
    }
  }
}

void SetFLDFixedX1(AthenaArray<Real> &u_rad, bool inner,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {
  const ShockState &s = inner ? left_state : right_state;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int n = 1; n <= ngh; ++n) {
        const int i = inner ? is - n : ie + n;
        u_rad(k, j, i) = s.erad;
      }
    }
  }
}

void SetHydroFixedX1(AthenaArray<Real> &prim, bool inner,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  const ShockState &s = inner ? left_state : right_state;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int n = 1; n <= ngh; ++n) {
        const int i = inner ? is - n : ie + n;
        prim(IDN, k, j, i) = s.rho;
        prim(IVX, k, j, i) = s.vel;
        prim(IVY, k, j, i) = 0.0;
        prim(IVZ, k, j, i) = 0.0;
        prim(IPR, k, j, i) = s.pres;
      }
    }
  }
}
}  // namespace

void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
                              const AthenaArray<Real> &prim,
                              const AthenaArray<Real> &prim_scalar,
                              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                              AthenaArray<Real> &cons_scalar);

void NRInnerX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)w; (void)time; (void)dt; (void)ie;
  SetNRFixedX1(u_rad, u_gas, true, is, ie, js, je, ks, ke, ngh);
}

void NROuterX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)w; (void)time; (void)dt; (void)is;
  SetNRFixedX1(u_rad, u_gas, false, is, ie, js, je, ks, ke, ngh);
}

void FLDInnerX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)pfld; (void)w; (void)time; (void)dt; (void)ie;
  SetFLDFixedX1(u_rad_fld, true, is, ie, js, je, ks, ke, ngh);
}

void FLDOuterX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)pfld; (void)w; (void)time; (void)dt; (void)is;
  SetFLDFixedX1(u_rad_fld, false, is, ie, js, je, ks, ke, ngh);
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)b; (void)time; (void)dt; (void)ie;
  SetHydroFixedX1(prim, true, is, ie, js, je, ks, ke, ngh);
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)b; (void)time; (void)dt; (void)is;
  SetHydroFixedX1(prim, false, is, ie, js, je, ks, ke, ngh);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!pin->GetBoolean("fld", "is_couple")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "is_couple must be true for nrfld_radiative_shock.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "only_rad must be false for full radiation hydrodynamics.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "cut_diff")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "cut_diff must be false for this diffusion shock.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "cut_Pnablav")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "cut_Pnablav must be false so radiation pressure work is retained.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "fixed_flux_limitter")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "fixed_flux_limitter must be true to force lambda=1/3.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetOrAddBoolean("problem", "force_lambda_one_third", true)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "force_lambda_one_third must be true for the Lowrie-Edwards comparison.";
    ATHENA_ERROR(msg);
  }

  rho_unit = pin->GetReal("hydro", "rho_unit");
  egas_unit = pin->GetReal("hydro", "egas_unit");
  time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }
  vel_unit = std::sqrt(egas_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  mu = pin->GetReal("hydro", "mu");
  temp_unit = egas_unit/rho_unit*mu/kGasConst;

  std::string problem_type = pin->GetOrAddString("problem", "problem_type", "mach2");
  if (problem_type != "mach2" && problem_type != "mach5") {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "problem_type must be 'mach2' or 'mach5'.";
    ATHENA_ERROR(msg);
  }

  Real rho_l_default, temp_l_default, vel_l_default;
  Real rho_r_default, temp_r_default, vel_r_default;
  if (problem_type == "mach2") {
    rho_l_default = 5.45887e-13;
    temp_l_default = 100.0;
    vel_l_default = 2.35435e5;
    rho_r_default = 1.24794e-12;
    temp_r_default = 207.757;
    vel_r_default = 1.02987e5;
  } else {
    rho_l_default = 5.45887e-13;
    temp_l_default = 100.0;
    vel_l_default = 5.88588e5;
    rho_r_default = 1.96405e-12;
    temp_r_default = 855.720;
    vel_r_default = 1.63592e5;
  }

  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real rho_l = pin->GetOrAddReal("problem", "rho_L", rho_l_default);
  const Real temp_l = pin->GetOrAddReal("problem", "T_L", temp_l_default);
  const Real vel_l = pin->GetOrAddReal("problem", "u_L", vel_l_default);
  const Real rho_r = pin->GetOrAddReal("problem", "rho_R", rho_r_default);
  const Real temp_r = pin->GetOrAddReal("problem", "T_R", temp_r_default);
  const Real vel_r = pin->GetOrAddReal("problem", "u_R", vel_r_default);
  left_state = MakeState(rho_l, temp_l, vel_l, gamma);
  right_state = MakeState(rho_r, temp_r, vel_r, gamma);

  x_discont_phys = pin->GetOrAddReal("problem", "x_discont", 0.0);
  const Real kappa_p_phys = pin->GetOrAddReal("problem", "kappa_P", 3.92664e-5);
  const Real chi_r_phys = pin->GetOrAddReal("problem", "chi_R", 0.848902);
  kappa_p_code = kappa_p_phys*leng_unit;
  chi_r_code = chi_r_phys*leng_unit;

  use_amr_refinement = pin->GetOrAddBoolean("problem", "use_amr_refinement", false);
  amr_curvature_threshold = pin->GetOrAddReal("problem", "amr_curvature_threshold", 0.8);
  amr_curvature_eps = pin->GetOrAddReal("problem", "amr_curvature_eps", 0.02);
  amr_derefine_threshold = pin->GetOrAddReal("problem", "amr_derefine_threshold",
                                             0.25*amr_curvature_threshold);
  shock_grad_threshold = pin->GetOrAddReal("problem", "shock_grad_threshold", 0.05);
  if (use_amr_refinement) {
    EnrollUserRefinementCondition(RefinementCondition);
  }

  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDInnerX1);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  EnrollUserExplicitSourceFunction(AddRadiativeForceAndWork);

  AllocateUserHistoryOutput(15);
  EnrollUserHistoryOutput(0, HistoryTgMax, "Tgas_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTrMax, "Trad_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryRhoMin, "rho_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(3, HistoryRhoMax, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryErMax, "Erad_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, HistoryCycleRelMax, "rel_dcycle", UserHistoryOperation::max);
  EnrollUserHistoryOutput(8, HistoryCycleRelRho, "rel_rho", UserHistoryOperation::max);
  EnrollUserHistoryOutput(9, HistoryCycleRelTgas, "rel_Tgas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(10, HistoryCycleRelTrad, "rel_Trad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(11, HistoryShockPosition, "x_shock", UserHistoryOperation::min);
  EnrollUserHistoryOutput(12, HistoryMassFluxResidual, "res_mdot",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(13, HistoryMomentumFluxResidual, "res_mom",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(14, HistoryEnergyFluxResidual, "res_etot",
                          UserHistoryOperation::max);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  (void)pin;
  AllocateRealUserMeshBlockDataField(NUSER_REAL_DATA);
  ruser_meshblock_data[PREV_PROFILE].NewAthenaArray(NPREV_PROFILE, ncells3, ncells2, ncells1);
  ruser_meshblock_data[CYCLE_REL_DIFF].NewAthenaArray(NREL_DIFF);
  AllocateUserOutputVariables(9);
  SetUserOutputVariableName(0, "rho");
  SetUserOutputVariableName(1, "vel_x");
  SetUserOutputVariableName(2, "P_gas");
  SetUserOutputVariableName(3, "T_gas");
  SetUserOutputVariableName(4, "E_rad");
  SetUserOutputVariableName(5, "T_rad");
  SetUserOutputVariableName(6, "P_rad");
  SetUserOutputVariableName(7, "P_tot");
  SetUserOutputVariableName(8, "lambda");
  prfld2->EnrollOpacityFunction(ConstantInverseLengthOpacity);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  (void)pin;
  const int kl = ks - NGHOST;
  const int ku = ke + NGHOST;
  const int jl = js - NGHOST;
  const int ju = je + NGHOST;
  const int il = is - NGHOST;
  const int iu = ie + NGHOST;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const Real x_phys = pcoord->x1v(i)*leng_unit;
        const ShockState &s = (x_phys < x_discont_phys) ? left_state : right_state;
        phydro->u(IDN, k, j, i) = s.rho;
        phydro->u(IM1, k, j, i) = s.rho*s.vel;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = s.egas + 0.5*s.rho*s.vel*s.vel;
        }
        prfld2->u_gas(k, j, i) = s.egas;
        prfld2->u_rad(k, j, i) = s.erad;
      }
    }
  }

  AthenaArray<Real> &prev = ruser_meshblock_data[PREV_PROFILE];
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const Real x_phys = pcoord->x1v(i)*leng_unit;
        const ShockState &s = (x_phys < x_discont_phys) ? left_state : right_state;
        prev(PREV_RHO, k, j, i) = phydro->u(IDN, k, j, i)*rho_unit;
        prev(PREV_TGAS, k, j, i) = s.temp*temp_unit;
        prev(PREV_TRAD, k, j, i) = std::pow(std::max(prfld2->u_rad(k, j, i)*egas_unit,
                                                     TINY_NUMBER)/kRadiationConst, 0.25);
      }
    }
  }
  for (int n = 0; n < NREL_DIFF; ++n) {
    ruser_meshblock_data[CYCLE_REL_DIFF](n) = 0.0;
  }

  if (gid == 0) {
    std::stringstream msg;
    msg << ">>> Zhang et al. (2011) section 6.6 radiative shock <<<" << std::endl;
    msg << "rho_unit       = " << rho_unit << " g cm^-3" << std::endl;
    msg << "egas_unit      = " << egas_unit << " erg cm^-3" << std::endl;
    msg << "leng_unit      = " << leng_unit << " cm" << std::endl;
    msg << "time_unit      = " << time_unit << " s" << std::endl;
    msg << "vel_unit       = " << vel_unit << " cm s^-1" << std::endl;
    msg << "T_unit         = " << temp_unit << " K" << std::endl;
    msg << "x_discont      = " << x_discont_phys << " cm" << std::endl;
    msg << "kappa_P_code   = " << kappa_p_code << std::endl;
    msg << "chi_R_code     = " << chi_r_code << std::endl;
    msg << "left rho,T,u   = " << left_state.rho*rho_unit << ", "
        << left_state.temp*temp_unit << ", " << left_state.vel*vel_unit << std::endl;
    msg << "right rho,T,u  = " << right_state.rho*rho_unit << ", "
        << right_state.temp*temp_unit << ", " << right_state.vel*vel_unit << std::endl;
    std::cout << msg.str();

    std::ofstream ofs("problem_parameters.txt");
    ofs << msg.str();
    ofs.close();
  }
}

void MeshBlock::UserWorkInLoop() {
  AthenaArray<Real> &prev = ruser_meshblock_data[PREV_PROFILE];
  AthenaArray<Real> &rel = ruser_meshblock_data[CYCLE_REL_DIFF];
  Real rel_rho = 0.0;
  Real rel_tgas = 0.0;
  Real rel_trad = 0.0;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real rho = phydro->w(IDN, k, j, i)*rho_unit;
        const Real tgas = LocalGasTemperaturePhys(this, k, j, i);
        const Real trad = LocalRadiationTemperaturePhys(this, k, j, i);
        rel_rho = std::max(rel_rho, std::abs(rho - prev(PREV_RHO, k, j, i))
                           /std::max(std::abs(prev(PREV_RHO, k, j, i)), TINY_NUMBER));
        rel_tgas = std::max(rel_tgas, std::abs(tgas - prev(PREV_TGAS, k, j, i))
                            /std::max(std::abs(prev(PREV_TGAS, k, j, i)), TINY_NUMBER));
        rel_trad = std::max(rel_trad, std::abs(trad - prev(PREV_TRAD, k, j, i))
                            /std::max(std::abs(prev(PREV_TRAD, k, j, i)), TINY_NUMBER));
        prev(PREV_RHO, k, j, i) = rho;
        prev(PREV_TGAS, k, j, i) = tgas;
        prev(PREV_TRAD, k, j, i) = trad;
      }
    }
  }

  rel(REL_RHO) = rel_rho;
  rel(REL_TGAS) = rel_tgas;
  rel(REL_TRAD) = rel_trad;
  rel(REL_MAX) = std::max(rel_rho, std::max(rel_tgas, rel_trad));
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  (void)pin;
  const Real gm1 = peos->GetGamma() - 1.0;
  const int kl = ks - NGHOST;
  const int ku = ke + NGHOST;
  const int jl = js - NGHOST;
  const int ju = je + NGHOST;
  const int il = is - NGHOST;
  const int iu = ie + NGHOST;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const Real rho_code = std::max(phydro->w(IDN, k, j, i), TINY_NUMBER);
        const Real vx_code = phydro->w(IVX, k, j, i);
        const Real pgas_code = phydro->w(IPR, k, j, i);
        const Real egas_code = pgas_code/gm1;
        const Real erad_code = std::max(prfld2->u_rad(k, j, i), TINY_NUMBER);
        const Real rho_phys = rho_code*rho_unit;
        const Real pgas_phys = pgas_code*egas_unit;
        const Real erad_phys = erad_code*egas_unit;
        user_out_var(0, k, j, i) = rho_phys;
        user_out_var(1, k, j, i) = vx_code*vel_unit;
        user_out_var(2, k, j, i) = pgas_phys;
        user_out_var(3, k, j, i) = egas_code/rho_code*gm1*temp_unit;
        user_out_var(4, k, j, i) = erad_phys;
        user_out_var(5, k, j, i) = std::pow(erad_phys/kRadiationConst, 0.25);
        user_out_var(6, k, j, i) = ONE_3RD*erad_phys;
        user_out_var(7, k, j, i) = pgas_phys + ONE_3RD*erad_phys;
        user_out_var(8, k, j, i) = ONE_3RD;
      }
    }
  }
}

void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
                              const AthenaArray<Real> &prim,
                              const AthenaArray<Real> &prim_scalar,
                              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                              AthenaArray<Real> &cons_scalar) {
  (void)time; (void)prim_scalar; (void)bcc; (void)cons_scalar;
  FLD2 *prfld = pmb->prfld2;
  AthenaArray<Real> &erad = prfld->u_rad;

  const int il = pmb->is;
  const int iu = pmb->ie;
  const int jl = pmb->js;
  const int ju = pmb->je;
  const int kl = pmb->ks;
  const int ku = pmb->ke;
  const Real hidx1 = 0.5/pmb->pcoord->dx1f(pmb->is);
  const Real hidx2 = (pmb->block_size.nx2 > 1) ? 0.5/pmb->pcoord->dx2f(pmb->js) : 0.0;
  const Real hidx3 = (pmb->block_size.nx3 > 1) ? 0.5/pmb->pcoord->dx3f(pmb->ks) : 0.0;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const Real dEr1 = hidx1*(erad(k, j, i+1) - erad(k, j, i-1));
        const Real dEr2 = hidx2*(erad(k, j+1, i) - erad(k, j-1, i));
        const Real dEr3 = hidx3*(erad(k+1, j, i) - erad(k-1, j, i));
        cons(IM1, k, j, i) += -ONE_3RD*dt*dEr1;
        cons(IM2, k, j, i) += -ONE_3RD*dt*dEr2;
        cons(IM3, k, j, i) += -ONE_3RD*dt*dEr3;
        const Real nabla_e_dot_v = dEr1*prim(IVX, k, j, i)
                                    + dEr2*prim(IVY, k, j, i)
                                    + dEr3*prim(IVZ, k, j, i);
        cons(IEN, k, j, i) += -ONE_3RD*dt*nabla_e_dot_v;
      }
    }
  }
}

namespace {
enum class FluxDiagnostic {
  mass,
  momentum,
  energy
};

Real RelativeSpread(Real fmin, Real fmax) {
  const Real scale = std::max(0.5*(std::abs(fmax) + std::abs(fmin)), TINY_NUMBER);
  return (fmax - fmin)/scale;
}

Real LocalFluxDiagnostic(MeshBlock *pmb, int k, int j, int i, FluxDiagnostic type) {
  const Real gamma = pmb->peos->GetGamma();
  const Real rho = pmb->phydro->w(IDN, k, j, i)*rho_unit;
  const Real vx = pmb->phydro->w(IVX, k, j, i)*vel_unit;
  const Real pgas = pmb->phydro->w(IPR, k, j, i)*egas_unit;
  const Real erad = pmb->prfld2->u_rad(k, j, i)*egas_unit;

  if (type == FluxDiagnostic::mass) {
    return rho*vx;
  }
  if (type == FluxDiagnostic::momentum) {
    return rho*vx*vx + pgas + ONE_3RD*erad;
  }

  const Real egas = pgas/(gamma - 1.0);
  const Real dErdx = (pmb->prfld2->u_rad(k, j, i+1) - pmb->prfld2->u_rad(k, j, i-1))
                     *egas_unit/(pmb->pcoord->x1v(i+1) - pmb->pcoord->x1v(i-1))
                     /leng_unit;
  const Real rad_flux = -kLightSpeed*ONE_3RD*dErdx/std::max(chi_r_code/leng_unit,
                                                            TINY_NUMBER);
  const Real prad = ONE_3RD*erad;
  return vx*(egas + 0.5*rho*vx*vx + pgas) + vx*(erad + prad) + rad_flux;
}

Real FluxResidual(MeshBlock *pmb, FluxDiagnostic type) {
  Real fmin = std::numeric_limits<Real>::max();
  Real fmax = -std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real flux = LocalFluxDiagnostic(pmb, k, j, i, type);
        fmin = std::min(fmin, flux);
        fmax = std::max(fmax, flux);
      }
    }
  }
  return RelativeSpread(fmin, fmax);
}

Real HistoryCycleRelMax(MeshBlock *pmb, int iout) {
  (void)iout;
  return pmb->ruser_meshblock_data[CYCLE_REL_DIFF](REL_MAX);
}

Real HistoryCycleRelRho(MeshBlock *pmb, int iout) {
  (void)iout;
  return pmb->ruser_meshblock_data[CYCLE_REL_DIFF](REL_RHO);
}

Real HistoryCycleRelTgas(MeshBlock *pmb, int iout) {
  (void)iout;
  return pmb->ruser_meshblock_data[CYCLE_REL_DIFF](REL_TGAS);
}

Real HistoryCycleRelTrad(MeshBlock *pmb, int iout) {
  (void)iout;
  return pmb->ruser_meshblock_data[CYCLE_REL_DIFF](REL_TRAD);
}

Real HistoryShockPosition(MeshBlock *pmb, int iout) {
  (void)iout;
  Real x_shock = kNoShockPosition;
  Real max_jump = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is + 1; i <= pmb->ie - 1; ++i) {
        const Real rho_l = pmb->phydro->w(IDN, k, j, i - 1)*rho_unit;
        const Real rho_r = pmb->phydro->w(IDN, k, j, i + 1)*rho_unit;
        const Real jump = std::abs(rho_r - rho_l)
                          /std::max(0.5*(std::abs(rho_r) + std::abs(rho_l)),
                                    TINY_NUMBER);
        if (jump <= max_jump) continue;
        max_jump = jump;
        x_shock = pmb->pcoord->x1v(i)*leng_unit;
      }
    }
  }
  return (max_jump >= shock_grad_threshold) ? x_shock : kNoShockPosition;
}

Real HistoryMassFluxResidual(MeshBlock *pmb, int iout) {
  (void)iout;
  return FluxResidual(pmb, FluxDiagnostic::mass);
}

Real HistoryMomentumFluxResidual(MeshBlock *pmb, int iout) {
  (void)iout;
  return FluxResidual(pmb, FluxDiagnostic::momentum);
}

Real HistoryEnergyFluxResidual(MeshBlock *pmb, int iout) {
  (void)iout;
  return FluxResidual(pmb, FluxDiagnostic::energy);
}

int RefinementCondition(MeshBlock *pmb) {
  Real max_metric = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rho_m = pmb->phydro->w(IDN, k, j, i - 2);
        const Real rho_0 = pmb->phydro->w(IDN, k, j, i);
        const Real rho_p = pmb->phydro->w(IDN, k, j, i + 2);
        const Real temp_m = LocalGasTemperaturePhys(pmb, k, j, i - 2);
        const Real temp_0 = LocalGasTemperaturePhys(pmb, k, j, i);
        const Real temp_p = LocalGasTemperaturePhys(pmb, k, j, i + 2);
        max_metric = std::max(max_metric, NormalizedSecondDerivative(rho_m, rho_0, rho_p));
        max_metric = std::max(max_metric, NormalizedSecondDerivative(temp_m, temp_0, temp_p));
      }
    }
  }
  if (max_metric > amr_curvature_threshold) return 1;
  if (max_metric < amr_derefine_threshold) return -1;
  return 0;
}

Real HistoryTgMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = 0.0;
  const Real gm1 = pmb->peos->GetGamma() - 1.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rho = std::max(pmb->phydro->w(IDN, k, j, i), TINY_NUMBER);
        const Real egas = pmb->phydro->w(IPR, k, j, i)/gm1;
        out = std::max(out, egas/rho*gm1*temp_unit);
      }
    }
  }
  return out;
}

Real HistoryTrMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real erad = std::max(pmb->prfld2->u_rad(k, j, i)*egas_unit, TINY_NUMBER);
        out = std::max(out, std::pow(erad/kRadiationConst, 0.25));
      }
    }
  }
  return out;
}

Real HistoryRhoMin(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::min(out, pmb->phydro->w(IDN, k, j, i)*rho_unit);
      }
    }
  }
  return out;
}

Real HistoryRhoMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, pmb->phydro->w(IDN, k, j, i)*rho_unit);
      }
    }
  }
  return out;
}

Real HistoryErMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, pmb->prfld2->u_rad(k, j, i)*egas_unit);
      }
    }
  }
  return out;
}

Real HistoryRtime(MeshBlock *pmb, int iout) {
  (void)iout;
  return pmb->pmy_mesh->time*time_unit;
}

Real HistoryEall(MeshBlock *pmb, int iout) {
  (void)iout;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((pmb->ie - pmb->is) + 2*NGHOST);
  Real e = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        e += (pmb->phydro->u(IEN, k, j, i) + pmb->prfld2->u_rad(k, j, i))*vol(i);
      }
    }
  }
  return e*egas_unit;
}
}  // namespace
