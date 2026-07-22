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
//! \file nrfld_streaming_front.cpp
//! \brief Problem generator for the Zhang et al. (2011) optically thin streaming test.

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
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
constexpr Real kDefaultLightSpeed = 2.99792458e10;

Real rho_unit, egas_unit, leng_unit, time_unit;
Real rho0, er_left_phys, er_right_phys, x_front0_phys;
Real chi_r_phys, kappa_p_phys, c_light_phys;
Real er_left_code, er_right_code, egas_init, hydro_p_init;
Real chi_r_code, kappa_p_code, fixed_dt_code;

Real FixedTimeStep(MeshBlock *pmb);
Real HistoryErMax(MeshBlock *pmb, int iout);
Real HistoryErMin(MeshBlock *pmb, int iout);
Real HistoryVMax(MeshBlock *pmb, int iout);
Real HistoryFrontX(MeshBlock *pmb, int iout);

void ConstantOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld, AthenaArray<Real> &prim) {
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

void SetRadiationDirichlet(AthenaArray<Real> &u_rad, bool inner,
                           int is, int ie, int js, int je, int ks, int ke, int ngh) {
  const Real er_bc = inner ? er_left_code : er_right_code;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int n = 1; n <= ngh; ++n) {
        const int i = inner ? is - n : ie + n;
        u_rad(k, j, i) = er_bc;
      }
    }
  }
}

void SetNRDirichlet(AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas, bool inner,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetRadiationDirichlet(u_rad, inner, is, ie, js, je, ks, ke, ngh);
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int n = 1; n <= ngh; ++n) {
        const int i = inner ? is - n : ie + n;
        u_gas(k, j, i) = egas_init;
      }
    }
  }
}

void SetHydroStaticBoundary(AthenaArray<Real> &prim, bool inner,
                            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int n = 1; n <= ngh; ++n) {
        const int i = inner ? is - n : ie + n;
        prim(IDN, k, j, i) = rho0;
        prim(IVX, k, j, i) = 0.0;
        prim(IVY, k, j, i) = 0.0;
        prim(IVZ, k, j, i) = 0.0;
        prim(IPR, k, j, i) = hydro_p_init;
      }
    }
  }
}

Real CenteredGradX(const MeshBlock *pmb, int k, int j, int i) {
  const int im = std::max(i - 1, pmb->is - NGHOST);
  const int ip = std::min(i + 1, pmb->ie + NGHOST);
  const Real dx = pmb->pcoord->x1v(ip) - pmb->pcoord->x1v(im);
  if (dx <= 0.0) return 0.0;
  return (pmb->prfld2->u_rad(k, j, ip) - pmb->prfld2->u_rad(k, j, im))/dx;
}

}  // namespace

void FLDInnerX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)pfld; (void)w; (void)time; (void)dt; (void)ie;
  SetRadiationDirichlet(u_rad_fld, true, is, ie, js, je, ks, ke, ngh);
}

void FLDOuterX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)pfld; (void)w; (void)time; (void)dt; (void)is;
  SetRadiationDirichlet(u_rad_fld, false, is, ie, js, je, ks, ke, ngh);
}

void NRInnerX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)w; (void)time; (void)dt; (void)ie;
  SetNRDirichlet(u_rad, u_gas, true, is, ie, js, je, ks, ke, ngh);
}

void NROuterX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)w; (void)time; (void)dt; (void)is;
  SetNRDirichlet(u_rad, u_gas, false, is, ie, js, je, ks, ke, ngh);
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)b; (void)time; (void)dt; (void)ie;
  SetHydroStaticBoundary(prim, true, is, ie, js, je, ks, ke, ngh);
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  (void)pmb; (void)pco; (void)b; (void)time; (void)dt; (void)is;
  SetHydroStaticBoundary(prim, false, is, ie, js, je, ks, ke, ngh);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!pin->GetBoolean("fld", "is_couple")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "is_couple must be true for this problem so the NR-FLD solve is active.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "only_rad must be true for this problem.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "cut_diff")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_diff must be false for this problem.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "cut_Pnablav")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_Pnablav must be true for this problem.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "fixed_flux_limitter")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "fixed_flux_limitter must be false for the optically thin streaming test.";
    ATHENA_ERROR(msg);
  }

  rho_unit = pin->GetReal("hydro", "rho_unit");
  egas_unit = pin->GetReal("hydro", "egas_unit");
  time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }
  const Real vel_unit = std::sqrt(egas_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  er_left_phys = pin->GetOrAddReal("problem", "Er_left", 1.0);
  er_right_phys = pin->GetOrAddReal("problem", "Er_right", 1.0e-10);
  x_front0_phys = pin->GetOrAddReal("problem", "x_front0", 10.0);
  chi_r_phys = pin->GetOrAddReal("problem", "chi_R", 1.0e-4);
  kappa_p_phys = pin->GetOrAddReal("problem", "kappa_P", 0.0);
  c_light_phys = pin->GetOrAddReal("problem", "c_light", kDefaultLightSpeed);
  const Real gas_pressure = pin->GetOrAddReal("problem", "gas_pressure", 1.0e-20);
  const Real dt_light_divisor = pin->GetOrAddReal("problem", "dt_light_divisor", 2.0);
  const Real fixed_dt_phys = pin->GetOrAddReal("problem", "fixed_dt", -1.0);

  er_left_code = er_left_phys/egas_unit;
  er_right_code = er_right_phys/egas_unit;
  hydro_p_init = std::max(gas_pressure/egas_unit, TINY_NUMBER);
  const Real gamma_gas = pin->GetReal("hydro", "gamma");
  egas_init = hydro_p_init/(gamma_gas - 1.0);
  chi_r_code = chi_r_phys*leng_unit;
  kappa_p_code = kappa_p_phys*leng_unit;

  const Real dx_phys = (mesh_size.x1max - mesh_size.x1min)
                       /static_cast<Real>(mesh_size.nx1)*leng_unit;
  if (fixed_dt_phys > 0.0) {
    fixed_dt_code = fixed_dt_phys/time_unit;
  } else {
    fixed_dt_code = dx_phys/(dt_light_divisor*c_light_phys*time_unit);
  }

  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDInnerX1);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  EnrollUserTimeStepFunction(FixedTimeStep);

  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, HistoryErMax, "Er_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryErMin, "Er_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(2, HistoryVMax, "vmax", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryFrontX, "x_front", UserHistoryOperation::max);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  (void)pin;
  AllocateUserOutputVariables(6);
  SetUserOutputVariableName(0, "x");
  SetUserOutputVariableName(1, "Erad");
  SetUserOutputVariableName(2, "chiR");
  SetUserOutputVariableName(3, "Rlim");
  SetUserOutputVariableName(4, "lambda");
  SetUserOutputVariableName(5, "Frad_x");
  prfld2->EnrollOpacityFunction(ConstantOpacity);
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
        const Real erad = (x_phys < x_front0_phys) ? er_left_code : er_right_code;
        phydro->u(IDN, k, j, i) = rho0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        if (NON_BAROTROPIC_EOS) phydro->u(IEN, k, j, i) = egas_init;
        prfld2->u_gas(k, j, i) = egas_init;
        prfld2->u_rad(k, j, i) = erad;
      }
    }
  }

  if (gid == 0) {
    const Real vel_unit = leng_unit/time_unit;
    const Real expected_front = x_front0_phys
        + kDefaultLightSpeed*pmy_mesh->tlim;
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "vel_unit = " << vel_unit << " cm s^-1" << std::endl;
    std::cout << "c_light_input = " << c_light_phys << " cm s^-1" << std::endl;
    std::cout << "c_light_solver = " << kDefaultLightSpeed << " cm s^-1" << std::endl;
    std::cout << "Er_left = " << er_left_phys << " erg cm^-3" << std::endl;
    std::cout << "Er_right = " << er_right_phys << " erg cm^-3" << std::endl;
    std::cout << "x_front0 = " << x_front0_phys << " cm" << std::endl;
    std::cout << "chi_R = " << chi_r_phys << " cm^-1" << std::endl;
    std::cout << "kappa_P = " << kappa_p_phys << " cm^-1" << std::endl;
    std::cout << "fixed_dt = " << fixed_dt_code << " [code]" << std::endl;
    std::cout << "fixed_dt = " << fixed_dt_code*time_unit << " s" << std::endl;
    std::cout << "expected_front_at_tlim = " << expected_front << " cm" << std::endl;

    std::ofstream ofs("problem_parameters.txt");
    ofs << ">>> Problem parameters <<<" << std::endl;
    ofs << "rho_unit        = " << rho_unit << " g cm^-3" << std::endl;
    ofs << "egas_unit       = " << egas_unit << " erg cm^-3" << std::endl;
    ofs << "leng_unit       = " << leng_unit << " cm" << std::endl;
    ofs << "time_unit       = " << time_unit << " s" << std::endl;
    ofs << "Er_left         = " << er_left_phys << " erg cm^-3" << std::endl;
    ofs << "Er_right        = " << er_right_phys << " erg cm^-3" << std::endl;
    ofs << "x_front0        = " << x_front0_phys << " cm" << std::endl;
    ofs << "chi_R           = " << chi_r_phys << " cm^-1" << std::endl;
    ofs << "kappa_P         = " << kappa_p_phys << " cm^-1" << std::endl;
    ofs << "chi_R_code      = " << chi_r_code << std::endl;
    ofs << "kappa_P_code    = " << kappa_p_code << std::endl;
    ofs << "fixed_dt_code   = " << fixed_dt_code << std::endl;
    ofs << "fixed_dt_phys   = " << fixed_dt_code*time_unit << " s" << std::endl;
    ofs << "expected_front  = " << expected_front << " cm" << std::endl;
    ofs.close();
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
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
        const Real erad = std::max(prfld2->u_rad(k, j, i), TINY_NUMBER);
        const Real grad = CenteredGradX(this, k, j, i);
        const Real rlim = std::abs(grad)/(std::max(chi_r_code, TINY_NUMBER)*erad);
        const Real lambda = RadFLD2::FluxLimiter(rlim, false);
        const Real flux_code = -prfld2->c_ph*lambda*grad/std::max(chi_r_code, TINY_NUMBER);
        user_out_var(0, k, j, i) = pcoord->x1v(i)*leng_unit;
        user_out_var(1, k, j, i) = prfld2->u_rad(k, j, i)*egas_unit;
        user_out_var(2, k, j, i) = chi_r_phys;
        user_out_var(3, k, j, i) = rlim;
        user_out_var(4, k, j, i) = lambda;
        user_out_var(5, k, j, i) = flux_code*egas_unit*(leng_unit/time_unit);
      }
    }
  }
}

namespace {
Real FixedTimeStep(MeshBlock *pmb) {
  (void)pmb;
  return fixed_dt_code;
}

Real HistoryErMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = -std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, pmb->prfld2->u_rad(k, j, i)*egas_unit);
      }
    }
  }
  return out;
}

Real HistoryErMin(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::min(out, pmb->prfld2->u_rad(k, j, i)*egas_unit);
      }
    }
  }
  return out;
}

Real HistoryVMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rho = std::max(pmb->phydro->u(IDN, k, j, i), TINY_NUMBER);
        const Real vx = pmb->phydro->u(IM1, k, j, i)/rho;
        const Real vy = pmb->phydro->u(IM2, k, j, i)/rho;
        const Real vz = pmb->phydro->u(IM3, k, j, i)/rho;
        out = std::max(out, std::sqrt(vx*vx + vy*vy + vz*vz));
      }
    }
  }
  return out;
}

Real HistoryFrontX(MeshBlock *pmb, int iout) {
  (void)iout;
  const Real threshold = 0.5*(er_left_phys + er_right_phys);
  Real out = pmb->pmy_mesh->mesh_size.x1min*leng_unit;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (pmb->prfld2->u_rad(k, j, i)*egas_unit > threshold) {
          out = std::max(out, pmb->pcoord->x1v(i)*leng_unit);
        }
      }
    }
  }
  return out;
}
}  // namespace
