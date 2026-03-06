//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../fld/fld.hpp"


#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit;
  Real a_r_dim, Rgas, mu;
  int dir;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real Er0_L, Er0_R, rho0, p0;
  Real chi;

  Real CoordAt(const Coordinates *pco, int i, int j, int k) {
    if (dir == 1) return pco->x1v(i);
    if (dir == 2) return pco->x2v(j);
    return pco->x3v(k);
  }

  Real DxAt(const Coordinates *pco, int idx) {
    if (dir == 1) return pco->dx1f(idx);
    if (dir == 2) return pco->dx2f(idx);
    return pco->dx3f(idx);
  }

  Real MeshMin(const Mesh *pm) {
    if (dir == 1) return pm->mesh_size.x1min;
    if (dir == 2) return pm->mesh_size.x2min;
    return pm->mesh_size.x3min;
  }

  Real MeshMax(const Mesh *pm) {
    if (dir == 1) return pm->mesh_size.x1max;
    if (dir == 2) return pm->mesh_size.x2max;
    return pm->mesh_size.x3max;
  }

  void SetLinearBoundary(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                         int is, int ie, int js, int je, int ks, int ke, int ngh,
                         int axis, bool inner) {
    Real x_L = MeshMin(pmb->pmy_mesh) - 0.5*DxAt(pco, 0);
    Real x_R = MeshMax(pmb->pmy_mesh) + 0.5*DxAt(pco, 0);
    Real slope = (Er0_R-Er0_L)/(x_R-x_L);
    Real cons = Er0_L - slope*x_L;
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? is-i : ie+i;
            Real x = pco->x1v(ii);
            pfld->u_rad(k,j,ii) = slope*x + cons;
          }
        }
      }
    } else if (axis == 2) {
      for (int k=ks; k<=ke; k++) {
        for (int i=is; i<=ie; i++) {
          for (int j=1; j<=ngh; j++) {
            int jj = inner ? js-j : je+j;
            Real x = pco->x2v(jj);
            pfld->u_rad(k,jj,i) = slope*x + cons;
          }
        }
      }
    } else {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int k=1; k<=ngh; k++) {
            int kk = inner ? ks-k : ke+k;
            Real x = pco->x3v(kk);
            pfld->u_rad(kk,j,i) = slope*x + cons;
          }
        }
      }
    }
  }

  void HydroCopyBoundary(MeshBlock *pmb, AthenaArray<Real> &prim,
                         int is, int ie, int js, int je, int ks, int ke, int ngh,
                         int axis, bool inner) {
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? is-i : ie+i;
            int ir = inner ? is : ie;
            prim(IDN,k,j,ii) = prim(IDN,k,j,ir);
            prim(IVX,k,j,ii) = prim(IVX,k,j,ir);
            prim(IVY,k,j,ii) = prim(IVY,k,j,ir);
            prim(IVZ,k,j,ii) = prim(IVZ,k,j,ir);
            prim(IPR,k,j,ii) = prim(IPR,k,j,ir);
          }
        }
      }
    } else if (axis == 2) {
      for (int k=ks; k<=ke; k++) {
        for (int i=is; i<=ie; i++) {
          for (int j=1; j<=ngh; j++) {
            int jj = inner ? js-j : je+j;
            int jr = inner ? js : je;
            prim(IDN,k,jj,i) = prim(IDN,k,jr,i);
            prim(IVX,k,jj,i) = prim(IVX,k,jr,i);
            prim(IVY,k,jj,i) = prim(IVY,k,jr,i);
            prim(IVZ,k,jj,i) = prim(IVZ,k,jr,i);
            prim(IPR,k,jj,i) = prim(IPR,k,jr,i);
          }
        }
      }
    } else {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int k=1; k<=ngh; k++) {
            int kk = inner ? ks-k : ke+k;
            int kr = inner ? ks : ke;
            prim(IDN,kk,j,i) = prim(IDN,kr,j,i);
            prim(IVX,kk,j,i) = prim(IVX,kr,j,i);
            prim(IVY,kk,j,i) = prim(IVY,kr,j,i);
            prim(IVZ,kk,j,i) = prim(IVZ,kr,j,i);
            prim(IPR,kk,j,i) = prim(IPR,kr,j,i);
          }
        }
      }
    }
  }
}

void FLDFixedInnerX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetLinearBoundary(pmb, pco, pfld, is, ie, js, je, ks, ke, ngh, 1, true);
  return;
}

void FLDFixedOuterX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetLinearBoundary(pmb, pco, pfld, is, ie, js, je, ks, ke, ngh, 1, false);
  return;
}

void FLDFixedInnerX2(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetLinearBoundary(pmb, pco, pfld, is, ie, js, je, ks, ke, ngh, 2, true);
  return;
}

void FLDFixedOuterX2(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetLinearBoundary(pmb, pco, pfld, is, ie, js, je, ks, ke, ngh, 2, false);
  return;
}

void FLDFixedInnerX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetLinearBoundary(pmb, pco, pfld, is, ie, js, je, ks, ke, ngh, 3, true);
  return;
}

void FLDFixedOuterX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetLinearBoundary(pmb, pco, pfld, is, ie, js, je, ks, ke, ngh, 3, false);
  return;
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroCopyBoundary(pmb, prim, is, ie, js, je, ks, ke, ngh, 1, true);
  return;
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroCopyBoundary(pmb, prim, is, ie, js, je, ks, ke, ngh, 1, false);
  return;
}

void HydroInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroCopyBoundary(pmb, prim, is, ie, js, je, ks, ke, ngh, 2, true);
  return;
}

void HydroOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroCopyBoundary(pmb, prim, is, ie, js, je, ks, ke, ngh, 2, false);
  return;
}

void HydroInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroCopyBoundary(pmb, prim, is, ie, js, je, ks, ke, ngh, 3, true);
  return;
}

void HydroOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroCopyBoundary(pmb, prim, is, ie, js, je, ks, ke, ngh, 3, false);
  return;
}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  /*
  is_couple       = false
  only_rad        = true
  cut_diff        = false
  cut_Pnablav     = true
  fixed_flux_limitter  = true
  */
  // check input
  if (pin->GetBoolean("fld", "is_couple")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "is_couple must be false for this problem.";
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

  if (!pin->GetBoolean("fld", "fixed_flux_limitter")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "fixed_flux_limitter must be true for this problem.";
    ATHENA_ERROR(msg);
  }

  dir = pin->GetOrAddInteger("problem", "dir", 1);
  if (dir < 1 || dir > 3) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "dir must be 1, 2, or 3.";
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
  Real pres_unit = egas_unit;
  // Rgas in cgs
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4

  Real vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  Real const_opasity = pin->GetReal("fld", "const_opacity");
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  chi = c_ph_sim*mfp_sim/3.0;

  rho0 = pin->GetReal("problem", "rho0");
  p0 = pin->GetReal("problem", "p0");
  Er0_L = pin->GetReal("problem", "Er0_L");
  Er0_R = pin->GetReal("problem", "Er0_R");


  if (dir == 1) {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  } else if (dir == 2) {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x2, FLDFixedInnerX2);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x2, FLDFixedOuterX2);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, HydroInnerX2);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, HydroOuterX2);
  } else {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroInnerX3);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroOuterX3);
  }
  AllocateUserHistoryOutput(8);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, HistoryL1norm, "L1norm", UserHistoryOperation::sum);
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(5);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");
  SetUserOutputVariableName(4, "L1norm");
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real dx = DxAt(pcoord, 4);
  Real courant = pin->GetReal("time", "cfl_number");
  Real sound = std::sqrt(gamma*p0/rho0);
  Real dt_exp = courant*dx/sound*time_unit;
  Real const_opasity = pin->GetReal("fld", "const_opacity");
  Real const_opasity_sim = const_opasity*leng_unit*rho_unit;
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  Real L = MeshMax(pmy_mesh) - MeshMin(pmy_mesh);
  // Real tau_diff = L*L/chi;
  // Real tau_diff_dt = tau_diff/dt_exp;
  Real Er_mean = 0.5*(Er0_L+Er0_R), Er_dif = std::abs(Er0_L-Er0_R);
  Real tau_diff = 3.0*(L/c_ph_sim)*(L*const_opasity_sim)*(Er_mean/Er_dif);
  tau_diff *= time_unit;
  Real tau_diff_dt = tau_diff/dt_exp;
  if (gid == 0) {
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "vel_unit = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    std::cout << "T_unit = " << T_unit << " K" << std::endl;
    std::cout << "c_ph_sim = " << c_ph_sim << " cm s^-1" << std::endl;
    std::cout << "chi = " << chi*leng_unit*leng_unit/time_unit << " cm^2 s^-1" << std::endl;
    std::cout << "dx = " << dx*leng_unit << " cm" << std::endl;
    std::cout << "dt = " << dt_exp << " s" << std::endl;
    std::cout << "dt_sim = " << dt_exp/time_unit << std::endl;
    std::cout << "tau_diff = " << tau_diff << " s" << std::endl;
    std::cout << "tau_diff in sim = " << tau_diff/time_unit << std::endl;
    std::cout << "tau_diff/dt = " << tau_diff/dt_exp << std::endl;

    Real ideal_step = 100.0;
    Real ideal_cfl = (tau_diff/time_unit)*sound/(dx*ideal_step);
    std::cout << "If you want to finish the diffusion in " << ideal_step
              << " steps, the cfl_number should be " << ideal_cfl << std::endl;
  }

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;
  Real x_L = MeshMin(pmy_mesh) - DxAt(pcoord, 0)/2.0;
  Real x_R = MeshMax(pmy_mesh) + DxAt(pcoord, 0)/2.0;
  Real slope = (Er0_R-Er0_L)/(x_R-x_L);
  Real cons = Er0_L - slope*x_L;

  Real x_mid = 0.5*(x_L + x_R);

  for(int k=kl; k<=ku; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=il; i<=iu; ++i) {
        Real x1 = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = p0*igm1;
      }
    }
  }
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        prfld2->u_gas(k,j,i) = p0*igm1;
        // if (pcoord->x1v(i) < x_mid) {
        // prfld2->u_rad(k,j,i) = Er0_L;
        // } else {
        // prfld2->u_rad(k,j,i) = Er0_R;
        // }

        // put tanh profile to avoid initial strong diffusion flux
        Real x = CoordAt(pcoord, i, j, k);
        Real an = 0.5*(Er0_R + Er0_L) + 0.5*(Er0_R - Er0_L)*std::tanh((x - x_mid)/(L/4.0));
        prfld2->u_rad(k,j,i) = an;

        if ((dir == 1 && (i == il || i == iu)) ||
            (dir == 2 && (j == jl || j == ju)) ||
            (dir == 3 && (k == kl || k == ku))) {
          Real an = slope*x + cons;
          prfld2->u_rad(k,j,i) = an;
        }
      }
    }
  }

  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real temp_coef = gm1*mu/Rgas*egas_unit/rho_unit;
  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        // assume cal in E
        user_out_var(0,k,j,i) = prfld2->u_gas(k,j,i)*egas_unit;
        user_out_var(1,k,j,i) = prfld2->u_rad(k,j,i)*egas_unit;
        user_out_var(2,k,j,i) = prfld2->u_gas(k,j,i)/phydro->w(IDN,k,j,i)*temp_coef;
        user_out_var(3,k,j,i) = std::pow(prfld2->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
      }
    }
  }

  
  Real x_L = MeshMin(pmy_mesh) - DxAt(pcoord, 0)/2.0;
  Real x_R = MeshMax(pmy_mesh) + DxAt(pcoord, 0)/2.0;
  Real slope = (Er0_R-Er0_L)/(x_R-x_L);
  Real cons = Er0_L - slope*x_L;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x = CoordAt(pcoord, i, j, k);
        Real an = slope*x + cons;
        Real L1norm = std::abs(prfld2->u_rad(k,j,i) - an)/std::abs(an);
        user_out_var(4,k,j,i) = L1norm;
      }
    }
  }
  return;
}

namespace {

Real HistoryTg(MeshBlock *pmb, int iout) {
  const Real gm1  = pmb->peos->GetGamma() - 1.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += pmb->prfld2->u_gas(k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit;
        num++;
      }
    }
  }
  T /= num;
  return T;
}

Real HistoryTr(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += std::pow(pmb->prfld2->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
        num++;
      }
    }
  }
  T /= num;
  return T;
}

// caution! this is for a mean of gas energy density.
Real HistoryEg(MeshBlock *pmb, int iout) {
  const Real gm1  = pmb->peos->GetGamma() - 1.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real e = 0;
  // AthenaArray<Real> vol;
  // vol.NewAthenaArray((ie-is)+2*NGHOST);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      // pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        e += pmb->prfld2->u_gas(k,j,i);//*vol(i);
        num++;
      }
    }
  }
  e /= num;
  return e*egas_unit;
}

// caution! this is for a mean of radiation energy density.
Real HistoryEr(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real E = 0;
  // AthenaArray<Real> vol;
  // vol.NewAthenaArray((ie-is)+2*NGHOST);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      // pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        E += pmb->prfld2->u_rad(k,j,i);//*vol(i);
        num++;
      }
    }
  }
  E /= num;
  return E*egas_unit;
}

Real HistoryaTg4(MeshBlock *pmb, int iout) {
  const Real gm1  = pmb->peos->GetGamma() - 1.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real aT4 = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        aT4 += std::pow(pmb->prfld2->u_gas(k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit, 4);
        num++;
      }
    }
  }
  aT4 *= a_r_dim;
  aT4 /= num;
  return aT4;
}

Real HistoryRtime(MeshBlock *pmb, int iout) {
  return pmb->pmy_mesh->time*time_unit;
}

// caution! this is for a sum of all energy.
Real HistoryEall(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((ie-is)+2*NGHOST);
  int num = 0;
  Real E = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        E += pmb->prfld2->u_gas(k,j,i)*vol(i);
        E += pmb->prfld2->u_rad(k,j,i)*vol(i);
      }
    }
  }
  return E*egas_unit;
}

Real HistoryL1norm(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  Real L1norm = 0;
  Real x_L = MeshMin(pmb->pmy_mesh) - DxAt(pmb->pcoord, 0)/2.0;
  Real x_R = MeshMax(pmb->pmy_mesh) + DxAt(pmb->pcoord, 0)/2.0;
  Real slope = (Er0_R-Er0_L)/(x_R-x_L);
  Real cons = Er0_L - slope*x_L;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x = CoordAt(pmb->pcoord, i, j, k);
        Real an = slope*x + cons;
        L1norm += std::abs(pmb->prfld2->u_rad(k,j,i) - an)/std::abs(an);
      }
    }
  }
  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
  L1norm /= ncells*nbtotal;
  return L1norm;
}

} // namespace
