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
#include <vector>

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
  int dim;
  int dir;
  Real init_ratio, init_time;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real HistoryL1normRel(MeshBlock *pmb, int iout);
  Real HistoryTransL1(MeshBlock *pmb, int iout);
  Real HistoryTransL1Rel(MeshBlock *pmb, int iout);
  Real HistoryTransMaxRel(MeshBlock *pmb, int iout);
  Real Er0, rho0, p0;
  Real chi;

  Real CoordAt(const Coordinates *pco, int i, int j, int k) {
    if (dir == 1) return pco->x1v(i);
    if (dir == 2) return pco->x2v(j);
    return pco->x3v(k);
  }

  Real DxAt(const Coordinates *pco) {
    if (dir == 1) return pco->dx1f(4);
    if (dir == 2) return pco->dx2f(4);
    return pco->dx3f(4);
  }

  void SetGaussianBoundary(AthenaArray<Real> &u_rad, Coordinates *pco,
                           int axis, bool inner,
                           int is, int ie, int js, int je, int ks, int ke, int ngh,
                           Real time, Real dt) {
    Real chi_t = chi * (time+dt+init_time);
    if (dim != 1) return;
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? (is-i) : (ie+i);
            Real x = pco->x1v(ii);
            Real r_sq = SQR(x-0.5);
            u_rad(k,j,ii) = coef*std::exp(-r_sq/(4*chi_t));
          }
        }
      }
    } else if (axis == 2) {
      for (int k=ks; k<=ke; k++) {
        for (int j=1; j<=ngh; j++) {
          int jj = inner ? (js-j) : (je+j);
          Real x = pco->x2v(jj);
          Real r_sq = SQR(x-0.5);
          for (int i=is; i<=ie; i++) {
            u_rad(k,jj,i) = coef*std::exp(-r_sq/(4*chi_t));
          }
        }
      }
    } else {
      for (int k=1; k<=ngh; k++) {
        int kk = inner ? (ks-k) : (ke+k);
        Real x = pco->x3v(kk);
        Real r_sq = SQR(x-0.5);
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            u_rad(kk,j,i) = coef*std::exp(-r_sq/(4*chi_t));
          }
        }
      }
    }
  }

  void HydroOutflowBoundary(AthenaArray<Real> &prim, int axis, bool inner,
                            int is, int ie, int js, int je, int ks, int ke, int ngh) {
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? (is-i) : (ie+i);
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
        for (int j=1; j<=ngh; j++) {
          int jj = inner ? (js-j) : (je+j);
          int jr = inner ? js : je;
          for (int i=is; i<=ie; i++) {
            prim(IDN,k,jj,i) = prim(IDN,k,jr,i);
            prim(IVX,k,jj,i) = prim(IVX,k,jr,i);
            prim(IVY,k,jj,i) = prim(IVY,k,jr,i);
            prim(IVZ,k,jj,i) = prim(IVZ,k,jr,i);
            prim(IPR,k,jj,i) = prim(IPR,k,jr,i);
          }
        }
      }
    } else {
      for (int k=1; k<=ngh; k++) {
        int kk = inner ? (ks-k) : (ke+k);
        int kr = inner ? ks : ke;
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
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

  void ComputeTransverseMean(MeshBlock *pmb, std::vector<Real> &mean) {
    int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
    if (dir == 1) {
      int nx = ie - is + 1;
      int ntr = (je - js + 1) * (ke - ks + 1);
      mean.assign(nx, 0.0);
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            mean[i - is] += pmb->prfld->u_rad(k, j, i);
          }
        }
      }
      for (int i = 0; i < nx; ++i) mean[i] /= ntr;
    } else if (dir == 2) {
      int nx = je - js + 1;
      int ntr = (ie - is + 1) * (ke - ks + 1);
      mean.assign(nx, 0.0);
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            mean[j - js] += pmb->prfld->u_rad(k, j, i);
          }
        }
      }
      for (int j = 0; j < nx; ++j) mean[j] /= ntr;
    } else {
      int nx = ke - ks + 1;
      int ntr = (ie - is + 1) * (je - js + 1);
      mean.assign(nx, 0.0);
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            mean[k - ks] += pmb->prfld->u_rad(k, j, i);
          }
        }
      }
      for (int k = 0; k < nx; ++k) mean[k] /= ntr;
    }
  }
}

void NRInnerX1(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad, pco, 1, true, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void NROuterX1(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad, pco, 1, false, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void NRInnerX2(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad, pco, 2, true, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void NROuterX2(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad, pco, 2, false, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void NRInnerX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad, pco, 3, true, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void NROuterX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad, pco, 3, false, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void FLDInnerX1(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad_fld, pco, 1, true, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void FLDOuterX1(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad_fld, pco, 1, false, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void FLDInnerX2(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad_fld, pco, 2, true, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void FLDOuterX2(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad_fld, pco, 2, false, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void FLDInnerX3(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad_fld, pco, 3, true, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void FLDOuterX3(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetGaussianBoundary(u_rad_fld, pco, 3, false, is, ie, js, je, ks, ke, ngh, time, dt);
  return;
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroOutflowBoundary(prim, 1, true, is, ie, js, je, ks, ke, ngh);
  return;
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroOutflowBoundary(prim, 1, false, is, ie, js, je, ks, ke, ngh);
  return;
}

void HydroInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroOutflowBoundary(prim, 2, true, is, ie, js, je, ks, ke, ngh);
  return;
}

void HydroOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroOutflowBoundary(prim, 2, false, is, ie, js, je, ks, ke, ngh);
  return;
}

void HydroInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroOutflowBoundary(prim, 3, true, is, ie, js, je, ks, ke, ngh);
  return;
}

void HydroOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroOutflowBoundary(prim, 3, false, is, ie, js, je, ks, ke, ngh);
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
  include_radiation_force = false
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

  if (pin->GetBoolean("fld", "include_radiation_force")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "include_radiation_force must be false for this problem.";
    ATHENA_ERROR(msg);
  }

  dim = pin->GetInteger("problem", "dim");
  if (dim != 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "dim should be 1.";
    ATHENA_ERROR(msg);
  }
  dir = pin->GetOrAddInteger("problem", "dir", 1);
  if (dir < 1 || dir > 3) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "dir must be 1, 2, or 3.";
    ATHENA_ERROR(msg);
  }
  init_ratio = pin->GetReal("problem", "init_ratio");
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
  Real vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  // Rgas in cgs
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4


  Real const_opasity = pin->GetReal("fld", "const_opacity");
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  chi = c_ph_sim*mfp_sim/3.0;

  Er0 = 1e+5, rho0 = 1.0, p0 = 100.0;

  Real tau_diff = leng_unit*leng_unit*rho_unit*const_opasity/(4.0*c_ph_dim);
  init_time = tau_diff/(time_unit)*init_ratio;


  if (dir == 1) {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDInnerX1);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterX1);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  } else if (dir == 2) {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x2, FLDInnerX2);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x2, FLDOuterX2);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x2, NRInnerX2);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x2, NROuterX2);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, HydroInnerX2);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, HydroOuterX2);
  } else {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, FLDInnerX3);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, FLDOuterX3);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x3, NRInnerX3);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x3, NROuterX3);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroInnerX3);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroOuterX3);
  }

  AllocateUserHistoryOutput(12);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, HistoryL1norm, "L1norm", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(8, HistoryL1normRel, "L1norm_rel", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(9, HistoryTransL1, "L1norm_trans", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(10, HistoryTransL1Rel, "L1norm_trans_rel", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(11, HistoryTransMaxRel, "max_trans_rel", UserHistoryOperation::max);
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");

  // prfld->EnrollOpacityFunction(NoCoupleOpacity);
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real dx = DxAt(pcoord);
  Real courant = pin->GetReal("time", "cfl_number");
  Real dt_exp = courant*dx*std::sqrt(rho0/(gamma*p0))*time_unit;
  Real const_opasity = pin->GetReal("fld", "const_opacity");
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  Real tau_diff = leng_unit*leng_unit*rho_unit*const_opasity/(4.0*c_ph_dim);
  Real tau_diff_dt = tau_diff/dt_exp;
  if (gid == 0) {
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "T_unit = " << T_unit << " K" << std::endl;
    std::cout << "chi = " << chi*leng_unit*leng_unit/time_unit << " cm^2 s^-1" << std::endl;
    std::cout << "init_time = " << init_time * time_unit << " s" << std::endl;
    std::cout << "dx = " << dx*leng_unit << " cm" << std::endl;
    std::cout << "dt = " << dt_exp << " s" << std::endl;
    std::cout << "dt_sim = " << dt_exp/time_unit << std::endl;
    std::cout << "tau_diff = " << tau_diff << " s" << std::endl;
    std::cout << "tau_diff in sim = " << tau_diff/time_unit << std::endl;
    std::cout << "tau_diff/dt = " << tau_diff_dt << std::endl;
    std::cout << "opacity(L^-1) in sim = " << const_opasity*rho_unit/(1/leng_unit) << std::endl;
  }

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;

  for(int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = p0*igm1;
      }
    }
  }

  if (dim == 3) {
    for(int k=kl; k<=ku; ++k) {
      Real z = pcoord->x3v(k);
      for(int j=jl; j<=ju; ++j) {
        Real y = pcoord->x2v(j);
        for(int i=il; i<=iu; ++i) {
          Real x = pcoord->x1v(i);
          Real r_sq = SQR(x-0.5)+SQR(y-0.5)+SQR(z-0.5);
          prfld->u_gas(k,j,i) = p0*igm1;
          Real res = Er0/(8*std::pow(M_PI*chi*init_time, 1.5))*std::exp(-r_sq/(4*chi*init_time));
          prfld->u_rad(k,j,i) = res;
        }
      }
    }
  } else if (dim == 2) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real r_sq = SQR(pcoord->x1v(i)-0.5)+SQR(pcoord->x2v(j)-0.5);
          prfld->u_gas(k,j,i) = p0*igm1;
          Real res = Er0/(4*M_PI*chi*init_time)*std::exp(-r_sq/(4*chi*init_time));
          prfld->u_rad(k,j,i) = res;
        }
      }
    }
  } else if (dim == 1) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real r_sq = SQR(CoordAt(pcoord, i, j, k) - 0.5);
          prfld->u_gas(k,j,i) = p0*igm1;
          Real res = Er0/(2*std::sqrt(M_PI*chi*init_time))*std::exp(-r_sq/(4*chi*init_time));
          prfld->u_rad(k,j,i) = res;
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
        user_out_var(0,k,j,i) = prfld->u_gas(k,j,i)*egas_unit;
        user_out_var(1,k,j,i) = prfld->u_rad(k,j,i)*egas_unit;
        user_out_var(2,k,j,i) = prfld->u_gas(k,j,i)/phydro->w(IDN,k,j,i)*temp_coef;
        user_out_var(3,k,j,i) = std::pow(prfld->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
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
        T += pmb->prfld->u_gas(k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit;
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
        T += std::pow(pmb->prfld->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
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
        e += pmb->prfld->u_gas(k,j,i);//*vol(i);
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
        E += pmb->prfld->u_rad(k,j,i);//*vol(i);
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
        aT4 += std::pow(pmb->prfld->u_gas(k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit, 4);
        num++;
      }
    }
  }
  aT4 *= a_r_dim;
  aT4 /= num;
  return aT4;
}

Real HistoryRtime(MeshBlock *pmb, int iout) {
  return (pmb->pmy_mesh->time+init_time)*time_unit;
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
        E += pmb->prfld->u_gas(k,j,i)*vol(i);
        E += pmb->prfld->u_rad(k,j,i)*vol(i);
      }
    }
  }
  return E*egas_unit;
}

Real HistoryL1norm(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  Real L1norm = 0;
  Real chi_t = chi * (pmb->pmy_mesh->time+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real x = CoordAt(pmb->pcoord, i, j, k);
          Real r_sq = SQR(x-0.5);
          Real an = coef*std::exp(-r_sq/(4*chi_t));
          L1norm += std::abs(pmb->prfld->u_rad(k,j,i)-an);
        }
      }
    }
  }
  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
  L1norm /= ncells*nbtotal;
  return L1norm;
}

Real HistoryL1normRel(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  Real L1norm = 0;
  Real chi_t = chi * (pmb->pmy_mesh->time+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real x = CoordAt(pmb->pcoord, i, j, k);
          Real r_sq = SQR(x-0.5);
          Real an = coef*std::exp(-r_sq/(4*chi_t));
          L1norm += std::abs(pmb->prfld->u_rad(k,j,i)-an)/an;
        }
      }
    }
  }
  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
  L1norm /= ncells*nbtotal;
  return L1norm;
}

Real HistoryTransL1(MeshBlock *pmb, int iout) {
  if (dim != 1) return 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  std::vector<Real> mean;
  ComputeTransverseMean(pmb, mean);

  Real norm = 0.0;
  if (dir == 1) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          norm += std::abs(pmb->prfld->u_rad(k, j, i) - mean[i - is]);
        }
      }
    }
  } else if (dir == 2) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          norm += std::abs(pmb->prfld->u_rad(k, j, i) - mean[j - js]);
        }
      }
    }
  } else {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          norm += std::abs(pmb->prfld->u_rad(k, j, i) - mean[k - ks]);
        }
      }
    }
  }

  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie - is + 1) * (je - js + 1) * (ke - ks + 1);
  norm /= ncells * nbtotal;
  return norm;
}

Real HistoryTransL1Rel(MeshBlock *pmb, int iout) {
  if (dim != 1) return 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  std::vector<Real> mean;
  ComputeTransverseMean(pmb, mean);

  Real norm = 0.0;
  if (dir == 1) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real denom = std::max(std::abs(mean[i - is]), static_cast<Real>(1.0e-30));
          norm += std::abs(pmb->prfld->u_rad(k, j, i) - mean[i - is]) / denom;
        }
      }
    }
  } else if (dir == 2) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real denom = std::max(std::abs(mean[j - js]), static_cast<Real>(1.0e-30));
          norm += std::abs(pmb->prfld->u_rad(k, j, i) - mean[j - js]) / denom;
        }
      }
    }
  } else {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real denom = std::max(std::abs(mean[k - ks]), static_cast<Real>(1.0e-30));
          norm += std::abs(pmb->prfld->u_rad(k, j, i) - mean[k - ks]) / denom;
        }
      }
    }
  }

  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie - is + 1) * (je - js + 1) * (ke - ks + 1);
  norm /= ncells * nbtotal;
  return norm;
}

Real HistoryTransMaxRel(MeshBlock *pmb, int iout) {
  if (dim != 1) return 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  std::vector<Real> mean;
  ComputeTransverseMean(pmb, mean);

  Real norm = 0.0;
  if (dir == 1) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real denom = std::max(std::abs(mean[i - is]), static_cast<Real>(1.0e-30));
          norm = std::max(norm, std::abs(pmb->prfld->u_rad(k, j, i) - mean[i - is]) / denom);
        }
      }
    }
  } else if (dir == 2) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real denom = std::max(std::abs(mean[j - js]), static_cast<Real>(1.0e-30));
          norm = std::max(norm, std::abs(pmb->prfld->u_rad(k, j, i) - mean[j - js]) / denom);
        }
      }
    }
  } else {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real denom = std::max(std::abs(mean[k - ks]), static_cast<Real>(1.0e-30));
          norm = std::max(norm, std::abs(pmb->prfld->u_rad(k, j, i) - mean[k - ks]) / denom);
        }
      }
    }
  }

  return norm;
}

} // namespace
