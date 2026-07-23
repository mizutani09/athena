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
//! \file nrfld_shock_tube.cpp
//! \brief Problem generator for radiative shock test
//! REFERENCE: W. Zhang, L. Howell, A. Almgren, A. Burrows, J. Bell, Astrophys. J. Suppl. Ser. 196, 20 (2011).
//!            for section 6.5: Shock Tube Problem in the Strong Equilibrium Regime
//======================================================================================

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
#include "../fld/fld.hpp"
#include "../fld/opacity_table.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

// #if !GENERAL_EOS
// #error "The general EOS must be enabled (--eos=general/eos_table)."
// #endif

namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit, vel_unit;
  Real opacity_unit;
  Real a_r_dim, Rgas, mu;
  Real a_r_sim;
  int dir;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  // Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real rho0_L, rho0_R;
  Real T0_L, T0_R;
  Real v0_L, v0_R;
  Real p0_L, p0_R;
  Real egas0_L, egas0_R;
  Real Er0_L, Er0_R;
  Real sigma_P, sigma_R;
  int rk_cycle;
  bool use_opacity_table;
  UserOpacityTable *puser_table = nullptr;

  // for iuser_meshblock
  int TSTEP_COUNTER = 0;

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

  void SetNRFixedBoundary(AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
                          int is, int ie, int js, int je, int ks, int ke, int ngh,
                          int axis, bool inner) {
    Real u_rad_val = inner ? Er0_L : Er0_R;
    Real u_gas_val = inner ? egas0_L : egas0_R;
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? is-i : ie+i;
            u_rad(k,j,ii) = u_rad_val;
            u_gas(k,j,ii) = u_gas_val;
          }
        }
      }
    } else if (axis == 2) {
      for (int k=ks; k<=ke; k++) {
        for (int i=is; i<=ie; i++) {
          for (int j=1; j<=ngh; j++) {
            int jj = inner ? js-j : je+j;
            u_rad(k,jj,i) = u_rad_val;
            u_gas(k,jj,i) = u_gas_val;
          }
        }
      }
    } else {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int k=1; k<=ngh; k++) {
            int kk = inner ? ks-k : ke+k;
            u_rad(kk,j,i) = u_rad_val;
            u_gas(kk,j,i) = u_gas_val;
          }
        }
      }
    }
  }

  void SetFLDFixedBoundary(AthenaArray<Real> &u_rad_fld,
                           int is, int ie, int js, int je, int ks, int ke, int ngh,
                           int axis, bool inner) {
    Real u_rad_val = inner ? Er0_L : Er0_R;
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? is-i : ie+i;
            u_rad_fld(k,j,ii) = u_rad_val;
          }
        }
      }
    } else if (axis == 2) {
      for (int k=ks; k<=ke; k++) {
        for (int i=is; i<=ie; i++) {
          for (int j=1; j<=ngh; j++) {
            int jj = inner ? js-j : je+j;
            u_rad_fld(k,jj,i) = u_rad_val;
          }
        }
      }
    } else {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int k=1; k<=ngh; k++) {
            int kk = inner ? ks-k : ke+k;
            u_rad_fld(kk,j,i) = u_rad_val;
          }
        }
      }
    }
  }

  void SetHydroFixedBoundary(AthenaArray<Real> &prim,
                             int is, int ie, int js, int je, int ks, int ke, int ngh,
                             int axis, bool inner) {
    Real rho_val = inner ? rho0_L : rho0_R;
    Real v_val = inner ? v0_L : v0_R;
    Real p_val = inner ? p0_L : p0_R;
    int vel = (axis == 1) ? IVX : (axis == 2) ? IVY : IVZ;
    if (axis == 1) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=1; i<=ngh; i++) {
            int ii = inner ? is-i : ie+i;
            prim(IDN,k,j,ii) = rho_val;
            prim(IVX,k,j,ii) = 0.0;
            prim(IVY,k,j,ii) = 0.0;
            prim(IVZ,k,j,ii) = 0.0;
            prim(vel,k,j,ii) = v_val;
            prim(IPR,k,j,ii) = p_val;
          }
        }
      }
    } else if (axis == 2) {
      for (int k=ks; k<=ke; k++) {
        for (int i=is; i<=ie; i++) {
          for (int j=1; j<=ngh; j++) {
            int jj = inner ? js-j : je+j;
            prim(IDN,k,jj,i) = rho_val;
            prim(IVX,k,jj,i) = 0.0;
            prim(IVY,k,jj,i) = 0.0;
            prim(IVZ,k,jj,i) = 0.0;
            prim(vel,k,jj,i) = v_val;
            prim(IPR,k,jj,i) = p_val;
          }
        }
      }
    } else {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int k=1; k<=ngh; k++) {
            int kk = inner ? ks-k : ke+k;
            prim(IDN,kk,j,i) = rho_val;
            prim(IVX,kk,j,i) = 0.0;
            prim(IVY,kk,j,i) = 0.0;
            prim(IVZ,kk,j,i) = 0.0;
            prim(vel,kk,j,i) = v_val;
            prim(IPR,kk,j,i) = p_val;
          }
        }
      }
    }
  }
}


void NRInnerX1(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  SetNRFixedBoundary(u_rad, u_gas, is, ie, js, je, ks, ke, ngh, 1, true);
  return;
}

void NROuterX1(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  SetNRFixedBoundary(u_rad, u_gas, is, ie, js, je, ks, ke, ngh, 1, false);
  return;
}

void FLDFixedInnerX1(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // std::cout << "Apply fixed inner x1 FLD BC" << std::endl;
  // for fixed boundary condition
  SetFLDFixedBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh, 1, true);
  return;
}

void FLDFixedOuterX1(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  SetFLDFixedBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh, 1, false);
  return;
}

void HydroFixedInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  SetHydroFixedBoundary(prim, is, ie, js, je, ks, ke, ngh, 1, true);
  return;
}

void HydroFixedOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  SetHydroFixedBoundary(prim, is, ie, js, je, ks, ke, ngh, 1, false);
  return;
}

void NRInnerX2(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetNRFixedBoundary(u_rad, u_gas, is, ie, js, je, ks, ke, ngh, 2, true);
  return;
}

void NROuterX2(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetNRFixedBoundary(u_rad, u_gas, is, ie, js, je, ks, ke, ngh, 2, false);
  return;
}

void NRInnerX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetNRFixedBoundary(u_rad, u_gas, is, ie, js, je, ks, ke, ngh, 3, true);
  return;
}

void NROuterX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetNRFixedBoundary(u_rad, u_gas, is, ie, js, je, ks, ke, ngh, 3, false);
  return;
}

void FLDFixedInnerX2(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetFLDFixedBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh, 2, true);
  return;
}

void FLDFixedOuterX2(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetFLDFixedBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh, 2, false);
  return;
}

void FLDFixedInnerX3(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetFLDFixedBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh, 3, true);
  return;
}

void FLDFixedOuterX3(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetFLDFixedBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh, 3, false);
  return;
}

void HydroFixedInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetHydroFixedBoundary(prim, is, ie, js, je, ks, ke, ngh, 2, true);
  return;
}

void HydroFixedOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetHydroFixedBoundary(prim, is, ie, js, je, ks, ke, ngh, 2, false);
  return;
}

void HydroFixedInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetHydroFixedBoundary(prim, is, ie, js, je, ks, ke, ngh, 3, true);
  return;
}

void HydroFixedOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetHydroFixedBoundary(prim, is, ie, js, je, ks, ke, ngh, 3, false);
  return;
}

void GetOpacityFromUserTable(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD *prfld = pmb->prfld;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        Real rho = prim(IDN,k,j,i);
        Real egas = prim(IEN,k,j,i);
        Real rho_phys = rho*rho_unit;
        Real temp_phys = pmb->peos->TempFromRhoEg(rho, egas)*T_unit;
        prfld->sigma_p(k,j,i) =
            puser_table->GetOpacity(RadFLD::SIGMA_P, rho_phys, temp_phys)/opacity_unit*rho;
        prfld->sigma_r(k,j,i) =
            puser_table->GetOpacity(RadFLD::SIGMA_R, rho_phys, temp_phys)/opacity_unit*rho;
      }
    }
  }
}

void ConstantOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD *prfld = pmb->prfld;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        prfld->sigma_p(k,j,i) = sigma_P;
        prfld->sigma_r(k,j,i) = sigma_R;
      }
    }
  }
}

Real FindPFromRhoT(MeshBlock *pmb, Real rho, Real T) {
  // std::cout << "FindPFromRhoT is called with rho = " << rho << ", T = " << T << std::endl;
  Real p;
#if GENERAL_EOS
  // find p from rho and T using bisection method
  p = 1.0; // dummy value to initialize p
  Real p_min = 1e-20;
  Real p_max = 1e10;
  for (int iter=0; iter<100; iter++) {
    Real egas = pmb->peos->EgasFromRhoP(rho, p);
    Real T_guess = pmb->peos->TempFromRhoEg(rho, egas);
    // std::cout << "iter = " << iter << ", p = " << p << ", T_guess = " << T_guess << ", egas = " << egas << std::endl;
    if (std::abs(T_guess - T) < 1e-10*T) {
      break;
    }
    if (T_guess > T) {
      p_max = p;
      p = 0.5*(p + p_min);
    } else {
      p_min = p;
      p = 0.5*(p + p_max);
    }
  }
#else
  p = rho*T;
#endif
  return p;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  std::cout << "multilevel = " << multilevel << std::endl;
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
#if GENERAL_EOS
  T_unit = pin->GetReal("hydro", "T_unit");
#else
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
#endif
  // if (Globals::my_rank == 0) {
  //   std::cout << "T_unit from input file: " << T_unit << std::endl;
  //   std::cout << "T_unit from unit conversion: " << T_unit_tmp << std::endl;
  // }
  // T_unit = pin->GetOrAddReal("hydro", "T_unit", pres_unit/rho_unit*mu/R_gas);

  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  a_r_sim = a_r_dim/(egas_unit/std::pow(T_unit, 4));

  vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;
  opacity_unit = 1.0/(rho_unit*leng_unit);

  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  rho0_L = pin->GetReal("problem", "rho0_L") / rho_unit;
  rho0_R = pin->GetReal("problem", "rho0_R") / rho_unit;
  T0_L = pin->GetReal("problem", "T0_L") / T_unit;
  T0_R = pin->GetReal("problem", "T0_R") / T_unit;
  v0_L = pin->GetReal("problem", "v0_L") / vel_unit;
  v0_R = pin->GetReal("problem", "v0_R") / vel_unit;

  if (dir == 1) {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroFixedInnerX1);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroFixedOuterX1);
  } else if (dir == 2) {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x2, FLDFixedInnerX2);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x2, FLDFixedOuterX2);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x2, NRInnerX2);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x2, NROuterX2);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, HydroFixedInnerX2);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, HydroFixedOuterX2);
  } else {
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x3, NRInnerX3);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x3, NROuterX3);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroFixedInnerX3);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroFixedOuterX3);
  }


  std::string integrator = pin->GetString("time","integrator");
  if (integrator == "rk3") {
    rk_cycle = 3;
  } else if (integrator == "rk2") {
    rk_cycle = 2;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR about integrator " << std::endl
        << "now only support the rk2 or rk3 integrator" << std::endl;
    ATHENA_ERROR(msg);
  }

  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTg, "Tgas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "Trad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "egas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "Erad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  // EnrollUserHistoryOutput(7, HistoryL1norm, "L1norm", UserHistoryOperation::sum);

}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // p0_L = rho0_L*T0_L;
  // p0_R = rho0_R*T0_R;
  p0_L = FindPFromRhoT(this, rho0_L, T0_L);
  p0_R = FindPFromRhoT(this, rho0_R, T0_R);
  Er0_L = a_r_sim*std::pow(T0_L, 4);
  Er0_R = a_r_sim*std::pow(T0_R, 4);
  egas0_L = peos->EgasFromRhoP(rho0_L, p0_L);
  egas0_R = peos->EgasFromRhoP(rho0_R, p0_R);

  int idata_size = 0;
  idata_size += 1; // for test counter
  AllocateIntUserMeshBlockDataField(idata_size);

  iuser_meshblock_data[TSTEP_COUNTER].NewAthenaArray(1);
  iuser_meshblock_data[TSTEP_COUNTER](0) = 0;

  AllocateUserOutputVariables(5);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");
  SetUserOutputVariableName(4, "P_tot");

  use_opacity_table = pin->GetBoolean("fld", "use_opacity_table");
  std::cout << use_opacity_table << std::endl;
  if (use_opacity_table) {
    puser_table = new UserOpacityTable(pin);
    prfld->EnrollOpacityFunction(GetOpacityFromUserTable);
  } else {
    sigma_P = pin->GetReal("fld", "const_opacity_P");
    sigma_R = pin->GetReal("fld", "const_opacity_R");
    prfld->EnrollOpacityFunction(ConstantOpacity);
  }
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real dx = DxAt(pcoord, 4);
  Real courant = pin->GetReal("time", "cfl_number");
  Real Cs_L = std::sqrt(peos->AsqFromRhoP(rho0_L, p0_L));
  Real Cs_R = std::sqrt(peos->AsqFromRhoP(rho0_R, p0_R));
  Real max_vel = std::max(std::abs(v0_L+Cs_L), std::abs(v0_R+Cs_R));
  Real dt_exp = courant*dx/max_vel*time_unit;
  // Real const_opasity = pin->GetReal("fld", "const_opacity");
  // Real const_opasity_sim = const_opasity*leng_unit*rho_unit;
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  Real t_lim = 1e-6; // in s
  Real exp_cycle = t_lim/dt_exp;

  Real L = MeshMax(pmy_mesh) - MeshMin(pmy_mesh);
  if (gid == 0) {
    std::cout << "use_opacity_table = " << (use_opacity_table ? "true" : "false") << std::endl;
    std::cout << "eos_type = " << EQUATION_OF_STATE << std::endl;
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "vel_unit = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    std::cout << "T_unit = " << T_unit << " K" << std::endl;
    std::cout << "c_ph_sim = " << c_ph_sim << " cm s^-1" << std::endl;
    std::cout << "dx = " << dx*leng_unit << " cm" << std::endl;
    std::cout << "dt = " << dt_exp << " s" << std::endl;
    std::cout << "dt_sim = " << dt_exp/time_unit << std::endl;
    std::cout << "p0_L = " << p0_L << std::endl;
    std::cout << "p0_R = " << p0_R << std::endl;
    std::cout << "Er0_L = " << Er0_L << std::endl;
    std::cout << "Er0_R = " << Er0_R << std::endl;
    std::cout << "v0_L = " << v0_L << std::endl;
    std::cout << "v0_R = " << v0_R << std::endl;
    std::cout << "expected cycle = " << exp_cycle << std::endl;
    std::cout << "sigma_P = " << sigma_P << std::endl;
    std::cout << "sigma_R = " << sigma_R << std::endl;
    std::cout << "T0_L = " << T0_L << std::endl;
    Real T0_L_check = peos->TempFromRhoEg(rho0_L, egas0_L);
    std::cout << "T0_L_check = " << T0_L_check << std::endl;

    // also output the upper values in txt file
    std::ofstream ofs("problem_parameters.txt");
    ofs << ">>> Problem parameters <<<" << std::endl;
    ofs << "- Units" << std::endl;
    ofs << "rhoUnit        = " << rho_unit << " g cm^-3" << std::endl;
    ofs << "egasUnit       = " << egas_unit << " erg cm^-3" << std::endl;
    ofs << "timeUnit       = " << time_unit << " s" << std::endl;
    ofs << "lengUnit       = " << leng_unit << " cm" << std::endl;
    ofs << "velUnit        = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    ofs << "TUnit          = " << T_unit << " K" << std::endl;
    ofs << std::endl;

    ofs << "- Simulation parameters" << std::endl;
    ofs << "c_ph_sim        = " << c_ph_sim << std::endl;
    ofs << "dx_dim          = " << dx*leng_unit << " cm" << std::endl;
    ofs << "dt_dim          = " << dt_exp << " s" << std::endl;
    ofs << "dt_sim          = " << dt_exp/time_unit << std::endl;
    ofs << "rho0_L          = " << rho0_L << std::endl;
    ofs << "rho0_R          = " << rho0_R << std::endl;
    ofs << "T0_L            = " << T0_L << std::endl;
    ofs << "T0_R            = " << T0_R << std::endl;
    ofs << "v0_L            = " << v0_L << std::endl;
    ofs << "v0_R            = " << v0_R << std::endl;
    ofs << "p0_L            = " << p0_L << std::endl;
    ofs << "p0_R            = " << p0_R << std::endl;
    ofs << "Er0_L           = " << Er0_L << std::endl;
    ofs << "Er0_R           = " << Er0_R << std::endl;
    ofs << "sigma_P         = " << sigma_P << std::endl;
    ofs << "sigma_R         = " << sigma_R << std::endl;
    ofs.close();
  }

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;

  int mom = (dir == 1) ? IM1 : (dir == 2) ? IM2 : IM3;
  Real x_mid = 0.5*(MeshMin(pmy_mesh) + MeshMax(pmy_mesh));

  bool flag = true;
  bool flag2 = true;
  for(int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real x = CoordAt(pcoord, i, j, k);
        if (x < x_mid) {
          phydro->u(IDN,k,j,i) = rho0_L;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(mom,k,j,i) = rho0_L*v0_L;
          if (NON_BAROTROPIC_EOS)
            phydro->u(IEN,k,j,i) = egas0_L + 0.5*rho0_L*v0_L*v0_L;

          // for FLD
          prfld->u_gas(k,j,i) = egas0_L;
          prfld->u_rad(k,j,i) = Er0_L;

          if (flag) {
            std::cout << "Initial condition at x = " << x << ": " << std::endl;
            std::cout << "rho = " << phydro->u(IDN,k,j,i) << std::endl;
            std::cout << "v = " << phydro->u(mom,k,j,i)/phydro->u(IDN,k,j,i) << std::endl;
            std::cout << "egas = " << prfld->u_gas(k,j,i) << std::endl;
            std::cout << "Erad = " << prfld->u_rad(k,j,i) << std::endl;
            flag = false;
          }
        } else {
          phydro->u(IDN,k,j,i) = rho0_R;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(mom,k,j,i) = rho0_R*v0_R;
          if (NON_BAROTROPIC_EOS)
            phydro->u(IEN,k,j,i) = egas0_R + 0.5*rho0_R*v0_R*v0_R;

          // for FLD
          prfld->u_gas(k,j,i) = egas0_R;
          prfld->u_rad(k,j,i) = Er0_R;

          if (flag2) {
            std::cout << "Initial condition at x = " << x << ": " << std::endl;
            std::cout << "rho = " << phydro->u(IDN,k,j,i) << std::endl;
            std::cout << "v = " << phydro->u(mom,k,j,i)/phydro->u(IDN,k,j,i) << std::endl;
            std::cout << "egas = " << prfld->u_gas(k,j,i) << std::endl;
            std::cout << "Erad = " << prfld->u_rad(k,j,i) << std::endl;
            flag2 = false;
          }
        }
      }
    }
  }
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // std::cout << "UserWorkBeforeOutput is called." << std::endl;
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
        Real egas = prfld->u_gas(k,j,i);
        user_out_var(0,k,j,i) = egas*egas_unit;
        user_out_var(1,k,j,i) = prfld->u_rad(k,j,i)*egas_unit;
        // user_out_var(2,k,j,i) = prfld->u_gas(k,j,i)/phydro->w(IDN,k,j,i)*temp_coef;
        user_out_var(2,k,j,i) = peos->TempFromRhoEg(phydro->w(IDN,k,j,i), egas)*T_unit;
        user_out_var(3,k,j,i) = std::pow(prfld->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
        user_out_var(4,k,j,i) = peos->PresFromRhoEg(phydro->w(IDN,k,j,i), egas) + ONE_3RD*user_out_var(1,k,j,i);
      }
    }
  }
  return;
}



namespace {

Real HistoryTg(MeshBlock *pmb, int iout) {
  // std::cout << "HistoryTg is called." << std::endl;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += pmb->peos->TempFromRhoEg(pmb->phydro->w(IDN,k,j,i), pmb->prfld->u_gas(k,j,i))*T_unit;
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
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real aT4 = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        aT4 += std::pow(pmb->peos->TempFromRhoEg(pmb->phydro->w(IDN,k,j,i), pmb->prfld->u_gas(k,j,i)), 4);
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
        E += pmb->prfld->u_gas(k,j,i)*vol(i);
        E += pmb->prfld->u_rad(k,j,i)*vol(i);
      }
    }
  }
  return E*egas_unit;
}

} // namespace
