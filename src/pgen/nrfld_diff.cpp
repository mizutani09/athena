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
#include "../field/field.hpp"
#include "../globals.hpp"
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
  Real init_ratio, init_time;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real Er0, rho0, p0;
  Real chi;
}

void NRInnerX1(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // put analytic solution
  Real chi_t = chi * (time+dt+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=1; i<=ngh; i++) {
          Real x = pco->x1v(is-i);
          Real r_sq = SQR(x-0.5);
          u_rad(k,j,is-i) = coef*std::exp(-r_sq/(4*chi_t));
        }
      }
    }
  }
  return;
}

void NROuterX1(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // put analytic solution
  Real chi_t = chi * (time+dt+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=1; i<=ngh; i++) {
            Real x = pco->x1v(ie+i);
            Real r_sq = SQR(x-0.5);
            u_rad(k,j,ie+i) = coef*std::exp(-r_sq/(4*chi_t));
        }
      }
    }
  }
  return;
}

void FLDInnerX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // put analytic solution
  Real chi_t = chi * (time+dt+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=1; i<=ngh; i++) {
          Real x = pco->x1v(is-i);
          Real r_sq = SQR(x-0.5);
          u_rad_fld(k,j,is-i) = coef*std::exp(-r_sq/(4*chi_t));
        }
      }
    }
  }
  return;
}

void FLDOuterX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // put analytic solution
  Real chi_t = chi * (time+dt+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=1; i<=ngh; i++) {
            Real x = pco->x1v(ie+i);
            Real r_sq = SQR(x-0.5);
            u_rad_fld(k,j,ie+i) = coef*std::exp(-r_sq/(4*chi_t));
        }
      }
    }
  }
  return;
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for outflow boundary condition
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is);
        prim(IVX,k,j,is-i) = prim(IVX,k,j,is);
        prim(IVY,k,j,is-i) = prim(IVY,k,j,is);
        prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
        prim(IPR,k,j,is-i) = prim(IPR,k,j,is);
      }
    }
  }
  return;
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for outflow boundary condition
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie);
        prim(IVX,k,j,ie+i) = prim(IVX,k,j,ie);
        prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
        prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
        prim(IPR,k,j,ie+i) = prim(IPR,k,j,ie);
      }
    }
  }
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
  */
  // check input
  if (pin->GetBoolean("nrfld", "is_couple")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "is_couple must be false for this problem.";
    ATHENA_ERROR(msg);
  }

  if (!pin->GetBoolean("nrfld", "only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "only_rad must be true for this problem.";
    ATHENA_ERROR(msg);
  }

  if (pin->GetBoolean("nrfld", "cut_diff")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_diff must be false for this problem.";
    ATHENA_ERROR(msg);
  }

  if (!pin->GetBoolean("nrfld", "cut_Pnablav")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_Pnablav must be true for this problem.";
    ATHENA_ERROR(msg);
  }

  dim = pin->GetInteger("problem", "dim");
  if (dim != 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "dim should be 1.";
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


  Real const_opasity = pin->GetReal("nrfld", "const_opacity");
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  chi = c_ph_sim*mfp_sim/3.0;

  Er0 = 1e+5, rho0 = 1.0, p0 = 100.0;

  Real tau_diff = leng_unit*leng_unit*rho_unit*const_opasity/(4.0*c_ph_dim);
  init_time = tau_diff/(time_unit)*init_ratio;


  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDInnerX1);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);

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
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");

  // prfld2->EnrollOpacityFunction(NoCoupleOpacity);
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real dx1 = pcoord->dx1f(4);
  Real courant = pin->GetReal("time", "cfl_number");
  Real dt_exp = courant*dx1*std::sqrt(rho0/(gamma*p0))*time_unit;
  Real const_opasity = pin->GetReal("nrfld", "const_opacity");
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
    std::cout << "dx = " << dx1*leng_unit << " cm" << std::endl;
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
          prfld2->u_gas(k,j,i) = p0*igm1;
          Real res = Er0/(8*std::pow(M_PI*chi*init_time, 1.5))*std::exp(-r_sq/(4*chi*init_time));
          prfld2->u_rad(k,j,i) = res;
        }
      }
    }
  } else if (dim == 2) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real r_sq = SQR(pcoord->x1v(i)-0.5)+SQR(pcoord->x2v(j)-0.5);
          prfld2->u_gas(k,j,i) = p0*igm1;
          Real res = Er0/(4*M_PI*chi*init_time)*std::exp(-r_sq/(4*chi*init_time));
          prfld2->u_rad(k,j,i) = res;
        }
      }
    }
  } else if (dim == 1) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real r_sq = SQR(pcoord->x1v(i)-0.5);
          prfld2->u_gas(k,j,i) = p0*igm1;
          Real res = Er0/(2*std::sqrt(M_PI*chi*init_time))*std::exp(-r_sq/(4*chi*init_time));
          prfld2->u_rad(k,j,i) = res;
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
  Real chi_t = chi * (pmb->pmy_mesh->time+init_time);
  if (dim == 1) {
    Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real x = pmb->pcoord->x1v(i);
          Real r_sq = SQR(x-0.5);
          Real an = coef*std::exp(-r_sq/(4*chi_t));
          L1norm += std::abs(pmb->prfld2->u_rad(k,j,i)-an)/an;
        }
      }
    }
  }
  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
  L1norm /= ncells*nbtotal;
  return L1norm;
}

} // namespace
