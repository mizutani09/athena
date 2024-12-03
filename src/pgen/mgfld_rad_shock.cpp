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
//! \file mgfld_rad_shock.cpp
//! \brief Problem generator for radiative shock test
//! REFERENCE: B. Commerçon, et al., Astron. Astrophys. Suppl. Ser. 529, A35 (2011).
//!            for section 4.4: 1D full RHD tests: radiative shocks
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
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../rad_fld/rad_fld.hpp"
#include "../rad_fld/mg_rad_fld.hpp"


#if !MGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-mgfld)."
#endif

namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit, vel_unit;
  Real a_r_dim, Rgas, mu;
  Real a_r_sim;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  // Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real rho0, T0;
  Real p0, egas0, Er0;
  Real vel_piston;
}

void FLDInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  // for reflective boundary condition
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        dst(RadFLD::GAS,k,j,is-i) = dst(RadFLD::GAS,k,j,is+i-1);
        dst(RadFLD::RAD,k,j,is-i) = dst(RadFLD::RAD,k,j,is+i-1);
      }
    }
  }
  return;
}

// void FLDOuterFixedX1(AthenaArray<Real> &dst, Real time, int nvar,
//                     int is, int ie, int js, int je, int ks, int ke, int ngh,
//                     const MGCoordinates &coord) {
//   // for fixed boundary condition
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=1; i<=ngh; i++) {
//         dst(RadFLD::GAS,k,j,ie+i) = egas0;
//         dst(RadFLD::RAD,k,j,ie+i) = Er0;
//       }
//     }
//   }
//   return;
// }

void FLDOuterOutflowX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  // for outflow boundary condition
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
        dst(RadFLD::GAS,k,j,ie+i) = dst(RadFLD::GAS,k,j,ie);
        dst(RadFLD::RAD,k,j,ie+i) = dst(RadFLD::RAD,k,j,ie);
      }
    }
  }
  return;
}

// void FLDAdvInnerX1(MeshBlock *pmb, Coordinates *pco, FLD *prfld,
//     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &r_fld,
//     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   // for reflective boundary condition
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=1; i<=ngh; i++) {
//         r_fld(k,j,is-i) = r_fld(k,j,is+i-1);
//       }
//     }
//   }
//   return;
// }

// void FLDAdvOuterX1(MeshBlock *pmb, Coordinates *pco, FLD *prfld,
//     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &r_fld,
//     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   // for fixed boundary condition
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=1; i<=ngh; i++) {
//         r_fld(k,j,ie+i) = Er0;
//       }
//     }
//   }
//   return;
// }

// void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   // for reflective boundary condition
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=1; i<=ngh; i++) {
//         prim(IDN,k,j,is-i) = prim(IDN,k,j,is+i-1);
//         prim(IVX,k,j,is-i) = -prim(IVX,k,j,is+i-1);
//         prim(IVY,k,j,is-i) = prim(IVY,k,j,is+i-1);
//         prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is+i-1);
//         prim(IPR,k,j,is-i) = prim(IPR,k,j,is+i-1);
//       }
//     }
//   }
//   return;
// }

// void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//     Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   // for fixed boundary condition
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=1; i<=ngh; i++) {
//         prim(IDN,k,j,ie+i) = rho0;
//         prim(IVX,k,j,ie+i) = -vel_piston;
//         prim(IVY,k,j,ie+i) = 0.0;
//         prim(IVZ,k,j,ie+i) = 0.0;
//         prim(IPR,k,j,ie+i) = p0;
//       }
//     }
//   }
//   return;
// }

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
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
  Rgas = 8.31451e+7; // erg mol^-1 K^-1
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  a_r_sim = a_r_dim/(egas_unit/std::pow(T_unit, 4));

  vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  Real const_opasity = pin->GetReal("mgfld", "const_opacity");
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  rho0 = pin->GetReal("problem", "rho0") / rho_unit;
  T0 = pin->GetReal("problem", "T0") / T_unit;
  vel_piston = pin->GetReal("problem", "vel_piston") * 1e5 / vel_unit; // from km s^-1 to cm s^-1

  // std::string ix1_bc = pin->GetString("mgfld", "ix1_bc");
  // std::string ox1_bc = pin->GetString("mgfld", "ox1_bc");
  // if (ix1_bc == "user")
    EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x1, FLDInnerX1);
  // if (ox1_bc == "user")
    EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterOutflowX1);

  // ix1_bc = pin->GetString("mesh", "ix1_bc");
  // ox1_bc = pin->GetString("mesh", "ox1_bc");
  // if (ix1_bc == "user") {
  //   EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  //   EnrollUserFLDAdvBoundaryFunction(BoundaryFace::inner_x1, FLDAdvInnerX1);
  // }
  // if (ox1_bc == "user"){
  //   EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  //   EnrollUserFLDAdvBoundaryFunction(BoundaryFace::outer_x1, FLDAdvOuterX1);
  //   EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterFixedX1); // caution
  // }

  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  // EnrollUserHistoryOutput(7, HistoryL1norm, "L1norm", UserHistoryOperation::sum);
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  Real igm1 = 1.0/(peos->GetGamma()-1.0);
  p0 = rho0*T0;
  Er0 = a_r_sim*std::pow(T0, 4);
  egas0 = p0*igm1;

  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real dx1 = pcoord->dx1f(0);
  Real courant = pin->GetReal("time", "cfl_number");
  Real sound = std::sqrt(gamma*p0/rho0);
  Real dt_exp = courant*dx1/(sound+vel_piston)*time_unit;
  Real const_opasity = pin->GetReal("mgfld", "const_opacity");
  Real const_opasity_sim = const_opasity*leng_unit*rho_unit;
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  Real L = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  if (gid == 0) {
    std::cout << "rho_unit       = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit      = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit      = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit      = " << leng_unit << " cm" << std::endl;
    std::cout << "vel_unit       = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    std::cout << "T_unit         = " << T_unit << " K" << std::endl;
    std::cout << "c_ph_sim       = " << c_ph_sim << std::endl;
    std::cout << "dx_dim         = " << dx1*leng_unit << " cm" << std::endl;
    std::cout << "dt_dim         = " << dt_exp << " s" << std::endl;
    std::cout << "dt_sim         = " << dt_exp/time_unit << std::endl;
    std::cout << "rho0_sim       = " << rho0 << std::endl;
    std::cout << "T0_sim         = " << T0 << std::endl;
    std::cout << "vel_piston_sim = " << vel_piston << std::endl;
    std::cout << "p0_sim         = " << p0 << std::endl;
    std::cout << "Er0_sim        = " << Er0 << std::endl;
    Real tmp = p0/rho0*T_unit;
    std::cout << "T0_dim         = " << tmp << std::endl;
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
        Real x1 = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = -rho0*vel_piston;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = egas0 + 0.5*rho0*SQR(vel_piston);

        // for FLD
        prfld->u(RadFLD::GAS,k,j,i) = egas0;
        prfld->u(RadFLD::RAD,k,j,i) = Er0;
      }
    }
  }
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real igm1 = 1.0/gm1;
//   Real temp_coef = gm1*mu/Rgas*egas_unit/rho_unit;
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

        // for egas
        // user_out_var(0,k,j,i) = prfld->u(RadFLD::GAS,k,j,i);//*egas_unit;
        user_out_var(0,k,j,i) = phydro->w(IPR,k,j,i)*igm1;//*egas_unit;

        // for Erad
        user_out_var(1,k,j,i) = prfld->u(RadFLD::RAD,k,j,i);//*egas_unit;

        // for Tgas
        user_out_var(2,k,j,i) = phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i)*T_unit;

        // for Trad
        user_out_var(3,k,j,i) = std::pow(prfld->u(RadFLD::RAD,k,j,i)/a_r_sim, 0.25)*T_unit;
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
        T += pmb->prfld->u(RadFLD::GAS,k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit;
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
        T += std::pow(pmb->prfld->u(RadFLD::RAD,k,j,i)*egas_unit/a_r_dim, 0.25);
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
        e += pmb->prfld->u(RadFLD::GAS,k,j,i);//*vol(i);
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
        E += pmb->prfld->u(RadFLD::RAD,k,j,i);//*vol(i);
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
        aT4 += std::pow(pmb->prfld->u(RadFLD::GAS,k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit, 4);
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
        E += pmb->prfld->u(RadFLD::GAS,k,j,i)*vol(i);
        E += pmb->prfld->u(RadFLD::RAD,k,j,i)*vol(i);
      }
    }
  }
  return E*egas_unit;
}

// Real HistoryL1norm(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   Real L1norm = 0;
//   Real x_L = pmb->pmy_mesh->mesh_size.x1min - pmb->pcoord->dx1f(0)/2.0;
//   Real x_R = pmb->pmy_mesh->mesh_size.x1max + pmb->pcoord->dx1f(0)/2.0;
//   Real slope = (Er0_R-Er0_L)/(x_R-x_L);
//   Real cons = Er0_L - slope*x_L;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         Real x = pmb->pcoord->x1v(i);
//         Real an = slope*x + cons;
//         L1norm += std::abs(pmb->prfld->u(RadFLD::RAD,k,j,i) - an)/std::abs(an);
//       }
//     }
//   }
//   int nbtotal = pmb->pmy_mesh->nbtotal;
//   int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
//   L1norm /= ncells*nbtotal;
//   return L1norm;
// }

} // namespace
