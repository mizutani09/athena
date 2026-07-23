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

// NRMGFLD_ENABLED must be true
#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
  Real Tr_0, Tg_0, eg_0, Er_0;
  bool calc_in_temp;
  bool is_gauss;
  Real r0, rho0;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit;
  Real a_r_dim, Rgas, mu;
  Real dt0_phys, dt_growth;
  Real GrowingTimeStep(MeshBlock *pmb);
}

// void FLDFixedInnerX1(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
//                      const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
//                      Real time, Real dt,
//                      int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=0; i<ngh; i++) {
//         pfld->u_gas(k,j,is-i-1) = eg_0; //should be in arguments
//         u_rad_fld(k,j,is-i-1) = Er_0;
//       }
//     }
//   }
//   return;
// }

// void FLDFixedOuterX1(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
//                      const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
//                      Real time, Real dt,
//                      int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=0; i<ngh; i++) {
//         pfld->u_gas(k,j,ie+i+1) = eg_0;
//         u_rad_fld(k,j,ie+i+1) = Er_0;
//       }
//     }
//   }
//   return;
// }

// void FLDFixedInnerX2(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
//                      const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
//                      Real time, Real dt,
//                      int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   for (int k=ks; k<=ke; k++) {
//     for (int j=0; j<ngh; j++) {
//       for (int i=is; i<=ie; i++) {
//         pfld->u_gas(k,js-j-1,i) = eg_0;
//         u_rad_fld(k,js-j-1,i) = Er_0;
//       }
//     }
//   }
//   return;
// }

// void FLDFixedOuterX2(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
//                      const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
//                      Real time, Real dt,
//                      int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   for (int k=ks; k<=ke; k++) {
//     for (int j=0; j<ngh; j++) {
//       for (int i=is; i<=ie; i++) {
//         pfld->u_gas(k,je+j+1,i) = eg_0;
//         u_rad_fld(k,je+j+1,i) = Er_0;
//       }
//     }
//   }
//   return;
// }

// void FLDFixedInnerX3(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
//                      const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
//                      Real time, Real dt,
//                      int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   for (int k=0; k<ngh; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         pfld->u_gas(ks-k-1,j,i) = eg_0;
//         u_rad_fld(ks-k-1,j,i) = Er_0;
//       }
//     }
//   }
//   return;
// }

// void FLDFixedOuterX3(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
//                      const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
//                      Real time, Real dt,
//                      int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   for (int k=0; k<ngh; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         pfld->u_gas(ke+k+1,j,i) = eg_0;
//         u_rad_fld(ke+k+1,j,i) = Er_0;
//       }
//     }
//   }
//   return;
// }

// void HydroFixedInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   Real gm1 = pmb->peos->GetGamma() - 1.0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=0; i<ngh; i++) {
//         prim(IDN,k,j,is-i-1) = rho0;
//         prim(IVX,k,j,is-i-1) = 0.0;
//         prim(IVY,k,j,is-i-1) = 0.0;
//         prim(IVZ,k,j,is-i-1) = 0.0;
//         prim(IPR,k,j,is-i-1) = (eg_0)*gm1;
//       }
//     }
//   }
//   return;
// }

// void HydroFixedOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   Real gm1 = pmb->peos->GetGamma() - 1.0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=0; i<ngh; i++) {
//         prim(IDN,k,j,ie+i+1) = rho0;
//         prim(IVX,k,j,ie+i+1) = 0.0;
//         prim(IVY,k,j,ie+i+1) = 0.0;
//         prim(IVZ,k,j,ie+i+1) = 0.0;
//         prim(IPR,k,j,ie+i+1) = (eg_0)*gm1;
//       }
//     }
//   }
//   return;
// }

// void HydroFixedInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   Real gm1 = pmb->peos->GetGamma() - 1.0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=0; j<=ngh; j++) {
//       for (int i=is; i<ie; i++) {
//         prim(IDN,k,js-j-1,i) = rho0;
//         prim(IVX,k,js-j-1,i) = 0.0;
//         prim(IVY,k,js-j-1,i) = 0.0;
//         prim(IVZ,k,js-j-1,i) = 0.0;
//         prim(IPR,k,js-j-1,i) = (eg_0)*gm1;
//       }
//     }
//   }
//   return;
// }

// void HydroFixedOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   Real gm1 = pmb->peos->GetGamma() - 1.0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=0; j<=ngh; j++) {
//       for (int i=is; i<ie; i++) {
//         prim(IDN,k,je+j+1,i) = rho0;
//         prim(IVX,k,je+j+1,i) = 0.0;
//         prim(IVY,k,je+j+1,i) = 0.0;
//         prim(IVZ,k,je+j+1,i) = 0.0;
//         prim(IPR,k,je+j+1,i) = (eg_0)*gm1;
//       }
//     }
//   }
//   return;
// }

// void HydroFixedInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   Real gm1 = pmb->peos->GetGamma() - 1.0;
//   for (int k=0; k<=ngh; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<ie; i++) {
//         prim(IDN,ks-k-1,j,i) = rho0;
//         prim(IVX,ks-k-1,j,i) = 0.0;
//         prim(IVY,ks-k-1,j,i) = 0.0;
//         prim(IVZ,ks-k-1,j,i) = 0.0;
//         prim(IPR,ks-k-1,j,i) = (eg_0)*gm1;
//       }
//     }
//   }
//   return;
// }

// void HydroFixedOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh) {
//   Real gm1 = pmb->peos->GetGamma() - 1.0;
//   for (int k=0; k<=ngh; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<ie; i++) {
//         prim(IDN,ke+k+1,j,i) = rho0;
//         prim(IVX,ke+k+1,j,i) = 0.0;
//         prim(IVY,ke+k+1,j,i) = 0.0;
//         prim(IVZ,ke+k+1,j,i) = 0.0;
//         prim(IPR,ke+k+1,j,i) = (eg_0)*gm1;
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
  /*
  is_couple       = true
  only_rad        = true
  cut_diff        = true
  include_radiation_force = false
  */
  // check input
  if (!pin->GetBoolean("fld", "is_couple")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "is_couple must be true for this problem.";
    ATHENA_ERROR(msg);
  }

  if (!pin->GetBoolean("fld", "only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "only_rad must be true for this problem.";
    ATHENA_ERROR(msg);
  }

  if (!pin->GetBoolean("fld", "cut_diff")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_diff must be true for this problem.";
    ATHENA_ERROR(msg);
  }

  if (pin->GetBoolean("fld", "include_radiation_force")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "include_radiation_force must be false for this problem.";
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

  dt0_phys = pin->GetOrAddReal("problem", "dt0", 1.0e-11);
  dt_growth = pin->GetOrAddReal("problem", "dt_growth", 1.05);
  if (dt0_phys <= 0.0 || dt_growth <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "problem/dt0 and problem/dt_growth must both be positive.";
    ATHENA_ERROR(msg);
  }

  calc_in_temp = pin->GetOrAddBoolean("fld", "calc_in_temp", false);
  if (calc_in_temp) {
    // raise error
    std::stringstream msg;
    msg << "calc_in_temp is not supported in this version.";
    ATHENA_ERROR(msg);
  }

  // EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroFixedInnerX1);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroFixedOuterX1);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x2, HydroFixedInnerX2);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x2, HydroFixedOuterX2);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroFixedInnerX3);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroFixedOuterX3);

  // EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
  // EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
  // EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x2, FLDFixedInnerX2);
  // EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x2, FLDFixedOuterX2);
  // EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
  // EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);

  EnrollUserTimeStepFunction(GrowingTimeStep);

  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
}

namespace {
Real GrowingTimeStep(MeshBlock *pmb) {
  // Athena++ starts ncycle from zero, so the first step uses dt0_phys.
  const Real dt_phys = dt0_phys
                       * std::pow(dt_growth,
                                  static_cast<Real>(pmb->pmy_mesh->ncycle));
  return dt_phys/time_unit;
}
}  // namespace


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real igm1 = 1.0/gm1;
  eg_0 = pin->GetReal("problem", "eg_0")/egas_unit;
  Er_0 = pin->GetReal("problem", "Er_0")/egas_unit;
  rho0 = pin->GetReal("problem", "rho0")/rho_unit;
  Tg_0 = gm1*eg_0/rho0;
  Tr_0 = std::pow(Er_0*egas_unit/a_r_dim,0.25)/T_unit;

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
  if (gid == 0) {
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    Real sound = std::sqrt(Tg_0*peos->GetGamma());
    Real dx = pcoord->dx1f(0);
    Real dt_exp = dx / sound * pmy_mesh->cfl_number;
    std::cout << "sound = " << sound << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dt_exp = " << dt_exp << std::endl;


    std::cout << "eg_0 in sim = " << eg_0 << std::endl;
    std::cout << "Er_0 in sim = " << Er_0 << std::endl;
    std::cout << "Tg_0 in sim = " << Tg_0 << " " << Tg_0 * T_unit << std::endl;
    std::cout << "Tr_0 in sim = " << Tr_0 << " " << Tr_0 * T_unit << std::endl;
  }
  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;

  for(int k=kl; k<=ku; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=il; i<=iu; ++i) {
        Real x1 = pcoord->x1v(i);
        Real r2 = SQR(x1)+SQR(x2)+SQR(x3);
        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = eg_0;
      }
    }
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
      for(int i=il; i<=iu; ++i) {
        prfld->u_gas(k,j,i) = eg_0;
        prfld->u_rad(k,j,i) = Er_0;
      }
    }
  }
  std::cout << "ProblemGenerator finished" << std::endl;
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
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        e += pmb->prfld->u_gas(k,j,i);
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
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        E += pmb->prfld->u_rad(k,j,i);
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
