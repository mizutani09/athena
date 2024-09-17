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
#include "../rad_fld/rad_fld.hpp"
#include "../rad_fld/mg_rad_fld.hpp"


#if !MGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-mgfld)."
#endif

namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit;
  Real a_r_dim, Rgas, mu;
  int dim;
  Real init_ratio;
}

void FLDFixedInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
      for (int i=0; i<ngh; i++) {
          // dst(RadFLD::GAS,k,j,is-i-1) = eg_0;
          // dst(RadFLD::RAD,k,j,is-i-1) = Er_0;
          dst(RadFLD::GAS,k,j,is-i-1) = dst(RadFLD::GAS,k,j,is);
          dst(RadFLD::RAD,k,j,is-i-1) = dst(RadFLD::RAD,k,j,is);
      }
      }
  }
  return;
}

void FLDFixedOuterX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=0; i<ngh; i++) {
        // dst(RadFLD::GAS,k,j,ie+i+1) = eg_0;
        // dst(RadFLD::RAD,k,j,ie+i+1) = Er_0;
        dst(RadFLD::GAS,k,j,ie+i+1) = dst(RadFLD::GAS,k,j,ie);
        dst(RadFLD::RAD,k,j,ie+i+1) = dst(RadFLD::RAD,k,j,ie);
      }
    }
  }
  return;
}

void FLDFixedInnerX2(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=ks; k<=ke; k++) {
    for (int j=0; j<ngh; j++) {
      for (int i=is; i<=ie; i++) {
        // dst(RadFLD::GAS,k,js-j-1,i) = eg_0;
        // dst(RadFLD::RAD,k,js-j-1,i) = Er_0;
        dst(RadFLD::GAS,k,js-j-1,i) = dst(RadFLD::GAS,k,js,i);
        dst(RadFLD::RAD,k,js-j-1,i) = dst(RadFLD::RAD,k,js,i);
      }
    }
  }
  return;
}

void FLDFixedOuterX2(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=ks; k<=ke; k++) {
    for (int j=0; j<ngh; j++) {
      for (int i=is; i<=ie; i++) {
        // dst(RadFLD::GAS,k,je+j+1,i) = eg_0;
        // dst(RadFLD::RAD,k,je+j+1,i) = Er_0;
        dst(RadFLD::GAS,k,je+j+1,i) = dst(RadFLD::GAS,k,je,i);
        dst(RadFLD::RAD,k,je+j+1,i) = dst(RadFLD::RAD,k,je,i);
      }
    }
  }
  return;
}

void FLDFixedInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=0; k<ngh; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // dst(RadFLD::GAS,ks-k-1,j,i) = eg_0;
        // dst(RadFLD::RAD,ks-k-1,j,i) = Er_0;
        dst(RadFLD::GAS,ks-k-1,j,i) = dst(RadFLD::GAS,ks,j,i);
        dst(RadFLD::RAD,ks-k-1,j,i) = dst(RadFLD::RAD,ks,j,i);
      }
    }
  }
  return;
}

void FLDFixedOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=0; k<ngh; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // dst(RadFLD::GAS,ke+k+1,j,i) = eg_0;
        // dst(RadFLD::RAD,ke+k+1,j,i) = Er_0;
        dst(RadFLD::GAS,ke+k+1,j,i) = dst(RadFLD::GAS,ke,j,i);
        dst(RadFLD::RAD,ke+k+1,j,i) = dst(RadFLD::RAD,ke,j,i);
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
  dim = pin->GetInteger("problem", "dim");
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
  // Rgas in cgs
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  std::cout << "T_unit = " << T_unit << " K" << std::endl;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4

  Real vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;
  std::cout << "rho_unit = " << rho_unit << " g/cm^3" << std::endl;
  std::cout << "egas_unit = " << egas_unit << " erg/cm^3" << std::endl;
  std::cout << "time_unit = " << time_unit << " s" << std::endl;
  std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;


  // EnrollUserRefinementCondition(AMRCondition);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x2, FLDFixedInnerX2);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x2, FLDFixedOuterX2);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);
  // AllocateUserHistoryOutput(1);
  // EnrollUserHistoryOutput(0, HistoryRtime, "Rtime", UserHistoryOperation::max);
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rho0 = 1.0, p0 = 1.0, Er0 = 1e+5;
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real dx1 = pcoord->dx1f(4);
  Real courant = pin->GetReal("time", "cfl_number");
  Real dt_exp = courant*dx1*std::sqrt(rho0/(gamma*p0))*time_unit;
  Real const_opasity = pin->GetReal("mgfld", "const_opacity");
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real tau_diff = leng_unit*leng_unit*rho_unit*const_opasity/(4.0*c_ph_dim);
  Real tau_diff_dt = tau_diff/dt_exp;

  if (gid == 0) {
    std::cout << "dx = " << dx1*leng_unit << " cm" << std::endl;
    std::cout << "dt = " << dt_exp << " s" << std::endl;
    std::cout << "tau_diff = " << tau_diff << " s" << std::endl;
    std::cout << "tau_diff/dt = " << tau_diff_dt << std::endl;

    std::cout << "tmp = " << const_opasity*rho_unit/(1/leng_unit) << std::endl;
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
          phydro->u(IEN,k,j,i) = p0*igm1;
      }
    }
  }

  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  Real chi = c_ph_sim*mfp_sim/3.0;
  std::cout << "chi = " << chi*leng_unit*leng_unit/time_unit << " cm^2 s^-1" << std::endl;
  Real t_diff = tau_diff/(time_unit)*init_ratio;
  std::cout << "init_time = " << t_diff * time_unit << " s" << std::endl;
  if (dim == 3) {
    for(int k=kl; k<=ku; ++k) {
      Real z = pcoord->x3v(k);
      for(int j=jl; j<=ju; ++j) {
        Real y = pcoord->x2v(j);
        for(int i=il; i<=iu; ++i) {
          Real x = pcoord->x1v(i);
          Real r_sq = SQR(x-0.5)+SQR(y-0.5)+SQR(z-0.5);
          prfld->u(RadFLD::GAS,k,j,i) = phydro->u(IEN,k,j,i);
          Real res = Er0/(8*std::pow(M_PI*chi*t_diff, 1.5))*std::exp(-r_sq/(4*chi*t_diff));
          prfld->u(RadFLD::RAD,k,j,i) = res;
        }
      }
    }
  } else if (dim == 2) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real r_sq = SQR(pcoord->x1v(i)-0.5)+SQR(pcoord->x2v(j)-0.5);
          prfld->u(RadFLD::GAS,k,j,i) = phydro->u(IEN,k,j,i);
          Real res = Er0/(4*M_PI*chi*t_diff)*std::exp(-r_sq/(4*chi*t_diff));
          prfld->u(RadFLD::RAD,k,j,i) = res;
        }
      }
    }
  } else if (dim == 1) {
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real r_sq = SQR(pcoord->x1v(i)-0.5);
          prfld->u(RadFLD::GAS,k,j,i) = phydro->u(IEN,k,j,i);
          Real res = Er0/(2*std::sqrt(M_PI*chi*t_diff))*std::exp(-r_sq/(4*chi*t_diff));
          prfld->u(RadFLD::RAD,k,j,i) = res;
        }
      }
    }
  }

  return;
}




void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");
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
        user_out_var(0,k,j,i) = prfld->u(RadFLD::GAS,k,j,i)*egas_unit;
        user_out_var(1,k,j,i) = prfld->u(RadFLD::RAD,k,j,i)*egas_unit;
        user_out_var(2,k,j,i) = prfld->u(RadFLD::GAS,k,j,i)/phydro->w(IDN,k,j,i)*temp_coef;
        user_out_var(3,k,j,i) = std::pow(prfld->u(RadFLD::RAD,k,j,i)*egas_unit/a_r_dim, 0.25);
      }
    }
  }
  return;
}

// namespace {

// /*
// Real HistoryTg(MeshBlock *pmb, int iout) {
//   const Real gm1  = pmb->peos->GetGamma() - 1.0;
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   int num = 0;
//   Real T = 0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         T += pmb->prfld->u(RadFLD::GAS,k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit;
//         num++;
//       }
//     }
//   }
//   T /= num;
//   return T;
// }

// Real HistoryTr(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   int num = 0;
//   Real T = 0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         T += std::pow(pmb->prfld->u(RadFLD::RAD,k,j,i)*egas_unit/a_r_dim, 0.25);
//         num++;
//       }
//     }
//   }
//   T /= num;
//   return T;
// }

// // caution! this is for a mean of gas energy density.
// Real HistoryEg(MeshBlock *pmb, int iout) {
//   const Real gm1  = pmb->peos->GetGamma() - 1.0;
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   int num = 0;
//   Real e = 0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         e += pmb->prfld->u(RadFLD::GAS,k,j,i);
//         num++;
//       }
//     }
//   }
//   e /= num;
//   return e*egas_unit;
// }

// // caution! this is for a mean of radiation energy density.
// Real HistoryEr(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   int num = 0;
//   Real E = 0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         E += pmb->prfld->u(RadFLD::RAD,k,j,i);
//         num++;
//       }
//     }
//   }
//   E /= num;
//   return E*egas_unit;
// }

// Real HistoryaTg4(MeshBlock *pmb, int iout) {
//   const Real gm1  = pmb->peos->GetGamma() - 1.0;
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   int num = 0;
//   Real aT4 = 0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         aT4 += std::pow(pmb->prfld->u(RadFLD::GAS,k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit, 4);
//         num++;
//       }
//     }
//   }
//   aT4 *= a_r_dim;
//   aT4 /= num;
//   return aT4;
// }

// Real HistoryRtime(MeshBlock *pmb, int iout) {
//   return pmb->pmy_mesh->time*time_unit;
// }

// // caution! this is for a sum of all energy.
// Real HistoryEall(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   AthenaArray<Real> vol;
//   vol.NewAthenaArray((ie-is)+2*NGHOST);
//   int num = 0;
//   Real E = 0;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       pmb->pcoord->CellVolume(k, j, is, ie, vol);
//       for (int i=is; i<=ie; i++) {
//         E += pmb->prfld->u(RadFLD::GAS,k,j,i)*vol(i);
//         E += pmb->prfld->u(RadFLD::RAD,k,j,i)*vol(i);
//       }
//     }
//   }
//   return E*egas_unit;
// }

// */
// } // namespace
