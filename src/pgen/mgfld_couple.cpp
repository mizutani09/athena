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
}

void FLDFixedInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  if (calc_in_temp) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++) {
          dst(RadFLD::GAS,k,j,is-i-1) = Tg_0;
          dst(RadFLD::RAD,k,j,is-i-1) = Tr_0;
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++) {
          dst(RadFLD::GAS,k,j,is-i-1) = dst(RadFLD::GAS,k,j,is);
          dst(RadFLD::RAD,k,j,is-i-1) = dst(RadFLD::RAD,k,j,is);
        }
      }
    }
  }
  return;
}

void FLDFixedOuterX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  if (calc_in_temp) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++) {
          dst(RadFLD::GAS,k,j,ie+i+1) = Tg_0;
          dst(RadFLD::RAD,k,j,ie+i+1) = Tr_0;
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++) {
          dst(RadFLD::GAS,k,j,ie+i+1) = dst(RadFLD::GAS,k,j,ie);
          dst(RadFLD::RAD,k,j,ie+i+1) = dst(RadFLD::RAD,k,j,ie);
        }
      }
    }
  }
  return;
}

void FLDFixedInnerX2(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  if (calc_in_temp) {
    for (int k=ks; k<=ke; k++) {
      for (int j=0; j<ngh; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,k,js-j-1,i) = Tg_0;
          dst(RadFLD::RAD,k,js-j-1,i) = Tr_0;
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; k++) {
      for (int j=0; j<ngh; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,k,js-j-1,i) = dst(RadFLD::GAS,k,js,i);
          dst(RadFLD::RAD,k,js-j-1,i) = dst(RadFLD::RAD,k,js,i);
        }
      }
    }
  }
  return;
}

void FLDFixedOuterX2(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  if (calc_in_temp) {
    for (int k=ks; k<=ke; k++) {
      for (int j=0; j<ngh; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,k,je+j+1,i) = Tg_0;
          dst(RadFLD::RAD,k,je+j+1,i) = Tr_0;
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; k++) {
      for (int j=0; j<ngh; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,k,je+j+1,i) = dst(RadFLD::GAS,k,je,i);
          dst(RadFLD::RAD,k,je+j+1,i) = dst(RadFLD::RAD,k,je,i);
        }
      }
    }
  }
  return;
}

void FLDFixedInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  if (calc_in_temp) {
    for (int k=0; k<ngh; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,ks-k-1,j,i) = Tg_0;
          dst(RadFLD::RAD,ks-k-1,j,i) = Tr_0;
        }
      }
    }
  } else {
    for (int k=0; k<ngh; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,ks-k-1,j,i) = dst(RadFLD::GAS,ks,j,i);
          dst(RadFLD::RAD,ks-k-1,j,i) = dst(RadFLD::RAD,ks,j,i);
        }
      }
    }
  }
  return;
}

void FLDFixedOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  if (calc_in_temp) {
    for (int k=0; k<ngh; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,ke+k+1,j,i) = Tg_0;
          dst(RadFLD::RAD,ke+k+1,j,i) = Tr_0;
        }
      }
    }
  } else {
    for (int k=0; k<ngh; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          dst(RadFLD::GAS,ke+k+1,j,i) = dst(RadFLD::GAS,ke,j,i);
          dst(RadFLD::RAD,ke+k+1,j,i) = dst(RadFLD::RAD,ke,j,i);
        }
      }
    }
  }
  return;
}


// int AMRCondition(MeshBlock *pmb) {
//   if (pmb->block_size.x1min >= 0.25 && pmb->block_size.x1min <=0.251
//   &&  pmb->block_size.x2min >= 0.25 && pmb->block_size.x2min <=0.251
//   &&  pmb->block_size.x3min >= 0.25 && pmb->block_size.x3min <=0.251) {
//     if (pmb->pmy_mesh->ncycle >= pmb->loc.level - 1)
//       return 1;
//   }
//   return 0;
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
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4

  Real vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  calc_in_temp = pin->GetOrAddBoolean("mgfld", "calc_in_temp", false);
  if (calc_in_temp) {
    // raise error
    std::stringstream msg;
    msg << "calc_in_temp is not supported in this version.";
    ATHENA_ERROR(msg);
  }

  // EnrollUserRefinementCondition(AMRCondition);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x2, FLDFixedInnerX2);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x2, FLDFixedOuterX2);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);
  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  Real igm1 = 1.0/gm1;
  eg_0 = pin->GetReal("problem", "eg_0")/egas_unit;
  Er_0 = pin->GetReal("problem", "Er_0")/egas_unit;
  rho0 = pin->GetReal("problem", "rho0")/rho_unit;
  Tg_0 = gm1*eg_0/rho0;
  Tr_0 = std::pow(Er_0*egas_unit/a_r_dim,0.25)/T_unit;
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
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*Tg_0*igm1;
      }
    }
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
      for(int i=il; i<=iu; ++i) {
        prfld->u(RadFLD::GAS,k,j,i) = phydro->u(IEN,k,j,i);
        prfld->u(RadFLD::RAD,k,j,i) = Er_0;
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
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        e += pmb->prfld->u(RadFLD::GAS,k,j,i);
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
        E += pmb->prfld->u(RadFLD::RAD,k,j,i);
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

} // namespace
