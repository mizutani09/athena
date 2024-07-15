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
  Real a_r;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real rho_unit, x_unit, t_unit;
  Real v_unit, e_unit, sigma_unit, T_unit, a_r_unit;
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
          // dst(RadFLD::GAS,k,j,is-i-1) = eg_0;
          // dst(RadFLD::RAD,k,j,is-i-1) = Er_0;
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
          // dst(RadFLD::GAS,k,j,ie+i+1) = eg_0;
          // dst(RadFLD::RAD,k,j,ie+i+1) = Er_0;
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
          // dst(RadFLD::GAS,k,js-j-1,i) = eg_0;
          // dst(RadFLD::RAD,k,js-j-1,i) = Er_0;
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
          // dst(RadFLD::GAS,k,je+j+1,i) = eg_0;
          // dst(RadFLD::RAD,k,je+j+1,i) = Er_0;
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
          // dst(RadFLD::GAS,ks-k-1,j,i) = eg_0;
          // dst(RadFLD::RAD,ks-k-1,j,i) = Er_0;
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
          // dst(RadFLD::GAS,ke+k+1,j,i) = eg_0;
          // dst(RadFLD::RAD,ke+k+1,j,i) = Er_0;
          dst(RadFLD::GAS,ke+k+1,j,i) = dst(RadFLD::GAS,ke,j,i);
          dst(RadFLD::RAD,ke+k+1,j,i) = dst(RadFLD::RAD,ke,j,i);
        }
      }
    }
  }
  return;
}


int AMRCondition(MeshBlock *pmb) {
  if (pmb->block_size.x1min >= 0.25 && pmb->block_size.x1min <=0.251
  &&  pmb->block_size.x2min >= 0.25 && pmb->block_size.x2min <=0.251
  &&  pmb->block_size.x3min >= 0.25 && pmb->block_size.x3min <=0.251) {
    if (pmb->pmy_mesh->ncycle >= pmb->loc.level - 1)
      return 1;
  }
  return 0;
}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho_unit = pin->GetReal("problem", "rho_unit");
  rho0 = 1.0;
  x_unit = pin->GetReal("problem", "x_unit");
  r0 = 0.5;
  t_unit = pin->GetReal("problem", "t_unit");
  v_unit = x_unit/t_unit;
  e_unit = rho_unit*v_unit*v_unit;
  // Rgas in cgs
  Real Rgas = 8.31451e+7; // erg/(mol*K)
  Real mu = pin->GetOrAddReal("problem", "mu", 0.5);
  T_unit = mu/Rgas*v_unit*v_unit;
  a_r_unit = std::pow(Rgas/mu, 4)*rho_unit*std::pow(v_unit, -6);
  a_r = 7.57e-15/a_r_unit;

  // output units
  std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
  std::cout << "x_unit = " << x_unit << " cm" << std::endl;
  std::cout << "t_unit = " << t_unit << " s" << std::endl;
  std::cout << "v_unit = " << v_unit << " cm s^-1" << std::endl;
  std::cout << "e_unit = " << e_unit << " erg g^-1" << std::endl;
  std::cout << "T_unit = " << T_unit << " K" << std::endl;
  std::cout << "a_r_unit = " << a_r_unit << " erg cm^-3 K^-4" << std::endl;

  calc_in_temp = pin->GetOrAddBoolean("mgfld", "calc_in_temp", false);
  is_gauss = pin->GetOrAddBoolean("problem", "is_gauss", false);
  Real gamma = 5.0/3.0;
  Real gm1 = gamma - 1.0;
  Real igm1 = 1.0/(gamma-1.0);
  if (calc_in_temp) {
    Tg_0 = pin->GetOrAddReal("problem", "Tg_0", 1.0)/T_unit;
    Tr_0 = pin->GetOrAddReal("problem", "Tr_0", 2.0)/T_unit;
    eg_0 = rho0*Tg_0*igm1;
    Er_0 = a_r*std::pow(Tr_0,4);
  } else {
    eg_0 = pin->GetOrAddReal("problem", "eg_0", 1.0)/e_unit;
    Er_0 = pin->GetOrAddReal("problem", "Er_0", 1.0)/e_unit;
    Tg_0 = gm1*eg_0/rho0;
    Tr_0 = std::pow(Er_0/a_r,0.25);

    std::cout << "eg_0 in sim = " << eg_0 << std::endl;
    std::cout << "Er_0 in sim = " << Er_0 << std::endl;
    std::cout << "Tg_0 in sim = " << Tg_0 << std::endl;
    std::cout << "Tr_0 in sim = " << Tr_0 << std::endl;
  }
  // EnrollUserRefinementCondition(AMRCondition);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x2, FLDFixedInnerX2);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x2, FLDFixedOuterX2);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
  EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);
  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas");
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad");
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas");
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad");
  std::cout << "End of InitUserMeshData" << std::endl;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  Real r_0_sq = SQR(r0);
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
        phydro->u(IDN,k,j,i) = rho0;//rho0/std::min(r2,r0*r0);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*Tg_0*igm1;
      }
    }
  }

  if (calc_in_temp) { // for temp
    for(int k=kl; k<=ku; ++k) {
      for(int j=jl; j<=ju; ++j) {
        for(int i=il; i<=iu; ++i) {
            prfld->u(RadFLD::GAS,k,j,i) = Tg_0;
            prfld->u(RadFLD::RAD,k,j,i) = Tr_0;
        }
      }
    }
  } else { // for energy
    for(int k=kl; k<=ku; ++k) {
      Real z = pcoord->x3v(k);
      for(int j=jl; j<=ju; ++j) {
        Real y = pcoord->x2v(j);
        for(int i=il; i<=iu; ++i) {
          Real x = pcoord->x1v(i);
          prfld->u(RadFLD::GAS,k,j,i) = phydro->u(IEN,k,j,i);
          if (is_gauss) {
            Real r_sq = SQR(x)+SQR(y)+SQR(z);
            prfld->u(RadFLD::RAD,k,j,i) = Er_0 * std::exp(-r_sq/r_0_sq);
          } else {
            prfld->u(RadFLD::RAD,k,j,i) = Er_0;
          }
        }
      }
    }
  }

  return;
}




void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  sigma_unit = 1.0/x_unit;
  Real const_opacity = pin->GetOrAddReal("problem", "const_opacity", -1.0); // -1.0 for non-const
  prfld->const_opacity = const_opacity/sigma_unit;
  Real c_ph = 3.00e+10; // cm s^-1
  prfld->c_ph = c_ph/v_unit;
  prfld->a_r = a_r;
  std::cout << "opacity in sim = " << prfld->const_opacity << std::endl;
  std::cout << "c_ph in sim = " << prfld->c_ph << std::endl;
  std::cout << "a_r in sim = " << prfld->a_r << std::endl;
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gm1 = peos->GetGamma() - 1.0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // assume cal in E
        user_out_var(0,k,j,i) = prfld->u(RadFLD::GAS,k,j,i)*e_unit;
        user_out_var(1,k,j,i) = prfld->u(RadFLD::RAD,k,j,i)*e_unit;
        user_out_var(2,k,j,i) = prfld->u(RadFLD::GAS,k,j,i)*gm1/phydro->w(IDN,k,j,i)*T_unit;
        user_out_var(3,k,j,i) = std::pow(prfld->u(RadFLD::RAD,k,j,i)/a_r, 0.25)*T_unit;
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
        T += pmb->prfld->u(RadFLD::GAS,k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i);
        num++;
      }
    }
  }
  T /= num;
  return T*T_unit;
}

Real HistoryTr(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += std::pow(pmb->prfld->u(RadFLD::RAD,k,j,i)/a_r, 0.25);
        num++;
      }
    }
  }
  T /= num;
  return T*T_unit;
}

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
  return e*e_unit;
}

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
  return E*e_unit;
}

} // namespace
