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
  Real sigma_P, sigma_R;
  Real T0;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  // Real HistoryL1norm(MeshBlock *pmb, int iout);
}

void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar);


void ConstantOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD2 *prfld = pmb->prfld2;
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

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  /*
  is_couple       = true
  only_rad        = false
  cut_diff        = false
  include_radiation_force = false
  */
  // // check input
  // if (!pin->GetBoolean("fld", "is_couple")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "is_couple must be true for this problem.";
  //   ATHENA_ERROR(msg);
  // }

  // if (pin->GetBoolean("fld", "only_rad")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "only_rad must be false for this problem.";
  //   ATHENA_ERROR(msg);
  // }

  // if (pin->GetBoolean("fld", "cut_diff")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "cut_diff must be false for this problem.";
  //   ATHENA_ERROR(msg);
  // }

  // if (pin->GetBoolean("fld", "include_radiation_force")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "include_radiation_force must be false for this problem.";
  //   ATHENA_ERROR(msg);
  // }

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

  Real rho0 = pin->GetReal("problem", "rho0"); // in code unit
  Real sigma_P_sim = pin->GetReal("problem", "kappa_P")*rho0; // in code unit
  Real sigma_R_sim = pin->GetReal("problem", "kappa_R")*rho0; // in code unit
  sigma_P = sigma_P_sim/(leng_unit); // in cm^-1
  sigma_R = sigma_R_sim/(leng_unit); // in cm^-1

  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real mfp_sim = 1.0/(sigma_P)/leng_unit;
  Real tau_diff = leng_unit*leng_unit*rho_unit*sigma_P/(4.0*c_ph_dim);

  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  // EnrollUserHistoryOutput(7, HistoryL1norm, "L1norm", UserHistoryOperation::sum);

  EnrollUserExplicitSourceFunction(AddRadiativeForceAndWork);
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");

  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(2, ncells3, ncells2, ncells1);
  prfld2->EnrollOpacityFunction(ConstantOpacity);
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rho0 = pin->GetReal("problem", "rho0");
  Real p0 = pin->GetReal("problem", "p0");
  Real amp = pin->GetReal("problem", "amp");
  Real num_wave = pin->GetOrAddReal("problem", "num_wave", 1.0);
  Real wave_number = 2.0*M_PI*num_wave/(pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min);
  T0 = p0/rho0;

  std::cout << pmy_mesh->mesh_size.x1max << " " << pmy_mesh->mesh_size.x1min << std::endl;
  std::cout << pcoord->x1v(is) << " " << pcoord->x1v(ie) << std::endl;

  Real gamma = peos->GetGamma();
  Real sound = std::sqrt(gamma*p0/rho0);
  Real sound_dim = sound*(leng_unit/time_unit);
  Real igm1 = 1.0/(gamma-1.0);
  Real dx1 = pcoord->dx1f(4);
  Real courant = pin->GetReal("time", "cfl_number");
  Real dt_exp = courant*dx1/sound*time_unit;
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  Real domain_size = (pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min)* leng_unit;
  Real kappa_P_sim = pin->GetReal("problem", "kappa_P");
  Real kappa_R_sim = pin->GetReal("problem", "kappa_R");
  Real kappa_P = kappa_P_sim/(leng_unit);
  Real kappa_R = kappa_R_sim/(leng_unit);
  Real mfp_dim = 1.0/(rho0*rho_unit*kappa_P);
  Real mfp_sim = mfp_dim/leng_unit;
  Real optical_depth = domain_size/mfp_dim;
  Real tau_couple = 1.0/(c_ph_dim*sigma_P);
  Real tau_diff = domain_size*domain_size*sigma_R/c_ph_dim;
  Real tau_adv = domain_size/sound_dim;

  if (gid == 0) {
    std::stringstream msg;
    msg << ">>> Problem parameters <<<" << std::endl;
    msg << "- Units" << std::endl;
    msg << "rho_unit             = " << rho_unit << " g cm^-3" << std::endl;
    msg << "egas_unit            = " << egas_unit << " erg cm^-3" << std::endl;
    msg << "time_unit            = " << time_unit << " s" << std::endl;
    msg << "leng_unit            = " << leng_unit << " cm" << std::endl;
    msg << "vel_unit             = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    msg << "T_unit               = " << T_unit << " K" << std::endl;
    msg << std::endl;

    msg << "- Simulation parameters" << std::endl;
    msg << "c_ph_sim             = " << c_ph_sim << std::endl;
    msg << "dx_dim               = " << dx1*leng_unit << " cm" << std::endl;
    msg << "dt_dim               = " << dt_exp << " s" << std::endl;
    msg << "dx_sim               = " << dx1 << std::endl;
    msg << "dt_sim               = " << dt_exp/time_unit << std::endl;
    msg << "mfp_sim              = " << mfp_sim << std::endl;
    msg << "tau_couple           = " << tau_couple << " s" << std::endl;
    msg << "tau_diff             = " << tau_diff << " s" << std::endl;
    msg << "tau_adv              = " << tau_adv << " s" << std::endl;
    msg << "tau_couple in sim    = " << tau_couple/time_unit << std::endl;
    msg << "tau_diff in sim      = " << tau_diff/time_unit << std::endl;
    msg << "tau_adv in sim       = " << tau_adv/time_unit << std::endl;
    msg << "optical_depth        = " << optical_depth << std::endl;
    msg << "rho0                 = " << rho0*rho_unit << " g cm^-3" << std::endl;
    msg << "p0                   = " << p0*egas_unit << " erg cm^-3" << std::endl;
    msg << "T0                   = " << T0*T_unit << " K" << std::endl;
    msg << "amp                  = " << amp << std::endl;
    msg << "num_wave             = " << num_wave << std::endl;
    msg << "wave_number          = " << wave_number/(1/leng_unit) << " cm^-1" << std::endl;
    msg << "E0sim                = " << prfld2->a_r*std::pow(T0, 4) << std::endl;
    msg << std::endl;

    std::cout << msg.str();
    std::ofstream ofs("problem_parameters.txt");
    ofs << msg.str();
    ofs.close();
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
        Real x = pcoord->x1v(i);
        Real delta = amp*std::sin(wave_number*x);
        Real rho = rho0*(1.0 + delta);
        Real p = p0*(1.0 + gamma*delta);
        Real v1 = sound*delta;
        Real T = p/rho;

        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho*v1;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = p*igm1 + 0.5*rho*v1*v1;
        prfld2->u_rad(k,j,i) = prfld2->a_r*std::pow(T, 4);
      }
    }
  }

  // record initial profile
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        ruser_meshblock_data[0](0,k,j,i) = phydro->u(IDN,k,j,i);
        ruser_meshblock_data[0](1,k,j,i) = phydro->u(IEN,k,j,i);
      }
    }
  }

  return;
}

void MeshBlock::UserWorkInLoop() {
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


void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar) {
#if NRMGFLD_ENABLED
  (void)pmb; (void)time; (void)dt; (void)prim; (void)prim_scalar;
  (void)bcc; (void)cons; (void)cons_scalar;
  return;
#endif
  std::cout << "Add radiative force and work" << std::endl;
  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0;
  Real igm1 = 1.0 / gm1;

  // if ((pmb->iuser_meshblock_data[TSTEP_COUNTER](0) + 1) % rk_cycle == 0) {
    int il = pmb->is - NGHOST, iu = pmb->ie + NGHOST;
    int jl = pmb->js - NGHOST, ju = pmb->je + NGHOST;
    int kl = pmb->ks - NGHOST, ku = pmb->ke + NGHOST;
    Real idx = 1.0/pmb->pcoord->dx1f(pmb->is);
    Real hidx = 0.5*idx;

    FLD2 *prfld = pmb->prfld2;
    AthenaArray<Real> &fld_u = prfld->u_rad;

    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          Real lambda;
          Real dEr[3];
          dEr[0] = hidx*(fld_u(k,j,i+1) - fld_u(k,j,i-1));
          dEr[1] = hidx*(fld_u(k,j+1,i) - fld_u(k,j-1,i));
          dEr[2] = hidx*(fld_u(k+1,j,i) - fld_u(k-1,j,i));

          Real gradE = std::sqrt(SQR(dEr[0]) + SQR(dEr[1]) + SQR(dEr[2]));
          Real R = gradE/(prfld->sigma_r(k,j,i)*fld_u(k,j,i)); // center
          lambda = RadFLD2::FluxLimiter(R, prfld->fixed_flux_limitter);

          cons(IM1,k,j,i) += -lambda*dt*dEr[0];
          cons(IM2,k,j,i) += -lambda*dt*dEr[1];
          cons(IM3,k,j,i) += -lambda*dt*dEr[2];
          Real nablaE_v = dEr[0]*prim(IVX,k,j,i) + dEr[1]*prim(IVY,k,j,i) + dEr[2]*prim(IVZ,k,j,i);
          cons(IEN,k,j,i) += -lambda*dt*nablaE_v;
        }
      }
    }
  // }
  // pmb->iuser_meshblock_data[TSTEP_COUNTER](0)++;
  // pmb->iuser_meshblock_data[TSTEP_COUNTER](0) %= rk_cycle;
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

// Real HistoryL1norm(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   Real L1norm = 0;
//   Real chi_t = chi * (pmb->pmy_mesh->time+init_time);
//   if (dim == 1) {
//     Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
//     for (int k=ks; k<=ke; k++) {
//       for (int j=js; j<=je; j++) {
//         for (int i=is; i<=ie; i++) {
//           Real x = pmb->pcoord->x1v(i);
//           Real r_sq = SQR(x-0.5);
//           Real an = coef*std::exp(-r_sq/(4*chi_t));
//           L1norm += std::abs(pmb->prfld2->u_rad(k,j,i)-an)/an;
//         }
//       }
//     }
//   }
//   int nbtotal = pmb->pmy_mesh->nbtotal;
//   int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
//   L1norm /= ncells*nbtotal;
//   return L1norm;
// }

} // namespace
