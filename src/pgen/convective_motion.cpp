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
//! \file convective_motion.cpp
//! \brief Problem generator for convection in the Sun
//! REFERENCE: M. Rempel, Numerical simulations of quiet sun magnetism: On the contribution from a small-scale dynamo. Astrophys. J. 789, 22 (2014).

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
#include "../fld/fld.hpp"
#include "../fld/opacity_table.hpp"
#include "../mg_fld/mg_rad_fld.hpp"


#if !MGFLD_ENABLED && !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-mgfld or -nrmgfld)."
#endif


namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit, grav_unit;
  Real T_unit, time_unit, vel_unit, opacity_unit;
  Real a_r_dim, Rgas, mu;
  Real a_r_sim;
  Real dt_initial;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  // Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real poly_n;
  Real grav_acc;
  // Real rho_bottom, T_bottom;
  // Real press_bottom, egas_bottom, Er_bottom;
  // Real rho_top, T_top;
  // Real press_top, egas_top, Er_top;
  Real z_ref, rho_ref, T_ref;
  Real igm1;
  Real sigma_P, sigma_R;
  int rk_cycle;
  bool use_opacity_table;
  UserOpacityTable *puser_table = nullptr;
  int iuov_max;

  // for fixed boundaries
  enum BIDX {
    RHO=0,
    PRESS=1,
    EGAS=2,
    ERAD=3,
    NBIDX,
  };
  Real bottom_boundary[NGHOST*NBIDX];
  Real top_boundary[NGHOST*NBIDX];

  // for ruser_meshblock
  int UBTOP = 0;
  int UBBOTTOM = 1;

  // for iuser_meshblock
  int TSTEP_COUNTER = 0;
}

void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar);

void FLDFixedInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  for (int k=1; k<=ngh; k++) {
    Real z = coord.x3v(ks-k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T_bottom = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real rho_bottom = rho_ref*std::pow(tmp, poly_n);

    Real press_bottom = rho_bottom*T_bottom;
    Real Er_bottom = a_r_sim*std::pow(T_bottom, 4);
    Real egas_bottom = press_bottom*igm1;
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dst(RadFLD::GAS,ks-k,j,i) = egas_bottom;
        dst(RadFLD::RAD,ks-k,j,i) = Er_bottom;
      }
    }
  }
  return;
}

void FLDFixedOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  // for fixed boundary condition
  for (int k=1; k<=ngh; k++) {
    Real z = coord.x3v(ke+k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T_top = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real rho_top = rho_ref*std::pow(tmp, poly_n);

    Real press_top = rho_top*T_top;
    Real Er_top = a_r_sim*std::pow(T_top, 4);
    Real egas_top = press_top*igm1;
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dst(RadFLD::GAS,ke+k,j,i) = egas_top;
        dst(RadFLD::RAD,ke+k,j,i) = Er_top;
      }
    }
  }
  return;
}

void FLDAdvFixedInnerX3(MeshBlock *pmb, Coordinates *pco, FLD2 *prfld,
    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &r_fld,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  for (int k=1; k<=ngh; k++) {
    Real z = pco->x3v(ks-k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T_bottom = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real Er_bottom = a_r_sim*std::pow(T_bottom, 4);
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        r_fld(ks-k,j,i) = Er_bottom;
      }
    }
  }
  return;
}

void FLDAdvFixedOuterX3(MeshBlock *pmb, Coordinates *pco, FLD2 *prfld,
    const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &r_fld,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  for (int k=1; k<=ngh; k++) {
    Real z = pco->x3v(ke+k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T_top = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real Er_top = a_r_sim*std::pow(T_top, 4);
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        r_fld(ke+k,j,i) = Er_top;
      }
    }
  }
  return;
}

void HydroFixedInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  for (int k=1; k<=ngh; k++) {
    Real z = pco->x3v(ks-k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T_bottom = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real rho_bottom = rho_ref*std::pow(tmp, poly_n);
    Real press_bottom = rho_bottom*T_bottom;
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        prim(IDN,ks-k,j,i) = rho_bottom;
        prim(IVX,ks-k,j,i) = 0.0;
        prim(IVY,ks-k,j,i) = 0.0;
        prim(IVZ,ks-k,j,i) = 0.0;
        prim(IPR,ks-k,j,i) = press_bottom;
      }
    }
  }
  return;
}

void HydroFixedOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // for fixed boundary condition
  for (int k=1; k<=ngh; k++) {
    Real z = pco->x3v(ke+k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T_top = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real rho_top = rho_ref*std::pow(tmp, poly_n);
    Real press_top = rho_top*T_top;
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        prim(IDN,ke+k,j,i) = rho_top;
        prim(IVX,ke+k,j,i) = 0.0;
        prim(IVY,ke+k,j,i) = 0.0;
        prim(IVZ,ke+k,j,i) = 0.0;
        prim(IPR,ke+k,j,i) = press_top;
      }
    }
  }
//   Real gamma_ad = 5.0/3.0;
//   for (int k=1; k<=ngh; ++k) {
//     Real z = pco->x3v(ku+k); // CAUTION!!
//     for (int j=jl; j<=ju; ++j) {
//       Real y = pco->x2v(j);
// #pragma omp simd
//       for (int i=il; i<=iu; ++i) {
//         Real x = pco->x1v(i);
//         Real r = std::sqrt(x*x + y*y + z*z);
//         Real vin_lim = -vel_lim; // < 0
//         // diode boundary
//         if (prim(IVZ,ku,j,i) > vin_lim) { // outflow
//           prim(IDN,ku+k,j,i) = prim(IDN,ku,j,i);
//           prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
//           prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
//           prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
//           prim(IPR,ku+k,j,i) = prim(IPR,ku,j,i);
//         } else {
//           // restore the initial state about rho
//           Real a = GM / tem_out / detailed_scale;
//           Real rho_res = pmb->ruser_meshblock_data[RHOREF](0) * std::exp(a*(detailed_scale/r-1.0));
//           // Real pres_res = rho_res * tem_out / (gamma - 1.0);
//           prim(IDN,ku+k,j,i) = rho_res;
//           prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
//           prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
//           prim(IVZ,ku+k,j,i) = vin_lim;
//           prim(IPR,ku+k,j,i) = prim(IPR,ku,j,i);
//           // prim(IPR,ku+k,j,i) = pres_res;
//         }
//       }
//     }
//   }

  return;
}

void GetOpacityFromUserTable(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  std::cout << "Updating opacity using table..." << std::endl;
  FLD2 *prfld = pmb->prfld2;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  // int il=pmb->is, iu=pmb->ie;
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
        Real press = prim(IPR,k,j,i);
        Real temp = press/rho;
        Real rho_phys = rho*rho_unit;
        Real temp_phys = temp*T_unit;
        prfld->sigma_p(k,j,i) =
            puser_table->GetOpacity(RadFLD2::SIGMA_P, rho_phys, temp_phys)/opacity_unit*rho;
        prfld->sigma_r(k,j,i) =
            puser_table->GetOpacity(RadFLD2::SIGMA_R, rho_phys, temp_phys)/opacity_unit*rho;
      }
    }
  }
  std::cout << "Opacity updated." << std::endl;
}

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
  // Real fixed_flux_limitter = pin->GetOrAddBoolean("mgfld", "fixed_flux_limitter", false);
  // if (!fixed_flux_limitter) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "fixed_flux_limitter must be used in this problem." << std::endl;
  //   msg << "Please set fixed_flux_limitter = true in block 'mgfld'.";
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
  // Rgas in cgs
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  a_r_sim = a_r_dim/(egas_unit/std::pow(T_unit, 4));

  vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;
  grav_unit = vel_unit / time_unit;
  opacity_unit = 1.0/(rho_unit*leng_unit); // opacity unit in cm^2/g

  // Real const_opasity = pin->GetReal("fld", "const_opacity");
  // sigma_P = pin->GetReal("fld", "const_opacity_P") * (leng_unit);
  // sigma_R = pin->GetReal("fld", "const_opacity_R") * (leng_unit);
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  poly_n = pin->GetReal("problem", "poly_n");
  rho_ref = pin->GetReal("problem", "rho_top") / rho_unit;
  T_ref = pin->GetReal("problem", "T_top") / T_unit;
  grav_acc = pin->GetReal("problem", "grav_acc") / grav_unit;
  z_ref = mesh_size.x3max; // reference is top

  std::string ix3_bc = pin->GetString("mgfld", "ix3_bc");
  std::string ox3_bc = pin->GetString("mgfld", "ox3_bc");
  if (ix3_bc == "user") EnrollUserMGFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
  if (ox3_bc == "user") EnrollUserMGFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);

  ix3_bc = pin->GetString("mesh", "ix3_bc");
  ox3_bc = pin->GetString("mesh", "ox3_bc");
  if (ix3_bc == "user") {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroFixedInnerX3);
    EnrollUserFLDAdvBoundaryFunction(BoundaryFace::inner_x3, FLDAdvFixedInnerX3);
  }
  if (ox3_bc == "user"){
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroFixedOuterX3);
    EnrollUserFLDAdvBoundaryFunction(BoundaryFace::outer_x3, FLDAdvFixedOuterX3);
  }

  EnrollUserExplicitSourceFunction(AddRadiativeForceAndWork);

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
  igm1 = 1.0/(peos->GetGamma()-1.0);

  // Real T_ref = T_top;
  // Real rho_ref = rho_top;

  // press_top = rho_top*T_top;
  // egas_top = igm1*press_top;
  // Er_top = a_r_sim*std::pow(T_top, 4);

  int idata_size = 0;
  idata_size += 1; // for test counter
  AllocateIntUserMeshBlockDataField(idata_size);

  iuser_meshblock_data[TSTEP_COUNTER].NewAthenaArray(1);
  iuser_meshblock_data[TSTEP_COUNTER](0) = 0;

  // user output variables
  int iuov = 0; // initialize
  iuov_max = 0;
  iuov_max += 2; // for e_gas, E_rad
  iuov_max += 2; // for T_gas, T_rad
  iuov_max += 1; // for P_tot
  iuov_max += 1; // for ent
  iuov_max += 1; // for sound speed
  iuov_max += 1; // for Mach number
  iuov_max += 2; // for opacity

  AllocateUserOutputVariables(iuov_max);
  SetUserOutputVariableName(iuov, "e_gas"), iuov++;
  SetUserOutputVariableName(iuov, "E_rad"), iuov++;
  SetUserOutputVariableName(iuov, "T_gas"), iuov++;
  SetUserOutputVariableName(iuov, "T_rad"), iuov++;
  SetUserOutputVariableName(iuov, "P_tot"), iuov++;
  SetUserOutputVariableName(iuov, "ent"), iuov++;
  SetUserOutputVariableName(iuov, "sound"), iuov++;
  SetUserOutputVariableName(iuov, "Mach"), iuov++;
  SetUserOutputVariableName(iuov, "sigma_P"), iuov++;
  SetUserOutputVariableName(iuov, "sigma_R"), iuov++;

  bool use_opacity_table = pin->GetBoolean("fld", "use_opacity_table");
  if (use_opacity_table) {
    puser_table = new UserOpacityTable(pin);
    prfld2->EnrollOpacityFunction(GetOpacityFromUserTable);
  } else {
    sigma_P = pin->GetReal("fld", "const_opacity_P");
    sigma_R = pin->GetReal("fld", "const_opacity_R");
    prfld2->EnrollOpacityFunction(ConstantOpacity);
  }

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
  Real z = pmy_mesh->mesh_size.x3min;
  Real dz = z-z_ref;
  Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
  Real T_bottom = -grav_acc*dz/(poly_n+1.0)+T_ref;
  Real rho_bottom = rho_ref*std::pow(tmp, poly_n);

  Real press_bottom = rho_bottom*T_bottom;
  Real Er_bottom = a_r_sim*std::pow(T_bottom, 4);
  Real egas_bottom = press_bottom*igm1;
  Real Cs_bottom = std::sqrt(gamma*press_bottom/rho_bottom);
  Real dt_exp = courant*dx1/Cs_bottom;
  dt_initial = dt_exp;
  // Real const_opasity = pin->GetReal("fld", "const_opacity");
  // Real const_opasity_sim = const_opasity*leng_unit*rho_unit;
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  Real L = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real t_sc = L/Cs_bottom;
  Real exp_cycle = t_sc/dt_exp;

  Real optical_depth = sigma_P*L;
  if (gid == 0) {
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "vel_unit = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    std::cout << "T_unit = " << T_unit << " K" << std::endl;
    std::cout << "c_ph_sim = " << c_ph_sim << " cm s^-1" << std::endl;
    std::cout << "dx = " << dx1*leng_unit << " cm" << std::endl;
    std::cout << "dt = " << dt_exp*time_unit << " s" << std::endl;
    std::cout << "dt_sim = " << dt_exp << std::endl;
    std::cout << "t_sc = " << L/Cs_bottom*time_unit << " s" << std::endl;
    std::cout << "t_sc_sim = " << L/Cs_bottom << std::endl;
    std::cout << "T_bottom = " << T_bottom*T_unit << " K" << std::endl;
    std::cout << "T_top = " << T_ref*T_unit << " K" << std::endl;
    std::cout << "press_bottom = " << press_bottom << std::endl;
    std::cout << "Er_bottom = " << Er_bottom << std::endl;
    std::cout << "expected cycle for t_sc = " << exp_cycle << std::endl;
    std::cout << "sigma_P = " << sigma_P << std::endl;
    std::cout << "sigma_R = " << sigma_R << std::endl;
    std::cout << "optical depth = " << optical_depth << std::endl;

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
    ofs << "c_ph_sim           = " << c_ph_sim << std::endl;
    ofs << "dx_dim             = " << dx1*leng_unit << " cm" << std::endl;
    ofs << "dt_dim             = " << dt_exp*time_unit << " s" << std::endl;
    ofs << "dt_sim             = " << dt_exp << std::endl;
    ofs << "t_sc               = " << t_sc*time_unit << " s" << std::endl;
    ofs << "t_sc_sim           = " << t_sc << std::endl;
    ofs << "exp_cycle for t_sc = " << exp_cycle << std::endl;
    ofs << "grav_acc           = " << grav_acc << std::endl;
    ofs << "poly_n             = " << poly_n << std::endl;
    ofs << "rho_bottom         = " << rho_bottom << std::endl;
    ofs << "rho_top            = " << rho_ref << std::endl;
    ofs << "T_bottom           = " << T_bottom << std::endl;
    ofs << "T_top              = " << T_ref << std::endl;
    ofs << "press_bottom       = " << press_bottom << std::endl;
    Real press_top = rho_ref*T_ref;
    Real egas_top = press_top*igm1;
    Real Er_top = a_r_sim*std::pow(T_ref, 4);
    ofs << "press_top          = " << press_top << std::endl;
    ofs << "egas_bottom        = " << egas_bottom << std::endl;
    ofs << "egas_top           = " << egas_top << std::endl;
    ofs << "Er_bottom          = " << Er_bottom << std::endl;
    ofs << "Er_top             = " << Er_top << std::endl;
    ofs << "sigma_P            = " << sigma_P << std::endl;
    ofs << "sigma_R            = " << sigma_R << std::endl;
    ofs << "optical_depth      = " << optical_depth << std::endl;
    ofs.close();
  }

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;


  for(int k=kl; k<=ku; ++k) {
    Real z = pcoord->x3v(k);
    Real dz = z-z_ref;
    Real tmp = -grav_acc*dz/((poly_n+1.0)*T_ref)+1.0;
    Real T = -grav_acc*dz/(poly_n+1.0)+T_ref;
    Real rho = rho_ref*std::pow(tmp, poly_n);
    Real pres = rho*T;

    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real x1 = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = pres*igm1;

        // for FLD
        prfld->u_gas(k,j,i) = pres*igm1;
        prfld->u_rad(k,j,i) = a_r_sim*std::pow(T, 4);
      }
    }
  }
  std::cout << "ProblemGenerator completed." << std::endl;
  return;
}

void Mesh::UserWorkInLoop() {
  if (dt > 1e2*dt_initial || dt < 1e-2*dt_initial) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::UserWorkInLoop]" << std::endl;
    msg << "The calculation is crushed" << std::endl;
    ATHENA_ERROR(msg);
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;
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
        int iuov = 0; // initialize
        // assume cal in E
        user_out_var(iuov,k,j,i) = prfld->u_gas(k,j,i)*egas_unit; iuov++;
        user_out_var(iuov,k,j,i) = prfld->u_rad(k,j,i)*egas_unit; iuov++;
        user_out_var(iuov,k,j,i) = prfld->u_gas(k,j,i)/phydro->w(IDN,k,j,i)*temp_coef; iuov++;
        user_out_var(iuov,k,j,i) = std::pow(prfld->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25); iuov++;
        user_out_var(iuov,k,j,i) = gm1*user_out_var(0,k,j,i) + ONE_3RD*user_out_var(1,k,j,i); iuov++;

        Real vx = phydro->w(IVX,k,j,i);
        Real vy = phydro->w(IVY,k,j,i);
        Real vz = phydro->w(IVZ,k,j,i);
        Real dens = phydro->w(IDN,k,j,i);
        Real idens = 1.0 / dens;
        Real pres = phydro->w(IPR,k,j,i);

        // for entropy
        user_out_var(iuov,k,j,i) =
          std::log(pres * std::pow(idens, gamma)); iuov++;

        // for sound speed
        Real sound = std::sqrt(gm1*pres*idens);
        user_out_var(iuov,k,j,i) = sound*vel_unit; iuov++; // cm/s

        // for Mach number
        Real v_sq = SQR(vx) + SQR(vy) + SQR(vz);
        Real mach = std::sqrt(v_sq) / sound;
        user_out_var(iuov,k,j,i) = mach; iuov++;

        // for opacity
        user_out_var(iuov,k,j,i) = prfld->sigma_p(k,j,i); iuov++;
        user_out_var(iuov,k,j,i) = prfld->sigma_r(k,j,i); iuov++;
      }
    }
  }
  return;
}


void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar) {
  Real gamma = pmb->peos->GetGamma();
  Real gm1 = gamma - 1.0;
  Real igm1 = 1.0 / gm1;

  // if ((pmb->iuser_meshblock_data[TSTEP_COUNTER](0) + 1) % rk_cycle == 0) {
    int il = pmb->is, iu = pmb->ie;
    int jl = pmb->js, ju = pmb->je;
    int kl = pmb->ks, ku = pmb->ke;
    Real idx = 1.0/pmb->pcoord->dx1f(pmb->is);
    Real hidx = 0.5*idx;
    Real dEr[3]; // caution when you use simd

    FLD2 *prfld = pmb->prfld2;
    AthenaArray<Real> &fld_u = prfld->u_rad;

    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          for (int ii = 0; ii < 3; ++ii) {
            int di = (ii == 0) ? 1 : 0;
            int dj = (ii == 1) ? 1 : 0;
            int dk = (ii == 2) ? 1 : 0;
            dEr[ii] = hidx*(fld_u(k+dk,j+dj,i+di) - fld_u(k-dk,j-dj,i-di));
          }
          Real gradE = std::sqrt(SQR(dEr[0]) + SQR(dEr[1]) + SQR(dEr[2]));

          Real R = gradE/(prfld->sigma_r(k,j,i)*fld_u(k,j,i)); // center
          Real lambda = (2.0+R)/(6.0+2.0*R+R*R);

          cons(IM1,k,j,i) += -lambda*dt*dEr[0];
          cons(IM2,k,j,i) += -lambda*dt*dEr[1];
          cons(IM3,k,j,i) += -lambda*dt*dEr[2];
          Real nablaE_v = dEr[0]*prim(IVX,k,j,i) + dEr[1]*prim(IVY,k,j,i) + dEr[2]*prim(IVZ,k,j,i);
          cons(IEN,k,j,i) += -lambda*dt*nablaE_v;
        }
      }
    }

    // for gravity
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          const Real &den  = prim(IDN,k,j,i);
          Real src_z = -grav_acc;
          src_z *= dt*den;
          cons(IM3,k,j,i) += src_z;
          cons(IEN,k,j,i) += src_z*prim(IVZ,k,j,i);
        }
      }
    }
  // }
  pmb->iuser_meshblock_data[TSTEP_COUNTER](0)++;
  pmb->iuser_meshblock_data[TSTEP_COUNTER](0) %= rk_cycle;
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
//   Real x_L = pmb->pmy_mesh->mesh_size.x1min - pmb->pcoord->dx1f(0)/2.0;
//   Real x_R = pmb->pmy_mesh->mesh_size.x1max + pmb->pcoord->dx1f(0)/2.0;
//   Real slope = (Er0_R-Er0_L)/(x_R-x_L);
//   Real cons = Er0_L - slope*x_L;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         Real x = pmb->pcoord->x1v(i);
//         Real an = slope*x + cons;
//         L1norm += std::abs(pmb->prfld2->u_rad(k,j,i) - an)/std::abs(an);
//       }
//     }
//   }
//   int nbtotal = pmb->pmy_mesh->nbtotal;
//   int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
//   L1norm /= ncells*nbtotal;
//   return L1norm;
// }

} // namespace
