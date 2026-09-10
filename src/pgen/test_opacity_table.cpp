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
//! \file test_opacity_table.cpp
//! \brief Problem generator for radiative shock test
//! REFERENCE: W. Zhang, L. Howell, A. Almgren, A. Burrows, J. Bell, Astrophys. J. Suppl. Ser. 196, 20 (2011).
//!            for section 6.6: Non-equilibrium Radiative Shock
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


#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit, vel_unit;
  Real opacity_unit;
  Real a_r_dim, Rgas, mu;
  Real a_r_sim;
  Real sigma_P, sigma_R;
  Real t_lim_dim;
  UserOpacityTable *puser_table = nullptr;
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
  a_r_sim = a_r_dim/(egas_unit/std::pow(T_unit, 4));

  vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  opacity_unit = 1.0/(rho_unit*leng_unit); // cm^2/g

  // // Real const_opasity = pin->GetReal("fld", "const_opacity");
  // sigma_P = pin->GetReal("fld", "const_opacity_P") * (leng_unit);
  // sigma_R = pin->GetReal("fld", "const_opacity_R") * (leng_unit);
  // Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  // Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
//   Real igm1 = 1.0/(peos->GetGamma()-1.0);
//   p0_L = rho0_L*T0_L;
//   p0_R = rho0_R*T0_R;
//   Er0_L = a_r_sim*std::pow(T0_L, 4);
//   Er0_R = a_r_sim*std::pow(T0_R, 4);
//   egas0_L = p0_L*igm1;
//   egas0_R = p0_R*igm1;
  if (puser_table == nullptr) puser_table = new UserOpacityTable(pin);


  prfld->EnrollOpacityFunction(ConstantOpacity);
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Real rho=1e-10, temp=1e5; // g/cm³, K
  Real rho = pin->GetReal("problem", "rho"); // in cgs
  Real temp = pin->GetReal("problem", "temp"); // in K
  Real press = rho*temp*Rgas/mu; // in cgs
  // Real rho=7e-11, temp=6e3;
  // Real rho=8.89e-02, temp=3.16e+07;
  // Real rho=2.81e-08, temp=1.00e+05;
  // std::cout << "Input fluid parameters and retrieve the corresponding opacity value." << '\n'
  //           << "Non-positive inputs will exit loop." << '\n';

  // while(true) {
  //   std::cout << "Input density in g cm^-3: ";
  //   std::cin >> rho;
  //   std::cout << "Input temperature in K: ";
  //   std::cin >> temp;

  //   if (rho <= 0.0 || temp <= 0.0) {
  //     std::cout << "Exiting..." << std::endl;
  //     break;
  //   }

    std::cout << ">>> Input parameters <<<" << std::endl;
    std::cout << "Density: " << rho << " g cm^-3" << std::endl;
    std::cout << "Temperature: " << temp << " K" << std::endl;
    std::cout << "Pressure: " << press << " erg cm^-3" << std::endl;

    // Repeat the same lookup so the regression suite can exercise thread-safe
    // diagnostic counters without running a time integration.
    const int repetitions = pin->GetOrAddInteger("problem", "opacity_probe_repetitions", 1);
    if (repetitions < 1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl
          << "problem/opacity_probe_repetitions must be positive." << std::endl;
      ATHENA_ERROR(msg);
    }
#pragma omp parallel for
    for (int probe = 0; probe < repetitions; ++probe) {
      const Real planck = puser_table->GetOpacity(RadFLD::SIGMA_P, rho, temp);
      const Real rosseland = puser_table->GetOpacity(RadFLD::SIGMA_R, rho, temp);
      if (probe == 0) {
        sigma_P = planck;
        sigma_R = rosseland;
      }
    }
    // std::cout << "opacity_unit: " << opacity_unit << " cm^2/g" << std::endl;

    std::cout << "Rosseland opacity: " << sigma_R << " cm^2/g" << std::endl;
    std::cout << "Planck opacity: " << sigma_P << " cm^2/g" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    puser_table->ReportDiagnostics(std::cout);
  // }

  std::stringstream msg;
  msg << "### INTENTIONAL ERROR in function [MeshBlock::ProblemGenerator]" << std::endl;
  msg << "This pgen is only for testing purposes." << std::endl;
  ATHENA_ERROR(msg);

//   Real gamma = peos->GetGamma();
//   Real igm1 = 1.0/(gamma-1.0);
//   Real dx1 = pcoord->dx1f(4);
//   Real courant = pin->GetReal("time", "cfl_number");
//   Real Cs_L = std::sqrt(gamma*p0_L/rho0_L);
//   Real Cs_R = std::sqrt(gamma*p0_R/rho0_R);
//   Real max_vel = std::max(std::abs(v0_L+Cs_L), std::abs(v0_R+Cs_R));
//   Real dt_exp = courant*dx1/max_vel;
//   // Real const_opasity = pin->GetReal("fld", "const_opacity");
//   // Real const_opasity_sim = const_opasity*leng_unit*rho_unit;
//   Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
//   Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
//   // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

//   Real L = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
//   Real t_sc = L/max_vel;
//   Real t_lim_dim = pin->GetReal("problem", "t_lim");
//   Real t_lim = t_lim_dim / time_unit; // in s
//   Real exp_cycle = t_lim / dt_exp;
//   Real exp_cycle_sc = t_sc / dt_exp;
//   if (gid == 0) {
//     std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
//     std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
//     std::cout << "time_unit = " << time_unit << " s" << std::endl;
//     std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
//     std::cout << "vel_unit = " << leng_unit/time_unit << " cm s^-1" << std::endl;
//     std::cout << "T_unit = " << T_unit << " K" << std::endl;
//     std::cout << "c_ph_sim = " << c_ph_sim << " cm s^-1" << std::endl;
//     std::cout << "dx = " << dx1*leng_unit << " cm" << std::endl;
//     std::cout << "dt = " << dt_exp*time_unit << " s" << std::endl;
//     std::cout << "dt_sim = " << dt_exp << std::endl;
//     std::cout << "t_lim_dim = " << t_lim*time_unit << " s" << std::endl;
//     std::cout << "t_lim_sim = " << t_lim << std::endl;
//     std::cout << "exp_cycle = " << exp_cycle << std::endl;
//     std::cout << "t_sc_dim = " << t_sc*time_unit << " s" << std::endl;
//     std::cout << "t_sc_sim = " << t_sc << std::endl;
//     std::cout << "exp_cycle for t_sc = " << exp_cycle_sc << std::endl;
//     std::cout << "p0_L = " << p0_L << std::endl;
//     std::cout << "p0_R = " << p0_R << std::endl;
//     std::cout << "Er0_L = " << Er0_L << std::endl;
//     std::cout << "Er0_R = " << Er0_R << std::endl;
//     std::cout << "v0_L = " << v0_L << std::endl;
//     std::cout << "v0_R = " << v0_R << std::endl;
//     std::cout << "expected cycle = " << exp_cycle << std::endl;
//     std::cout << "sigma_P = " << sigma_P << std::endl;
//     std::cout << "sigma_R = " << sigma_R << std::endl;

//     // also output the upper values in txt file
//     std::ofstream ofs("problem_parameters.txt");
//     ofs << ">>> Problem parameters <<<" << std::endl;
//     ofs << "- Units" << std::endl;
//     ofs << "rhoUnit        = " << rho_unit << " g cm^-3" << std::endl;
//     ofs << "egasUnit       = " << egas_unit << " erg cm^-3" << std::endl;
//     ofs << "timeUnit       = " << time_unit << " s" << std::endl;
//     ofs << "lengUnit       = " << leng_unit << " cm" << std::endl;
//     ofs << "velUnit        = " << leng_unit/time_unit << " cm s^-1" << std::endl;
//     ofs << "TUnit          = " << T_unit << " K" << std::endl;
//     ofs << std::endl;

//     ofs << "- Simulation parameters" << std::endl;
//     ofs << "c_ph_sim         = " << c_ph_sim << std::endl;
//     ofs << "dx_dim           = " << dx1*leng_unit << " cm" << std::endl;
//     ofs << "dt_dim           = " << dt_exp*time_unit << " s" << std::endl;
//     ofs << "dt_sim           = " << dt_exp << std::endl;
//     ofs << "t_lim_dim        = " << t_lim*time_unit << " s" << std::endl;
//     ofs << "t_lim_sim        = " << t_lim << std::endl;
//     ofs << "exp_cycle(t_lim) = " << exp_cycle << std::endl;
//     ofs << "t_sc_dim         = " << t_sc*time_unit << " s" << std::endl;
//     ofs << "t_sc_sim         = " << t_sc << std::endl;
//     ofs << "exp_cycle(t_sc)  = " << exp_cycle << std::endl;
//     ofs << "rho0_L           = " << rho0_L << std::endl;
//     ofs << "rho0_R           = " << rho0_R << std::endl;
//     ofs << "T0_L             = " << T0_L << std::endl;
//     ofs << "T0_R             = " << T0_R << std::endl;
//     ofs << "v0_L             = " << v0_L << std::endl;
//     ofs << "v0_R             = " << v0_R << std::endl;
//     ofs << "p0_L             = " << p0_L << std::endl;
//     ofs << "p0_R             = " << p0_R << std::endl;
//     ofs << "Er0_L            = " << Er0_L << std::endl;
//     ofs << "Er0_R            = " << Er0_R << std::endl;
//     ofs << "sigma_P          = " << sigma_P << std::endl;
//     ofs << "sigma_R          = " << sigma_R << std::endl;
//     ofs.close();
//   }

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;


  for(int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
          phydro->u(IDN,k,j,i) = 0.0;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            phydro->u(IEN,k,j,i) = 0.0;

          // for FLD
          prfld->u_gas(k,j,i) = 0.0;
          prfld->u_rad(k,j,i) = 0.0;
      }
    }
  }
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  return;
}
