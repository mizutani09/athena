//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ps_test.cpp
//! \brief scalar advection test
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

namespace {
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit;
  Real a_r_dim, Rgas, mu;

  Real rho0, p0, v0, Er0;
  Real r0, r_sigma;
  Real delta_ratio;

  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryL1norm(MeshBlock *pmb, int iout);
}


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

  rho0 = pin->GetReal("problem", "rho0");
  p0 = pin->GetReal("problem", "p0");
  Er0 = pin->GetReal("problem", "Er0");
  v0 = pin->GetReal("problem", "v0");
  r0 = pin->GetReal("problem", "r0");
  r_sigma = pin->GetReal("problem", "r_sigma");
  delta_ratio = pin->GetReal("problem", "delta_ratio");

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryL1norm, "L1norm", UserHistoryOperation::sum);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;

  Real r_sigma_sq = SQR(r_sigma);
  for(int k=kl; k<=ku; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=il; i<=iu; ++i) {
        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = rho0*v0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = p0*igm1 + 0.5*rho0*v0*v0;

        Real r_sq = SQR(pcoord->x1v(i)-r0);
        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n,k,j,i) = Er0*(1.0+delta_ratio*std::exp(-r_sq/r_sigma_sq));
          }
        }
      }
    }
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
  // to fix hydro variables
  Real igm1 = 1.0/(peos->GetGamma()-1.0);
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
        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = rho0*v0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = p0*igm1 + 0.5*rho0*v0*v0;

        phydro->w(IDN,k,j,i) = rho0;
        phydro->w(IVX,k,j,i) = v0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
        phydro->w(IPR,k,j,i) = p0;
      }
    }
  }
  return;
}

namespace {

Real HistoryRtime(MeshBlock *pmb, int iout) {
  return pmb->pmy_mesh->time*time_unit;
}

Real HistoryL1norm(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  Real L1norm = 0;
  Real r_sigma_sq = SQR(r_sigma);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x = (pmb->pcoord->x1v(i) - r0) - v0*pmb->pmy_mesh->time;
        Real r_sq = SQR(x);
        Real an;
        if (r_sigma < 0.0) {
          an = Er0;
        } else {
          an = Er0*(1.0+std::exp(-r_sq/r_sigma_sq));
        }
        L1norm += std::abs(pmb->pscalars->s(0,k,j,i) - an)/std::abs(an);
      }
    }
  }
  int nbtotal = pmb->pmy_mesh->nbtotal;
  int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
  L1norm /= ncells*nbtotal;
  return L1norm;
}

} // namespace
