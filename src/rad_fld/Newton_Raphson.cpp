//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson.cpp
//! \brief implementation of the functions commonly used in Newton-Raphson

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>    // memset, memcpy
#include <iostream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"

#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "Newton_Raphson.hpp"
#include "linear_multigrid.hpp"

NewtonRaphson::NewtonRaphson(Mesh *pm, ParameterInput *pin)
 : pmy_mesh(pm) {
  max_iter_ = pin->GetOrAddInteger("rad_fld", "nr_maxiter", 100);
  plinmg = new linearMG(pmb->pmy_mesh->pmnr, pmb, pin);
}

NewtonRaphson::~NewtonRaphson() {
  delete plinmg;
}

void NewtonRaphson::Solve(int stage, Real dt) {
  Hydro *phydro = pmy_block->phydro;
  for (int iter = 0; iter < max_iter_; iter++) {
    // make linear eq. to be solved with linear multigrid
    CalculateCoefficients(u_work, u_pre, phydro->w, dt);
    // solve linear eq. with linear multigrid
    plinmgdriver->Solve(stage, dt);
    UpdateRadEnergy(u_work, delta_u);
  }
  Hydro *phydro = pmy_block->phydro;
  if (!only_rad)
    UpdateHydroVariables(phydro->w, phydro->u, u_work);
}

void NewtonRaphson::UpdateHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u,
                             const AthenaArray<Real> &u_fld) {
  // int is = pmy_block->is, ie = pmy_block->ie;
  // int js = pmy_block->js, je = pmy_block->je;
  // int ks = pmy_block->ks, ke = pmy_block->ke;
  // for (int k=ks; k<=ke; k++) {
  //   for (int j=js; j<=je; j++) {
  //     for (int i=is; i<=ie; i++) {
  //       Real rho = w(IDN,k,j,i);
  //       Real egas = w(IEN,k,j,i) - 0.5*rho*(SQR(w(IVX,k,j,i))
  //                   + SQR(w(IVY,k,j,i)) + SQR(w(IVZ,k,j,i)));
  //       Real erad_old = u_fld(k,j,i);
  //       Real erad_new = u(k,j,i);
  //       Real dedt = (erad_new - erad_old)/dt;
  //       Real egas_new = egas - dedt;
  //       if (egas_new < 0.0) {
  //         std::stringstream msg;
  //         msg << "### FATAL ERROR in NewtonRaphson::UpdateHydroVariables" << std::endl
  //             << "Negative gas energy density is found at "
  //             << "k=" << k << " j=" << j << " i=" << i << std::endl
  //             << "egas=" << egas << " erad_old=" << erad_old
  //             << " erad_new=" << erad_new << " dt=" << dt << std::endl;
  //         ATHENA_ERROR(msg);
  //       }
  //       u(IDN,k,j,i) = rho;
  //       u(IEN,k,j,i) = egas_new + 0.5*rho*(SQR(w(IVX,k,j,i))
  //                     + SQR(w(IVY,k,j,i)) + SQR(w(IVZ,k,j,i)));
  //       // u(IEN,k,j,i) -= dedt;
  //     }
  //   }
  // }
}

void NewtonRaphson::CalculateCoefficients(const AthenaArray<Real> &work,
                                          const AthenaArray<Real> &pre,
                                          const AthenaArray<Real> &w, Real dt) {
  int is = pmy_block->is, ie = pmy_block->ie;
  int js = pmy_block->js, je = pmy_block->je;
  int ks = pmy_block->ks, ke = pmy_block->ke;
  AthenaArray<Real> dcp;
  dcp.NewAthenaArray(6);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        delta_u(k,j,i) = 0.0; // reset correction
        Real erad = work(k,j,i);
        Real erad_pre = pre(k,j,i);

        Real Fg = 1.0;
        Real Fr = 1.0;
        Real dFg_deg = 0.0;
        Real dFg_dEr = 0.0;
        Real dFr_deg = 0.0;
        Real dFr_dEr = 0.0;
        AthenaArray<Real> dEr;
        dEr.NewAthenaArray(3);
        for (int ii = 0; ii < 3; ++ii) {
          int di = (ii == 0) ? 1 : 0;
          int dj = (ii == 1) ? 1 : 0;
          int dk = (ii == 2) ? 1 : 0;
          dEr(ii) = hidx*(u(RadFLD::RAD,k+dk,j+dj,i+di) - u(RadFLD::RAD,k-dk,j-dj,i-di));
        }
        // Real dEr1 = hidx*(u(RadFLD::RAD,k,j,i+1) - u(RadFLD::RAD,k,j,i-1));
        // Real dEr2 = hidx*(u(RadFLD::RAD,k,j+1,i) - u(RadFLD::RAD,k,j-1,i));
        // Real dEr3 = hidx*(u(RadFLD::RAD,k+1,j,i) - u(RadFLD::RAD,k-1,j,i));
        // Real gradE = std::sqrt(SQR(dEr1) + SQR(dEr2) + SQR(dEr3));
        Real gradE = std::sqrt(SQR(dEr(0)) + SQR(dEr(1)) + SQR(dEr(2)));

        Real sigma_rface, R, lambda;
        if (fixed_flux_limitter) lambda = ONE_3RD;

        for (int ii = 0; ii < 6; ++ii) {
          int di = (ii == 0) ? -1 : (ii == 1) ? 1 : 0;
          int dj = (ii == 2) ? -1 : (ii == 3) ? 1 : 0;
          int dk = (ii == 4) ? -1 : (ii == 5) ? 1 : 0;
          sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k+dk,j+dj,i+di)),
                std::max(2.0*sigma_r(k,j,i)*sigma_r(k+dk,j+dj,i+di)/(sigma_r(k,j,i) + sigma_r(k+dk,j+dj,i+di)),
                2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
          R = gradE/(sigma_rface*u(RadFLD::RAD,k,j,i));
          if (!fixed_flux_limitter) lambda = (2.0+R)/(6.0+2.0*R+R*R);
          coeff(RadFLD::DXM+ii,k,j,i) = pmg->c_ph*lambda/sigma_rface;
        }
        Real sum_dcp = 0.0;
        for (int n = 0; n < 6; n++) {
          dcp(n) = 0.0;
          sum_dcp += dcp(n);
        }

        // for P:\nabla v
        R = gradE/(sigma_r(k,j,i)*u(RadFLD::RAD,k,j,i)); // center
        if (!fixed_flux_limitter) lambda = (2.0+R)/(6.0+2.0*R+R*R);
        Real chi = lambda+std::pow(lambda*R,2);

        AthenaArray<Real> ngrad;
        ngrad.NewAthenaArray(3);
        for (int ii = 0; ii < 3; ++ii) ngrad(ii) = dEr(ii)/(gradE+TINY_NUMBER);

        Real Dnablav = 0.0;
        for (int jj = 0; jj < 3; ++jj) {
          for (int ii = 0; ii < 3; ++ii) {
            Real D_edd = 0.0;
            if (ii == jj) D_edd += .5*(1.-chi);
            D_edd += .5*(3.*chi-1.)*ngrad(jj)*ngrad(ii); //caution

            int di = ii == 0 ? 1 : 0;
            int dj = ii == 1 ? 1 : 0;
            int dk = ii == 2 ? 1 : 0;

            Real dv_dx = hidx*(w(IVX+jj,k+dk,j+dj,i+di) - w(IVX+jj,k-dk,j-dj,i-di));
            Dnablav += D_edd * dv_dx;
          }
        }

        coeff(NewtonRaphsonFLD::DCCF,k,j,i) = sum_dcp;
        coeff(NewtonRaphsonFLD::DCCS,k,j,i) = 1.0 + dt*(c_ph_sim*sigma_p(k,j,i)+Dnablav); // from dFr_dEr
        coeff(NewtonRaphsonFLD::DCCS,k,j,i) += -(dFr_deg/dFg_deg)*dFg_dEr;
        for (int n = 0; n < 6; n++) coeff(NewtonRaphsonFLD::DXM+n,k,j,i) = dcp(n);

        rhs(k,j,i) = -Fr + (dFr_deg/dFg_deg)*Fg;
      }
    }
  }
}

void NewtonRaphson::UpdateRadEnergy(AthenaArray<Real> &work, const AthenaArray<Real> &delta) {
  int is = pmy_block->is, ie = pmy_block->ie;
  int js = pmy_block->js, je = pmy_block->je;
  int ks = pmy_block->ks, ke = pmy_block->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        work(k,j,i) += delta(k,j,i);
      }
    }
  }
}



