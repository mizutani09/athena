//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file NRFLD.cpp
//! \brief implementation of the functions used in FLD

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
#include "NRFLD.hpp"




//----------------------------------------------------------------------------------------
//! \fn NRFLDDriver::NRFLDDriver(Mesh *pm, ParameterInput *pin)
//! \brief NRFLDDriver constructor

NRFLDDriver::NRFLDDriver(Mesh *pm, ParameterInput *pin)
    : NewtonRaphsonDriver(pm, 1, linearSolver::NCOEFF, linearSolver::NMATRIX) {
//   eps_ = pin->GetOrAddReal("mgfld", "threshold", -1.0);
//   niter_ = pin->GetOrAddInteger("mgfld", "niteration", -1);
//   ffas_ = pin->GetOrAddBoolean("mgfld", "fas", ffas_);
//   omega_ = pin->GetOrAddReal("mgfld", "omega", 1.0);
//   fsteady_ = pin->GetOrAddBoolean("mgfld", "steady", false);
//   npresmooth_ = pin->GetOrAddReal("mgfld", "npresmooth", 2);
//   npostsmooth_ = pin->GetOrAddReal("mgfld", "npostsmooth", 2);
//   fshowdef_ = pin->GetOrAddBoolean("mgfld", "show_defect", fshowdef_);
//   std::string smoother = pin->GetOrAddString("mgfld", "smoother", "jacobi-rb");
//   matrixmode_ = 1;
//   if (smoother == "jacobi-rb") {
//     fsmoother_ = 1;
//     redblack_ = true;
//   } else if (smoother == "jacobi-double") {
//     fsmoother_ = 0;
//     redblack_ = true;
//   } else { // jacobi
//     fsmoother_ = 0;
//     redblack_ = false;
//   }
//   std::string prol = pin->GetOrAddString("mgfld", "prolongation", "trilinear");
//   if (prol == "tricubic")
//     fprolongation_ = 1;

//   std::string m = pin->GetOrAddString("mgfld", "mgmode", "none");
//   std::transform(m.begin(), m.end(), m.begin(), ::tolower);
//   if (m == "fmg") {
//     mode_ = 0;
//   } else if (m == "mgi") {
//     mode_ = 1; // Iterative
//   } else {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in MGFLDDriver::MGFLDDriver" << std::endl
//         << "The \"mgmode\" parameter in the <mgfld> block is invalid." << std::endl
//         << "FMG: Full Multigrid + Multigrid iteration (default)" << std::endl
//         << "MGI: Multigrid Iteration" << std::endl;
//     ATHENA_ERROR(msg);
//   }
//   if (eps_ < 0.0 && niter_ < 0) {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in MGFLDDriver::MGFLDDriver" << std::endl
//         << "Either \"threshold\" or \"niteration\" parameter must be set "
//         << "in the <mgfld> block." << std::endl
//         << "When both parameters are specified, \"niteration\" is ignored." << std::endl
//         << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
//     ATHENA_ERROR(msg);
//   }
//   mg_mesh_bcs_[inner_x1] =
//               GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ix1_bc", "none"));
//   mg_mesh_bcs_[outer_x1] =
//               GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ox1_bc", "none"));
//   mg_mesh_bcs_[inner_x2] =
//               GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ix2_bc", "none"));
//   mg_mesh_bcs_[outer_x2] =
//               GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ox2_bc", "none"));
//   mg_mesh_bcs_[inner_x3] =
//               GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ix3_bc", "none"));
//   mg_mesh_bcs_[outer_x3] =
//               GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ox3_bc", "none"));
//   CheckBoundaryFunctions();
//   fsubtract_average_ = false; // override the subtract average flag

//   mgtlist_ = new MultigridTaskList(this);

//   // Allocate the root multigrid
//   mgroot_ = new MGFLD(this, nullptr, pin);

//   fldtlist_ = new FLDBoundaryTaskList(pin, pm);

//   int nth = 1;
// #ifdef OPENMP_PARALLEL
//   nth = omp_get_max_threads();
// #endif
//   temp = new AthenaArray<Real>[nth];
//   int nx = std::max(pmy_mesh_->block_size.nx1, pmy_mesh_->nrbx1) + 2*mgroot_->ngh_;
//   int ny = std::max(pmy_mesh_->block_size.nx2, pmy_mesh_->nrbx2) + 2*mgroot_->ngh_;
//   int nz = std::max(pmy_mesh_->block_size.nx3, pmy_mesh_->nrbx3) + 2*mgroot_->ngh_;
//   for (int n = 0; n < nth; ++n)
//     temp[n].NewAthenaArray(nz, ny, nx);
}


//----------------------------------------------------------------------------------------
//! \fn NRFLDDriver::~NRFLDDriver()
//! \brief NRFLDDriver destructor

NRFLDDriver::~NRFLDDriver() {
  // delete fldtlist_;
  // delete mgroot_;
  // delete mgtlist_;
  // delete [] temp;
}


NRFLD::NRFLD(MeshBlock *pmb, ParameterInput *pin) :
    NewtonRaphson(pmy_driver_, pmb, ngh_),
    pmy_block_(pmb)
    {
    // pmy_mesh(pm),
    // coarse_u_(1, pmb->ncc3, pmb->ncc2, pmb->ncc1,
    //              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
    //               AthenaArray<Real>::DataStatus::empty)), // ? caution!
    // nrbvar(pmb, &u_, &coarse_u_, u_flux_, true),
 }

NRFLD::~NRFLD() {

}

// void NewtonRaphson::Solve(int stage, Real dt) {
//   Hydro *phydro = pmy_block->phydro;
//   for (int iter = 0; iter < max_iter_; iter++) {
//     // make linear eq. to be solved with linear multigrid
//     CalculateCoefficients(u_work, u_pre, phydro->w, dt);
//     // solve linear eq. with linear multigrid
//     plinmgdriver->Solve(stage, dt);
//     UpdateRadEnergy(u_work, delta_u);
//   }
//   Hydro *phydro = pmy_block->phydro;
//   if (!only_rad)
//     UpdateHydroVariables(phydro->w, phydro->u, u_work);
// }

void NRFLD::LoadHydroVariables() {
  pmy_block_->prfld2->LoadHydroVariables(pmy_block_->phydro->w, u_); // caution!
}

void NRFLD::CalculateCoefficients(const AthenaArray<Real> &work,
                                  const AthenaArray<Real> &u_pre,
                                  const AthenaArray<Real> &w, Real dt) {
  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real idx = 1.0/dx;
  Real hidx = 0.5*idx;
  AthenaArray<Real> dcp;
  dcp.NewAthenaArray(6);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        delta_u_(k,j,i) = 0.0; // reset correction
        Real erad = work(k,j,i);
        Real erad_pre = u_pre(k,j,i);

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
          dEr(ii) = hidx*(u_pre(k+dk,j+dj,i+di) - u_pre(k-dk,j-dj,i-di));
        }
        Real gradE = std::sqrt(SQR(dEr(0)) + SQR(dEr(1)) + SQR(dEr(2)));

        Real sigma_rface, R, lambda;
        if (pfld->fixed_flux_limitter) lambda = ONE_3RD;

        Real sum_dcp = 0.0;
        for (int ii = 0; ii < 6; ++ii) {
          int di = (ii == 0) ? -1 : (ii == 1) ? 1 : 0;
          int dj = (ii == 2) ? -1 : (ii == 3) ? 1 : 0;
          int dk = (ii == 4) ? -1 : (ii == 5) ? 1 : 0;
          sigma_rface = std::min(0.5*(pfld->sigma_r(k,j,i) + pfld->sigma_r(k+dk,j+dj,i+di)),
                std::max(2.0*pfld->sigma_r(k,j,i)*pfld->sigma_r(k+dk,j+dj,i+di)/(pfld->sigma_r(k,j,i) + pfld->sigma_r(k+dk,j+dj,i+di)),
                2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
          R = gradE/(sigma_rface*u_pre(k,j,i));
          if (!pfld->fixed_flux_limitter) lambda = (2.0+R)/(6.0+2.0*R+R*R);
          dcp(ii) = pfld->c_ph*lambda/sigma_rface;
          sum_dcp += dcp(ii);
        }

        // for P:\nabla v
        R = gradE/(pfld->sigma_r(k,j,i)*u_pre(k,j,i)); // center
        if (!pfld->fixed_flux_limitter) lambda = (2.0+R)/(6.0+2.0*R+R*R);
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

        coeff_(linearSolver::DCCF,k,j,i) = sum_dcp;
        coeff_(linearSolver::DCCS,k,j,i) = 1.0 + dt*(pfld->c_ph*pfld->sigma_p(k,j,i)+Dnablav); // from dFr_dEr
        coeff_(linearSolver::DCCS,k,j,i) += -(dFr_deg/dFg_deg)*dFg_dEr;
        for (int n = 0; n < 6; n++) coeff_(linearSolver::DXMF+n,k,j,i) = dcp(n);

        src_(k,j,i) = -Fr + (dFr_deg/dFg_deg)*Fg;
      }
    }
  }
}


Real NewtonRaphson::CalculateDefectNorm(NRNormType nrm, int n) {
  Real norm=0.0;
  return norm;
}

void NRFLDDriver::SetLinearSolver() {
  plinsolver_ = nullptr;
}
