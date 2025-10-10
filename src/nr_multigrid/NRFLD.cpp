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
#include "../eos/eos.hpp"
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

  // // Allocate the root multigrid
  // mgroot_ = new linearMG(this, nullptr, pin);

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

  if (NRMGFLD_ENABLED) {
    plmgd_ = new linearMGDriver(pm, pin, this);
    pmy_mesh_->pmlmd = plmgd_;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in NewtonRaphsonDriver::NewtonRaphsonDriver" << std::endl
        << "Failed to allocate linear solver" << std::endl;
    ATHENA_ERROR(msg);
  }
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
    pmy_block_(pmb),
    derivetive_(NewtonRaphsonFLD::NNRDIV, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u_gas_(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    def_coeff_(NewtonRaphsonFLD::NDCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1)
    {
    // pmy_mesh(pm),
    // coarse_u_(1, pmb->ncc3, pmb->ncc2, pmb->ncc1,
    //              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
    //               AthenaArray<Real>::DataStatus::empty)), // ? caution!
    // nrbvar(pmb, &u_, &coarse_u_, u_flux_, true),
    // plmg_ = new linearMG(pmy_driver_->plinsolver_->Solve(), pmb, pin);
    if (NRMGFLD_ENABLED) {
      plmg_ = new linearMG(pmy_driver_->plmgd_, pmb, pin, this);
      pmb->plmg = plmg_;
    }
 }

NRFLD::~NRFLD() {

}

void NRFLD::LoadHydroVariables() {
  FLD2 *pfld = pmy_block_->prfld2;
  pfld->LoadHydroVariables(pmy_block_->phydro->w, u_gas_);
  return;
}

void NRFLD::UpdateHydroVariables() {
  FLD2 *pfld = pmy_block_->prfld2;
  pfld->UpdateHydroVariables(pmy_block_->phydro->w,
                               pmy_block_->phydro->u, u_gas_);
  return;
}

void NRFLD::CalculateCoefficientsOnce(const AthenaArray<Real> &u_pre,
                                      const AthenaArray<Real> &w) {
  FLD2 *prfld = pmy_block_->prfld2;
  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real idx = 1.0/dx;
  Real idx2 = 1.0/(dx*dx);
  Real hidx = 0.5*idx;
  Real gm1 = pmy_block_->peos->GetGamma() - 1.0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // for coefficient from e_gas to T_gas
        def_coeff_(NewtonRaphsonFLD::DCOUPLE,k,j,i) = gm1/w(IDN,k,j,i);

        // for lambda and coefficient of diff term
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
          derivetive_(NewtonRaphsonFLD::dFr_dEr_xm+ii,k,j,i) = pfld->c_ph*lambda/sigma_rface;
          sum_dcp += derivetive_(NewtonRaphsonFLD::dFr_dEr_xm+ii,k,j,i);
        }

        // for P:\nabla v
        R = gradE/(pfld->sigma_r(k,j,i)*u_pre(k,j,i)); // center
        if (!pfld->fixed_flux_limitter) lambda = (2.0+R)/(6.0+2.0*R+R*R);
        Real chi = lambda+std::pow(lambda*R,2);

        AthenaArray<Real> ngrad;
        ngrad.NewAthenaArray(3);
        for (int ii = 0; ii < 3; ++ii) ngrad(ii) = dEr(ii)/(gradE+TINY_NUMBER);

        def_coeff_(NewtonRaphsonFLD::DDV,k,j,i) = 0.0;
        for (int jj = 0; jj < 3; ++jj) {
          for (int ii = 0; ii < 3; ++ii) {
            Real D_edd = 0.0;
            if (ii == jj) D_edd += .5*(1.-chi);
            D_edd += .5*(3.*chi-1.)*ngrad(jj)*ngrad(ii); //caution

            int di = ii == 0 ? 1 : 0;
            int dj = ii == 1 ? 1 : 0;
            int dk = ii == 2 ? 1 : 0;

            Real dv_dx = hidx*(w(IVX+jj,k+dk,j+dj,i+di) - w(IVX+jj,k-dk,j-dj,i-di));
            def_coeff_(NewtonRaphsonFLD::DDV,k,j,i) += D_edd * dv_dx;
          }
        }
      }
    }
  }
}

void NRFLD::CalculateCoefficients(const AthenaArray<Real> &u_rad_old,
                                  const AthenaArray<Real> &u_rad_new,
                                  // const AthenaArray<Real> &u_gas_old,
                                  // const AthenaArray<Real> &u_gas_new,
                                  Real dt) {
  FLD2 *prfld = pmy_block_->prfld2;
  AthenaArray<Real> &u_gas_new = u_gas_; // caution! this should be in argument
  AthenaArray<Real> &u_gas_old = prfld->u_gas; // caution! this should be in argument

  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real idx2 = 1.0/(dx*dx);


  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        delta_u_(k,j,i) = 0.0; // caution! reset correction should be in different function
        Real c_sigma_p = prfld->c_ph*prfld->sigma_p(k,j,i);

        // caution! following can be optimized (only once per step)
        Real sum_dcp = 0.0;
        for (int n = 0; n < 6; n++) {
          sum_dcp += coeff_(NewtonRaphsonFLD::dFr_dEr_xm+n,k,j,i);
        }

        Real T_gas_new = def_coeff_(NewtonRaphsonFLD::DCOUPLE,k,j,i)*u_gas_new(k,j,i);
        Real src_term = c_sigma_p*(prfld->a_r*std::pow(T_gas_new,4) - u_rad_new(k,j,i));
        Real Pnablav = def_coeff_(NewtonRaphsonFLD::DDV,k,j,i)*u_rad_new(k,j,i);
        Real diff_term = 0.0;
        for (int n = 0; n < 6; n++) {
          int di = (n == 0) ? -1 : (n == 1) ? 1 : 0;
          int dj = (n == 2) ? -1 : (n == 3) ? 1 : 0;
          int dk = (n == 4) ? -1 : (n == 5) ? 1 : 0;
          diff_term += coeff_(linearSolver::DXMF+n,k,j,i)*(u_rad_new(k+dk,j+dj,i+di) - u_rad_new(k,j,i));
        }
        diff_term *= idx2;


        derivetive_(NewtonRaphsonFLD::Fg,k,j,i) = (u_gas_new(k,j,i) - u_gas_old(k,j,i)) + dt * src_term;
        derivetive_(NewtonRaphsonFLD::Fr,k,j,i) = (u_rad_new(k,j,i) - u_rad_old(k,j,i)) - dt * src_term - dt *(-Pnablav + diff_term);

        derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i) = 1.0 + 4.0*dt*c_sigma_p*prfld->a_r*std::pow(T_gas_new,3)*def_coeff_(NewtonRaphsonFLD::DCOUPLE,k,j,i);
        derivetive_(NewtonRaphsonFLD::dFg_dEr,k,j,i) = -dt*c_sigma_p;
        derivetive_(NewtonRaphsonFLD::dFr_deg,k,j,i) = -4.0*dt*c_sigma_p*prfld->a_r*std::pow(T_gas_new,3)*def_coeff_(NewtonRaphsonFLD::DCOUPLE,k,j,i);
        derivetive_(NewtonRaphsonFLD::dFr_dEr,k,j,i) = 1.0 + dt*(c_sigma_p + def_coeff_(NewtonRaphsonFLD::DDV,k,j,i) + idx2*sum_dcp);


        coeff_(linearSolver::DCCF,k,j,i) = sum_dcp;
        coeff_(linearSolver::DCCS,k,j,i) = 1.0 + dt*(pfld->c_ph*pfld->sigma_p(k,j,i)+def_coeff_(NewtonRaphsonFLD::DDV,k,j,i)); // from dFr_dEr
        coeff_(linearSolver::DCCS,k,j,i) += -(derivetive_(NewtonRaphsonFLD::dFr_deg,k,j,i)/derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i))*derivetive_(NewtonRaphsonFLD::dFg_dEr,k,j,i);
        for (int n = 0; n < 6; n++) coeff_(linearSolver::DXMF+n,k,j,i) = derivetive_(NewtonRaphsonFLD::dFr_dEr_xm+n,k,j,i);

        src_(k,j,i) = -derivetive_(NewtonRaphsonFLD::Fr,k,j,i) + (derivetive_(NewtonRaphsonFLD::dFr_deg,k,j,i)/derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i))*derivetive_(NewtonRaphsonFLD::Fg,k,j,i);
      }
    }
  }
}

void NRFLD::CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
              const AthenaArray<Real> &u_old, const AthenaArray<Real> &coeff,
              bool th) {
  FLD2 *prfld = pmy_block_->prfld2;
  MeshBlock *pmb = pmy_block_;
  int il = pmb->is, iu = pmb->ie;
  int jl = pmb->js, ju = pmb->je;
  int kl = pmb->ks, ku = pmb->ke;
  Real dx = pmb->pcoord->dx1f(il);
  Real idx2 = 1.0/SQR(dx);
  Real dt = pmy_driver_->dt_;

#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        Real T_gas = def_coeff_(NewtonRaphsonFLD::DCOUPLE,k,j,i)*u_gas_(k,j,i);
        Real src_term = prfld->c_ph*prfld->sigma_p(k,j,i)*(prfld->a_r*std::pow(T_gas,4) - u(k,j,i));
        Real Pnablav = def_coeff_(NewtonRaphsonFLD::DDV,k,j,i)*u(k,j,i);
        Real diff_term = 0.0;
        for (int n = 0; n < 6; n++) {
          int di = (n == 0) ? -1 : (n == 1) ? 1 : 0;
          int dj = (n == 2) ? -1 : (n == 3) ? 1 : 0;
          int dk = (n == 4) ? -1 : (n == 5) ? 1 : 0;
          diff_term += coeff_(linearSolver::DXMF+n,k,j,i)*(u(k+dk,j+dj,i+di) - u(k,j,i));
        }
        diff_term *= idx2;

        Real Fg = (u_gas_(k,j,i) - prfld->u_gas(k,j,i)) + dt* src_term;
        Real Fr = (u(k,j,i)      - u_old(k,j,i))        - dt*(src_term - Pnablav + diff_term);
        def(k,j,i) = Fg + Fr;
      }
    }
  }

  return;
}

void NRFLD::AddDifference(AthenaArray<Real> &u_rad, const AthenaArray<Real> &delta_u) {

  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;
  FLD2 *prfld = pmy_block_->prfld2;

  // assuming single variable
  // for (int v=0; v<nvar_; ++v) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // dst(v,k,j,i) += delta(v,k,j,i);
          u_rad(k,j,i) += delta_u(k,j,i);
          prfld->u_gas(k,j,i) += -(derivetive_(NewtonRaphsonFLD::Fg,k,j,i) + derivetive_(NewtonRaphsonFLD::dFg_dEr,k,j,i)*delta_u(k,j,i)) / derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i);
        }
      }
    }
  // }
  return;
}
