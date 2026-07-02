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
  eps_ = pin->GetOrAddReal("nrfld", "nr_threshold", -1.0);
  niter_ = pin->GetOrAddInteger("nrfld", "nr_niteration", -1);
  fshowdef_ = pin->GetOrAddBoolean("nrfld", "show_defect", fshowdef_);
//   omega_ = pin->GetOrAddReal("mgfld", "omega", 1.0);
//   fshowdef_ = pin->GetOrAddBoolean("mgfld", "show_defect", fshowdef_);
  if (eps_ < 0.0 && niter_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <nrfld> block." << std::endl
      << "When both parameters are specified, \"niteration\" is ignored." << std::endl  
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
    ATHENA_ERROR(msg);
  }
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

    nr_mesh_bcs_[inner_x1] =
                GetBoundaryFlag(pin->GetOrAddString("nrfld", "ix1_bc", "none"));
    nr_mesh_bcs_[outer_x1] =
                GetBoundaryFlag(pin->GetOrAddString("nrfld", "ox1_bc", "none"));
    nr_mesh_bcs_[inner_x2] =
                GetBoundaryFlag(pin->GetOrAddString("nrfld", "ix2_bc", "none"));
    nr_mesh_bcs_[outer_x2] =
                GetBoundaryFlag(pin->GetOrAddString("nrfld", "ox2_bc", "none"));
    nr_mesh_bcs_[inner_x3] =
                GetBoundaryFlag(pin->GetOrAddString("nrfld", "ix3_bc", "none"));
    nr_mesh_bcs_[outer_x3] =
                GetBoundaryFlag(pin->GetOrAddString("nrfld", "ox3_bc", "none"));
    CheckBoundaryFunctions();
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
  if (NRMGFLD_ENABLED) {
    delete plmgd_;
  }
}


NRFLD::NRFLD(MeshBlock *pmb, ParameterInput *pin) :
    NewtonRaphson(pmb->pmy_mesh->pmnr, pmb, NGHOST,
                  NewtonRaphsonFLD::NNRDIV,
                  NewtonRaphsonFLD::NDCOEFF),
    pmy_driver_(pmb->pmy_mesh->pmnr),
    pmy_block_(pmb),
    u_gas_(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    ngh_(NGHOST)
    {
    if (pmy_driver_->fshowdef_ && pmy_block_->gid == 0) std::cout << ngh_ << std::endl;

    // check pointer
    if (pmy_driver_ == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in NRFLD::NRFLD" << std::endl
          << "NewtonRaphsonDriver pointer is null" << std::endl;
      ATHENA_ERROR(msg);
    }
    if (pmy_driver_->plmgd_ == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in NRFLD::NRFLD" << std::endl
          << "linearMGDriver pointer is null" << std::endl;
      ATHENA_ERROR(msg);
    }
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
  if (NRMGFLD_ENABLED) {
    delete plmg_;
  }
}

void NRFLD::LoadVariables() {
  FLD2 *pfld = pmy_block_->prfld2;
  pfld->LoadHydroVariables(pmy_block_->phydro->w, pfld->u_gas);
  int il = pmy_block_->is - NGHOST, iu = pmy_block_->ie + NGHOST;
  int jl = pmy_block_->js, ju = pmy_block_->je;
  int kl = pmy_block_->ks, ku = pmy_block_->ke;
  if (pmy_block_->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block_->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        u_gas_(k,j,i) = pfld->u_gas(k,j,i);
        u_(k,j,i) = pfld->u_rad(k,j,i);
        uold_(k,j,i) = pfld->u_rad(k,j,i);

      }
    }
  }
  return;
}

void NRFLD::UpdateHydroVariables() {
  FLD2 *pfld = pmy_block_->prfld2;
  pfld->UpdateHydroVariables(pmy_block_->phydro->w,
                             pmy_block_->phydro->u,
                             u_, u_gas_);
  return;
}

void NRFLD::CalculateCoefficientsOnce(const AthenaArray<Real> &u_pre,
                                      const AthenaArray<Real> &w,
                                      AthenaArray<Real> &def_coeff,
                                      AthenaArray<Real> &derivetive) {
  FLD2 *pfld = pmy_block_->prfld2;
  AthenaArray<Real> sigma_r = pfld->sigma_r;
  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real dy = pmy_block_->pcoord->dx2f(js);
  Real dz = pmy_block_->pcoord->dx3f(ks);
  Real idx = 1.0/dx;
  Real idy = 1.0/dy;
  Real idz = 1.0/dz;
  Real hidx = 0.5*idx;
  Real hidy = 0.5*idy;
  Real hidz = 0.5*idz;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        def_coeff(NewtonRaphsonFLD::DRHO,k,j,i) = w(IDN,k,j,i);
        // for derivetive of temperature to gas energy
#if GENERAL_EOS
        Real rho = w(IDN,k,j,i);
        Real egas = u_gas_(k,j,i);
        Real temperature = pmy_block_->peos->TempFromRhoEg(rho, egas);
        Real dlnT_dlnE = pmy_block_->peos->DlnTDlnEgasFromRhoEg(rho, egas);
        def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i) = temperature/egas * dlnT_dlnE;
#else
        Real gm1 = pmy_block_->peos->GetGamma() - 1.0;
        def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i) = gm1/w(IDN,k,j,i);
#endif

        // for lambda and coefficient of diff term
        AthenaArray<Real> dEr;
        dEr.NewAthenaArray(3);
        dEr(0) = hidx*(u_pre(k,j,i+1) - u_pre(k,j,i-1));
        dEr(1) = hidy*(u_pre(k,j+1,i) - u_pre(k,j-1,i));
        dEr(2) = hidz*(u_pre(k+1,j,i) - u_pre(k-1,j,i));
        Real gradE = std::sqrt(SQR(dEr(0)) + SQR(dEr(1)) + SQR(dEr(2)));

        // not to use loop for better performance
        // compute derivetive at faces and store in derivetive array

        Real sigma_rface, R_face, lambda_face;
        if (pfld->fixed_flux_limitter) lambda_face = ONE_3RD;
        Real gx, gy, gz, gradE_face, E_face;

        // for i-1/2 face
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j,i-1)),
                      std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j,i-1)/(sigma_r(k,j,i) + sigma_r(k,j,i-1)),
                      2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        gx = (u_pre(k,j,i-1) - u_pre(k,j,i))*idx;
        gy = 0.25*idy*((u_pre(k,j+1,i-1) - u_pre(k,j-1,i-1)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
        gz = 0.25*idz*((u_pre(k+1,j,i-1) - u_pre(k-1,j,i-1)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
        gradE_face = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        E_face = 0.5*(u_pre(k,j,i) + u_pre(k,j,i-1));
        R_face = gradE_face/(sigma_rface*E_face);
        if (!pfld->fixed_flux_limitter) lambda_face = (2.0+R_face)/(6.0+2.0*R_face+R_face*R_face);
        derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i) = pfld->c_ph*lambda_face/sigma_rface;

         // for i+1/2 face
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j,i+1)),
                      std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j,i+1)/(sigma_r(k,j,i) + sigma_r(k,j,i+1)),
                      2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        gx = (u_pre(k,j,i+1) - u_pre(k,j,i))*idx;
        gy = 0.25*idy*((u_pre(k,j+1,i+1) - u_pre(k,j-1,i+1)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
        gz = 0.25*idz*((u_pre(k+1,j,i+1) - u_pre(k-1,j,i+1)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
        gradE_face = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        E_face = 0.5*(u_pre(k,j,i) + u_pre(k,j,i+1));
        R_face = gradE_face/(sigma_rface*E_face);
        if (!pfld->fixed_flux_limitter) lambda_face = (2.0+R_face)/(6.0+2.0*R_face+R_face*R_face);
        derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i) = pfld->c_ph*lambda_face/sigma_rface;

        // for j-1/2 face
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j-1,i)),
                      std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j-1,i)/(sigma_r(k,j,i) + sigma_r(k,j-1,i)),
                      2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        gx = 0.25*idx*((u_pre(k,j-1,i+1) - u_pre(k,j-1,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
        gy = (u_pre(k,j-1,i) - u_pre(k,j,i))*idy;
        gz = 0.25*idz*((u_pre(k+1,j-1,i) - u_pre(k-1,j-1,i)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
        gradE_face = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        E_face = 0.5*(u_pre(k,j,i) + u_pre(k,j-1,i));
        R_face = gradE_face/(sigma_rface*E_face);
        if (!pfld->fixed_flux_limitter) lambda_face = (2.0+R_face)/(6.0+2.0*R_face+R_face*R_face);
        derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i) = pfld->c_ph*lambda_face/sigma_rface;

        // for j+1/2 face
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j+1,i)),
                      std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j+1,i)/(sigma_r(k,j,i) + sigma_r(k,j+1,i)),
                      2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        gx = 0.25*idx*((u_pre(k,j+1,i+1) - u_pre(k,j+1,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
        gy = (u_pre(k,j+1,i) - u_pre(k,j,i))*idy;
        gz = 0.25*idz*((u_pre(k+1,j+1,i) - u_pre(k-1,j+1,i)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
        gradE_face = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        E_face = 0.5*(u_pre(k,j,i) + u_pre(k,j+1,i));
        R_face = gradE_face/(sigma_rface*E_face);
        if (!pfld->fixed_flux_limitter) lambda_face = (2.0+R_face)/(6.0+2.0*R_face+R_face*R_face);
        derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i) = pfld->c_ph*lambda_face/sigma_rface;

        // for k-1/2 face
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k-1,j,i)),
                      std::max(2.0*sigma_r(k,j,i)*sigma_r(k-1,j,i)/(sigma_r(k,j,i) + sigma_r(k-1,j,i)),
                      2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        gx = 0.25*idx*((u_pre(k-1,j,i+1) - u_pre(k-1,j,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
        gy = 0.25*idy*((u_pre(k-1,j+1,i) - u_pre(k-1,j-1,i)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
        gz = (u_pre(k-1,j,i) - u_pre(k,j,i))*idz;
        gradE_face = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        E_face = 0.5*(u_pre(k,j,i) + u_pre(k-1,j,i));
        R_face = gradE_face/(sigma_rface*E_face);
        if (!pfld->fixed_flux_limitter) lambda_face = (2.0+R_face)/(6.0+2.0*R_face+R_face*R_face);
        derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i) = pfld->c_ph*lambda_face/sigma_rface;

        // for k+1/2 face
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k+1,j,i)),
                      std::max(2.0*sigma_r(k,j,i)*sigma_r(k+1,j,i)/(sigma_r(k,j,i) + sigma_r(k+1,j,i)),
                      2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        gx = 0.25*idx*((u_pre(k+1,j,i+1) - u_pre(k+1,j,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
        gy = 0.25*idy*((u_pre(k+1,j+1,i) - u_pre(k+1,j-1,i)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
        gz = (u_pre(k+1,j,i) - u_pre(k,j,i))*idz;
        gradE_face = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        E_face = 0.5*(u_pre(k,j,i) + u_pre(k+1,j,i));
        R_face = gradE_face/(sigma_rface*E_face);
        if (!pfld->fixed_flux_limitter) lambda_face = (2.0+R_face)/(6.0+2.0*R_face+R_face*R_face);
        derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i) = pfld->c_ph*lambda_face/sigma_rface;

        Real R_center, lambda_center;
        if (pfld->fixed_flux_limitter) lambda_center = ONE_3RD;

        // for P:\nabla v
        R_center = gradE/(sigma_r(k,j,i)*u_pre(k,j,i)); // center
        if (!pfld->fixed_flux_limitter) lambda_center = (2.0+R_center)/(6.0+2.0*R_center+R_center*R_center);
        Real chi = lambda_center+std::pow(lambda_center*R_center,2);

        AthenaArray<Real> ngrad;
        ngrad.NewAthenaArray(3);
        ngrad(0) = dEr(0)/(gradE+TINY_NUMBER);
        ngrad(1) = dEr(1)/(gradE+TINY_NUMBER);
        ngrad(2) = dEr(2)/(gradE+TINY_NUMBER);

        AthenaArray<Real> dv_dx;
        dv_dx.NewAthenaArray(3);
        dv_dx(0) = hidx*(w(IVX,k,j,i+1) - w(IVX,k,j,i-1));
        dv_dx(1) = hidy*(w(IVY,k,j+1,i) - w(IVY,k,j-1,i));
        dv_dx(2) = hidz*(w(IVZ,k+1,j,i) - w(IVZ,k-1,j,i));

        def_coeff(NewtonRaphsonFLD::DDV,k,j,i) = 0.0;
        Real DDV_sum = 0.0;
        Real chi_term_diag = 0.5*(1.-chi), chi_term_all = 0.5*(3.*chi-1.);
        // not to use loop for better performance
        DDV_sum += (chi_term_diag + chi_term_all*ngrad(0)*ngrad(0)) * dv_dx(0);
        DDV_sum += (                chi_term_all*ngrad(1)*ngrad(0)) * dv_dx(1);
        DDV_sum += (                chi_term_all*ngrad(2)*ngrad(0)) * dv_dx(2);
        DDV_sum += (                chi_term_all*ngrad(0)*ngrad(1)) * dv_dx(0);
        DDV_sum += (chi_term_diag + chi_term_all*ngrad(1)*ngrad(1)) * dv_dx(1);
        DDV_sum += (                chi_term_all*ngrad(2)*ngrad(1)) * dv_dx(2);
        DDV_sum += (                chi_term_all*ngrad(0)*ngrad(2)) * dv_dx(0);
        DDV_sum += (                chi_term_all*ngrad(1)*ngrad(2)) * dv_dx(1);
        DDV_sum += (chi_term_diag + chi_term_all*ngrad(2)*ngrad(2)) * dv_dx(2);
        def_coeff(NewtonRaphsonFLD::DDV,k,j,i) = DDV_sum;
      }
    }
  }

  if (pfld->cut_diff) {
    if (pmy_driver_->fshowdef_ && pmy_block_->gid == 0)
      std::cout << "Cutting diffusion term coefficients to zero." << std::endl;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int n = 0; n < 6; n++) {
            derivetive(NewtonRaphsonFLD::dFr_dEr_xm+n,k,j,i) = 0.0;
          }
        }
      }
    }
  }

  if (pfld->cut_Pnablav) {
    if (pmy_driver_->fshowdef_ && pmy_block_->gid == 0)
      std::cout << "Cutting P:nabla v term coefficients to zero." << std::endl;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          def_coeff(NewtonRaphsonFLD::DDV,k,j,i) = 0.0;
        }
      }
    }
  }

  if (pmy_driver_->fshowdef_ && pmy_block_->gid == 0) {
    // print everything
    int i = (is + ie) / 2;
    int j = (js + je) / 2;
    int k = (ks + ke) / 2;
    std::cout << "At (k,j,i) = (" << k << "," << j << "," << i << "):" << std::endl;
    std::cout << "  sigma_p = " << pfld->sigma_p(k,j,i) << ", sigma_r = " << pfld->sigma_r(k,j,i) << std::endl;
    std::cout << "  def_coeff.DCOUPLE = " << def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i)
              << ", def_coeff.DDV = " << def_coeff(NewtonRaphsonFLD::DDV,k,j,i) << std::endl;
    for (int ii = 0; ii < 6; ++ii)
      std::cout << "  derivetive.dFr_dEr_xm+"<< ii <<" = " << derivetive(NewtonRaphsonFLD::dFr_dEr_xm+ii,k,j,i) << std::endl;
  }
}

void NRFLD::CalculateCoefficients(const AthenaArray<Real> &u_rad_old,
                                  const AthenaArray<Real> &u_rad_new,
                                  // const AthenaArray<Real> &u_gas_old,
                                  // const AthenaArray<Real> &u_gas_new,
                                  const AthenaArray<Real> &def_coeff,
                                  AthenaArray<Real> &coeff,
                                  AthenaArray<Real> &derivetive,
                                  AthenaArray<Real> &src,
                                  Real dt) {
  FLD2 *pfld = pmy_block_->prfld2;
  AthenaArray<Real> &u_gas_new = u_gas_; // caution! this should be in argument
  AthenaArray<Real> &u_gas_old = pfld->u_gas; // caution! this should be in argument

  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real idx2 = 1.0/(dx*dx);


  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real c_sigma_p = pfld->c_ph*pfld->sigma_p(k,j,i);

        // caution! following can be optimized (only once per step)
        Real sum_dcp = 0.0;
        sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i);
        sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i);
        sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i);
        sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i);
        sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i);
        sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i);

        Real T_gas_new;
#if GENERAL_EOS
        T_gas_new = pmy_block_->peos->TempFromRhoEg(def_coeff(NewtonRaphsonFLD::DRHO,k,j,i), u_gas_new(k,j,i));
#else
        T_gas_new = def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i)*u_gas_new(k,j,i);
#endif
        Real src_term = c_sigma_p*(pfld->a_r*std::pow(T_gas_new,4) - u_rad_new(k,j,i));
        Real Pnablav = def_coeff(NewtonRaphsonFLD::DDV,k,j,i)*u_rad_new(k,j,i);
        Real diff_term = 0.0;
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i)*(u_rad_new(k,j,i-1) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i)*(u_rad_new(k,j,i+1) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i)*(u_rad_new(k,j-1,i) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i)*(u_rad_new(k,j+1,i) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i)*(u_rad_new(k-1,j,i) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i)*(u_rad_new(k+1,j,i) - u_rad_new(k,j,i));
        diff_term *= idx2;

        derivetive(NewtonRaphsonFLD::Fg,k,j,i) = (u_gas_new(k,j,i) - u_gas_old(k,j,i)) + dt * src_term;
        derivetive(NewtonRaphsonFLD::Fr,k,j,i) = (u_rad_new(k,j,i) - u_rad_old(k,j,i)) - dt *(src_term - Pnablav + diff_term);

        derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i) = 1.0 + 4.0*dt*c_sigma_p*pfld->a_r*std::pow(T_gas_new,3)*def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i);
        derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i) = -dt*c_sigma_p;
        derivetive(NewtonRaphsonFLD::dFr_deg,k,j,i) = -4.0*dt*c_sigma_p*pfld->a_r*std::pow(T_gas_new,3)*def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i);
        derivetive(NewtonRaphsonFLD::dFr_dEr,k,j,i) = 1.0 + dt*(c_sigma_p + def_coeff(NewtonRaphsonFLD::DDV,k,j,i) + idx2*sum_dcp);


        coeff(linearSolver::DCCF,k,j,i) = sum_dcp;
        coeff(linearSolver::DCCS,k,j,i) = 1.0 + dt*(pfld->c_ph*pfld->sigma_p(k,j,i)+def_coeff(NewtonRaphsonFLD::DDV,k,j,i)); // from dFr_dEr
        coeff(linearSolver::DCCS,k,j,i) += -(derivetive(NewtonRaphsonFLD::dFr_deg,k,j,i)/derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i))*derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i);

        
        coeff(linearSolver::DXMF,k,j,i) = -derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i);
        coeff(linearSolver::DXPF,k,j,i) = -derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i);
        coeff(linearSolver::DYMF,k,j,i) = -derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i);
        coeff(linearSolver::DYPF,k,j,i) = -derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i);
        coeff(linearSolver::DZMF,k,j,i) = -derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i);
        coeff(linearSolver::DZPF,k,j,i) = -derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i);

        src(k,j,i) = -derivetive(NewtonRaphsonFLD::Fr,k,j,i) + (derivetive(NewtonRaphsonFLD::dFr_deg,k,j,i)/derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i))*derivetive(NewtonRaphsonFLD::Fg,k,j,i);

        // output
        if (pmy_driver_->fshowdef_ && pmy_block_->gid == 0 &&
            k == (ks+ke) / 2 && j == (js+je) / 2 && i == (is+ie) / 2) {
          Real T_gas_old;
#if GENERAL_EOS
          T_gas_old = pmy_block_->peos->TempFromRhoEg(def_coeff(NewtonRaphsonFLD::DRHO,k,j,i), u_gas_old(k,j,i));
#else
          T_gas_old = def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i)*u_gas_old(k,j,i);
#endif
          Real T_rad_new = std::pow(u_rad_new(k,j,i)/pfld->a_r, 0.25);
          Real T_rad_old = std::pow(u_rad_old(k,j,i)/pfld->a_r, 0.25);
          std::cout << "At (" << k << "," << j << "," << i << "):" << std::endl;
          std::cout << "  dt = " << dt << std::endl;
          std::cout << "  u_gas_new = " << u_gas_new(k,j,i) << ", u_gas_old = " << u_gas_old(k,j,i) << std::endl;
          std::cout << "  u_rad_new = " << u_rad_new(k,j,i) << ", u_rad_old = " << u_rad_old(k,j,i) << std::endl;
          std::cout << "  T_gas_new = " << T_gas_new << ", T_gas_old = " << T_gas_old << std::endl;
          std::cout << "  T_rad_new = " << T_rad_new << ", T_rad_old = " << T_rad_old << std::endl; 
          std::cout << "  src_term = " << src_term << ", Pnablav = " << Pnablav << ", diff_term = " << diff_term << std::endl;
          std::cout << "  derivetive.Fg = " << derivetive(NewtonRaphsonFLD::Fg,k,j,i)
                    << ", derivetive.Fr = " << derivetive(NewtonRaphsonFLD::Fr,k,j,i) << std::endl;
          std::cout << "  derivetive.dFg_deg = " << derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i)
                    << ", derivetive.dFg_dEr = " << derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i) << std::endl;
          std::cout << "  sum_dcp = " << sum_dcp << std::endl;
          std::cout << "  derivetive.dFr_deg = " << derivetive(NewtonRaphsonFLD::dFr_deg,k,j,i)
                    << ", derivetive.dFr_dEr = " << derivetive(NewtonRaphsonFLD::dFr_dEr,k,j,i) << std::endl;
          std::cout << "  coeff.DCCF = " << coeff(linearSolver::DCCF,k,j,i)
                    << ", coeff.DCCS = " << coeff(linearSolver::DCCS,k,j,i) << std::endl;
          // for (int n = 0; n < 6; ++n) {
          //   std::cout << "  coeff.DXMF+"<< n <<" = " << coeff(linearSolver::DXMF+n,k,j,i) << std::endl;
          //   std::cout << "  coeff.DXMS+"<< n <<" = " << coeff(linearSolver::DXMS+n,k,j,i) << std::endl;
          // }
          std::cout << "  i-face coeffs: " << coeff(linearSolver::DXMF,k,j,i) << ", " << coeff(linearSolver::DXPF,k,j,i) << std::endl;
          std::cout << "  j-face coeffs: " << coeff(linearSolver::DYMF,k,j,i) << ", " << coeff(linearSolver::DYPF,k,j,i) << std::endl;
          std::cout << "  k-face coeffs: " << coeff(linearSolver::DZMF,k,j,i) << ", " << coeff(linearSolver::DZPF,k,j,i) << std::endl;
          std::cout << "  src = " << src(k,j,i) << std::endl;
          std::cout << "  delta_u = " << delta_u_(k,j,i) << std::endl;
        }
      }
    }
  }
}

void NRFLD::CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
              const AthenaArray<Real> &u_old,
              const AthenaArray<Real> &coeff,
              const AthenaArray<Real> &def_coeff,
              bool th) {
  AthenaArray<Real> &u_gas = u_gas_; // caution! this should be in argument
  FLD2 *pfld = pmy_block_->prfld2;
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
        Real T_gas;
#if GENERAL_EOS
        T_gas = pmy_block_->peos->TempFromRhoEg(def_coeff(NewtonRaphsonFLD::DRHO,k,j,i), u_gas(k,j,i));
#else
        T_gas = def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i)*u_gas(k,j,i);
#endif
        Real src_term = pfld->c_ph*pfld->sigma_p(k,j,i)*(pfld->a_r*std::pow(T_gas,4) - u(k,j,i));
        Real Pnablav = def_coeff(NewtonRaphsonFLD::DDV,k,j,i)*u(k,j,i);
        Real diff_term = 0.0;
        diff_term += -coeff(linearSolver::DXMF,k,j,i)*(u(k,j,i-1) - u(k,j,i));
        diff_term += -coeff(linearSolver::DXPF,k,j,i)*(u(k,j,i+1) - u(k,j,i));
        diff_term += -coeff(linearSolver::DYMF,k,j,i)*(u(k,j-1,i) - u(k,j,i));
        diff_term += -coeff(linearSolver::DYPF,k,j,i)*(u(k,j+1,i) - u(k,j,i));
        diff_term += -coeff(linearSolver::DZMF,k,j,i)*(u(k-1,j,i) - u(k,j,i));
        diff_term += -coeff(linearSolver::DZPF,k,j,i)*(u(k+1,j,i) - u(k,j,i));
        diff_term *= idx2;

        Real Fg = (u_gas(k,j,i) - pfld->u_gas(k,j,i)) + dt* src_term;
        Real Fr = (u(k,j,i)     - u_old(k,j,i))       - dt*(src_term - Pnablav + diff_term);
        // def(k,j,i) = std::abs(Fg) + std::abs(Fr);
        def(k,j,i) = Fr;

        if (pmy_driver_->fshowdef_ && pmy_block_->gid == 0 &&
            k==(kl+ku)/2 && j==(jl+ju)/2 && i==(il+iu)/2) {
          Real T_rad = std::pow(u(k,j,i)/pfld->a_r, 0.25);
          std::cout << "At (" << k << "," << j << "," << i << "):" << std::endl;
          std::cout << "  u_gas = " << u_gas(k,j,i) << ", pfld->u_gas = " << pfld->u_gas(k,j,i) << std::endl;
          std::cout << "  u_rad = " << u(k,j,i) << ", u_rad_old = " << u_old(k,j,i) << std::endl;
          std::cout << "  T_gas = " << T_gas << ", T_rad = " << T_rad << std::endl;
          std::cout << "  src_term = " << src_term << ", Pnablav = " << Pnablav << ", diff_term = " << diff_term << std::endl;
          std::cout << "  Fg = " << Fg << ", Fr = " << Fr << std::endl;
          std::cout << "  defect = " << def(k,j,i) << std::endl;
        }
      }
    }
  }

  return;
}


// delta is reset to zero in this function
void NRFLD::AddDifference(AthenaArray<Real> &u_rad,
                          AthenaArray<Real> &delta_u,
                          const AthenaArray<Real> &derivetive) {

  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;
  FLD2 *pfld = pmy_block_->prfld2;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (!pfld->fixed_u_rad)
          u_rad(k,j,i) += delta_u(k,j,i);
        u_gas_(k,j,i) += -(derivetive(NewtonRaphsonFLD::Fg,k,j,i) + derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i)*delta_u(k,j,i)) / derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i);
        
        // reset delta_u to zero for next iteration
        delta_u(k,j,i) = 0.0;
      }
    }
  }
  // if(pmy_driver_->fshowdef_)  {
  //   int k = (ks + ke) / 2;
  //   int j = (js + je) / 2;
  //   int i = (is + ie) / 2;
  //   Real u_gas_delta = -(derivetive(NewtonRaphsonFLD::Fg,k,j,i) + derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i)*delta_u(k,j,i)) / derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i);
  //   std::cout << "At (" << k << "," << j << "," << i << "):" << std::endl;
  //   std::cout << "u_rad_delta = " << delta_u(k,j,i) << std::endl;
  //   std::cout << "u_gas_delta = " << u_gas_delta << std::endl;
  //   std::cout << "derivetive.Fg = " << derivetive(NewtonRaphsonFLD::Fg,k,j,i)
  //             << ", derivetive.dFg_dEr = " << derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i)
  //             << ", derivetive.dFg_deg = " << derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i) << std::endl;
  // }
  return;
}

void NRFLD::ApplyPhysicalBoundary() {
  for (int dir = 0; dir < 6; dir++) {
    // check if neighbor block exists
    if (pmy_block_->pbval->block_bcs[dir] == BoundaryFlag::block) continue;
    BoundaryFace face = static_cast<BoundaryFace>(dir);
    BoundaryFlag bflag = pmy_driver_->nr_mesh_bcs_[dir];
    // To do: add other boundary conditions like outflow
    if (bflag == BoundaryFlag::user) {
      pmy_driver_->NRBoundaryFunction_[dir](pmy_block_, u_, u_gas_,
      pmy_block_->pcoord, pmy_block_->phydro->w,
      pmy_block_->pmy_mesh->time, pmy_driver_->dt_,
      pmy_block_->is, pmy_block_->ie,
      pmy_block_->js, pmy_block_->je,
      pmy_block_->ks, pmy_block_->ke,
      NGHOST);
    }
  }
  
  return;
}
