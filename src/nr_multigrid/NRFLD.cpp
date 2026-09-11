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
#include <cstdlib>
#include <cstring>    // memset, memcpy
#include <iostream>
#include <limits>
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


namespace {

// In the optically thick LTE limit, the Newton solve can only place a*T^4 and
// Er on adjacent floating-point numbers.  Their subtraction then changes sign
// from one Newton iteration to the next, and c*sigma amplifies that irreducible
// roundoff into a large, grid-dependent source.  Treat differences within the
// rounding error of the temperature-to-energy conversion as zero.  The factor
// allows for the multiply chain T = DCOUPLE*egas and a*T*T*T*T.
inline Real ResolvableThermalDifference(Real equilibrium_rad, Real u_rad) {
  const Real difference = equilibrium_rad - u_rad;
  const Real scale = std::max(std::abs(equilibrium_rad), std::abs(u_rad));
  const Real roundoff = 32.0*std::numeric_limits<Real>::epsilon()*scale;
  return (std::abs(difference) <= roundoff) ? 0.0 : difference;
}

inline RadFLD::ClosureValues RequireFaceClosure(const Real gradient,
                                                const Real opacity,
                                                const Real erad,
                                                const bool fixed,
                                                const char *face, const int k,
                                                const int j, const int i) {
  const RadFLD::ClosureValues closure =
      RadFLD::EvaluateClosure(gradient, opacity, erad, fixed);
  if (!closure.valid) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NR-FLD face closure at " << face
        << " cell (k,j,i)=(" << k << "," << j << "," << i << ")"
        << ": gradient=" << gradient << " opacity=" << opacity
        << " Er=" << erad
        << ". Opacity and Er must be finite and non-negative;"
        << " fixed limiter with zero opacity and nonzero gradient is unsupported.\n";
    ATHENA_ERROR(msg);
  }
  return closure;
}

inline Real FaceGradient(const Real gx, const Real gy, const Real gz) {
  return std::hypot(gx, std::hypot(gy, gz));
}

inline Real FaceEnergy(const Real left, const Real right) {
  return 0.5*left + 0.5*right;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn NRFLDDriver::NRFLDDriver(Mesh *pm, ParameterInput *pin)
//! \brief NRFLDDriver constructor

NRFLDDriver::NRFLDDriver(Mesh *pm, ParameterInput *pin)
    : NewtonRaphsonDriver(pm, 1, linearSolver::NCOEFF, linearSolver::NMATRIX) {
  eps_ = pin->GetOrAddReal("nrfld", "nr_threshold", -1.0);
  niter_ = pin->GetOrAddInteger("nrfld", "nr_niteration", -1);
  nr_max_iterations_ = pin->GetOrAddInteger("nrfld", "nr_max_iterations", 100);
  const bool verbosity_in_input =
      pin->DoesParameterExist("nrfld", "diagnostic_verbosity") != 0;
  fshowdef_ = pin->GetOrAddBoolean("nrfld", "show_defect", fshowdef_);
  diagnostic_verbosity_ = pin->GetOrAddInteger(
      "nrfld", "diagnostic_verbosity", fshowdef_ ? 2 : 0);
  // show_defect remains a compatibility alias for the former all-or-nothing
  // diagnostics.  An explicit verbosity setting takes precedence.
  if (!verbosity_in_input && fshowdef_) diagnostic_verbosity_ = 2;
  fshowdef_ = diagnostic_verbosity_ > 0;
  use_mg_smoothing_fallback_ =
      pin->GetOrAddBoolean("nrfld", "nr_use_mg_smoothing_fallback", true);
  mg_coarse_retry_max_ = pin->GetOrAddInteger("nrfld", "nr_mg_coarse_retry_max", 4);
  mg_coarse_retry_factor_ =
      pin->GetOrAddReal("nrfld", "nr_mg_coarse_retry_factor", 0.5);
  mg_coarse_retry_min_scale_ =
      pin->GetOrAddReal("nrfld", "nr_mg_coarse_retry_min_scale", 0.0625);
  max_backtrack_ = pin->GetOrAddInteger("nrfld", "nr_backtrack_max", 4);
  backtrack_factor_ = pin->GetOrAddReal("nrfld", "nr_backtrack_factor", 0.5);
  min_step_scale_ = pin->GetOrAddReal("nrfld", "nr_min_step_scale", 0.05);
  if (eps_ < 0.0 && niter_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "Either \"nr_threshold\" or \"nr_niteration\" parameter must be set "
        << "in the <nrfld> block." << std::endl
        << "When both parameters are specified, \"nr_niteration\" is ignored." << std::endl
        << "Set \"nr_threshold = 0.0\" for automatic convergence control." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (diagnostic_verbosity_ < 0 || diagnostic_verbosity_ > 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"diagnostic_verbosity\" must be 0 (off), 1 (iteration "
        << "summary), or 2 (maximum-cell stencil)." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (niter_ < -1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_niteration\" must be >= 0 or omitted." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!std::isfinite(eps_)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_threshold\" must be finite." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (nr_max_iterations_ <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_max_iterations\" must be positive." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (max_backtrack_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_backtrack_max\" must be >= 0." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(backtrack_factor_ > 0.0 && backtrack_factor_ < 1.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_backtrack_factor\" must satisfy 0 < factor < 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(min_step_scale_ > 0.0 && min_step_scale_ <= 1.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_min_step_scale\" must satisfy 0 < scale <= 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mg_coarse_retry_max_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_mg_coarse_retry_max\" must be >= 0." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(mg_coarse_retry_factor_ > 0.0 && mg_coarse_retry_factor_ < 1.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_mg_coarse_retry_factor\" must satisfy 0 < factor < 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(mg_coarse_retry_min_scale_ > 0.0 && mg_coarse_retry_min_scale_ <= 1.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLDDriver::NRFLDDriver" << std::endl
        << "\"nr_mg_coarse_retry_min_scale\" must satisfy 0 < scale <= 1." << std::endl;
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
  const bool trace_dtor = (std::getenv("ATHENA_TRACE_DTOR") != nullptr);
  if (NRMGFLD_ENABLED) {
    if (trace_dtor) std::cout << "[DTOR] NRFLDDriver delete plmgd_" << std::endl;
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
    u_gas_iter_backup_(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    gas_defect_(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    ngh_(NGHOST),
    // A cell-by-cell cap destroys the spatial structure of the correction
    // returned by the linear solve, especially across SMR/AMR interfaces.
    // Positivity and nonlinear globalization are handled by the floors and
    // the driver-wide Newton backtracking, respectively.  Keep the local cap
    // available only as an explicitly requested diagnostic safeguard.
    max_update_fraction_(pin->GetOrAddReal("nrfld", "max_update_fraction", -1.0)),
    fixed_linear_coefficients_initialized_(false)
    {
    last_delta_rad_.NewAthenaArray(2, pmb->ncells3, pmb->ncells2, pmb->ncells1);
    last_delta_rad_.ZeroClear();

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
  const bool trace_dtor = (std::getenv("ATHENA_TRACE_DTOR") != nullptr);
  if (NRMGFLD_ENABLED) {
    if (trace_dtor) {
      std::cout << "[DTOR] NRFLD gid=" << pmy_block_->gid
                << " delete plmg_ ptr=" << plmg_
                << " block.plmg=" << pmy_block_->plmg << std::endl;
    }
    delete plmg_;
  }
}

void NRFLD::LoadVariables() {
  FLD *pfld = pmy_block_->prfld;
  fixed_linear_coefficients_initialized_ = false;
  pfld->LoadHydroVariables(pmy_block_->phydro->w, pfld->u_gas);
  if (last_delta_rad_.data() == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NRFLD::LoadVariables" << std::endl
        << "Debug delta array is not allocated."
        << " ptr=" << static_cast<const void*>(last_delta_rad_.data())
        << " dims=(" << last_delta_rad_.GetDim4() << ","
        << last_delta_rad_.GetDim3() << ","
        << last_delta_rad_.GetDim2() << ","
        << last_delta_rad_.GetDim1() << ")" << std::endl;
    ATHENA_ERROR(msg);
  }
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
        last_delta_rad_(0,k,j,i) = 0.0;
        last_delta_rad_(1,k,j,i) = 0.0;

      }
    }
  }
  return;
}

void NRFLD::UpdateHydroVariables() {
  FLD *pfld = pmy_block_->prfld;
  pfld->UpdateHydroVariables(pmy_block_->phydro->w,
                             pmy_block_->phydro->u,
                             u_, u_gas_);
  return;
}

void NRFLD::StoreIterate() {
  NewtonRaphson::StoreIterate();
  u_gas_iter_backup_ = u_gas_;
  user_diagnostic_counters_iter_backup_ =
      pmy_block_->prfld->user_diagnostic_counters;
}

void NRFLD::RestoreIterate() {
  NewtonRaphson::RestoreIterate();
  u_gas_ = u_gas_iter_backup_;
  pmy_block_->prfld->user_diagnostic_counters =
      user_diagnostic_counters_iter_backup_;
}

void NRFLD::CalculateCoefficientsOnce(const AthenaArray<Real> &u_pre,
                                      const AthenaArray<Real> &w,
                                      AthenaArray<Real> &def_coeff,
                                      AthenaArray<Real> &derivetive) {
  FLD *pfld = pmy_block_->prfld;
  const AthenaArray<Real> &sigma_r = pfld->sigma_r;
  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real dy = pmy_block_->pcoord->dx2f(js);
  Real dz = pmy_block_->pcoord->dx3f(ks);
  Real idx = 1.0/dx;
  Real idy = 1.0/dy;
  Real idz = 1.0/dz;
  const int check_il = std::max(0, is-1);
  const int check_iu = std::min(u_pre.GetDim1()-1, ie+1);
  const int check_jl = std::max(0, js-1);
  const int check_ju = std::min(u_pre.GetDim2()-1, je+1);
  const int check_kl = std::max(0, ks-1);
  const int check_ku = std::min(u_pre.GetDim3()-1, ke+1);
  for (int k=check_kl; k<=check_ku; ++k) {
    for (int j=check_jl; j<=check_ju; ++j) {
      for (int i=check_il; i<=check_iu; ++i) {
        if (!std::isfinite(u_pre(k,j,i)) || u_pre(k,j,i) < 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in NR-FLD radiation energy" << std::endl
              << "Er at (k,j,i)=(" << k << "," << j << "," << i << ") is "
              << u_pre(k,j,i)
              << "; radiation energy must be finite and non-negative.\n";
          ATHENA_ERROR(msg);
        }
      }
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      // The lower x-face reuses the upper face from i-1 below.  This loop has
      // a genuine loop-carried dependency and must not be declared SIMD.
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
        // not to use loop for better performance
        // compute derivetive at faces and store in derivetive array

        Real sigma_rface;
        Real gx, gy, gz, gradE_face, E_face;

        // The + face of the previous active cell is exactly the - face of
        // this cell.  Reuse it in block interiors; only the lower block face
        // needs a separate stencil evaluation.
        if (i == is) {
          // for i-1/2 face
          sigma_rface = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j,i-1), idx);
          gx = (u_pre(k,j,i-1) - u_pre(k,j,i))*idx;
          gy = 0.25*idy*((u_pre(k,j+1,i-1) - u_pre(k,j-1,i-1)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
          gz = 0.25*idz*((u_pre(k+1,j,i-1) - u_pre(k-1,j,i-1)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
          gradE_face = FaceGradient(gx, gy, gz);
          E_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j,i-1));
          const RadFLD::ClosureValues closure = RequireFaceClosure(
              gradE_face, sigma_rface, E_face, pfld->fixed_flux_limiter,
              "x-", k, j, i);
          derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i) =
              pfld->c_ph*closure.lambda_over_opacity;
        } else {
          derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i) =
              derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i-1);
        }

         // for i+1/2 face
        sigma_rface = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j,i+1), idx);
        gx = (u_pre(k,j,i+1) - u_pre(k,j,i))*idx;
        gy = 0.25*idy*((u_pre(k,j+1,i+1) - u_pre(k,j-1,i+1)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
        gz = 0.25*idz*((u_pre(k+1,j,i+1) - u_pre(k-1,j,i+1)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
        gradE_face = FaceGradient(gx, gy, gz);
        E_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j,i+1));
        const RadFLD::ClosureValues closure_xp = RequireFaceClosure(
            gradE_face, sigma_rface, E_face, pfld->fixed_flux_limiter,
            "x+", k, j, i);
        derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i) =
            pfld->c_ph*closure_xp.lambda_over_opacity;

        if (j == js) {
          // for j-1/2 face
          sigma_rface = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j-1,i), idx);
          gx = 0.25*idx*((u_pre(k,j-1,i+1) - u_pre(k,j-1,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
          gy = (u_pre(k,j-1,i) - u_pre(k,j,i))*idy;
          gz = 0.25*idz*((u_pre(k+1,j-1,i) - u_pre(k-1,j-1,i)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
          gradE_face = FaceGradient(gx, gy, gz);
          E_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j-1,i));
          const RadFLD::ClosureValues closure_ym = RequireFaceClosure(
              gradE_face, sigma_rface, E_face, pfld->fixed_flux_limiter,
              "y-", k, j, i);
          derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i) =
              pfld->c_ph*closure_ym.lambda_over_opacity;
        } else {
          derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i) =
              derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j-1,i);
        }

        // for j+1/2 face
        sigma_rface = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j+1,i), idx);
        gx = 0.25*idx*((u_pre(k,j+1,i+1) - u_pre(k,j+1,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
        gy = (u_pre(k,j+1,i) - u_pre(k,j,i))*idy;
        gz = 0.25*idz*((u_pre(k+1,j+1,i) - u_pre(k-1,j+1,i)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
        gradE_face = FaceGradient(gx, gy, gz);
        E_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j+1,i));
        const RadFLD::ClosureValues closure_yp = RequireFaceClosure(
            gradE_face, sigma_rface, E_face, pfld->fixed_flux_limiter,
            "y+", k, j, i);
        derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i) =
            pfld->c_ph*closure_yp.lambda_over_opacity;

        if (k == ks) {
          // for k-1/2 face
          sigma_rface = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k-1,j,i), idx);
          gx = 0.25*idx*((u_pre(k-1,j,i+1) - u_pre(k-1,j,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
          gy = 0.25*idy*((u_pre(k-1,j+1,i) - u_pre(k-1,j-1,i)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
          gz = (u_pre(k-1,j,i) - u_pre(k,j,i))*idz;
          gradE_face = FaceGradient(gx, gy, gz);
          E_face = FaceEnergy(u_pre(k,j,i), u_pre(k-1,j,i));
          const RadFLD::ClosureValues closure_zm = RequireFaceClosure(
              gradE_face, sigma_rface, E_face, pfld->fixed_flux_limiter,
              "z-", k, j, i);
          derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i) =
              pfld->c_ph*closure_zm.lambda_over_opacity;
        } else {
          derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i) =
              derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k-1,j,i);
        }

        // for k+1/2 face
        sigma_rface = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k+1,j,i), idx);
        gx = 0.25*idx*((u_pre(k+1,j,i+1) - u_pre(k+1,j,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
        gy = 0.25*idy*((u_pre(k+1,j+1,i) - u_pre(k+1,j-1,i)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
        gz = (u_pre(k+1,j,i) - u_pre(k,j,i))*idz;
        gradE_face = FaceGradient(gx, gy, gz);
        E_face = FaceEnergy(u_pre(k,j,i), u_pre(k+1,j,i));
        const RadFLD::ClosureValues closure_zp = RequireFaceClosure(
            gradE_face, sigma_rface, E_face, pfld->fixed_flux_limiter,
            "z+", k, j, i);
        derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i) =
            pfld->c_ph*closure_zp.lambda_over_opacity;
        // Preserve the lagged upper-face diffusion coefficient used by this
        // Newton solve.  The Marshak residual/Jacobian and diagnostics must
        // use this exact coefficient rather than an uninitialized side array
        // or an independently reconstructed approximation.
        pfld->marshak_dface(k,j,i) =
            derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i);

      }
    }
  }

  if (pfld->cut_diff) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          for (int n = 0; n < 6; n++) {
            derivetive(NewtonRaphsonFLD::dFr_dEr_xm+n,k,j,i) = 0.0;
          }
          pfld->marshak_dface(k,j,i) = 0.0;
        }
      }
    }
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
  FLD *pfld = pmy_block_->prfld;
  AthenaArray<Real> &u_gas_new = u_gas_; // caution! this should be in argument
  AthenaArray<Real> &u_gas_old = pfld->u_gas; // caution! this should be in argument

  int is = pmy_block_->is, ie = pmy_block_->ie;
  int js = pmy_block_->js, je = pmy_block_->je;
  int ks = pmy_block_->ks, ke = pmy_block_->ke;
  Real dx = pmy_block_->pcoord->dx1f(is);
  Real idx2 = 1.0/(dx*dx);
  // Apply the Marshak radiation condition only on the physical upper
  // boundary, never on the upper face of an interior z meshblock.
  const bool physical_top =
      pmy_block_->block_size.x3max == pmy_block_->pmy_mesh->mesh_size.x3max;


  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
#pragma omp simd
      for (int i=is; i<=ie; i++) {
        Real c_sigma_p = pfld->c_ph*pfld->sigma_p(k,j,i);

        Real sum_dcp = 0.0;
        if (fixed_linear_coefficients_initialized_) {
          sum_dcp = coeff(linearSolver::DCCF,k,j,i);
        } else {
          sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i);
          sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i);
          sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i);
          sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i);
          sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i);
          sum_dcp += derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i);
        }

        Real T_gas_new;
#if GENERAL_EOS
        T_gas_new = pmy_block_->peos->TempFromRhoEg(def_coeff(NewtonRaphsonFLD::DRHO,k,j,i), u_gas_new(k,j,i));
#else
        T_gas_new = def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i)*u_gas_new(k,j,i);
#endif
        const Real T_gas_new2 = T_gas_new*T_gas_new;
        const Real T_gas_new3 = T_gas_new2*T_gas_new;
        const Real T_gas_new4 = T_gas_new2*T_gas_new2;
        const Real equilibrium_rad = pfld->a_r*T_gas_new4;
        Real src_term = c_sigma_p*ResolvableThermalDifference(
            equilibrium_rad, u_rad_new(k,j,i));
        Real diff_term = 0.0;
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i)*(u_rad_new(k,j,i-1) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i)*(u_rad_new(k,j,i+1) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i)*(u_rad_new(k,j-1,i) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i)*(u_rad_new(k,j+1,i) - u_rad_new(k,j,i));
        diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i)*(u_rad_new(k-1,j,i) - u_rad_new(k,j,i));
        Real marshak_dflux_dE = 0.0;
        if (pfld->marshak_top_boundary && physical_top && k == ke) {
          const Real dface = std::max(pfld->marshak_dface(k,j,i), TINY_NUMBER);
          const Real acoef = pfld->marshak_top_alpha*pfld->c_ph*0.5*dx;
          const Real eb = (dface*u_rad_new(k,j,i)
                           + acoef*pfld->marshak_top_erad_ext)
                          /std::max(dface + acoef, TINY_NUMBER);
          const Real ftop = pfld->marshak_top_alpha*pfld->c_ph
                          *(eb - pfld->marshak_top_erad_ext);
          diff_term += -ftop*dx;
          marshak_dflux_dE = pfld->marshak_top_alpha*pfld->c_ph*dface
                           /std::max(dface + acoef, TINY_NUMBER);
        } else {
          diff_term += derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i)
                     *(u_rad_new(k+1,j,i) - u_rad_new(k,j,i));
        }
        diff_term *= idx2;

        // The implicit subsystem contains only matter-radiation thermal
        // exchange and radiation diffusion. The exchange appears with equal
        // and opposite signs in the gas and radiation residuals.
        derivetive(NewtonRaphsonFLD::Fg,k,j,i) =
            (u_gas_new(k,j,i) - u_gas_old(k,j,i)) + dt*src_term;
        derivetive(NewtonRaphsonFLD::Fr,k,j,i) = (u_rad_new(k,j,i) - u_rad_old(k,j,i))
            - dt*(src_term + diff_term);

        const Real exchange_jacobian = 4.0*dt*c_sigma_p*pfld->a_r*T_gas_new3
            *def_coeff(NewtonRaphsonFLD::DCOUPLE,k,j,i);
        derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i) = 1.0 + exchange_jacobian;
        derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i) = -dt*c_sigma_p;
        derivetive(NewtonRaphsonFLD::dFr_deg,k,j,i) = -exchange_jacobian;
        Real boundary_coeff = sum_dcp;
        if (pfld->marshak_top_boundary && physical_top && k == ke &&
            !fixed_linear_coefficients_initialized_)
          boundary_coeff = sum_dcp
                         - derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i);
        derivetive(NewtonRaphsonFLD::dFr_dEr,k,j,i) =
            1.0 + dt*(c_sigma_p + idx2*(boundary_coeff
                + ((pfld->marshak_top_boundary && physical_top && k == ke)
                   ? dx*marshak_dflux_dE : 0.0)));


        if (pfld->fixed_u_rad) {
          // Erad is prescribed rather than an NR unknown.  Leave the linear
          // correction at zero; AddDifference performs the local Newton update
          // of egas with delta_Erad = 0.
          coeff(linearSolver::DCCF,k,j,i) = 0.0;
          coeff(linearSolver::DCCS,k,j,i) = 1.0;
          coeff(linearSolver::DXMF,k,j,i) = 0.0;
          coeff(linearSolver::DXPF,k,j,i) = 0.0;
          coeff(linearSolver::DYMF,k,j,i) = 0.0;
          coeff(linearSolver::DYPF,k,j,i) = 0.0;
          coeff(linearSolver::DZMF,k,j,i) = 0.0;
          coeff(linearSolver::DZPF,k,j,i) = 0.0;
          src(k,j,i) = 0.0;
        } else {
          if (!fixed_linear_coefficients_initialized_) {
            coeff(linearSolver::DCCF,k,j,i) = boundary_coeff;
            coeff(linearSolver::DXMF,k,j,i) =
                -derivetive(NewtonRaphsonFLD::dFr_dEr_xm,k,j,i);
            coeff(linearSolver::DXPF,k,j,i) =
                -derivetive(NewtonRaphsonFLD::dFr_dEr_xp,k,j,i);
            coeff(linearSolver::DYMF,k,j,i) =
                -derivetive(NewtonRaphsonFLD::dFr_dEr_ym,k,j,i);
            coeff(linearSolver::DYPF,k,j,i) =
                -derivetive(NewtonRaphsonFLD::dFr_dEr_yp,k,j,i);
            coeff(linearSolver::DZMF,k,j,i) =
                -derivetive(NewtonRaphsonFLD::dFr_dEr_zm,k,j,i);
            coeff(linearSolver::DZPF,k,j,i) =
                (pfld->marshak_top_boundary && physical_top && k == ke) ? 0.0
                : -derivetive(NewtonRaphsonFLD::dFr_dEr_zp,k,j,i);
          }
          // Eliminate the gas-energy correction from the two-equation
          // Newton system.  Since thermal exchange enters the gas and
          // radiation equations with equal and opposite signs,
          //
          //   dFr/deg = 1 - dFg/deg
          //
          // for the local (non-diffusive) terms.  Writing the resulting
          // Schur complement directly as
          //
          //   (1 + C) - B*C/(1 + B)
          //
          // loses all useful digits when the matter-radiation coupling is
          // very stiff (B,C >> 1).  In the static-equilibrium test C can be
          // O(1e20), which previously produced spurious values such as
          // -65536, -256, or zero for this positive diagonal.  Use the
          // algebraically identical forms
          //
          //   1 - (dFg/dEr)/(dFg/deg)
          //   -(Fr + Fg) + Fg/(dFg/deg)
          //
          // for the local diagonal and reduced right-hand side.  These also
          // preserve the exact cancellation of equal-and-opposite exchange
          // residuals without subtracting two O(C) quantities.
          const Real inv_dFg_deg =
              1.0/derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i);
          coeff(linearSolver::DCCS,k,j,i) =
              1.0 - derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i)*inv_dFg_deg
              // The Robin boundary is a local face-reaction term in the
              // matrix, dt*(dF/dE)/dx.  Keep it in DCCS rather than DCCF:
              // DCCF is rescaled by dt/dx^2 at every MG level, which would
              // otherwise discard or incorrectly scale the boundary term
              // during coefficient restriction.
              + ((pfld->marshak_top_boundary && physical_top && k == ke)
                 ? dt*marshak_dflux_dE/dx : 0.0);
          src(k,j,i) =
              -(derivetive(NewtonRaphsonFLD::Fr,k,j,i)
                + derivetive(NewtonRaphsonFLD::Fg,k,j,i))
              + derivetive(NewtonRaphsonFLD::Fg,k,j,i)*inv_dFg_deg;
        }

      }
    }
  }
  fixed_linear_coefficients_initialized_ = true;
}

void NRFLD::CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
              const AthenaArray<Real> &u_old,
              const AthenaArray<Real> &coeff,
              const AthenaArray<Real> &def_coeff,
              bool th) {
  AthenaArray<Real> &u_gas = u_gas_; // caution! this should be in argument
  FLD *pfld = pmy_block_->prfld;
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
        const Real T_gas2 = T_gas*T_gas;
        const Real equilibrium_rad = pfld->a_r*T_gas2*T_gas2;
        Real src_term = pfld->c_ph*pfld->sigma_p(k,j,i)
            *ResolvableThermalDifference(equilibrium_rad, u(k,j,i));
        Real diff_term = 0.0;
        diff_term += -coeff(linearSolver::DXMF,k,j,i)*(u(k,j,i-1) - u(k,j,i));
        diff_term += -coeff(linearSolver::DXPF,k,j,i)*(u(k,j,i+1) - u(k,j,i));
        diff_term += -coeff(linearSolver::DYMF,k,j,i)*(u(k,j-1,i) - u(k,j,i));
        diff_term += -coeff(linearSolver::DYPF,k,j,i)*(u(k,j+1,i) - u(k,j,i));
        diff_term += -coeff(linearSolver::DZMF,k,j,i)*(u(k-1,j,i) - u(k,j,i));
        const bool physical_top =
            pmb->block_size.x3max == pmb->pmy_mesh->mesh_size.x3max;
        if (pfld->marshak_top_boundary && physical_top && k == ku) {
          const Real dface = std::max(pfld->marshak_dface(k,j,i), TINY_NUMBER);
          const Real acoef = pfld->marshak_top_alpha*pfld->c_ph*0.5*dx;
          const Real eb = (dface*u(k,j,i) + acoef*pfld->marshak_top_erad_ext)
                        /std::max(dface + acoef, TINY_NUMBER);
          const Real ftop = pfld->marshak_top_alpha*pfld->c_ph
                          *(eb - pfld->marshak_top_erad_ext);
          diff_term += -ftop*dx;
        } else {
          diff_term += -coeff(linearSolver::DZPF,k,j,i)
                     *(u(k+1,j,i) - u(k,j,i));
        }
        diff_term *= idx2;

        Real Fg = (u_gas(k,j,i) - pfld->u_gas(k,j,i)) + dt*src_term;
        Real Fr = (u(k,j,i)     - u_old(k,j,i))
            - dt*(src_term + diff_term);
        // Normalize each equation by its local transient/source scale, with
        // the current and old state as a floor.  This makes gas and radiation
        // residuals dimensionless without letting a nearly-zero update hide
        // an absolute residual near a floor.
        Real unsteady = u(k,j,i) - u_old(k,j,i);
        Real scale = std::abs(unsteady)
                   + dt*(std::abs(src_term) + std::abs(diff_term));
        scale = std::max(scale, std::max(std::abs(u(k,j,i)), std::abs(u_old(k,j,i))));
        scale = std::max(scale, static_cast<Real>(1.0e-30));
        Real gas_scale = std::abs(u_gas(k,j,i) - pfld->u_gas(k,j,i))
                       + dt*std::abs(src_term);
        gas_scale = std::max(gas_scale,
                             std::max(std::abs(u_gas(k,j,i)),
                                      std::abs(pfld->u_gas(k,j,i))));
        gas_scale = std::max(gas_scale, static_cast<Real>(1.0e-30));
        gas_defect_(k,j,i) = Fg/gas_scale;
        if (pfld->fixed_u_rad) {
          def(k,j,i) = gas_defect_(k,j,i);
        } else {
          def(k,j,i) = Fr/scale;
        }

      }
    }
  }

  return;
}


void NRFLD::CalculateAdditionalDefectNorms(Real &l2_sum, Real &max_norm,
                                           bool &active, bool &finite) const {
  const FLD *fld = pmy_block_->prfld;
  // The gas correction is eliminated locally from the one-variable radiation
  // solve.  In the ordinary coupled mode it is nevertheless an independent
  // nonlinear residual and must participate in convergence control.  It is
  // not an unknown equation in only_rad mode, while fixed_u_rad makes it the
  // primary (and only) residual instead.
  active = !fld->fixed_u_rad && !fld->only_rad;
  l2_sum = 0.0;
  max_norm = 0.0;
  finite = true;
  if (!active) return;

  const int il = pmy_block_->is, iu = pmy_block_->ie;
  const int jl = pmy_block_->js, ju = pmy_block_->je;
  const int kl = pmy_block_->ks, ku = pmy_block_->ke;
  const Real dx = pmy_block_->pcoord->dx1f(il);
  const Real dy = pmy_block_->pcoord->dx2f(jl);
  const Real dz = pmy_block_->pcoord->dx3f(kl);
  Real sum = 0.0;
  Real maximum = 0.0;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd reduction(+: sum) reduction(max: maximum)
      for (int i = il; i <= iu; ++i) {
        const Real value = gas_defect_(k,j,i);
        const Real abs_value = std::abs(value);
        sum += SQR(abs_value);
        maximum = std::max(maximum, abs_value);
      }
    }
  }
  l2_sum = sum*dx*dy*dz;
  max_norm = maximum;
  finite = std::isfinite(l2_sum) && std::isfinite(max_norm);
}

bool NRFLD::PrimaryDefectIsGas() const {
  return pmy_block_->prfld->fixed_u_rad;
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
  FLD *pfld = pmy_block_->prfld;

  if (pmy_driver_->diagnostic_verbosity_ >= 2) {
    for (int k = 0; k < delta_u.GetDim3(); ++k)
      for (int j = 0; j < delta_u.GetDim2(); ++j)
        for (int i = 0; i < delta_u.GetDim1(); ++i)
          last_delta_rad_(0,k,j,i) = last_delta_rad_(1,k,j,i) = delta_u(k,j,i);
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real raw_delta_ur = delta_u(k,j,i);
        Real delta_ur = pfld->fixed_u_rad ? 0.0 : raw_delta_ur;
        if (max_update_fraction_ > 0.0) {
          const Real ur_scale = std::max(std::abs(u_rad(k,j,i)), TINY_NUMBER);
          const Real ur_limit = max_update_fraction_*ur_scale;
          delta_ur = std::max(-ur_limit, std::min(ur_limit, delta_ur));
        }
        const Real step_scale = pmy_driver_->step_scale_;
        const Real scaled_delta_ur = step_scale*delta_ur;
        const Real u_rad_before = u_rad(k,j,i);
        if (!pfld->fixed_u_rad) {
          const Real ur_next = u_rad(k,j,i) + scaled_delta_ur;
          u_rad(k,j,i) = (std::isfinite(ur_next) && ur_next > TINY_NUMBER)
                         ? ur_next : TINY_NUMBER;
        } else {
          // Enforce the prescribed value explicitly on every NR iteration.
          u_rad(k,j,i) = uold_(k,j,i);
        }
        last_delta_rad_(0,k,j,i) = raw_delta_ur;
        last_delta_rad_(1,k,j,i) = u_rad(k,j,i) - u_rad_before;

        const Real denom = derivetive(NewtonRaphsonFLD::dFg_deg,k,j,i);
        const Real egas_delta = -(derivetive(NewtonRaphsonFLD::Fg,k,j,i)
            + derivetive(NewtonRaphsonFLD::dFg_dEr,k,j,i)*delta_ur)/denom;
        Real egas_floor = TINY_NUMBER;
#if GENERAL_EOS
        const Real rho = pmy_block_->phydro->w(IDN,k,j,i);
        if (std::isfinite(rho) && rho > pmy_block_->peos->GetDensityFloor()) {
          egas_floor = std::max(egas_floor,
              pmy_block_->peos->EgasFromRhoP(rho, pmy_block_->peos->GetPressureFloor()));
        }
#endif
        Real limited_egas_delta = egas_delta;
        if (max_update_fraction_ > 0.0) {
          const Real egas_scale = std::max(std::abs(u_gas_(k,j,i)), egas_floor);
          const Real egas_limit = max_update_fraction_*egas_scale;
          limited_egas_delta = std::max(-egas_limit, std::min(egas_limit, limited_egas_delta));
        }
        const Real egas_next = u_gas_(k,j,i) + step_scale*limited_egas_delta;
        u_gas_(k,j,i) = (std::isfinite(egas_next) && egas_next > egas_floor)
                        ? egas_next : egas_floor;
        
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

  if (pmy_block_->prfld->fixed_u_rad) {
    int il = pmy_block_->is - NGHOST, iu = pmy_block_->ie + NGHOST;
    int jl = pmy_block_->js, ju = pmy_block_->je;
    int kl = pmy_block_->ks, ku = pmy_block_->ke;
    if (pmy_block_->pmy_mesh->f2) jl -= NGHOST, ju += NGHOST;
    if (pmy_block_->pmy_mesh->f3) kl -= NGHOST, ku += NGHOST;
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          u_(k,j,i) = uold_(k,j,i);
        }
      }
    }
  }
  
  return;
}

void NRFLD::PrintCellPhysicsDebug(int k, int j, int i) {
  FLD *pfld = pmy_block_->prfld;
  const AthenaArray<Real> &u_pre = u_;
  const AthenaArray<Real> &sigma_r = pfld->sigma_r;
  const AthenaArray<Real> &coeff = coeff_;
  const AthenaArray<Real> &src = src_;
  const Real dx = pmy_block_->pcoord->dx1f(i);
  const Real dy = pmy_block_->pcoord->dx2f(j);
  const Real dz = pmy_block_->pcoord->dx3f(k);
  const Real idx = 1.0/dx;
  const Real idy = 1.0/dy;
  const Real idz = 1.0/dz;
  const Real fac = pmy_driver_->dt_/SQR(dx);

  auto print_face = [&](const char *label, Real sigma_face, Real gx, Real gy, Real gz,
                        Real e_face, Real coeff_face) {
    const Real grad_face = FaceGradient(gx, gy, gz);
    const RadFLD::ClosureValues closure = RequireFaceClosure(
        grad_face, sigma_face, e_face, pfld->fixed_flux_limiter,
        label, k, j, i);
    Real r_face = 0.0;
    if (grad_face > 0.0 && (sigma_face == 0.0 || e_face == 0.0)) {
      r_face = std::numeric_limits<Real>::infinity();
    } else if (grad_face > 0.0) {
      r_face = static_cast<Real>(static_cast<long double>(grad_face)
          / static_cast<long double>(sigma_face)
          / static_cast<long double>(e_face));
    }
    std::cout << "      " << label
              << " sigma_face=" << sigma_face
              << " E_face=" << e_face
              << " gradE_face=" << grad_face
              << " R_face=" << r_face
              << " lambda_face=" << closure.lambda
              << " dFr_coeff=" << coeff_face
              << std::endl;
  };

  Real sigma_face = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j,i-1), idx);
  Real gx = (u_pre(k,j,i-1) - u_pre(k,j,i))*idx;
  Real gy = 0.25*idy*((u_pre(k,j+1,i-1) - u_pre(k,j-1,i-1)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
  Real gz = 0.25*idz*((u_pre(k+1,j,i-1) - u_pre(k-1,j,i-1)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
  Real e_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j,i-1));
  print_face("face_xm", sigma_face, gx, gy, gz, e_face,
             derivetive_(NewtonRaphsonFLD::dFr_dEr_xm, k, j, i));

  sigma_face = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j,i+1), idx);
  gx = (u_pre(k,j,i+1) - u_pre(k,j,i))*idx;
  gy = 0.25*idy*((u_pre(k,j+1,i+1) - u_pre(k,j-1,i+1)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
  gz = 0.25*idz*((u_pre(k+1,j,i+1) - u_pre(k-1,j,i+1)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
  e_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j,i+1));
  print_face("face_xp", sigma_face, gx, gy, gz, e_face,
             derivetive_(NewtonRaphsonFLD::dFr_dEr_xp, k, j, i));

  sigma_face = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j-1,i), idx);
  gx = 0.25*idx*((u_pre(k,j-1,i+1) - u_pre(k,j-1,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
  gy = (u_pre(k,j-1,i) - u_pre(k,j,i))*idy;
  gz = 0.25*idz*((u_pre(k+1,j-1,i) - u_pre(k-1,j-1,i)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
  e_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j-1,i));
  print_face("face_ym", sigma_face, gx, gy, gz, e_face,
             derivetive_(NewtonRaphsonFLD::dFr_dEr_ym, k, j, i));

  sigma_face = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k,j+1,i), idx);
  gx = 0.25*idx*((u_pre(k,j+1,i+1) - u_pre(k,j+1,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
  gy = (u_pre(k,j+1,i) - u_pre(k,j,i))*idy;
  gz = 0.25*idz*((u_pre(k+1,j+1,i) - u_pre(k-1,j+1,i)) + (u_pre(k+1,j,i) - u_pre(k-1,j,i)));
  e_face = FaceEnergy(u_pre(k,j,i), u_pre(k,j+1,i));
  print_face("face_yp", sigma_face, gx, gy, gz, e_face,
             derivetive_(NewtonRaphsonFLD::dFr_dEr_yp, k, j, i));

  sigma_face = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k-1,j,i), idx);
  gx = 0.25*idx*((u_pre(k-1,j,i+1) - u_pre(k-1,j,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
  gy = 0.25*idy*((u_pre(k-1,j+1,i) - u_pre(k-1,j-1,i)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
  gz = (u_pre(k-1,j,i) - u_pre(k,j,i))*idz;
  e_face = FaceEnergy(u_pre(k,j,i), u_pre(k-1,j,i));
  print_face("face_zm", sigma_face, gx, gy, gz, e_face,
             derivetive_(NewtonRaphsonFLD::dFr_dEr_zm, k, j, i));

  sigma_face = RadFLD::FaceOpacity(sigma_r(k,j,i), sigma_r(k+1,j,i), idx);
  gx = 0.25*idx*((u_pre(k+1,j,i+1) - u_pre(k+1,j,i-1)) + (u_pre(k,j,i+1) - u_pre(k,j,i-1)));
  gy = 0.25*idy*((u_pre(k+1,j+1,i) - u_pre(k+1,j-1,i)) + (u_pre(k,j+1,i) - u_pre(k,j-1,i)));
  gz = (u_pre(k+1,j,i) - u_pre(k,j,i))*idz;
  e_face = FaceEnergy(u_pre(k,j,i), u_pre(k+1,j,i));
  print_face("face_zp", sigma_face, gx, gy, gz, e_face,
             derivetive_(NewtonRaphsonFLD::dFr_dEr_zp, k, j, i));

  const Real m_cc = fac*coeff(linearSolver::DCCF,k,j,i) + coeff(linearSolver::DCCS,k,j,i);
  const Real m_xm = fac*coeff(linearSolver::DXMF,k,j,i);
  const Real m_xp = fac*coeff(linearSolver::DXPF,k,j,i);
  const Real m_ym = fac*coeff(linearSolver::DYMF,k,j,i);
  const Real m_yp = fac*coeff(linearSolver::DYPF,k,j,i);
  const Real m_zm = fac*coeff(linearSolver::DZMF,k,j,i);
  const Real m_zp = fac*coeff(linearSolver::DZPF,k,j,i);

  const Real delta_raw_c = last_delta_rad_(0,k,j,i);
  const Real delta_raw_xm = last_delta_rad_(0,k,j,i-1);
  const Real delta_raw_xp = last_delta_rad_(0,k,j,i+1);
  const Real delta_raw_ym = last_delta_rad_(0,k,j-1,i);
  const Real delta_raw_yp = last_delta_rad_(0,k,j+1,i);
  const Real delta_raw_zm = last_delta_rad_(0,k-1,j,i);
  const Real delta_raw_zp = last_delta_rad_(0,k+1,j,i);
  const Real m_delta_raw = m_cc*delta_raw_c
      + m_xm*delta_raw_xm + m_xp*delta_raw_xp
      + m_ym*delta_raw_ym + m_yp*delta_raw_yp
      + m_zm*delta_raw_zm + m_zp*delta_raw_zp;

  const Real delta_app_c = last_delta_rad_(1,k,j,i);
  const Real delta_app_xm = last_delta_rad_(1,k,j,i-1);
  const Real delta_app_xp = last_delta_rad_(1,k,j,i+1);
  const Real delta_app_ym = last_delta_rad_(1,k,j-1,i);
  const Real delta_app_yp = last_delta_rad_(1,k,j+1,i);
  const Real delta_app_zm = last_delta_rad_(1,k-1,j,i);
  const Real delta_app_zp = last_delta_rad_(1,k+1,j,i);
  const Real m_delta_applied = m_cc*delta_app_c
      + m_xm*delta_app_xm + m_xp*delta_app_xp
      + m_ym*delta_app_ym + m_yp*delta_app_yp
      + m_zm*delta_app_zm + m_zp*delta_app_zp;

  std::cout << "      linear_row"
            << " m_cc=" << m_cc
            << " m_xm=" << m_xm
            << " m_xp=" << m_xp
            << " m_ym=" << m_ym
            << " m_yp=" << m_yp
            << " m_zm=" << m_zm
            << " m_zp=" << m_zp
            << std::endl;
  std::cout << "      delta_raw"
            << " c=" << delta_raw_c
            << " xm=" << delta_raw_xm
            << " xp=" << delta_raw_xp
            << " ym=" << delta_raw_ym
            << " yp=" << delta_raw_yp
            << " zm=" << delta_raw_zm
            << " zp=" << delta_raw_zp
            << std::endl;
  std::cout << "      delta_applied"
            << " c=" << delta_app_c
            << " xm=" << delta_app_xm
            << " xp=" << delta_app_xp
            << " ym=" << delta_app_ym
            << " yp=" << delta_app_yp
            << " zm=" << delta_app_zm
            << " zp=" << delta_app_zp
            << std::endl;
  std::cout << "      linear_balance"
            << " src=" << src(k,j,i)
            << " A_delta_raw=" << m_delta_raw
            << " src_minus_A_delta_raw=" << (src(k,j,i) - m_delta_raw)
            << " A_delta_applied=" << m_delta_applied
            << " src_minus_A_delta_applied=" << (src(k,j,i) - m_delta_applied)
            << std::endl;
}
