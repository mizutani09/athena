//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rad_fld.cpp
//! \brief implementation of functions in class FLD2

// C headers

// C++ headers
#include <cmath>
#include <sstream>    // sstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/bvals_interfaces.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../fld/fld.hpp"
#include "mg_rad_fld.hpp"
#include "rad_fld.hpp"

//----------------------------------------------------------------------------------------
//! \fn FLD2::FLD2(MeshBlock *pmb, ParameterInput *pin)
//! \brief FLD2 constructor
FLD2::FLD2(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb),
    pfld(pmb->prfld),
    u(RadFLD2::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coeff(RadFLD2::NCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_u(RadFLD2::NTEMP, pmb->ncc3, pmb->ncc2, pmb->ncc1),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    output_defect(false),
    mgfldbvar(pmb, &u, &coarse_u, empty_flux, 1, false),
    refinement_idx_() {
  output_defect = pin->GetOrAddBoolean("fld", "output_defect", false);
  if (output_defect)
    def.NewAthenaArray(RadFLD2::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  pmb->RegisterMeshBlockData(u);

  if (pmb->pmy_mesh->multilevel)
    refinement_idx_ = pmy_block->pmr->AddToRefinement(&u, &coarse_u);

  pmg = new MGFLD(pmb->pmy_mesh->pmfld, pmb, pin);
  pmg->SetRadiationConstants(pfld->c_ph, pfld->a_r);
  pmb->pmy_mesh->pmfld->SetRadiationConstantsOnce(pfld->c_ph, pfld->a_r);

  mgfldbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&mgfldbvar);
}

//----------------------------------------------------------------------------------------
//! \fn FLD2::~FLD2()
//! \brief FLD2 destructor
FLD2::~FLD2() {
  delete pmg;
}

//----------------------------------------------------------------------------------------
//! \fn void FLD2::SyncFromFld()
//! \brief Load hydro/radiation variables into MGFLD working array
void FLD2::SyncFromFld(const AthenaArray<Real> &w) {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  Real igm1 = 1.0/(pmy_block->peos->GetGamma() - 1.0);
  if (pmy_block->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        u(RadFLD2::RAD,k,j,i) = pfld->u_rad(k,j,i);
        if (!pfld->only_rad) {
          u(RadFLD2::GAS,k,j,i) = igm1*w(IPR,k,j,i);
        } else {
          u(RadFLD2::GAS,k,j,i) = pfld->u_gas(k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FLD2::SyncToFld()
//! \brief Write MGFLD results back to FLD
void FLD2::SyncToFld() {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  if (pmy_block->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        pfld->u_rad(k,j,i) = u(RadFLD2::RAD,k,j,i);
        pfld->u_gas(k,j,i) = u(RadFLD2::GAS,k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FLD2::CalculateCoefficients()
//! \brief Calculate coefficients required for MGFLD calculation
void FLD2::CalculateCoefficients(const AthenaArray<Real> &w) {
  int il = pmy_block->is - 1, iu = pmy_block->ie + 1;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  Real idx = 1.0/pmy_block->pcoord->dx1f(pmy_block->is);
  Real hidx = 0.5*idx;
  Real gm1 = pmy_block->peos->GetGamma() - 1.0;
  if (pmy_block->pmy_mesh->f2)
    jl -= 1, ju += 1;
  if (pmy_block->pmy_mesh->f3)
    kl -= 1, ku += 1;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        AthenaArray<Real> dEr;
        dEr.NewAthenaArray(3);
        for (int ii = 0; ii < 3; ++ii) {
          int di = (ii == 0) ? 1 : 0;
          int dj = (ii == 1) ? 1 : 0;
          int dk = (ii == 2) ? 1 : 0;
          dEr(ii) = hidx*(u(RadFLD2::RAD,k+dk,j+dj,i+di) - u(RadFLD2::RAD,k-dk,j-dj,i-di));
        }
        Real gradE = std::sqrt(SQR(dEr(0)) + SQR(dEr(1)) + SQR(dEr(2)));

        Real sigma_rface, R, lambda;

        for (int ii = 0; ii < 6; ++ii) {
          int di = (ii == 0) ? -1 : (ii == 1) ? 1 : 0;
          int dj = (ii == 2) ? -1 : (ii == 3) ? 1 : 0;
          int dk = (ii == 4) ? -1 : (ii == 5) ? 1 : 0;
          sigma_rface = std::min(0.5*(pfld->sigma_r(k,j,i)
                                      + pfld->sigma_r(k+dk,j+dj,i+di)),
                std::max(2.0*pfld->sigma_r(k,j,i)*pfld->sigma_r(k+dk,j+dj,i+di)
                         /(pfld->sigma_r(k,j,i) + pfld->sigma_r(k+dk,j+dj,i+di)),
                         2.0*TWO_3RD*idx));
          R = gradE/(sigma_rface*u(RadFLD2::RAD,k,j,i));
          lambda = RadFLD::FluxLimiter(R, pfld->fixed_flux_limiter);
          coeff(RadFLD2::DXM+ii,k,j,i) = pfld->c_ph*lambda/sigma_rface;
        }

        coeff(RadFLD2::DSIGMAP,k,j,i) = pfld->sigma_p(k,j,i)*w(IDN,k,j,i);
        coeff(RadFLD2::DCOUPLE,k,j,i) = gm1/w(IDN,k,j,i);

        R = gradE/(pfld->sigma_r(k,j,i)*u(RadFLD2::RAD,k,j,i));
        lambda = RadFLD::FluxLimiter(R, pfld->fixed_flux_limiter);
        Real chi = RadFLD::EddingtonFactor(R, pfld->fixed_flux_limiter);

        AthenaArray<Real> ngrad;
        ngrad.NewAthenaArray(3);
        for (int ii = 0; ii < 3; ++ii) ngrad(ii) = dEr(ii)/(gradE+TINY_NUMBER);

        coeff(RadFLD2::DPV,k,j,i) = 0.0;
        for (int jj = 0; jj < 3; ++jj) {
          for (int ii = 0; ii < 3; ++ii) {
            Real D_edd = 0.0;
            if (ii == jj) D_edd += .5*(1.-chi);
            D_edd += .5*(3.*chi-1.)*ngrad(jj)*ngrad(ii);

            int di = ii == 0 ? 1 : 0;
            int dj = ii == 1 ? 1 : 0;
            int dk = ii == 2 ? 1 : 0;

            Real dv_dx = hidx*(w(IVX+jj,k+dk,j+dj,i+di) - w(IVX+jj,k-dk,j-dj,i-di));
            coeff(RadFLD2::DPV,k,j,i) += D_edd * dv_dx;
          }
        }

        ngrad.DeleteAthenaArray();
      }
    }
  }

  if (pfld->cut_diff) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          for (int n = 0; n <= RadFLD2::DZP; ++n) {
            coeff(n,k,j,i) = 0.0;
          }
        }
      }
    }
  }

  if (!pfld->is_couple) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          coeff(RadFLD2::DSIGMAP,k,j,i) = 0.0;
          coeff(RadFLD2::DCOUPLE,k,j,i) = 0.0;
        }
      }
    }
  }

  if (!pfld->include_radiation_force) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          coeff(RadFLD2::DPV,k,j,i) = 0.0;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FLD2::UpdateHydroVariables()
//! \brief Update conserved variables from MGFLD results
void FLD2::UpdateHydroVariables(AthenaArray<Real> &w,
                                          AthenaArray<Real> &hydro_u) {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  Real gm1 = pmy_block->peos->GetGamma() - 1.0;
  Real igm1 = 1.0/gm1;
  if (pmy_block->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        hydro_u(IEN,k,j,i) += (u(RadFLD2::GAS,k,j,i) - igm1*w(IPR,k,j,i));
        w(IPR,k,j,i) = gm1*u(RadFLD2::GAS,k,j,i);
      }
    }
  }
}
