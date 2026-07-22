//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rad_fld.cpp
//! \brief implementation of functions in class MGFLDInterface

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
//! \fn MGFLDInterface::MGFLDInterface(MeshBlock *pmb, ParameterInput *pin)
//! \brief MGFLDInterface constructor
MGFLDInterface::MGFLDInterface(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb),
    pfld2(pmb->prfld2),
    u(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coeff(RadFLD::NCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_u(RadFLD::NTEMP, pmb->ncc3, pmb->ncc2, pmb->ncc1),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    output_defect(false),
    mgfldbvar(pmb, &u, &coarse_u, empty_flux, 1, false),
    refinement_idx_() {
  output_defect = pin->GetOrAddBoolean("fld", "output_defect", false);
  if (output_defect)
    def.NewAthenaArray(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  pmb->RegisterMeshBlockData(u);

  if (pmb->pmy_mesh->multilevel)
    refinement_idx_ = pmy_block->pmr->AddToRefinement(&u, &coarse_u);

  pmg = new MGFLD(pmb->pmy_mesh->pmfld, pmb, pin);
  pmg->SetRadiationConstants(pfld2->c_ph, pfld2->a_r);
  pmb->pmy_mesh->pmfld->SetRadiationConstantsOnce(pfld2->c_ph, pfld2->a_r);

  mgfldbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&mgfldbvar);
}

//----------------------------------------------------------------------------------------
//! \fn MGFLDInterface::~MGFLDInterface()
//! \brief MGFLDInterface destructor
MGFLDInterface::~MGFLDInterface() {
  delete pmg;
}

//----------------------------------------------------------------------------------------
//! \fn void MGFLDInterface::SyncFromFld2()
//! \brief Load hydro/radiation variables into MGFLD working array
void MGFLDInterface::SyncFromFld2(const AthenaArray<Real> &w) {
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
        u(RadFLD::RAD,k,j,i) = pfld2->u_rad(k,j,i);
        if (!pfld2->only_rad) {
          u(RadFLD::GAS,k,j,i) = igm1*w(IPR,k,j,i);
        } else {
          u(RadFLD::GAS,k,j,i) = pfld2->u_gas(k,j,i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MGFLDInterface::SyncToFld2()
//! \brief Write MGFLD results back to FLD2
void MGFLDInterface::SyncToFld2() {
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
        pfld2->u_rad(k,j,i) = u(RadFLD::RAD,k,j,i);
        pfld2->u_gas(k,j,i) = u(RadFLD::GAS,k,j,i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MGFLDInterface::CalculateCoefficients()
//! \brief Calculate coefficients required for MGFLD calculation
void MGFLDInterface::CalculateCoefficients(const AthenaArray<Real> &w) {
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
          dEr(ii) = hidx*(u(RadFLD::RAD,k+dk,j+dj,i+di) - u(RadFLD::RAD,k-dk,j-dj,i-di));
        }
        Real gradE = std::sqrt(SQR(dEr(0)) + SQR(dEr(1)) + SQR(dEr(2)));

        Real sigma_rface, R, lambda;
        if (pfld2->fixed_flux_limitter) lambda = ONE_3RD;

        for (int ii = 0; ii < 6; ++ii) {
          int di = (ii == 0) ? -1 : (ii == 1) ? 1 : 0;
          int dj = (ii == 2) ? -1 : (ii == 3) ? 1 : 0;
          int dk = (ii == 4) ? -1 : (ii == 5) ? 1 : 0;
          sigma_rface = std::min(0.5*(pfld2->sigma_r(k,j,i)
                                      + pfld2->sigma_r(k+dk,j+dj,i+di)),
                std::max(2.0*pfld2->sigma_r(k,j,i)*pfld2->sigma_r(k+dk,j+dj,i+di)
                         /(pfld2->sigma_r(k,j,i) + pfld2->sigma_r(k+dk,j+dj,i+di)),
                         2.0*TWO_3RD*idx));
          R = gradE/(sigma_rface*u(RadFLD::RAD,k,j,i));
          if (!pfld2->fixed_flux_limitter) lambda = (2.0+R)/(6.0+3.0*R+R*R);
          coeff(RadFLD::DXM+ii,k,j,i) = pfld2->c_ph*lambda/sigma_rface;
        }

        coeff(RadFLD::DSIGMAP,k,j,i) = pfld2->sigma_p(k,j,i)*w(IDN,k,j,i);
        coeff(RadFLD::DCOUPLE,k,j,i) = gm1/w(IDN,k,j,i);

        R = gradE/(pfld2->sigma_r(k,j,i)*u(RadFLD::RAD,k,j,i));
        if (!pfld2->fixed_flux_limitter) lambda = (2.0+R)/(6.0+3.0*R+R*R);
        Real chi = lambda+std::pow(lambda*R,2);

        AthenaArray<Real> ngrad;
        ngrad.NewAthenaArray(3);
        for (int ii = 0; ii < 3; ++ii) ngrad(ii) = dEr(ii)/(gradE+TINY_NUMBER);

        coeff(RadFLD::DPV,k,j,i) = 0.0;
        for (int jj = 0; jj < 3; ++jj) {
          for (int ii = 0; ii < 3; ++ii) {
            Real D_edd = 0.0;
            if (ii == jj) D_edd += .5*(1.-chi);
            D_edd += .5*(3.*chi-1.)*ngrad(jj)*ngrad(ii);

            int di = ii == 0 ? 1 : 0;
            int dj = ii == 1 ? 1 : 0;
            int dk = ii == 2 ? 1 : 0;

            Real dv_dx = hidx*(w(IVX+jj,k+dk,j+dj,i+di) - w(IVX+jj,k-dk,j-dj,i-di));
            coeff(RadFLD::DPV,k,j,i) += D_edd * dv_dx;
          }
        }

        ngrad.DeleteAthenaArray();
      }
    }
  }

  if (pfld2->cut_diff) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          for (int n = 0; n <= RadFLD::DZP; ++n) {
            coeff(n,k,j,i) = 0.0;
          }
        }
      }
    }
  }

  if (!pfld2->is_couple) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          coeff(RadFLD::DSIGMAP,k,j,i) = 0.0;
          coeff(RadFLD::DCOUPLE,k,j,i) = 0.0;
        }
      }
    }
  }

  if (pfld2->cut_Pnablav) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          coeff(RadFLD::DPV,k,j,i) = 0.0;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MGFLDInterface::UpdateHydroVariables()
//! \brief Update conserved variables from MGFLD results
void MGFLDInterface::UpdateHydroVariables(AthenaArray<Real> &w,
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
        hydro_u(IEN,k,j,i) += (u(RadFLD::GAS,k,j,i) - igm1*w(IPR,k,j,i));
        w(IPR,k,j,i) = gm1*u(RadFLD::GAS,k,j,i);
      }
    }
  }
}
