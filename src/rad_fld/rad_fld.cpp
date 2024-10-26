//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rad_fld.cpp
//! \brief implementation of functions in class FLD

// C headers

// C++ headers
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

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
#include "../utils/buffer_utils.hpp"
#include "mg_rad_fld.hpp"
#include "rad_fld.hpp"


inline void DefaultOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD *prfld = pmb->prfld;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-1, iu=pmb->ie+1;
  if (pmb->block_size.nx2 > 1) {
    jl -= 1;
    ju += 1;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= 1;
    ku += 1;
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        prfld->sigma_r(k,j,i) = prfld->pmg->const_opacity*prim(IDN,k,j,i); //temporary
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn FLD::FLD(MeshBlock *pmb, ParameterInput *pin)
//! \brief FLD constructor
FLD::FLD(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb), u(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    source(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coeff(RadFLD::NCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_u(RadFLD::NTEMP, pmb->ncc3, pmb->ncc2, pmb->ncc1), //!
    sigma_r(pmb->ncells3,pmb->ncells2,pmb->ncells1),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    output_defect(false), rfldbvar(pmb, &u, &coarse_u, empty_flux, false), //!
    refinement_idx_(), calc_in_temp(), is_couple(), only_rad() {
  is_couple = pin->GetOrAddBoolean("mgfld", "is_couple", false);
  output_defect = pin->GetOrAddBoolean("mgfld", "output_defect", false);
  calc_in_temp = pin->GetOrAddBoolean("mgfld", "calc_in_temp", false);
  only_rad = pin->GetOrAddBoolean("mgfld", "only_rad", false);
  if (calc_in_temp) {
    // raise error
    std::stringstream msg;
    msg << "Error: calc_in_temp is not implemented yet.";
    ATHENA_ERROR(msg);
  }
  if (output_defect)
    def.NewAthenaArray(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  pmb->RegisterMeshBlockData(u); //!
  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pmb->pmy_mesh->multilevel)
    refinement_idx_ = pmy_block->pmr->AddToRefinement(&u, &coarse_u); //!

  pmg = new MGFLD(pmb->pmy_mesh->pmfld, pmb, pin);

  // Enroll CellCenteredBoundaryVariable object
  rfldbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&rfldbvar);
  pmb->pbval->prfldbvar = &rfldbvar;

  // set a default opacity function
  UpdateOpacity = DefaultOpacity;
}

void FLD::EnrollOpacityFunction(FLDOpacityFunc MyOpacityFunction) {
  UpdateOpacity = MyOpacityFunction;
}


//----------------------------------------------------------------------------------------
//! \fn FLD::~FLD()
//! \brief FLD destructor
FLD::~FLD() {
  delete pmg;
}


//----------------------------------------------------------------------------------------
//! \fn void FLD::CalculateCoefficients()
//! \brief Calculate coefficients required for FLD calculation
void FLD::CalculateCoefficients(const AthenaArray<Real> &w,
                                const AthenaArray<Real> &u) {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  Real idx = 1.0/pmy_block->pcoord->dx1f(pmy_block->is);
  Real hidx = 0.5*idx;
  Real gm1 = pmy_block->peos->GetGamma() - 1.0;
  if (pmy_block->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real dEr1 = hidx*(u(RadFLD::RAD,k,j,i+1) - u(RadFLD::RAD,k,j,i-1));
        Real dEr2 = hidx*(u(RadFLD::RAD,k,j+1,i) - u(RadFLD::RAD,k,j-1,i));
        Real dEr3 = hidx*(u(RadFLD::RAD,k+1,j,i) - u(RadFLD::RAD,k-1,j,i));
        Real grad = sqrt(SQR(dEr1) + SQR(dEr2) + SQR(dEr3))/u(RadFLD::RAD,k,j,i);

        // Real sigma_rhere = sigma_r(k,j,i);
        // sigma_r(RadFLD::DXM) = sigma_r(k,j,i-1);
        // sigma_r(RadFLD::DXP) = sigma_r(k,j,i+1);
        // sigma_r(RadFLD::DYM) = sigma_r(k,j-1,i);
        // sigma_r(RadFLD::DYP) = sigma_r(k,j+1,i);
        // sigma_r(RadFLD::DZM) = sigma_r(k-1,j,i);
        // sigma_r(RadFLD::DZP) = sigma_r(k+1,j,i);

        // for (int n = 0; n <= RadFLD::DZP; ++n) {
        //   Real sigma_rface = std::min(0.5*(sigma_rhere + sigma_r(n)),
        //         std::max(2.0*sigma_rhere*sigma_r(n)/(sigma_rhere + sigma_r(n)),
        //         2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        //   Real R = grad/sigma_rface;
        //   Real lambda = (2.0+R)/(6.0+2.0*R+R*R);
        //   coeff(n,k,j,i) = pmg->c_ph*lambda/sigma_rface;
        // }

        Real sigma_rface, R, lambda;
        // for DXM
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j,i-1)),
              std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j,i-1)/(sigma_r(k,j,i) + sigma_r(k,j,i-1)),
              2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        R = grad/sigma_rface;
        lambda = (2.0+R)/(6.0+2.0*R+R*R);
        coeff(RadFLD::DXM,k,j,i) = pmg->c_ph*lambda/sigma_rface;

        // for DXP
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j,i+1)),
              std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j,i+1)/(sigma_r(k,j,i) + sigma_r(k,j,i+1)),
              2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        R = grad/sigma_rface;
        lambda = (2.0+R)/(6.0+2.0*R+R*R);
        coeff(RadFLD::DXP,k,j,i) = pmg->c_ph*lambda/sigma_rface;

        // for DYM
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j-1,i)),
              std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j-1,i)/(sigma_r(k,j,i) + sigma_r(k,j-1,i)),
              2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        R = grad/sigma_rface;
        lambda = (2.0+R)/(6.0+2.0*R+R*R);
        coeff(RadFLD::DYM,k,j,i) = pmg->c_ph*lambda/sigma_rface;

        // for DYP
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k,j+1,i)),
              std::max(2.0*sigma_r(k,j,i)*sigma_r(k,j+1,i)/(sigma_r(k,j,i) + sigma_r(k,j+1,i)),
              2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        R = grad/sigma_rface;
        lambda = (2.0+R)/(6.0+2.0*R+R*R);
        coeff(RadFLD::DYP,k,j,i) = pmg->c_ph*lambda/sigma_rface;

        // for DZM
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k-1,j,i)),
              std::max(2.0*sigma_r(k,j,i)*sigma_r(k-1,j,i)/(sigma_r(k,j,i) + sigma_r(k-1,j,i)),
              2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        R = grad/sigma_rface;
        lambda = (2.0+R)/(6.0+2.0*R+R*R);
        coeff(RadFLD::DZM,k,j,i) = pmg->c_ph*lambda/sigma_rface;

        // for DZP
        sigma_rface = std::min(0.5*(sigma_r(k,j,i) + sigma_r(k+1,j,i)),
              std::max(2.0*sigma_r(k,j,i)*sigma_r(k+1,j,i)/(sigma_r(k,j,i) + sigma_r(k+1,j,i)),
              2.0*TWO_3RD*idx)); // Howell & Greenough 2002 (after eq. 15)
        R = grad/sigma_rface;
        lambda = (2.0+R)/(6.0+2.0*R+R*R);
        coeff(RadFLD::DZP,k,j,i) = pmg->c_ph*lambda/sigma_rface;

        // for later calculation
        if (is_couple) {
          coeff(RadFLD::DSIGMAP,k,j,i) = sigma_r(k,j,i)*w(IDN,k,j,i);
          coeff(RadFLD::DCOUPLE,k,j,i) = gm1/w(IDN,k,j,i);
        } else {
          coeff(RadFLD::DSIGMAP,k,j,i) = 0.0;
          coeff(RadFLD::DCOUPLE,k,j,i) = 0.0;
        }
      }
    }
  }
  return;
}


// //----------------------------------------------------------------------------------------
// //! \fn Real FLD::CalculateSigmaR(const Real den, const Real egas)
// //! \brief Calculate Rosseland mean opacity
// Real FLD::CalculateSigmaR(const Real den, const Real egas) {
//   Real sigma_r = pmg->const_opacity*den; // temporary
//   return sigma_r;
// }


//----------------------------------------------------------------------------------------
//! \fn void FLD::LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u)
//! \brief Load hydro variables from conserved variables
void FLD::LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u) {
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
        u(RadFLD::GAS,k,j,i) = igm1*w(IPR,k,j,i);
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void FLD::UpdateHydroVariables(AthenaArray<Real> &w,
//!               AthenaArray<Real> &hydro_u, const AthenaArray<Real> &fld_u)
//! \brief Update conserved variables from hydro variables
void FLD::UpdateHydroVariables(AthenaArray<Real> &w, AthenaArray<Real> &hydro_u, const AthenaArray<Real> &fld_u) {
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
        hydro_u(IEN,k,j,i) += (fld_u(RadFLD::GAS,k,j,i) - igm1*w(IPR,k,j,i));
        w(IPR,k,j,i) = gm1*fld_u(RadFLD::GAS,k,j,i);
      }
    }
  }
  return;
}
