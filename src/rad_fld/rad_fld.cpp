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


//----------------------------------------------------------------------------------------
//! \fn FLD::FLD(MeshBlock *pmb, ParameterInput *pin)
//! \brief FLD constructor
FLD::FLD(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb), u(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    source(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coeff(RadFLD::NCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_u(RadFLD::NTEMP, pmb->ncc3, pmb->ncc2, pmb->ncc1), //!
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    output_defect(false), rfldbvar(pmb, &u, &coarse_u, empty_flux, false), //!
    refinement_idx_(), calc_in_temp(), is_couple(), only_rad() {
  is_couple = pin->GetOrAddBoolean("mgfld", "is_couple", false);
  output_defect = pin->GetOrAddBoolean("mgfld", "output_defect", false);
  calc_in_temp = pin->GetOrAddBoolean("mgfld", "calc_in_temp", false);
  only_rad = pin->GetOrAddBoolean("mgfld", "only_rad", false);
  InitFLDConstants(pin);
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

  pmg = new MGFLD(pmb->pmy_mesh->pmfld, pmb);

  // Enroll CellCenteredBoundaryVariable object
  rfldbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&rfldbvar);
  pmb->pbval->prfldbvar = &rfldbvar;
}


//----------------------------------------------------------------------------------------
//! \fn FLD::~FLD()
//! \brief FLD destructor
FLD::~FLD() {
  delete pmg;
}


//----------------------------------------------------------------------------------------
//! \fn void FLD::InitFLDConstants(ParameterInput *pin)
//! \brief Initialize constants required for FLD calculation
void FLD::InitFLDConstants(ParameterInput *pin) {
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  Real R_gas = 8.3144621e7; // gas constant in erg K^-1 mol^-1
  Real const_opacity_dim = pin->GetReal("mgfld", "const_opacity"); // caution: in cm^2 g^-1

  Real rho_unit = pin->GetReal("hydro", "rho_unit");
  Real egas_unit = pin->GetReal("hydro", "egas_unit");
  Real pres_unit = egas_unit;
  Real vel_unit = std::sqrt(pres_unit/rho_unit);

  Real time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  Real leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [FLD::InitFLDConstants]" << std::endl;
    msg << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  Real mu = pin->GetReal("hydro", "mu");
  Real T_unit = pres_unit/rho_unit*mu/R_gas;

  c_ph = c_ph_dim/vel_unit;
  a_r = a_r_dim/(egas_unit/std::pow(T_unit, 4));
  const_opacity_dim *= rho_unit;
  const_opacity = const_opacity_dim*leng_unit;

  std::cout << "c_ph in sim: " << c_ph << std::endl;
  std::cout << "a_r in sim: " << a_r << std::endl;
  std::cout << "const_opacity in sim: " << const_opacity << std::endl;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void FLD::CalculateCoefficients()
//! \brief Calculate coefficients required for FLD calculation
void FLD::CalculateCoefficients(const AthenaArray<Real> &w,
                                const AthenaArray<Real> &u) {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  Real hidx = 1.0/(2.0*pmy_block->pcoord->dx1f(pmy_block->is));
  AthenaArray<Real> sigma_r(RadFLD::NCOEFF);
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
        Real grad = sqrt(dEr1*dEr1 + dEr2*dEr2 + dEr3*dEr3)/u(RadFLD::RAD,k,j,i);

        for(int n=0; n<=RadFLD::DZP; ++n) {
          if (const_opacity > 0.0) {
            sigma_r(n) = const_opacity;
          } else {
            sigma_r(n) = 1.0; // calc_sigma_r(rho, T); on surface Howeell & Greenough 2003
          }
          Real R = grad/sigma_r(n);
          Real lambda = (2.0+R)/(6.0+2.0*R+R*R);
          // coeff(n,k,j,i) = c_ph*lambda/sigma_r(n);
          coeff(n,k,j,i) = 0.0; // caution!
        }

        // for later calculation
        if (is_couple) {
          coeff(RadFLD::DSIGMAP,k,j,i) = const_opacity; // calc_sigma_p(rho, T); on center?
          coeff(RadFLD::DCOUPLE,k,j,i) = gm1/w(IDN,k,j,i);
        } else {
          coeff(RadFLD::DSIGMAP,k,j,i) = 0.0;
          coeff(RadFLD::DCOUPLE,k,j,i) = 0.0;
        }
        coeff(RadFLD::DEGAS,k,j,i) = u(RadFLD::GAS,k,j,i); // we can just reuse src(RadFLD::GAS,k,j,i)

        coeff(RadFLD::DCPH,k,j,i) = c_ph; // should be different way
        coeff(RadFLD::DAR,k,j,i) = a_r;
      }
    }
  }
  return;
}


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
