//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fld.cpp
//! \brief implementation of functions in class FLD2

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
#include "fld.hpp"


inline void DefaultOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  std::cout << "DefaultOpacity is called!" << std::endl;
  FLD2 *prfld = pmb->prfld2;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        prfld->sigma_p(k,j,i) = prfld->const_opacity*prim(IDN,k,j,i);
        prfld->sigma_r(k,j,i) = prfld->const_opacity*prim(IDN,k,j,i);
      }
    }
  }

  if (!prfld->is_couple) {
    for(int k=kl; k<=ku; ++k) {
      for(int j=jl; j<=ju; ++j) {
        for(int i=il; i<=iu; ++i) {
          prfld->sigma_p(k,j,i) = 0.0;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn FLD2::FLD2(MeshBlock *pmb, ParameterInput *pin)
//! \brief FLD2 constructor
FLD2::FLD2(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb),
    u_gas(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u_rad(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u_rad1(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_u_rad(pmb->ncc3, pmb->ncc2, pmb->ncc1),
    sigma_p(pmb->ncells3,pmb->ncells2,pmb->ncells1),
    sigma_r(pmb->ncells3,pmb->ncells2,pmb->ncells1),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    // output_defect(false), mgfldbvar(pmb, &u, &coarse_u, empty_flux, false), //!
    u_rad_flux{ {pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
            {pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
             (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)},
            {pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
             (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)}
    },
    u_rad_fldbvar(pmb, &u_rad, &coarse_u_rad, u_rad_flux, true),
    // coarse_r_(RadFLD::NADV, pmb->ncc3, pmb->ncc2, pmb->ncc1,
    //           (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
    //            AthenaArray<Real>::DataStatus::empty)),
    refinement_idx_(),
    is_couple(), only_rad(), cut_diff(), cut_Pnablav(), fixed_u_rad()
    // is_adv(false)
    {
  is_couple = pin->GetOrAddBoolean("nrfld", "is_couple", true);
//   output_defect = pin->GetOrAddBoolean("nrfld", "output_defect", false);
//   calc_in_temp = pin->GetOrAddBoolean("nrfld", "calc_in_temp", false);
  only_rad = pin->GetOrAddBoolean("nrfld", "only_rad", false);
  cut_diff = pin->GetOrAddBoolean("nrfld", "cut_diff", false);
  cut_Pnablav = pin->GetOrAddBoolean("nrfld", "cut_Pnablav", false);
  fixed_flux_limitter = pin->GetOrAddBoolean("nrfld", "fixed_flux_limitter", false);
  fixed_u_rad = pin->GetOrAddBoolean("nrfld", "fixed_u_rad", false);
//   if (calc_in_temp) {
//     // raise error
//     std::stringstream msg;
//     msg << "Error: calc_in_temp is not implemented yet.";
//     ATHENA_ERROR(msg);
//   }
//   if (output_defect)
//     def.NewAthenaArray(RadFLD::NTEMP, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  pmb->RegisterMeshBlockData(u_gas);
  pmb->RegisterMeshBlockData(u_rad);

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED)
    // future extension may add "int nregister" to Hydro class
    u_rad2.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);

  // If STS RKL2, allocate additional memory registers
  if (STS_ENABLED) {
    std::string sts_integrator = pin->GetOrAddString("time", "sts_integrator", "rkl2");
    if (sts_integrator == "rkl2") {
      u_rad0.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
      u_rad_fl_div.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
    }
  }

//   // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
//   if (pmb->pmy_mesh->multilevel)
//     refinement_idx_ = pmy_block->pmr->AddToRefinement(&u, &coarse_u); //!

  // caution!
    // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pmb->pmy_mesh->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&u_rad, &coarse_u_rad);
  }

  // Enroll CellCenteredBoundaryVariable object for advection
  u_rad_fldbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&u_rad_fldbvar);
  pmb->pbval->prfldbvar = &u_rad_fldbvar;
  pmb->pbval->bvars_main_int.push_back(&u_rad_fldbvar); // for main integration

  // int tmp = pmb->pbval->prfldbvar->nu_;
  // std::cout << "tmp: " << tmp << std::endl;

  // Allocate memory for scratch arrays
  u_radl_.NewAthenaArray(pmb->ncells1);
  u_radr_.NewAthenaArray(pmb->ncells1);
  u_radlb_.NewAthenaArray(pmb->ncells1);
  x1face_area_.NewAthenaArray(pmb->ncells1+1);
  Mesh *pm = pmy_block->pmy_mesh;
  if (pm->f2) {
    x2face_area_.NewAthenaArray(pmb->ncells1);
    x2face_area_p1_.NewAthenaArray(pmb->ncells1);
  }
  if (pm->f3) {
    x3face_area_.NewAthenaArray(pmb->ncells1);
    x3face_area_p1_.NewAthenaArray(pmb->ncells1);
  }
  cell_volume_.NewAthenaArray(pmb->ncells1);
  dflx_.NewAthenaArray(pmb->ncells1);

  // set a default opacity function
  UpdateOpacity = DefaultOpacity;

  // set constants
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  Real R_gas = 8.3144621e7; // gas constant in erg K^-1 mol^-1
  Real const_opacity_dim = pin->GetOrAddReal("nrfld", "const_opacity", 0.4); //caution: in cm^2 g^-1

  Real rho_unit = pin->GetReal("hydro", "rho_unit");
  Real egas_unit = pin->GetReal("hydro", "egas_unit");
  Real pres_unit = egas_unit;
  Real vel_unit = std::sqrt(pres_unit/rho_unit);

  Real time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  Real leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [FLD2::FLD2]" << std::endl;
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
  const_opacity = const_opacity_dim*leng_unit*rho_unit; // to be multiplied by rho

  std::cout << "c_ph in sim: " << c_ph << std::endl;
  std::cout << "a_r in sim: " << a_r << std::endl;
  std::cout << "const_opacity in sim: " << const_opacity << std::endl;
}

void FLD2::EnrollOpacityFunction(FLDOpacityFunc MyOpacityFunction) {
  UpdateOpacity = MyOpacityFunction;
}


//----------------------------------------------------------------------------------------
//! \fn FLD2::~FLD2()
//! \brief FLD2 destructor
FLD2::~FLD2() {
//   delete pmg;
}


// //----------------------------------------------------------------------------------------
// //! \fn Real FLD2::CalculateSigmaR(const Real den, const Real egas)
// //! \brief Calculate Rosseland mean opacity
// Real FLD2::CalculateSigmaR(const Real den, const Real egas) {
//   Real sigma_r = pmg->const_opacity*den; // temporary
//   return sigma_r;
// }


//----------------------------------------------------------------------------------------
//! \fn void FLD2::LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u)
//! \brief Load hydro variables from conserved variables
void FLD2::LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &fld_u_gas) {
  if(only_rad && pmy_block->pmy_mesh->dt > 0.0) return;
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
        fld_u_gas(k,j,i) = igm1*w(IPR,k,j,i);
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void FLD2::UpdateHydroVariables(AthenaArray<Real> &w,
//!               AthenaArray<Real> &hydro_u, const AthenaArray<Real> &fld_u)
//! \brief Update conserved variables from hydro variables
void FLD2::UpdateHydroVariables(AthenaArray<Real> &w, AthenaArray<Real> &hydro_u,
                                const AthenaArray<Real> &fld_u_rad,
                                const AthenaArray<Real> &fld_u_gas) {
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
        if (!fixed_u_rad)
          u_rad(k,j,i) = fld_u_rad(k,j,i);
        u_gas(k,j,i) = fld_u_gas(k,j,i);
        if (!only_rad) {
          hydro_u(IEN,k,j,i) += (fld_u_gas(k,j,i) - igm1*w(IPR,k,j,i));
          w(IPR,k,j,i) = gm1*fld_u_gas(k,j,i);
        }
      }
    }
  }
  return;
}
