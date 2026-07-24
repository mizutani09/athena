//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fld.cpp
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
#include "fld.hpp"


inline void DefaultOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD *prfld = pmb->prfld;
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
//! \fn FLD::FLD(MeshBlock *pmb, ParameterInput *pin)
//! \brief FLD constructor
FLD::FLD(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb),
    u_gas(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u_rad(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u_rad1(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_u_rad(pmb->ncc3, pmb->ncc2, pmb->ncc1),
    sigma_p(pmb->ncells3,pmb->ncells2,pmb->ncells1),
    sigma_r(pmb->ncells3,pmb->ncells2,pmb->ncells1),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    u_rad_flux{ {pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
            {pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
             (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)},
            {pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
             (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)}
    },
    u_rad_fldbvar(pmb, &u_rad, &coarse_u_rad, u_rad_flux,
                  CellCenteredBoundaryVariable::max_phys_id, true),
    refinement_idx_(),
    is_couple(), only_rad(), cut_diff(), include_radiation_force(), fixed_u_rad()
    {
  is_couple = pin->GetOrAddBoolean("fld", "is_couple", true);
  only_rad = pin->GetOrAddBoolean("fld", "only_rad", false);
  cut_diff = pin->GetOrAddBoolean("fld", "cut_diff", false);
  include_radiation_force =
      pin->GetOrAddBoolean("fld", "include_radiation_force", true);
  fixed_flux_limiter = pin->GetOrAddBoolean("fld", "fixed_flux_limiter", false);
  fixed_u_rad = pin->GetOrAddBoolean("fld", "fixed_u_rad", false);
  include_mixed_frame_terms =
      pin->GetOrAddBoolean("fld", "include_mixed_frame_terms", false);

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

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pmb->pmy_mesh->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&u_rad, &coarse_u_rad);
  }

  // Enroll CellCenteredBoundaryVariable object for advection
  u_rad_fldbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&u_rad_fldbvar);
  pmb->pbval->prfldbvar = &u_rad_fldbvar;
  pmb->pbval->bvars_main_int.push_back(&u_rad_fldbvar); // for main integration

  // Allocate memory for scratch arrays
  Mesh *pm = pmy_block->pmy_mesh;
  rad_state_cc_.NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                               pmb->ncells3, pmb->ncells2, pmb->ncells1);
  rad_statel_.NewAthenaArray(RadFLD::NRAD_FACE_STATE, pmb->ncells1);
  rad_stater_.NewAthenaArray(RadFLD::NRAD_FACE_STATE, pmb->ncells1);
  rad_statelb_.NewAthenaArray(RadFLD::NRAD_FACE_STATE, pmb->ncells1);
  rad_face_l[X1DIR].NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                                   pmb->ncells3, pmb->ncells2, pmb->ncells1+1);
  rad_face_r[X1DIR].NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                                   pmb->ncells3, pmb->ncells2, pmb->ncells1+1);
  rad_face_g[X1DIR].NewAthenaArray(pmb->ncells3, pmb->ncells2,
                                  pmb->ncells1+1);
  if (pm->f2) {
    rad_face_l[X2DIR].NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                                     pmb->ncells3, pmb->ncells2+1, pmb->ncells1);
    rad_face_r[X2DIR].NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                                     pmb->ncells3, pmb->ncells2+1, pmb->ncells1);
    rad_face_g[X2DIR].NewAthenaArray(pmb->ncells3, pmb->ncells2+1,
                                    pmb->ncells1);
  }
  if (pm->f3) {
    rad_face_l[X3DIR].NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                                     pmb->ncells3+1, pmb->ncells2, pmb->ncells1);
    rad_face_r[X3DIR].NewAthenaArray(RadFLD::NRAD_FACE_STATE,
                                     pmb->ncells3+1, pmb->ncells2, pmb->ncells1);
    rad_face_g[X3DIR].NewAthenaArray(pmb->ncells3+1, pmb->ncells2,
                                    pmb->ncells1);
  }
  x1face_area_.NewAthenaArray(pmb->ncells1+1);
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
  Real const_opacity_dim = pin->GetOrAddReal("fld", "const_opacity", 0.4); //caution: in cm^2 g^-1

  Real rho_unit = pin->GetReal("hydro", "rho_unit");
  Real egas_unit = pin->GetReal("hydro", "egas_unit");
  Real pres_unit = egas_unit;
  Real vel_unit = std::sqrt(pres_unit/rho_unit);

  Real time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  Real leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [FLD::FLD]" << std::endl;
    msg << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  Real T_unit;
#if GENERAL_EOS
  T_unit = pin->GetReal("hydro", "T_unit");
#else
  Real mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/R_gas;
#endif
  c_ph = c_ph_dim/vel_unit;
  a_r = a_r_dim/(egas_unit/std::pow(T_unit, 4));
  const_opacity = const_opacity_dim*leng_unit*rho_unit; // to be multiplied by rho

  // std::cout << "c_ph in sim: " << c_ph << std::endl;
  // std::cout << "a_r in sim: " << a_r << std::endl;
  // std::cout << "const_opacity in sim: " << const_opacity << std::endl;
}

void FLD::EnrollOpacityFunction(FLDOpacityFunc MyOpacityFunction) {
  UpdateOpacity = MyOpacityFunction;
}


//----------------------------------------------------------------------------------------
//! \fn FLD::~FLD()
//! \brief FLD destructor
FLD::~FLD() {
}


//----------------------------------------------------------------------------------------
//! \fn void FLD::LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u)
//! \brief Load hydro variables from conserved variables
void FLD::LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &fld_u_gas) {
  if(only_rad && pmy_block->pmy_mesh->dt > 0.0) return;
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
#if GENERAL_EOS
        Real rho = w(IDN,k,j,i);
        Real pres = w(IPR,k,j,i);
        fld_u_gas(k,j,i) = pmy_block->peos->EgasFromRhoP(rho, pres);
#else
        Real igm1 = 1.0/(pmy_block->peos->GetGamma() - 1.0);
        fld_u_gas(k,j,i) = igm1*w(IEN,k,j,i);
#endif
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void FLD::UpdateHydroVariables(AthenaArray<Real> &w,
//!               AthenaArray<Real> &hydro_u, const AthenaArray<Real> &fld_u)
//! \brief Update conserved variables from hydro variables
void FLD::UpdateHydroVariables(AthenaArray<Real> &w, AthenaArray<Real> &hydro_u,
                                const AthenaArray<Real> &fld_u_rad,
                                const AthenaArray<Real> &fld_u_gas) {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  if (pmy_block->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  
  // first, update FLD quantities
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        if (!fixed_u_rad)
          u_rad(k,j,i) = fld_u_rad(k,j,i);
        u_gas(k,j,i) = fld_u_gas(k,j,i);
      }
    }
  }

  // then, update hydro variables
  if (!only_rad) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
#if GENERAL_EOS
          Real rho = w(IDN,k,j,i);
          Real egas_old = pmy_block->peos->EgasFromRhoP(rho, w(IPR,k,j,i));
          Real pres_new = pmy_block->peos->PresFromRhoEg(rho, fld_u_gas(k,j,i));
          hydro_u(IEN,k,j,i) += (fld_u_gas(k,j,i) - egas_old);
          w(IPR,k,j,i) = pres_new;
#else
          Real gm1 = pmy_block->peos->GetGamma() - 1.0;
          Real igm1 = 1.0/gm1;
          hydro_u(IEN,k,j,i) += (fld_u_gas(k,j,i) - igm1*w(IPR,k,j,i));
          w(IPR,k,j,i) = gm1*fld_u_gas(k,j,i);
#endif
        }
      }
    }
  }
  return;
}
