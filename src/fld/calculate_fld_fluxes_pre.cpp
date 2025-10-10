//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_fld_fluxes.cpp
//! \brief Calculate fld fluxes(advection term)
//!        mainly copied from scaler/calculate_scalar_fluxes.cpp

// C headers

// C++ headers
#include <algorithm>   // min,max

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"   // reapply floors to face-centered reconstructed states
#include "../hydro/hydro.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "fld.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void FLD2::CalculateFluxes
//! \brief Calculate FLD fluxes using reconstruction + weighted upwinding rule
//!
//! \note
//! design decision: do not pass Hydro::flux (for mass flux) via function parameters,
//! since
//! - it is unlikely that anything else would be passed,
//! - the current PassiveScalars class/feature implementation is inherently
//!   coupled to Hydro class
//! - high-order calculation of scalar fluxes will require other Hydro flux
//!   approximations (flux_fc in calculate_fluxes.cpp is currently not saved persistently
//!   in Hydro class but each flux dir is temp. stored in 4D scratch array scr1_nkji_)
void FLD2::CalculateFluxes(AthenaArray<Real> &u_rad, const int order) {
  MeshBlock *pmb = pmy_block;
  Hydro &hyd = *(pmb->phydro);

  AthenaArray<Real> &x1flux = u_rad_flux[X1DIR];
  AthenaArray<Real> vf;
  vf.InitWithShallowSlice(hyd.vf[X1DIR], 4, 0, 1);
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;

  //--------------------------------------------------------------------------------------
  // i-direction

  // set the loop limits
  jl = js, ju = je, kl = ks, ku = ke;
  if (pmb->block_size.nx2 > 1) {
    if (pmb->block_size.nx3 == 1) // 2D
      jl = js-1, ju = je+1, kl = ks, ku = ke;
    else // 3D
      jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      // reconstruct L/R states
      if (order == 1) {
        pmb->precon->DonorCellX1(k, j, is-1, ie+1, u_rad, u_radl_, u_radr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX1(k, j, is-1, ie+1, u_rad, u_radl_, u_radr_);
      } else {
        pmb->precon->PiecewiseParabolicX1(k, j, is-1, ie+1, u_rad, u_radl_, u_radr_);
//         for (int n=0; n<RadFLD::NADV; ++n) {
// #pragma omp simd
//           for (int i=is; i<=ie+1; ++i) {
//             pmb->peos->ApplyPassiveScalarFloors(rl_, n, k, j, i);
//             pmb->peos->ApplyPassiveScalarFloors(rr_, n, k, j, i);
//           }
//         }
      }

      ComputeUpwindFlux(k, j, is, ie+1, u_radl_, u_radr_, vf, x1flux);
    }
  }

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmb->pmy_mesh->f2) {
    AthenaArray<Real> &x2flux = u_rad_flux[X2DIR];
    vf.InitWithShallowSlice(hyd.vf[X2DIR], 4, 0, 1);

    // set the loop limits
    il = is-1, iu = ie+1, kl = ks, ku = ke;
    // TODO(felker): fix loop limits for fourth-order hydro
    //    if (MAGNETIC_FIELDS_ENABLED) {
    if (pmb->block_size.nx3 == 1) // 2D
      kl = ks, ku = ke;
    else // 3D
      kl = ks-1, ku = ke+1;
    //    }

    for (int k=kl; k<=ku; ++k) {
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX2(k, js-1, il, iu, u_rad, u_radl_, u_radr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX2(k, js-1, il, iu, u_rad, u_radl_, u_radr_);
      } else {
        pmb->precon->PiecewiseParabolicX2(k, js-1, il, iu, u_rad, u_radl_, u_radr_);
//         for (int n=0; n<RadFLD::NADV; ++n) {
// #pragma omp simd
//           for (int i=il; i<=iu; ++i) {
//             pmb->peos->ApplyPassiveScalarFloors(rl_, n, k, js-1, i);
//             //pmb->peos->ApplyPassiveScalarFloors(rr_, n, k, j, i);
//           }
//         }
      }
      for (int j=js; j<=je+1; ++j) {
        // reconstruct L/R states at j
        if (order == 1) {
          pmb->precon->DonorCellX2(k, j, il, iu, u_rad, u_radlb_, u_radr_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX2(k, j, il, iu, u_rad, u_radlb_, u_radr_);
        } else {
          pmb->precon->PiecewiseParabolicX2(k, j, il, iu, u_rad, u_radlb_, u_radr_);
//           for (int n=0; n<RadFLD::NADV; ++n) {
// #pragma omp simd
//             for (int i=il; i<=iu; ++i) {
//               pmb->peos->ApplyPassiveScalarFloors(rlb_, n, k, j, i);
//               pmb->peos->ApplyPassiveScalarFloors(rr_, n, k, j, i);
//             }
//           }
        }

        ComputeUpwindFlux(k, j, il, iu, u_radl_, u_radr_, vf, x2flux);

        // swap the arrays for the next step
        u_radl_.SwapAthenaArray(u_radlb_);
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->f3) {
    AthenaArray<Real> &x3flux = u_rad_flux[X3DIR];
    vf.InitWithShallowSlice(hyd.vf[X3DIR], 4, 0, 1);

    // set the loop limits
    // TODO(felker): fix loop limits for fourth-order hydro
    //    if (MAGNETIC_FIELDS_ENABLED)
    il = is-1, iu = ie+1, jl = js-1, ju = je+1;

    for (int j=jl; j<=ju; ++j) { // this loop ordering is intentional
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX3(ks-1, j, il, iu, u_rad, u_radl_, u_radr_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX3(ks-1, j, il, iu, u_rad, u_radl_, u_radr_);
      } else {
        pmb->precon->PiecewiseParabolicX3(ks-1, j, il, iu, u_rad, u_radl_, u_radr_);
//         for (int n=0; n<RadFLD::NADV; ++n) {
// #pragma omp simd
//           for (int i=il; i<=iu; ++i) {
//             pmb->peos->ApplyPassiveScalarFloors(rl_, n, ks-1, j, i);
//             //pmb->peos->ApplyPassiveScalarFloors(rr_, n, k, j, i);
//           }
//         }
      }
      for (int k=ks; k<=ke+1; ++k) {
        // reconstruct L/R states at k
        if (order == 1) {
          pmb->precon->DonorCellX3(k, j, il, iu, u_rad, u_radlb_, u_radr_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX3(k, j, il, iu, u_rad, u_radlb_, u_radr_);
        } else {
          pmb->precon->PiecewiseParabolicX3(k, j, il, iu, u_rad, u_radlb_, u_radr_);
//           for (int n=0; n<RadFLD::NADV; ++n) {
// #pragma omp simd
//             for (int i=il; i<=iu; ++i) {
//               pmb->peos->ApplyPassiveScalarFloors(rlb_, n, k, j, i);
//               pmb->peos->ApplyPassiveScalarFloors(rr_, n, k, j, i);
//             }
//           }
        }

        ComputeUpwindFlux(k, j, il, iu, u_radl_, u_radr_, vf, x3flux);

        // swap the arrays for the next step
        u_radl_.SwapAthenaArray(u_radlb_);
      }
    }
  }
  return;
}


void FLD2::ComputeUpwindFlux(const int k, const int j, const int il,
                                       const int iu, // CoordinateDirection dir,
                                       AthenaArray<Real> &rl, AthenaArray<Real> &rr, // 2D
                                       AthenaArray<Real> &vf,  // 3D
                                       AthenaArray<Real> &flx_out) { // 4D

#pragma omp simd
  for (int i=il; i<=iu; i++) {
    Real fluid_flx = vf(k,j,i);
    if (fluid_flx >= 0.0)
      flx_out(k,j,i) = fluid_flx*u_radl_(i);
    else
      flx_out(k,j,i) = fluid_flx*u_radr_(i);
  }
  return;
}
