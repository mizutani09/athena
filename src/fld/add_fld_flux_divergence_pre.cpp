//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file add_fld_flux_divergence.cpp
//! \brief Computes divergence of the fld fluxes and adds that to a conserved
//! variable register

// C headers

// C++ headers
#include <cmath>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "fld.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void FLD::AddFluxDivergence
//! \brief Adds flux divergence to weighted average of conservative variables from
//!  previous step(s) of time integrator algorithm
//!
//! \todo (felker): after the removal of AddCoordTermsDivergence() fn call from
//! Hydro::AddFluxDivergence(), the 2x fns could be trivially shared if:
//! - flux/s_flux renamed to the same class member name
//! - 7x below references of x1face_area_ ... dflx_ private class members (which are only
//! ever used in this fn and are members to prevent de/allocating each fn call)
//! - NHYDRO/NSCALARS is replaced with array_out.GetDim4()
//! - Hydro should be derived from PassiveScalars
//!
//! \todo (felker): remove the following unnecessary private class member?
//! field_diffusion.cpp:66:    cell_volume_.NewAthenaArray(nc1);

void FLD::AddFluxDivergence(const Real wght, AthenaArray<Real> &u_rad_out,
                            int stage, bool physical_state) {
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = u_rad_flux[X1DIR];
  AthenaArray<Real> &x2flux = u_rad_flux[X2DIR];
  AthenaArray<Real> &x3flux = u_rad_flux[X3DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &x1area = x1face_area_, &x2area = x2face_area_,
                 &x2area_p1 = x2face_area_p1_, &x3area = x3face_area_,
                 &x3area_p1 = x3face_area_p1_, &vol = cell_volume_, &dflx = dflx_;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      // calculate x1-flux divergence
      pmb->pcoord->Face1Area(k, j, is, ie+1, x1area);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        dflx(i) = (x1area(i+1)*x1flux(k,j,i+1) - x1area(i)*x1flux(k,j,i));
      }

      // calculate x2-flux divergence
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k, j  , is, ie, x2area   );
        pmb->pcoord->Face2Area(k, j+1, is, ie, x2area_p1);
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dflx(i) += (x2area_p1(i)*x2flux(k,j+1,i) - x2area(i)*x2flux(k,j,i));
        }
      }

      // calculate x3-flux divergence
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k  , j, is, ie, x3area   );
        pmb->pcoord->Face3Area(k+1, j, is, ie, x3area_p1);
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dflx(i) += (x3area_p1(i)*x3flux(k+1,j,i) - x3area(i)*x3flux(k,j,i));
        }
      }

      // update conserved variables
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      if (!explicit_positivity_diagnostics_ || !physical_state) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          u_rad_out(k,j,i) -= wght*dflx(i)/vol(i);
        }
        continue;
      }
      for (int i=is; i<=ie; ++i) {
        const Real old_er = u_rad_out(k,j,i);
        const Real delta_er = -wght*dflx(i)/vol(i);
        const Real next_er = old_er + delta_er;
        if (!std::isfinite(old_er) || old_er < 0.0
            || !std::isfinite(next_er) || next_er < 0.0) {
          const Real f1m = x1flux(k,j,i), f1p = x1flux(k,j,i+1);
          const Real f2m = pmb->block_size.nx2 > 1 ? x2flux(k,j,i) : 0.0;
          const Real f2p = pmb->block_size.nx2 > 1 ? x2flux(k,j+1,i) : 0.0;
          const Real f3m = pmb->block_size.nx3 > 1 ? x3flux(k,j,i) : 0.0;
          const Real f3p = pmb->block_size.nx3 > 1 ? x3flux(k+1,j,i) : 0.0;
          const Real outgoing_rate =
              x1area(i+1)*std::max(f1p, 0.0)
              + x1area(i)*std::max(-f1m, 0.0)
              + (pmb->block_size.nx2 > 1
                 ? x2area_p1(i)*std::max(f2p, 0.0)
                   + x2area(i)*std::max(-f2m, 0.0) : 0.0)
              + (pmb->block_size.nx3 > 1
                 ? x3area_p1(i)*std::max(f3p, 0.0)
                   + x3area(i)*std::max(-f3m, 0.0) : 0.0);
          const Real incoming_rate =
              x1area(i+1)*std::max(-f1p, 0.0)
              + x1area(i)*std::max(f1m, 0.0)
              + (pmb->block_size.nx2 > 1
                 ? x2area_p1(i)*std::max(-f2p, 0.0)
                   + x2area(i)*std::max(f2m, 0.0) : 0.0)
              + (pmb->block_size.nx3 > 1
                 ? x3area_p1(i)*std::max(-f3p, 0.0)
                   + x3area(i)*std::max(f3m, 0.0) : 0.0);
          std::stringstream msg;
          msg << "### FATAL ERROR in explicit FLD radiation update\n"
              << "phase=" << ((!std::isfinite(old_er) || old_er < 0.0)
                                  ? "before_transport" : "after_transport")
              << " cycle=" << pmb->pmy_mesh->ncycle
              << " time=" << pmb->pmy_mesh->time
              << " dt=" << pmb->pmy_mesh->dt << " stage=" << stage
              << " gid=" << pmb->gid << " level=" << pmb->loc.level
              << " logical=(" << pmb->loc.lx1 << ',' << pmb->loc.lx2 << ','
              << pmb->loc.lx3 << ") cell=(" << k << ',' << j << ',' << i
              << ") xyz=(" << pmb->pcoord->x1v(i) << ','
              << pmb->pcoord->x2v(j) << ',' << pmb->pcoord->x3v(k) << ")\n"
              << "Er_old=" << old_er << " delta_Er=" << delta_er
              << " Er_next=" << next_er << " subdt=" << wght
              << " flux_divergence=" << dflx(i)/vol(i) << '\n'
              << "F1m=" << f1m << " F1p=" << f1p
              << " F2m=" << f2m << " F2p=" << f2p
              << " F3m=" << f3m << " F3p=" << f3p << '\n'
              << "outgoing_delta=" << wght*outgoing_rate/vol(i)
              << " incoming_delta=" << wght*incoming_rate/vol(i)
              << " outgoing_over_Er="
              << wght*outgoing_rate/(vol(i)*std::max(old_er,TINY_NUMBER))
              << " incoming_over_Er="
              << wght*incoming_rate/(vol(i)*std::max(old_er,TINY_NUMBER)) << '\n'
              << "vf1m=" << pmb->phydro->vf[X1DIR](k,j,i)
              << " vf1p=" << pmb->phydro->vf[X1DIR](k,j,i+1)
              << " vf2m=" << (pmb->block_size.nx2 > 1
                                  ? pmb->phydro->vf[X2DIR](k,j,i) : 0.0)
              << " vf2p=" << (pmb->block_size.nx2 > 1
                                  ? pmb->phydro->vf[X2DIR](k,j+1,i) : 0.0)
              << " vf3m=" << (pmb->block_size.nx3 > 1
                                  ? pmb->phydro->vf[X3DIR](k,j,i) : 0.0)
              << " vf3p=" << (pmb->block_size.nx3 > 1
                                  ? pmb->phydro->vf[X3DIR](k+1,j,i) : 0.0) << '\n'
              << "Er_xm=" << u_rad_out(k,j,i-1)
              << " Er_xp=" << u_rad_out(k,j,i+1)
              << " Er_ym=" << (pmb->block_size.nx2 > 1
                                  ? u_rad_out(k,j-1,i) : old_er)
              << " Er_yp=" << (pmb->block_size.nx2 > 1
                                  ? u_rad_out(k,j+1,i) : old_er)
              << " Er_zm=" << (pmb->block_size.nx3 > 1
                                  ? u_rad_out(k-1,j,i) : old_er)
              << " Er_zp=" << (pmb->block_size.nx3 > 1
                                  ? u_rad_out(k+1,j,i) : old_er) << '\n';
          ATHENA_ERROR(msg);
        }
        u_rad_out(k,j,i) = next_er;
      }
    }
  }
  return;
}
