//========================================================================================
//! \file add_mixed_frame_source.cpp
//! \brief Explicit variable-limiter force and mixed-frame energy corrections.

#include <algorithm>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "fld.hpp"

void FLD2::AddExplicitSourceTerms(const Real dt, const AthenaArray<Real> &prim,
                                  AthenaArray<Real> &hydro_u) {
  if (!is_couple || only_rad) return;
  MeshBlock *pmb=pmy_block;
  // For a fixed limiter grad(lambda E)-lambda grad(E) vanishes identically.
  const bool do_force=include_radiation_force && !fixed_flux_limiter;
  const bool do_mixed=include_mixed_frame_terms;
  if (!do_force && !do_mixed) return;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real idx=0.5/pmb->pcoord->dx1f(i);
        const Real idy=(pmb->block_size.nx2>1)?0.5/pmb->pcoord->dx2f(j):0.0;
        const Real idz=(pmb->block_size.nx3>1)?0.5/pmb->pcoord->dx3f(k):0.0;
        const Real ex=idx*(u_rad(k,j,i+1)-u_rad(k,j,i-1));
        const Real ey=idy*(u_rad(k,j+1,i)-u_rad(k,j-1,i));
        const Real ez=idz*(u_rad(k+1,j,i)-u_rad(k-1,j,i));

        // CalculateRadiationFaceStates has already evaluated the closure at
        // this RK stage. Reuse it here instead of recomputing R and lambda.
        const Real lambda=rad_state_cc_(RadFLD2::LAMBDA,k,j,i);

        if (do_force) {
          auto prad = [&](int kk,int jj,int ii) {
            return rad_state_cc_(RadFLD2::LAMBDA,kk,jj,ii)
                   *rad_state_cc_(RadFLD2::ERAD,kk,jj,ii);
          };
          // HLLC supplies -grad(lambda E).  This correction makes the net
          // radiation force exactly -lambda grad(E).
          const Real cx=idx*(prad(k,j,i+1)-prad(k,j,i-1))-lambda*ex;
          const Real cy=(pmb->block_size.nx2>1)
              ? idy*(prad(k,j+1,i)-prad(k,j-1,i))-lambda*ey : 0.0;
          const Real cz=(pmb->block_size.nx3>1)
              ? idz*(prad(k+1,j,i)-prad(k-1,j,i))-lambda*ez : 0.0;
          hydro_u(IM1,k,j,i)+=dt*cx;
          hydro_u(IM2,k,j,i)+=dt*cy;
          hydro_u(IM3,k,j,i)+=dt*cz;
        }

        if (do_mixed) {
          // Zhang et al.: q=2 lambda kappa_P/chi_R. In FLD2, sigma_p and
          // sigma_r are the corresponding inverse-length coefficients.
          const Real q=2.0*lambda*sigma_p(k,j,i)
                       /std::max(sigma_r(k,j,i),TINY_NUMBER);
          const Real source=(q-lambda)
              *(prim(IVX,k,j,i)*ex+prim(IVY,k,j,i)*ey+prim(IVZ,k,j,i)*ez);
          if (NON_BAROTROPIC_EOS) hydro_u(IEN,k,j,i)+=dt*source;
          u_rad(k,j,i)=std::max(u_rad(k,j,i)-dt*source,TINY_NUMBER);
        }
      }
    }
  }
}
