//========================================================================================
//! \file add_mixed_frame_source.cpp
//! \brief Explicit radiation force and mixed-frame energy terms.

#include <algorithm>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "fld.hpp"

void FLD::AddExplicitSourceTerms(const Real dt, const AthenaArray<Real> &prim,
                                  AthenaArray<Real> &hydro_u) {
  if (!is_couple || only_rad) return;
  MeshBlock *pmb=pmy_block;
  const bool do_force=include_radiation_force;
  const bool do_mixed=include_mixed_frame_terms;
  if (!do_force && !do_mixed) return;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real idx=1.0/pmb->pcoord->dx1f(i);
        const Real idy=(pmb->block_size.nx2>1)?1.0/pmb->pcoord->dx2f(j):0.0;
        const Real idz=(pmb->block_size.nx3>1)?1.0/pmb->pcoord->dx3f(k):0.0;
        const Real ex=idx*(rad_face_g[X1DIR](k,j,i+1)
                           -rad_face_g[X1DIR](k,j,i));
        const Real ey=(pmb->block_size.nx2>1)
            ? idy*(rad_face_g[X2DIR](k,j+1,i)-rad_face_g[X2DIR](k,j,i)) : 0.0;
        const Real ez=(pmb->block_size.nx3>1)
            ? idz*(rad_face_g[X3DIR](k+1,j,i)-rad_face_g[X3DIR](k,j,i)) : 0.0;

        // CalculateRadiationFaceStates has already evaluated the closure at
        // this RK stage. Reuse it here instead of recomputing R and lambda.
        const Real lambda=rad_state_cc_(RadFLD::LAMBDA,k,j,i);

        if (do_force) {
          hydro_u(IM1,k,j,i)-=dt*lambda*ex;
          hydro_u(IM2,k,j,i)-=dt*lambda*ey;
          hydro_u(IM3,k,j,i)-=dt*lambda*ez;
        }

        if (do_mixed) {
          // Zhang et al.: q=2 lambda kappa_P/chi_R. In FLD, sigma_p and
          // sigma_r are the corresponding inverse-length coefficients.
          const Real q=2.0*lambda*sigma_p(k,j,i)
                       /std::max(sigma_r(k,j,i),TINY_NUMBER);
          const Real vx=0.5*(pmb->phydro->vf[X1DIR](k,j,i)
                             +pmb->phydro->vf[X1DIR](k,j,i+1));
          const Real vy=(pmb->block_size.nx2>1)
              ? 0.5*(pmb->phydro->vf[X2DIR](k,j,i)
                     +pmb->phydro->vf[X2DIR](k,j+1,i)) : prim(IVY,k,j,i);
          const Real vz=(pmb->block_size.nx3>1)
              ? 0.5*(pmb->phydro->vf[X3DIR](k,j,i)
                     +pmb->phydro->vf[X3DIR](k+1,j,i)) : prim(IVZ,k,j,i);
          const Real source=(q-lambda)*(vx*ex+vy*ey+vz*ez);
          if (NON_BAROTROPIC_EOS) hydro_u(IEN,k,j,i)+=dt*source;
          u_rad(k,j,i)=std::max(u_rad(k,j,i)-dt*source,TINY_NUMBER);
        }
      }
    }
  }
}
