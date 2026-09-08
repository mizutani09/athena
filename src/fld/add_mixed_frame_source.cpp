//========================================================================================
//! \file add_mixed_frame_source.cpp
//! \brief Explicit radiation force and mixed-frame energy terms.

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

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
  // Keep the existing source formulation as the default.  The opt-in
  // diagnostic mode puts the same face radiation pressure in the HLLC
  // momentum flux and must therefore suppress this explicit force.
  const bool radiation_pressure_in_flux =
      std::getenv("ATHENA_FLD_PRESSURE_IN_FLUX") != nullptr;
  const bool do_force=include_radiation_force && !radiation_pressure_in_flux;
  const bool do_mixed=include_mixed_frame_terms
      && std::getenv("ATHENA_FLD_DISABLE_MIXED_FRAME") == nullptr;
  if (!do_force && !do_mixed) return;

  // Optional, read-only diagnostic for comparing the two terms in the
  // discrete momentum update. Each block reports only on its first call, and
  // profile rows are restricted to the x1 line nearest positive x2=x3=0.
  static std::set<int> diagnosed_blocks;
  const bool force_diagnostic = std::getenv("ATHENA_FLD_FORCE_DIAGNOSTICS") != nullptr
                                && diagnosed_blocks.insert(pmb->gid).second;
  Real max_abs_residual = 0.0;
  Real max_relative_residual = 0.0;
  Real max_abs_gas = 0.0;
  Real max_abs_rad = 0.0;
  Real max_force_sum = 0.0;
  Real residual_at_max_force = 0.0;
  Real relative_at_max_force = 0.0;
  Real gas_at_max_residual = 0.0;
  Real rad_at_max_residual = 0.0;
  Real relative_at_max_residual = 0.0;
  int max_i = pmb->is, max_j = pmb->js, max_k = pmb->ks;
  int force_i = pmb->is, force_j = pmb->js, force_k = pmb->ks;
  if (force_diagnostic) {
    std::cout << std::setprecision(17)
              << "# FLD_FORCE_DIAGNOSTIC dt_code=" << dt
              << " columns: x1 x2 a_gas_x a_rad_x residual_x relative"
              << " v_flux_x v_after_source_x residual_dt" << std::endl;
  }

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

        if (force_diagnostic) {
          const Real rho = prim(IDN,k,j,i);
          const Real gas_flux_div_x =
              (pmb->phydro->flux[X1DIR](IM1,k,j,i+1)
               - pmb->phydro->flux[X1DIR](IM1,k,j,i))/pmb->pcoord->dx1f(i);
          const Real gas_flux_div_y = (pmb->block_size.nx2>1)
              ? (pmb->phydro->flux[X2DIR](IM1,k,j+1,i)
                 - pmb->phydro->flux[X2DIR](IM1,k,j,i))/pmb->pcoord->dx2f(j)
              : 0.0;
          const Real gas_flux_div_z = (pmb->block_size.nx3>1)
              ? (pmb->phydro->flux[X3DIR](IM1,k+1,j,i)
                 - pmb->phydro->flux[X3DIR](IM1,k,j,i))/pmb->pcoord->dx3f(k)
              : 0.0;
          const Real a_gas_x = -(gas_flux_div_x+gas_flux_div_y+gas_flux_div_z)/rho;
          const Real a_rad_x = -lambda*ex/rho;
          const Real residual_x = a_gas_x+a_rad_x;
          const Real relative = std::abs(residual_x)
              / std::max(std::abs(a_gas_x)+std::abs(a_rad_x), TINY_NUMBER);
          const Real v_flux_x = hydro_u(IM1,k,j,i)/hydro_u(IDN,k,j,i);
          const Real v_after_source_x =
              (hydro_u(IM1,k,j,i)-dt*lambda*ex)/hydro_u(IDN,k,j,i);
          max_abs_gas=std::max(max_abs_gas,std::abs(a_gas_x));
          max_abs_rad=std::max(max_abs_rad,std::abs(a_rad_x));
          const Real force_sum=std::abs(a_gas_x)+std::abs(a_rad_x);
          if (force_sum>max_force_sum) {
            max_force_sum=force_sum;
            residual_at_max_force=residual_x;
            relative_at_max_force=relative;
            force_i=i; force_j=j; force_k=k;
          }
          if (std::abs(residual_x)>max_abs_residual) {
            max_abs_residual=std::abs(residual_x);
            max_i=i; max_j=j; max_k=k;
            gas_at_max_residual=a_gas_x;
            rad_at_max_residual=a_rad_x;
            relative_at_max_residual=relative;
          }
          max_relative_residual=std::max(max_relative_residual,relative);
          if (pmb->pcoord->x2v(j)>0.0
              && pmb->pcoord->x2v(j)<=0.51*pmb->pcoord->dx2f(j)
              && pmb->pcoord->x3v(k)>0.0
              && pmb->pcoord->x3v(k)<=0.51*pmb->pcoord->dx3f(k)) {
            std::cout << "FLD_FORCE_PROFILE " << pmb->pcoord->x1v(i) << " "
                      << pmb->pcoord->x2v(j) << " " << a_gas_x << " "
                      << a_rad_x << " " << residual_x << " " << relative << " "
                      << v_flux_x << " " << v_after_source_x << " "
                      << residual_x*dt << std::endl;
          }
        }

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
  if (force_diagnostic) {
    std::cout << "FLD_FORCE_MAX gid=" << pmb->gid
              << " abs_gas=" << max_abs_gas << " abs_rad=" << max_abs_rad
              << " abs_residual=" << max_abs_residual
              << " max_relative=" << max_relative_residual
              << " max_force_sum=" << max_force_sum
              << " residual_at_max_force=" << residual_at_max_force
              << " relative_at_max_force=" << relative_at_max_force
              << " force_x1=" << pmb->pcoord->x1v(force_i)
              << " force_x2=" << pmb->pcoord->x2v(force_j)
              << " force_x3=" << pmb->pcoord->x3v(force_k)
              << " gas_at_max_residual=" << gas_at_max_residual
              << " rad_at_max_residual=" << rad_at_max_residual
              << " relative_at_max_residual=" << relative_at_max_residual
              << " i=" << max_i << " j=" << max_j << " k=" << max_k
              << " x1=" << pmb->pcoord->x1v(max_i)
              << " x2=" << pmb->pcoord->x2v(max_j)
              << " x3=" << pmb->pcoord->x3v(max_k) << std::endl;
  }
}
