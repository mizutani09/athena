//========================================================================================
//! \file add_mixed_frame_source.cpp
//! \brief Explicit radiation force and hyperbolic mixed-frame work.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "fld.hpp"

namespace {
// Match the fixed-limiter face coefficient and centered stencil used by the
// NR-FLD mixed-drift residual.  This branch is an experimental time-placement
// control, so every spatial coefficient is held identical between variants.
Real ExplicitMixedDrift(const FLD *fld, const MeshBlock *pmb,
                        const AthenaArray<Real> &prim,
                        const AthenaArray<Real> &erad, int k, int j, int i) {
  const Real idx = 1.0/pmb->pcoord->dx1f(i);
  auto face = [&](int ka, int ja, int ia, int kb, int jb, int ib, int velocity) {
    const Real opacity = RadFLD::FaceOpacity(
        fld->sigma_r(ka,ja,ia), fld->sigma_r(kb,jb,ib), idx);
    if (!(opacity > 0.0) || !std::isfinite(opacity)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in ExplicitMixedDrift: invalid face opacity.";
      ATHENA_ERROR(msg);
    }
    return 0.5*(fld->sigma_p(ka,ja,ia) + fld->sigma_p(kb,jb,ib))
        *(prim(velocity,ka,ja,ia) + prim(velocity,kb,jb,ib))
        *ONE_3RD/opacity;
  };
  const Real center = erad(k,j,i);
  Real term = face(k,j,i,k,j,i+1,IVX)*(erad(k,j,i+1)-center)
      + face(k,j,i-1,k,j,i,IVX)*(center-erad(k,j,i-1));
  if (pmb->block_size.nx2 > 1) {
    term += face(k,j,i,k,j+1,i,IVY)*(erad(k,j+1,i)-center)
          + face(k,j-1,i,k,j,i,IVY)*(center-erad(k,j-1,i));
  }
  if (pmb->block_size.nx3 > 1) {
    term += face(k,j,i,k+1,j,i,IVZ)*(erad(k+1,j,i)-center)
          + face(k-1,j,i,k,j,i,IVZ)*(center-erad(k-1,j,i));
  }
  return 0.5*idx*term;
}
}  // namespace

void FLD::EmitForceDiagnostic(const Real dt, const AthenaArray<Real> &prim,
                              const AthenaArray<Real> &hydro_u) {
  MeshBlock *pmb = pmy_block;
  const bool force_profile = force_diagnostic_verbosity_ >= 2;
  std::ostringstream diagnostic;
  diagnostic << std::setprecision(17);
  if (force_profile) {
    diagnostic << "# FLD_FORCE_DIAGNOSTIC rank=" << Globals::my_rank
               << " gid=" << pmb->gid << " dt_code=" << dt
               << " columns: x1 x2 a_gas_x a_rad_x residual_x relative"
               << " v_flux_x v_after_source_x residual_dt\n";
  }

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

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real ex = (rad_face_g[X1DIR](k,j,i+1)
                         - rad_face_g[X1DIR](k,j,i))/pmb->pcoord->dx1f(i);
        const Real lambda = rad_state_cc_(RadFLD::LAMBDA,k,j,i);
        const Real rho = prim(IDN,k,j,i);
        const Real gas_flux_div_x =
            (pmb->phydro->flux[X1DIR](IM1,k,j,i+1)
             - pmb->phydro->flux[X1DIR](IM1,k,j,i))/pmb->pcoord->dx1f(i);
        const Real gas_flux_div_y = (pmb->block_size.nx2 > 1)
            ? (pmb->phydro->flux[X2DIR](IM1,k,j+1,i)
               - pmb->phydro->flux[X2DIR](IM1,k,j,i))/pmb->pcoord->dx2f(j)
            : 0.0;
        const Real gas_flux_div_z = (pmb->block_size.nx3 > 1)
            ? (pmb->phydro->flux[X3DIR](IM1,k+1,j,i)
               - pmb->phydro->flux[X3DIR](IM1,k,j,i))/pmb->pcoord->dx3f(k)
            : 0.0;
        const Real a_gas_x =
            -(gas_flux_div_x + gas_flux_div_y + gas_flux_div_z)/rho;
        const Real a_rad_x = -lambda*ex/rho;
        const Real residual_x = a_gas_x + a_rad_x;
        const Real relative = std::abs(residual_x)
            / std::max(std::abs(a_gas_x) + std::abs(a_rad_x),
                       static_cast<Real>(TINY_NUMBER));
        const Real v_flux_x = hydro_u(IM1,k,j,i)/hydro_u(IDN,k,j,i);
        const Real v_after_source_x =
            (hydro_u(IM1,k,j,i) - dt*lambda*ex)/hydro_u(IDN,k,j,i);
        max_abs_gas = std::max(max_abs_gas, std::abs(a_gas_x));
        max_abs_rad = std::max(max_abs_rad, std::abs(a_rad_x));
        const Real force_sum = std::abs(a_gas_x) + std::abs(a_rad_x);
        if (force_sum > max_force_sum) {
          max_force_sum = force_sum;
          residual_at_max_force = residual_x;
          relative_at_max_force = relative;
          force_i = i; force_j = j; force_k = k;
        }
        if (std::abs(residual_x) > max_abs_residual) {
          max_abs_residual = std::abs(residual_x);
          max_i = i; max_j = j; max_k = k;
          gas_at_max_residual = a_gas_x;
          rad_at_max_residual = a_rad_x;
          relative_at_max_residual = relative;
        }
        max_relative_residual = std::max(max_relative_residual, relative);
        if (force_profile && pmb->pcoord->x2v(j) > 0.0
            && pmb->pcoord->x2v(j) <= 0.51*pmb->pcoord->dx2f(j)
            && pmb->pcoord->x3v(k) > 0.0
            && pmb->pcoord->x3v(k) <= 0.51*pmb->pcoord->dx3f(k)) {
          diagnostic << "FLD_FORCE_PROFILE " << pmb->pcoord->x1v(i) << " "
                     << pmb->pcoord->x2v(j) << " " << a_gas_x << " "
                     << a_rad_x << " " << residual_x << " " << relative << " "
                     << v_flux_x << " " << v_after_source_x << " "
                     << residual_x*dt << "\n";
        }
      }
    }
  }

  diagnostic << "FLD_FORCE_MAX rank=" << Globals::my_rank
             << " gid=" << pmb->gid
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
             << " x3=" << pmb->pcoord->x3v(max_k) << "\n";
#ifdef OPENMP_PARALLEL
#pragma omp critical(fld_diagnostic_output)
#endif
  std::cout << diagnostic.str() << std::flush;
}

void FLD::AddExplicitSourceTerms(const Real dt, const AthenaArray<Real> &prim,
                                 AthenaArray<Real> &hydro_u) {
  if (!is_couple || only_rad) return;
  MeshBlock *pmb = pmy_block;
  const bool do_force = PressureCoupled();
  const bool do_mixed = MixedFrameCoupled();
  const bool explicit_drift = do_mixed && !implicit_mixed_frame_drift;
  const bool ledger = do_mixed && mixed_frame_ledger_count < mixed_frame_ledger_limit;
  Real lambda_gas_volume = 0.0, drift_gas_volume = 0.0;
  Real lambda_abs_volume = 0.0, drift_abs_volume = 0.0;
  Real gas_delta_volume = 0.0, radiation_delta_volume = 0.0;
  Real radiation_floor_volume = 0.0;
  if (!do_force && !do_mixed) return;

  AthenaArray<Real> erad_before;
  if (explicit_drift) {
    erad_before.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
    for (int k = 0; k < pmb->ncells3; ++k)
      for (int j = 0; j < pmb->ncells2; ++j)
        for (int i = 0; i < pmb->ncells1; ++i)
          erad_before(k,j,i) = u_rad(k,j,i);
  }

  // Keep optional work out of the production cell loop.  State belongs to
  // this block rather than a process-wide gid set, so AMR gid reuse is safe;
  // a newly constructed restart block intentionally emits one new report.
  if (force_diagnostic_verbosity_ > 0
      && Globals::my_rank == force_diagnostic_rank_
      && pmb->gid == force_diagnostic_gid_
      && !force_diagnostic_printed_) {
    force_diagnostic_printed_ = true;
    EmitForceDiagnostic(dt, prim, hydro_u);
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real idx = 1.0/pmb->pcoord->dx1f(i);
        const Real idy = (pmb->block_size.nx2 > 1)
            ? 1.0/pmb->pcoord->dx2f(j) : 0.0;
        const Real idz = (pmb->block_size.nx3 > 1)
            ? 1.0/pmb->pcoord->dx3f(k) : 0.0;
        const Real ex = idx*(rad_face_g[X1DIR](k,j,i+1)
                             - rad_face_g[X1DIR](k,j,i));
        const Real ey = (pmb->block_size.nx2 > 1)
            ? idy*(rad_face_g[X2DIR](k,j+1,i) - rad_face_g[X2DIR](k,j,i))
            : 0.0;
        const Real ez = (pmb->block_size.nx3 > 1)
            ? idz*(rad_face_g[X3DIR](k+1,j,i) - rad_face_g[X3DIR](k,j,i))
            : 0.0;

        // CalculateRadiationFaceStates has already evaluated the closure at
        // this RK stage. Reuse it here instead of recomputing R and lambda.
        const Real lambda = rad_state_cc_(RadFLD::LAMBDA,k,j,i);

        // A density floor is a numerical atmosphere, not physical inertia.
        // Applying a finite radiation-force density to a cell pinned to that
        // floor can launch it across many cells in one otherwise valid fluid
        // CFL step.  An opt-in mask excludes only cells at the numerical lower
        // bound; zero preserves the historical behavior.
        const bool force_cell = RadiationForceActiveAtDensity(
            prim(IDN,k,j,i), pmb->peos->GetDensityFloor());
        if (do_force && force_cell) {
          hydro_u(IM1,k,j,i) -= dt*lambda*ex;
          hydro_u(IM2,k,j,i) -= dt*lambda*ey;
          hydro_u(IM3,k,j,i) -= dt*lambda*ez;
        }

        if (do_mixed) {
          // CASTRO's hyperbolic subsystem contains -lambda*v.grad(Er).
          // The q*v.grad(Er) Lorentz correction is implicit by default;
          // the comparison switch moves only that term to this source stage.
          const Real vx = 0.5*(pmb->phydro->vf[X1DIR](k,j,i)
                               + pmb->phydro->vf[X1DIR](k,j,i+1));
          const Real vy = (pmb->block_size.nx2 > 1)
              ? 0.5*(pmb->phydro->vf[X2DIR](k,j,i)
                     + pmb->phydro->vf[X2DIR](k,j+1,i)) : prim(IVY,k,j,i);
          const Real vz = (pmb->block_size.nx3 > 1)
              ? 0.5*(pmb->phydro->vf[X3DIR](k,j,i)
                     + pmb->phydro->vf[X3DIR](k+1,j,i)) : prim(IVZ,k,j,i);
          const Real source = -lambda*(vx*ex + vy*ey + vz*ez);
          const Real drift = explicit_drift
              ? ExplicitMixedDrift(this, pmb, prim, erad_before, k,j,i) : 0.0;
          const Real total_source = source + drift;
          const Real radiation_before = u_rad(k,j,i);
          const Real radiation_unclipped = radiation_before - dt*total_source;
          const Real radiation_after = std::max(
              radiation_unclipped, static_cast<Real>(TINY_NUMBER));
          if (NON_BAROTROPIC_EOS) hydro_u(IEN,k,j,i) += dt*total_source;
          u_rad(k,j,i) = radiation_after;
          if (ledger) {
            const Real volume = pmb->pcoord->GetCellVolume(k,j,i);
            lambda_gas_volume += dt*source*volume;
            drift_gas_volume += dt*drift*volume;
            lambda_abs_volume += std::abs(dt*source*volume);
            drift_abs_volume += std::abs(dt*drift*volume);
            gas_delta_volume += dt*total_source*volume;
            radiation_delta_volume += (radiation_after-radiation_before)*volume;
            radiation_floor_volume += (radiation_after-radiation_unclipped)*volume;
          }
        }
      }
    }
  }
  if (ledger) {
    ++mixed_frame_ledger_count;
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << std::scientific << std::setprecision(16)
              << "[MIXED_EXPLICIT_LEDGER] rank=" << Globals::my_rank
              << " gid=" << pmb->gid << " cycle=" << pmb->pmy_mesh->ncycle
              << " time=" << pmb->pmy_mesh->time << " dt=" << dt
              << " implicit_q=" << implicit_mixed_frame_drift
              << " lambda_gas_volume=" << lambda_gas_volume
              << " lambda_abs_volume=" << lambda_abs_volume
              << " q_gas_volume=" << drift_gas_volume
              << " q_abs_volume=" << drift_abs_volume
              << " gas_delta_volume=" << gas_delta_volume
              << " radiation_delta_volume=" << radiation_delta_volume
              << " radiation_floor_volume=" << radiation_floor_volume
              << " pair_balance_volume=" << gas_delta_volume+radiation_delta_volume
              << std::endl;
    std::cout.flags(flags);
    std::cout.precision(precision);
  }
}
