//========================================================================================
//! \file calculate_radiation_face_states.cpp
//! \brief Reconstruct the scalar FLD closure used by HLLC on hydro faces.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "fld.hpp"

namespace {
void ReconstructX1(Reconstruction *precon, int order, int k, int j, int il, int iu,
                   const AthenaArray<Real> &q, AthenaArray<Real> &ql,
                   AthenaArray<Real> &qr) {
  if (order == 1) precon->DonorCellX1(k, j, il, iu, q, ql, qr);
  else if (order == 2) precon->PiecewiseLinearX1(k, j, il, iu, q, ql, qr);
  else precon->PiecewiseParabolicX1(k, j, il, iu, q, ql, qr);
}

void ReconstructX2(Reconstruction *precon, int order, int k, int j, int il, int iu,
                   const AthenaArray<Real> &q, AthenaArray<Real> &ql,
                   AthenaArray<Real> &qr) {
  if (order == 1) precon->DonorCellX2(k, j, il, iu, q, ql, qr);
  else if (order == 2) precon->PiecewiseLinearX2(k, j, il, iu, q, ql, qr);
  else precon->PiecewiseParabolicX2(k, j, il, iu, q, ql, qr);
}

void ReconstructX3(Reconstruction *precon, int order, int k, int j, int il, int iu,
                   const AthenaArray<Real> &q, AthenaArray<Real> &ql,
                   AthenaArray<Real> &qr) {
  if (order == 1) precon->DonorCellX3(k, j, il, iu, q, ql, qr);
  else if (order == 2) precon->PiecewiseLinearX3(k, j, il, iu, q, ql, qr);
  else precon->PiecewiseParabolicX3(k, j, il, iu, q, ql, qr);
}

RadFLD::ClosureValues RequireClosure(const Real gradient, const Real opacity,
                                     const Real erad, const bool fixed,
                                     const char *context, const int k,
                                     const int j, const int i) {
  const RadFLD::ClosureValues closure =
      RadFLD::EvaluateClosure(gradient, opacity, erad, fixed);
  if (!closure.valid) {
    std::stringstream msg;
    msg << "### FATAL ERROR in FLD closure at " << context
        << " cell (k,j,i)=(" << k << "," << j << "," << i << ")"
        << ": gradient=" << gradient << " opacity=" << opacity
        << " Er=" << erad
        << ". Opacity and Er must be finite and non-negative;"
        << " fixed limiter with zero opacity and nonzero gradient is unsupported.\n";
    ATHENA_ERROR(msg);
  }
  return closure;
}
}  // namespace

void FLD::CalculateRadiationFaceStates(const int order) {
  MeshBlock *pmb = pmy_block;
  // Reconstruction is requested over [is-1,ie+1].  Donor-cell, PLM, and PPM
  // therefore require one, two, and three closure-state ghost layers,
  // respectively.  In particular PLM with the default NGHOST=2 must include
  // the outermost ghost cell at a MeshBlock boundary.
  const int stencil = (order <= 1) ? 1 : ((order == 2) ? 2 : 3);
  const int il = pmb->is - stencil, iu = pmb->ie + stencil;
  const int jl = pmb->pmy_mesh->f2 ? pmb->js-stencil : pmb->js;
  const int ju = pmb->pmy_mesh->f2 ? pmb->je+stencil : pmb->je;
  const int kl = pmb->pmy_mesh->f3 ? pmb->ks-stencil : pmb->ks;
  const int ku = pmb->pmy_mesh->f3 ? pmb->ke+stencil : pmb->ke;

  // Check the complete reconstruction stencil before taking differences.
  // Otherwise a negative/non-finite neighbor could be hidden by a finite
  // face average and only appear much later in the Riemann solve.
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        if (!std::isfinite(u_rad(k,j,i)) || u_rad(k,j,i) < 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in FLD radiation energy" << std::endl
              << "Er at (k,j,i)=(" << k << "," << j << "," << i << ") is "
              << u_rad(k,j,i)
              << "; radiation energy must be finite and non-negative.\n";
          ATHENA_ERROR(msg);
        }
      }
    }
  }

  // Reconstruct a compact, internally consistent state.  In particular total
  // pressure is not reconstructed independently: HLLC forms p_gas+lambda*E
  // from the gas and radiation face states.
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const int im = std::max(i-1, 0);
        const int ip = std::min(i+1, pmb->ncells1-1);
        const int jm = std::max(j-1, 0);
        const int jp = std::min(j+1, pmb->ncells2-1);
        const int km = std::max(k-1, 0);
        const int kp = std::min(k+1, pmb->ncells3-1);
        const Real idx = ((ip-im == 2) ? 0.5 : 1.0) / pmb->pcoord->dx1f(i);
        const Real idy = ((jp-jm == 2) ? 0.5 : 1.0) / pmb->pcoord->dx2f(j);
        const Real idz = ((kp-km == 2) ? 0.5 : 1.0) / pmb->pcoord->dx3f(k);
        const Real gx = idx * (u_rad(k, j, ip) - u_rad(k, j, im));
        const Real gy = idy * (u_rad(k, jp, i) - u_rad(k, jm, i));
        const Real gz = idz * (u_rad(kp, j, i) - u_rad(km, j, i));
        const Real grad = std::hypot(gx, std::hypot(gy, gz));
        const Real erad = u_rad(k, j, i);
        const Real sigma = sigma_r(k, j, i);
        const RadFLD::ClosureValues closure = RequireClosure(
            grad, sigma, erad, fixed_flux_limiter, "CalculateRadiationFaceStates",
            k, j, i);
        const Real lambda = closure.lambda;
        const Real chi = closure.chi;
        const Real ar = 0.5*(3.0-chi);

        rad_state_cc_(RadFLD::ERAD, k, j, i) = erad;
        rad_state_cc_(RadFLD::LAMBDA, k, j, i) = lambda;
        rad_state_cc_(RadFLD::ARAD, k, j, i) = ar;
      }
    }
  }

  // x1 faces
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      ReconstructX1(pmb->precon, order, k, j, pmb->is - 1, pmb->ie + 1,
                    rad_state_cc_,
                    rad_statel_, rad_stater_);
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
          rad_face_l[X1DIR](n, k, j, i) = rad_statel_(n, i);
          rad_face_r[X1DIR](n, k, j, i) = rad_stater_(n, i);
        }
      }
    }
  }

  // x2 faces
  if (pmb->pmy_mesh->f2) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      ReconstructX2(pmb->precon, order, k, pmb->js - 1,
                    pmb->is - 1, pmb->ie + 1,
                    rad_state_cc_, rad_statel_, rad_stater_);
      for (int j = pmb->js; j <= pmb->je + 1; ++j) {
        ReconstructX2(pmb->precon, order, k, j, pmb->is - 1, pmb->ie + 1,
                      rad_state_cc_, rad_statelb_, rad_stater_);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
            rad_face_l[X2DIR](n, k, j, i) = rad_statel_(n, i);
            rad_face_r[X2DIR](n, k, j, i) = rad_stater_(n, i);
          }
        }
        rad_statel_.SwapAthenaArray(rad_statelb_);
      }
    }
  }

  // x3 faces
  if (pmb->pmy_mesh->f3) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      ReconstructX3(pmb->precon, order, pmb->ks - 1, j,
                    pmb->is - 1, pmb->ie + 1,
                    rad_state_cc_, rad_statel_, rad_stater_);
      for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
        ReconstructX3(pmb->precon, order, k, j, pmb->is - 1, pmb->ie + 1,
                      rad_state_cc_, rad_statelb_, rad_stater_);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
            rad_face_l[X3DIR](n, k, j, i) = rad_statel_(n, i);
            rad_face_r[X3DIR](n, k, j, i) = rad_stater_(n, i);
          }
        }
        rad_statel_.SwapAthenaArray(rad_statelb_);
      }
    }
  }
}

Real FLD::RadiationSoundSpeedSquared(int k, int j, int i, int dir) const {
  MeshBlock *pmb = pmy_block;
  const Real gx = 0.5*(u_rad(k,j,i+1)-u_rad(k,j,i-1))/pmb->pcoord->dx1f(i);
  const Real gy = 0.5*(u_rad(k,j+1,i)-u_rad(k,j-1,i))/pmb->pcoord->dx2f(j);
  const Real gz = 0.5*(u_rad(k+1,j,i)-u_rad(k-1,j,i))/pmb->pcoord->dx3f(k);
  const Real grad = std::hypot(gx, std::hypot(gy, gz));
  const Real erad = u_rad(k,j,i);
  const RadFLD::ClosureValues closure = RequireClosure(
      grad, sigma_r(k,j,i), erad, fixed_flux_limiter,
      "RadiationSoundSpeedSquared", k, j, i);
  const Real lambda = closure.lambda;
  (void)dir;
  const Real p = std::max(lambda*erad, 0.0);
  const Real rho = std::max(pmb->phydro->w(IDN,k,j,i), TINY_NUMBER);
  return std::max(lambda, 0.0)*(erad+p)/rho;
}
