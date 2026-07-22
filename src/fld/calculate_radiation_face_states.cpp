//========================================================================================
//! \file calculate_radiation_face_states.cpp
//! \brief Reconstruct FLD radiation energy and pressure tensor on hydro faces.

#include <algorithm>
#include <cmath>

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
}  // namespace

void FLD2::CalculateRadiationFaceStates(const int order) {
  MeshBlock *pmb = pmy_block;
  const int il = pmb->is - 1, iu = pmb->ie + 1;
  const int jl = pmb->js - 1, ju = pmb->je + 1;
  const int kl = pmb->ks - 1, ku = pmb->ke + 1;

  // Use the same Levermore-Pomraning limiter and Eddington tensor as NRFLD:
  //   lambda=(2+R)/(6+3R+R^2), chi=lambda+(lambda R)^2,
  //   P_ij/E = (1-chi)delta_ij/2 + (3chi-1)n_i n_j/2.
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const Real idx = 0.5 / pmb->pcoord->dx1f(i);
        const Real idy = 0.5 / pmb->pcoord->dx2f(j);
        const Real idz = 0.5 / pmb->pcoord->dx3f(k);
        const Real gx = idx * (u_rad(k, j, i + 1) - u_rad(k, j, i - 1));
        const Real gy = idy * (u_rad(k, j + 1, i) - u_rad(k, j - 1, i));
        const Real gz = idz * (u_rad(k + 1, j, i) - u_rad(k - 1, j, i));
        const Real grad = std::sqrt(SQR(gx) + SQR(gy) + SQR(gz));
        const Real erad = std::max(u_rad(k, j, i), TINY_NUMBER);
        const Real sigma = std::max(sigma_r(k, j, i), TINY_NUMBER);
        const Real r = grad / (sigma * erad);
        const Real lambda = fixed_flux_limitter
                                ? ONE_3RD
                                : (2.0 + r) / (6.0 + 3.0*r + r*r);
        const Real chi = fixed_flux_limitter
                             ? ONE_3RD : lambda + SQR(lambda*r);
        const Real inv_grad = 1.0 / std::max(grad, TINY_NUMBER);
        const Real n[3] = {gx*inv_grad, gy*inv_grad, gz*inv_grad};
        const Real diag = 0.5 * (1.0 - chi);
        const Real beam = 0.5 * (3.0*chi - 1.0);

        rad_state_cc_(RadFLD2::ERAD, k, j, i) = erad;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            const int nvar = 1 + 3*a + b;
            const Real eddington = (a == b ? diag : 0.0) + beam*n[a]*n[b];
            rad_state_cc_(nvar, k, j, i) = erad*eddington;
          }
          // Reconstruct total normal pressure itself. Reconstructing gas and
          // radiation pressure independently does not preserve a cell-wise
          // constant total pressure when Hydro uses characteristic variables.
          rad_state_cc_(RadFLD2::PTOT1 + a, k, j, i) =
              pmb->phydro->w(IPR, k, j, i)
              + rad_state_cc_(1 + 3*a + a, k, j, i);
        }
      }
    }
  }

  // x1 faces
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      ReconstructX1(pmb->precon, order, k, j, il, iu, rad_state_cc_,
                    rad_statel_, rad_stater_);
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        for (int n = 0; n < RadFLD2::NRAD_FACE_STATE; ++n) {
          rad_face_l[X1DIR](n, k, j, i) = rad_statel_(n, i);
          rad_face_r[X1DIR](n, k, j, i) = rad_stater_(n, i);
        }
      }
    }
  }

  // x2 faces
  if (pmb->pmy_mesh->f2) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      ReconstructX2(pmb->precon, order, k, pmb->js - 1, il, iu,
                    rad_state_cc_, rad_statel_, rad_stater_);
      for (int j = pmb->js; j <= pmb->je + 1; ++j) {
        ReconstructX2(pmb->precon, order, k, j, il, iu,
                      rad_state_cc_, rad_statelb_, rad_stater_);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          for (int n = 0; n < RadFLD2::NRAD_FACE_STATE; ++n) {
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
      ReconstructX3(pmb->precon, order, pmb->ks - 1, j, il, iu,
                    rad_state_cc_, rad_statel_, rad_stater_);
      for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
        ReconstructX3(pmb->precon, order, k, j, il, iu,
                      rad_state_cc_, rad_statelb_, rad_stater_);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          for (int n = 0; n < RadFLD2::NRAD_FACE_STATE; ++n) {
            rad_face_l[X3DIR](n, k, j, i) = rad_statel_(n, i);
            rad_face_r[X3DIR](n, k, j, i) = rad_stater_(n, i);
          }
        }
        rad_statel_.SwapAthenaArray(rad_statelb_);
      }
    }
  }
}

Real FLD2::RadiationSoundSpeedSquared(int k, int j, int i, int dir) const {
  MeshBlock *pmb = pmy_block;
  const Real gx = 0.5*(u_rad(k,j,i+1)-u_rad(k,j,i-1))/pmb->pcoord->dx1f(i);
  const Real gy = 0.5*(u_rad(k,j+1,i)-u_rad(k,j-1,i))/pmb->pcoord->dx2f(j);
  const Real gz = 0.5*(u_rad(k+1,j,i)-u_rad(k-1,j,i))/pmb->pcoord->dx3f(k);
  const Real grad = std::sqrt(SQR(gx)+SQR(gy)+SQR(gz));
  const Real erad = std::max(u_rad(k,j,i), TINY_NUMBER);
  const Real r = grad/(std::max(sigma_r(k,j,i), TINY_NUMBER)*erad);
  const Real lambda = fixed_flux_limitter
                          ? ONE_3RD : (2.0+r)/(6.0+3.0*r+r*r);
  const Real chi = fixed_flux_limitter ? ONE_3RD : lambda+SQR(lambda*r);
  const Real g[3] = {gx, gy, gz};
  const Real nd = g[dir]/std::max(grad, TINY_NUMBER);
  const Real f = 0.5*(1.0-chi) + 0.5*(3.0*chi-1.0)*SQR(nd);
  const Real pnn = std::max(f*erad, 0.0);
  const Real rho = std::max(pmb->phydro->w(IDN,k,j,i), TINY_NUMBER);
  return std::max(f, 0.0)*(erad+pnn)/rho;
}
