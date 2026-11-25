//========================================================================================
//! \file calculate_radiation_face_states.cpp
//! \brief Reconstruct the scalar FLD closure used by HLLC on hydro faces.

#include <algorithm>
#include <cmath>
#include <limits>
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

bool PhysicalRadiationFaceState(const AthenaArray<Real> &state, int i) {
  const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon();
  const Real er = state(RadFLD::ERAD, i);
  const Real lambda = state(RadFLD::LAMBDA, i);
  const Real ar = state(RadFLD::ARAD, i);
  return er > 0.0 && std::isfinite(er) && std::isfinite(lambda)
      && lambda >= -tolerance && lambda <= 1.0/3.0+tolerance
      && std::isfinite(ar) && ar >= 1.0-tolerance && ar <= 4.0/3.0+tolerance;
}

Real BoundedRadiationFaceValue(int n, Real value) {
  if (n == RadFLD::ERAD) return std::max(value, TINY_NUMBER);
  if (n == RadFLD::LAMBDA) return std::max(0.0, std::min(value, 1.0/3.0));
  return std::max(1.0, std::min(value, 4.0/3.0));
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

Real TransverseLaplacian(const MeshBlock *pmb, const AthenaArray<Real> &face,
                         int dir, int k, int j, int i) {
  const Coordinates *coord = pmb->pcoord;
  Real lap = 0.0;
  if (dir != X1DIR) {
    const Real h = coord->dx1f(i);
    lap += (face(k,j,i-1) - 2.0*face(k,j,i) + face(k,j,i+1))/(h*h);
  }
  if (dir != X2DIR && pmb->pmy_mesh->f2) {
    const Real h = coord->dx2f(j);
    lap += (face(k,j-1,i) - 2.0*face(k,j,i) + face(k,j+1,i))/(h*h);
  }
  if (dir != X3DIR && pmb->pmy_mesh->f3) {
    const Real h = coord->dx3f(k);
    lap += (face(k-1,j,i) - 2.0*face(k,j,i) + face(k+1,j,i))/(h*h);
  }
  return lap;
}
}  // namespace

void FLD::PrepareFourthOrderPointFaces(int dir, int k, int j, int il, int iu,
                                       Real c) {
  const AthenaArray<Real> &left = rad_face_l[dir];
  const AthenaArray<Real> &right = rad_face_r[dir];
  for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
    if (dir == X1DIR) {
      pmy_block->pcoord->LaplacianX1(left, rad_lap_line_, n, k, j, il, iu);
    } else if (dir == X2DIR) {
      pmy_block->pcoord->LaplacianX2(left, rad_lap_line_, n, k, j, il, iu);
    } else {
      pmy_block->pcoord->LaplacianX3(left, rad_lap_line_, n, k, j, il, iu);
    }
    for (int i = il; i <= iu; ++i)
      rad_point_l_(n,i) = left(n,k,j,i) - c*rad_lap_line_(i);

    if (dir == X1DIR) {
      pmy_block->pcoord->LaplacianX1(right, rad_lap_line_, n, k, j, il, iu);
    } else if (dir == X2DIR) {
      pmy_block->pcoord->LaplacianX2(right, rad_lap_line_, n, k, j, il, iu);
    } else {
      pmy_block->pcoord->LaplacianX3(right, rad_lap_line_, n, k, j, il, iu);
    }
    for (int i = il; i <= iu; ++i)
      rad_point_r_(n,i) = right(n,k,j,i) - c*rad_lap_line_(i);
  }

  // De-averaging is unrestricted only where it remains a physical closure.
  // At a steep surface revert the complete state on that side together.
  for (int i = il; i <= iu; ++i) {
    for (int side = 0; side < 2; ++side) {
      AthenaArray<Real> &point = side == 0 ? rad_point_l_ : rad_point_r_;
      const AthenaArray<Real> &average = side == 0 ? left : right;
      const Real er = point(RadFLD::ERAD,i);
      const Real lambda = point(RadFLD::LAMBDA,i);
      const Real ar = point(RadFLD::ARAD,i);
      if (!(er > 0.0) || !std::isfinite(er) || !std::isfinite(lambda)
          || lambda < 0.0 || lambda > 1.0/3.0 || !std::isfinite(ar)
          || ar < 1.0 || ar > 4.0/3.0) {
        for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n)
          point(n,i) = average(n,k,j,i);
      }
    }
  }
}

void FLD::PrepareFourthOrderFaceLaplacian(int dir, int kl, int ku, int jl,
                                          int ju, int il, int iu) {
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        rad_flux_lap_[dir](k,j,i) = TransverseLaplacian(
            pmy_block, u_rad_flux[dir], dir, k, j, i);
        rad_g_lap_[dir](k,j,i) = TransverseLaplacian(
            pmy_block, rad_face_g[dir], dir, k, j, i);
      }
    }
  }
}

void FLD::CorrectFourthOrderFaceRow(int dir, int k, int j, int il, int iu,
                                    Real c) {
  for (int i = il; i <= iu; ++i) {
    CorrectFourthOrderFace(dir, k, j, i, c);
  }
}

void FLD::CorrectFourthOrderFace(int dir, int k, int j, int i, Real c) {
  u_rad_flux[dir](k,j,i) += c*rad_flux_lap_[dir](k,j,i);
  rad_face_g[dir](k,j,i) = std::max(
      rad_face_g[dir](k,j,i) + c*rad_g_lap_[dir](k,j,i), TINY_NUMBER);
}

void FLD::ReduceFourthOrderPointFace(int dir, int k, int j, int i) {
  for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
    rad_point_l_(n,i) = rad_face_l[dir](n,k,j,i);
    rad_point_r_(n,i) = rad_face_r[dir](n,k,j,i);
  }
}

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
  // face average and only appear much later in the Riemann solve.  Finite
  // negative ghost values can be produced by multidimensional AMR
  // prolongation or high-order boundary interpolation even though all active
  // cells are admissible.  Bound those interpolation values at the point of
  // use; retain a strict failure for every active cell and for non-finite
  // ghosts.  If a same-level neighbour supplied a genuinely negative active
  // value, that value is independently checked on its owning MeshBlock.
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const bool active = i >= pmb->is && i <= pmb->ie
                         && j >= pmb->js && j <= pmb->je
                         && k >= pmb->ks && k <= pmb->ke;
        if (std::isfinite(u_rad(k,j,i)) && u_rad(k,j,i) < 0.0 && !active) {
          u_rad(k,j,i) = TINY_NUMBER;
        } else if (!std::isfinite(u_rad(k,j,i)) || u_rad(k,j,i) < 0.0) {
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

  // The fourth-order hydro flux needs one transverse guard face on either
  // side to form its face Laplacian.  Its first Riemann sweep consumes the
  // radiation states on those faces as well.
  const int transverse = (order == 4) ? 1 : 0;

  // x1 faces
  for (int k = pmb->ks - (pmb->pmy_mesh->f3 ? transverse : 0);
       k <= pmb->ke + (pmb->pmy_mesh->f3 ? transverse : 0); ++k) {
    for (int j = pmb->js - (pmb->pmy_mesh->f2 ? transverse : 0);
         j <= pmb->je + (pmb->pmy_mesh->f2 ? transverse : 0); ++j) {
      ReconstructX1(pmb->precon, order, k, j, pmb->is - 1, pmb->ie + 1,
                    rad_state_cc_,
                    rad_statel_, rad_stater_);
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        const bool left_physical = !pmb->precon->berta_wenoz
            || PhysicalRadiationFaceState(rad_statel_, i);
        const bool right_physical = !pmb->precon->berta_wenoz
            || PhysicalRadiationFaceState(rad_stater_, i);
        for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
          rad_face_l[X1DIR](n, k, j, i) = BoundedRadiationFaceValue(
              n, left_physical ? rad_statel_(n, i) : rad_state_cc_(n, k, j, i-1));
          rad_face_r[X1DIR](n, k, j, i) = BoundedRadiationFaceValue(
              n, right_physical ? rad_stater_(n, i) : rad_state_cc_(n, k, j, i));
        }
      }
    }
  }

  // x2 faces
  if (pmb->pmy_mesh->f2) {
    for (int k = pmb->ks - (pmb->pmy_mesh->f3 ? transverse : 0);
         k <= pmb->ke + (pmb->pmy_mesh->f3 ? transverse : 0); ++k) {
      ReconstructX2(pmb->precon, order, k, pmb->js - 1,
                    pmb->is - 1, pmb->ie + 1,
                    rad_state_cc_, rad_statel_, rad_stater_);
      for (int j = pmb->js; j <= pmb->je + 1; ++j) {
        ReconstructX2(pmb->precon, order, k, j, pmb->is - 1, pmb->ie + 1,
                      rad_state_cc_, rad_statelb_, rad_stater_);
        for (int i = pmb->is - transverse; i <= pmb->ie + transverse; ++i) {
          const bool left_physical = !pmb->precon->berta_wenoz
              || PhysicalRadiationFaceState(rad_statel_, i);
          const bool right_physical = !pmb->precon->berta_wenoz
              || PhysicalRadiationFaceState(rad_stater_, i);
          for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
            rad_face_l[X2DIR](n, k, j, i) = BoundedRadiationFaceValue(
                n, left_physical ? rad_statel_(n, i) : rad_state_cc_(n, k, j-1, i));
            rad_face_r[X2DIR](n, k, j, i) = BoundedRadiationFaceValue(
                n, right_physical ? rad_stater_(n, i) : rad_state_cc_(n, k, j, i));
          }
        }
        rad_statel_.SwapAthenaArray(rad_statelb_);
      }
    }
  }

  // x3 faces
  if (pmb->pmy_mesh->f3) {
    for (int j = pmb->js - (pmb->pmy_mesh->f2 ? transverse : 0);
         j <= pmb->je + (pmb->pmy_mesh->f2 ? transverse : 0); ++j) {
      ReconstructX3(pmb->precon, order, pmb->ks - 1, j,
                    pmb->is - 1, pmb->ie + 1,
                    rad_state_cc_, rad_statel_, rad_stater_);
      for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
        ReconstructX3(pmb->precon, order, k, j, pmb->is - 1, pmb->ie + 1,
                      rad_state_cc_, rad_statelb_, rad_stater_);
        for (int i = pmb->is - transverse; i <= pmb->ie + transverse; ++i) {
          const bool left_physical = !pmb->precon->berta_wenoz
              || PhysicalRadiationFaceState(rad_statel_, i);
          const bool right_physical = !pmb->precon->berta_wenoz
              || PhysicalRadiationFaceState(rad_stater_, i);
          for (int n = 0; n < RadFLD::NRAD_FACE_STATE; ++n) {
            rad_face_l[X3DIR](n, k, j, i) = BoundedRadiationFaceValue(
                n, left_physical ? rad_statel_(n, i) : rad_state_cc_(n, k-1, j, i));
            rad_face_r[X3DIR](n, k, j, i) = BoundedRadiationFaceValue(
                n, right_physical ? rad_stater_(n, i) : rad_state_cc_(n, k, j, i));
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
