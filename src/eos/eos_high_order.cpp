//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos_high_order.cpp
//! \brief functions for variable conversion at greater than second-order spatial accuracy

// C headers

// C++ headers
#include <algorithm>
#include <cmath>   // sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "eos.hpp"

//---------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitiveCellAverage(AthenaArray<Real> &cons,
//!           const AthenaArray<Real> &prim_old, const FaceField &b,
//!           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts cell-averaged conserved variables to cell-averaged primitive variables
//! at fourth order accuracy. Wrapper function for specific pointwise conversion routine

void EquationOfState::ConservedToPrimitiveCellAverage(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  MeshBlock *pmb = pmy_block_;
  Hydro *ph = pmb->phydro;

  int nl = 0;
  int nu = NHYDRO - 1;
  // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this should factor out
  // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-centered
  Real h = pco->dx1f(il);  // pco->dx1f(i); inside loop
  Real C = (h*h)/24.0;

  // Fourth-order accurate approx to cell-centered conserved and primitive variables
  AthenaArray<Real> &u_cc = ph->u_cc, &w_cc = ph->w_cc;
  // Laplacians of cell-averaged conserved and 2nd order accurate primitive variables
  AthenaArray<Real> &laplacian_cc = ph->scr1_nkji_;

  // Compute and store Laplacian of cell-averaged conserved variables
  pco->Laplacian(cons, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);

  // Compute fourth-order approximation to cell-centered conserved variables
  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // We do not actually need to store all cell-centered conserved variables,
          // but the ConservedToPrimitive() implementation operates on 4D arrays
          u_cc(n,k,j,i) = cons(n,k,j,i) - C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  // Compute Laplacian of 2nd-order approximation to cell-averaged primitive variables
  pco->Laplacian(prim, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);

  // Convert cell-centered conserved values to cell-centered primitive values
  ConservedToPrimitive(u_cc, prim_old, b, w_cc, bcc, pco, il, iu,
                       jl, ju, kl, ku);

  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // De-averaging can produce a near-vacuum point value across an
          // unresolved density cliff.  Dividing the reconstructed momentum
          // by that value creates an enormous point velocity, even though the
          // cell-averaged primitive state is well behaved.  Retain the
          // second-order cell average in that cell; smooth cells keep the
          // fourth-order conversion.
          if (!(u_cc(IDN,k,j,i) >= 0.5*cons(IDN,k,j,i))) continue;
          // Compute fourth-order approximation to cell-averaged primitive variables
          prim(n,k,j,i) = w_cc(n,k,j,i) + C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  // Reapply primitive variable floors
  // Cannot fuse w/ above loop since floors are applied to all NHYDRO variables at once
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        ApplyPrimitiveConservedFloors(prim, cons, bcc, k, j, i);
      }
    }
  }

  return;
}

//---------------------------------------------------------------------------------------
//! \brief Recover cell-center point primitive values from evolved volume averages.
//!
//! This is the HD subset of Berta et al. (2024), Eq. (16) and Eq. (33)-(35). Density and
//! pressure curvature select smooth cells. A point-state quality gate then catches EOS
//! or density-cliff failures before the mask is dilated to direct neighbours.

void EquationOfState::ConservedToPrimitivePointValue(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  MeshBlock *pmb = pmy_block_;
  Hydro *ph = pmb->phydro;
  const int nl = 0;
  const int nu = NHYDRO - 1;
  const Real h = pco->dx1f(il);
  const Real C = (h*h)/24.0;

  AthenaArray<Real> &u_cc = ph->u_cc;
  AthenaArray<Real> &w_cc = ph->w_cc;
  AthenaArray<Real> &laplacian_cc = ph->scr1_nkji_;
  AthenaArray<Real> &theta = ph->scr2_nkji_;
  AthenaArray<int> &mask = ph->troubled_cell_mask_;

  // Publish exactly the point-recovery domain. Same-level/periodic ghost cells are
  // included by the caller, while physical-boundary ghosts that were not de-averaged
  // remain unavailable and therefore use the fail-closed face fallback.
  const int mask_il = il;
  const int mask_iu = iu;
  const int mask_jl = jl;
  const int mask_ju = ju;
  const int mask_kl = kl;
  const int mask_ku = ku;
  for (int k=mask_kl; k<=mask_ku; ++k) {
    for (int j=mask_jl; j<=mask_ju; ++j) {
      for (int i=mask_il; i<=mask_iu; ++i) mask(k,j,i) = 1;
    }
  }

  constexpr Real sensor_threshold = 1.0e-2;

  for (int k=mask_kl; k<=mask_ku; ++k) {
    for (int j=mask_jl; j<=mask_ju; ++j) {
      for (int i=mask_il; i<=mask_iu; ++i) {
        Real eta_density_sq = 0.0;
        Real eta_pressure_sq = 0.0;
        const auto accumulate = [](Real qm, Real q, Real qp, Real floor,
                                   Real &eta_sq) {
          if (!std::isfinite(qm) || !std::isfinite(q) || !std::isfinite(qp)) {
            eta_sq = std::numeric_limits<Real>::infinity();
            return;
          }
          const Real denominator = std::max(
              std::abs(qp) + 2.0*std::abs(q) + std::abs(qm),
              4.0*std::max(floor, std::numeric_limits<Real>::min()));
          eta_sq += SQR(std::abs(qp - 2.0*q + qm)/denominator);
        };
        accumulate(cons(IDN,k,j,i-1), cons(IDN,k,j,i), cons(IDN,k,j,i+1),
                   density_floor_, eta_density_sq);
#if NON_BAROTROPIC_EOS
        accumulate(prim(IPR,k,j,i-1), prim(IPR,k,j,i), prim(IPR,k,j,i+1),
                   pressure_floor_, eta_pressure_sq);
#endif
        if (pmb->block_size.nx2 > 1) {
          accumulate(cons(IDN,k,j-1,i), cons(IDN,k,j,i), cons(IDN,k,j+1,i),
                     density_floor_, eta_density_sq);
#if NON_BAROTROPIC_EOS
          accumulate(prim(IPR,k,j-1,i), prim(IPR,k,j,i), prim(IPR,k,j+1,i),
                     pressure_floor_, eta_pressure_sq);
#endif
        }
        if (pmb->block_size.nx3 > 1) {
          accumulate(cons(IDN,k-1,j,i), cons(IDN,k,j,i), cons(IDN,k+1,j,i),
                     density_floor_, eta_density_sq);
#if NON_BAROTROPIC_EOS
          accumulate(prim(IPR,k-1,j,i), prim(IPR,k,j,i), prim(IPR,k+1,j,i),
                     pressure_floor_, eta_pressure_sq);
#endif
        }
        const Real eta = std::sqrt(std::max(eta_density_sq, eta_pressure_sq));
        mask(k,j,i) = (eta < sensor_threshold) ? 0 : 1;
      }
    }
  }

  pco->Laplacian(cons, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);

  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          const Real smooth = static_cast<Real>(1-mask(k,j,i));
          u_cc(n,k,j,i) = cons(n,k,j,i) - smooth*C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  ConservedToPrimitive(u_cc, prim_old, b, w_cc, bcc, pco,
                       il, iu, jl, ju, kl, ku);

  // Reject point recovery that crosses an unresolved density or pressure cliff. Keep
  // the already-computed second-order primitive state in those cells.
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        const Real rho0 = prim(IDN,k,j,i);
        const Real rho = w_cc(IDN,k,j,i);
        bool valid = std::isfinite(rho) && rho >= 0.5*rho0 && rho <= 2.0*rho0;
#if NON_BAROTROPIC_EOS
        const Real p0 = prim(IPR,k,j,i);
        const Real p = w_cc(IPR,k,j,i);
        valid = valid && std::isfinite(p) && p >= 0.5*p0 && p <= 2.0*p0;
#endif
        if (!valid) mask(k,j,i) = 1;
      }
    }
  }
  // Berta et al. Sec. 3.4: also reduce direct neighbours of a troubled cell. Do not
  // manufacture trouble beyond this safe sensor domain: its outer ghost shell is still
  // part of a boundary-adjacent five-point WENO stencil on a uniform periodic mesh.
  const auto is_troubled = [&mask, mask_il, mask_iu, mask_jl, mask_ju,
                             mask_kl, mask_ku](int k, int j, int i) {
    if (i<mask_il || i>mask_iu || j<mask_jl || j>mask_ju
        || k<mask_kl || k>mask_ku) return false;
    return mask(k,j,i) != 0;
  };
  for (int k=mask_kl; k<=mask_ku; ++k) {
    for (int j=mask_jl; j<=mask_ju; ++j) {
      for (int i=mask_il; i<=mask_iu; ++i) {
        bool troubled = is_troubled(k,j,i) || is_troubled(k,j,i-1)
                      || is_troubled(k,j,i+1);
        if (pmb->block_size.nx2 > 1) {
          troubled = troubled || is_troubled(k,j-1,i) || is_troubled(k,j+1,i);
        }
        if (pmb->block_size.nx3 > 1) {
          troubled = troubled || is_troubled(k-1,j,i) || is_troubled(k+1,j,i);
        }
        theta(1,k,j,i) = troubled ? 1.0 : 0.0;
      }
    }
  }

  for (int k=mask_kl; k<=mask_ku; ++k) {
    for (int j=mask_jl; j<=mask_ju; ++j) {
      for (int i=mask_il; i<=mask_iu; ++i) {
        mask(k,j,i) = (theta(1,k,j,i) != 0.0) ? 1 : 0;
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        if (mask(k,j,i) == 0) {
          for (int n=nl; n<=nu; ++n) prim(n,k,j,i) = w_cc(n,k,j,i);
        }
      }
    }
  }
  ph->troubled_cell_mask_bounds_ = {mask_il, mask_iu, mask_jl, mask_ju,
                                    mask_kl, mask_ku};
}
