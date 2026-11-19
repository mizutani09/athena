//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wenoz_berta.cpp
//! \brief Berta et al. (2024) point-value primitive WENO-Z reconstruction for HD.
//!
//! Formula reference: V. Berta, A. Mignone, M. Bugli, and G. Mattia,
//! J. Comput. Phys. 499 (2024) 112701.

// C++ headers
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "reconstruction.hpp"
#include "troubled_fallback.hpp"

namespace {
inline bool MaskTroubledOrUnavailable(const Hydro *ph, int k, int j, int i) {
  const Hydro::TroubledCellMaskBounds &b = ph->GetTroubledCellMaskBounds();
  if (i<b.il || i>b.iu || j<b.jl || j>b.ju || k<b.kl || k>b.ku) return true;
  return ph->GetTroubledCellMask()(k,j,i) != 0;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \brief Point-value WENO-Z reconstruction, Berta et al. (2024), Eq. (24)-(26).

inline void Reconstruction::BertaWENOZ(
    const Real &q_im2, const Real &q_im1, const Real &q_i,
    const Real &q_ip1, const Real &q_ip2, Real &ql_ip1, Real &qr_i) noexcept {
  constexpr Real eps = 1.0e-40;

  const Real beta0 = (13.0/12.0)*SQR(q_im2 - 2.0*q_im1 + q_i)
                     + 0.25*SQR(q_im2 - 4.0*q_im1 + 3.0*q_i);
  const Real beta1 = (13.0/12.0)*SQR(q_im1 - 2.0*q_i + q_ip1)
                     + 0.25*SQR(q_im1 - q_ip1);
  const Real beta2 = (13.0/12.0)*SQR(q_i - 2.0*q_ip1 + q_ip2)
                     + 0.25*SQR(3.0*q_i - 4.0*q_ip1 + q_ip2);
  const Real tau5 = std::abs(beta0 - beta2);

  Real a0 = (1.0/16.0)*(1.0 + tau5/(beta0 + eps));
  Real a1 = (5.0/8.0)*(1.0 + tau5/(beta1 + eps));
  Real a2 = (5.0/16.0)*(1.0 + tau5/(beta2 + eps));
  Real asum = a0 + a1 + a2;

  Real p0 = (3.0*q_im2 - 10.0*q_im1 + 15.0*q_i)/8.0;
  Real p1 = (-q_im1 + 6.0*q_i + 3.0*q_ip1)/8.0;
  Real p2 = (3.0*q_i + 6.0*q_ip1 - q_ip2)/8.0;
  ql_ip1 = (a0*p0 + a1*p1 + a2*p2)/asum;

  a0 = (1.0/16.0)*(1.0 + tau5/(beta2 + eps));
  a1 = (5.0/8.0)*(1.0 + tau5/(beta1 + eps));
  a2 = (5.0/16.0)*(1.0 + tau5/(beta0 + eps));
  asum = a0 + a1 + a2;

  p0 = (3.0*q_ip2 - 10.0*q_ip1 + 15.0*q_i)/8.0;
  p1 = (-q_ip1 + 6.0*q_i + 3.0*q_im1)/8.0;
  p2 = (3.0*q_i + 6.0*q_im1 - q_im2)/8.0;
  qr_i = (a0*p0 + a1*p1 + a2*p2)/asum;
}

void Reconstruction::BertaWENOZX1(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      const bool right_troubled = MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i)
          || MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i+1);
      const bool left_troubled = MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i-1)
          || MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i);
      Real unused;
      if (right_troubled) {
        reconstruction::VanLeerPLM(w(n,k,j,i-1), w(n,k,j,i), w(n,k,j,i+1),
                                   wl(n,i+1), unused);
      } else {
        BertaWENOZ(w(n,k,j,i-2), w(n,k,j,i-1), w(n,k,j,i),
                   w(n,k,j,i+1), w(n,k,j,i+2), wl(n,i+1), unused);
      }
      if (left_troubled) {
        reconstruction::VanLeerPLM(w(n,k,j,i-1), w(n,k,j,i), w(n,k,j,i+1),
                                   unused, wr(n,i));
      } else {
        BertaWENOZ(w(n,k,j,i-2), w(n,k,j,i-1), w(n,k,j,i),
                   w(n,k,j,i+1), w(n,k,j,i+2), unused, wr(n,i));
      }
    }
  }
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    pmy_block_->peos->ApplyPrimitiveFloors(wl, k, j, i+1);
    pmy_block_->peos->ApplyPrimitiveFloors(wr, k, j, i);
  }
}

void Reconstruction::BertaWENOZX2(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      const bool right_troubled = MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i)
          || MaskTroubledOrUnavailable(pmy_block_->phydro,k,j+1,i);
      const bool left_troubled = MaskTroubledOrUnavailable(pmy_block_->phydro,k,j-1,i)
          || MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i);
      Real unused;
      if (right_troubled) {
        reconstruction::VanLeerPLM(w(n,k,j-1,i), w(n,k,j,i), w(n,k,j+1,i),
                                   wl(n,i), unused);
      } else {
        BertaWENOZ(w(n,k,j-2,i), w(n,k,j-1,i), w(n,k,j,i),
                   w(n,k,j+1,i), w(n,k,j+2,i), wl(n,i), unused);
      }
      if (left_troubled) {
        reconstruction::VanLeerPLM(w(n,k,j-1,i), w(n,k,j,i), w(n,k,j+1,i),
                                   unused, wr(n,i));
      } else {
        BertaWENOZ(w(n,k,j-2,i), w(n,k,j-1,i), w(n,k,j,i),
                   w(n,k,j+1,i), w(n,k,j+2,i), unused, wr(n,i));
      }
    }
  }
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    pmy_block_->peos->ApplyPrimitiveFloors(wl, k, j, i);
    pmy_block_->peos->ApplyPrimitiveFloors(wr, k, j, i);
  }
}

void Reconstruction::BertaWENOZX3(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  for (int n=0; n<NHYDRO; ++n) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      const bool right_troubled = MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i)
          || MaskTroubledOrUnavailable(pmy_block_->phydro,k+1,j,i);
      const bool left_troubled = MaskTroubledOrUnavailable(pmy_block_->phydro,k-1,j,i)
          || MaskTroubledOrUnavailable(pmy_block_->phydro,k,j,i);
      Real unused;
      if (right_troubled) {
        reconstruction::VanLeerPLM(w(n,k-1,j,i), w(n,k,j,i), w(n,k+1,j,i),
                                   wl(n,i), unused);
      } else {
        BertaWENOZ(w(n,k-2,j,i), w(n,k-1,j,i), w(n,k,j,i),
                   w(n,k+1,j,i), w(n,k+2,j,i), wl(n,i), unused);
      }
      if (left_troubled) {
        reconstruction::VanLeerPLM(w(n,k-1,j,i), w(n,k,j,i), w(n,k+1,j,i),
                                   unused, wr(n,i));
      } else {
        BertaWENOZ(w(n,k-2,j,i), w(n,k-1,j,i), w(n,k,j,i),
                   w(n,k+1,j,i), w(n,k+2,j,i), unused, wr(n,i));
      }
    }
  }
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    pmy_block_->peos->ApplyPrimitiveFloors(wl, k, j, i);
    pmy_block_->peos->ApplyPrimitiveFloors(wr, k, j, i);
  }
}
