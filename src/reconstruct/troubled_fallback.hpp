#ifndef RECONSTRUCT_TROUBLED_FALLBACK_HPP_
#define RECONSTRUCT_TROUBLED_FALLBACK_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Copyright(C) 2026 Shinsuke Takasao
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <algorithm>
#include <cmath>

#include "../athena.hpp"

namespace reconstruction {
//! Uniform-Cartesian van Leer PLM used by the reference 4wz troubled-face path.
inline void VanLeerPLM(Real qm, Real q, Real qp, Real &right, Real &left) {
  const Real dl = q-qm;
  const Real dr = qp-q;
  Real slope = 0.0;
  if ((dl > 0.0 && dr > 0.0) || (dl < 0.0 && dr < 0.0)) {
    const Real small = std::min(std::abs(dl), std::abs(dr));
    const Real large = std::max(std::abs(dl), std::abs(dr));
    slope = std::copysign(small*(2.0/(1.0 + small/large)), dl);
  }
  right = q + 0.5*slope;
  left = q - 0.5*slope;
}
}  // namespace reconstruction

#endif  // RECONSTRUCT_TROUBLED_FALLBACK_HPP_
