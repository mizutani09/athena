#ifndef PGEN_SOLAR_TOP_FACE_OPACITY_HPP_
#define PGEN_SOLAR_TOP_FACE_OPACITY_HPP_

#include <algorithm>
#include <cmath>

namespace solar_top_face_opacity {

// Return the average opacity between the outermost cell centre and the
// physical face, assuming log(opacity) varies linearly between the two
// outermost cell centres.  expm1 keeps the constant-opacity limit accurate.
template <typename T>
T HalfCellAverage(const T outer, const T inward) {
  const T slope = std::log(outer/inward);
  if (slope == T(0)) return outer;
  return T(2)*outer*std::expm1(T(0.5)*slope)/slope;
}

// In the optically thin solar upper boundary, Athena++'s capped face-opacity
// rule reduces to the arithmetic average of the cell and ghost values.  Pick
// the ghost value that gives the analytical half-cell average above, while
// retaining a positive finite fallback.
template <typename T>
T ArithmeticFaceGhost(const T outer, const T inward, const T floor) {
  if (!(outer > T(0)) || !(inward > T(0)) || !std::isfinite(outer) ||
      !std::isfinite(inward)) {
    return floor;
  }
  const T face = HalfCellAverage(outer, inward);
  if (!std::isfinite(face)) return floor;
  return std::max(T(2)*face - outer, floor);
}

}  // namespace solar_top_face_opacity

#endif  // PGEN_SOLAR_TOP_FACE_OPACITY_HPP_
