#include <cassert>
#include <cmath>
#include <limits>

#include "../../src/pgen/solar_top_face_opacity.hpp"

namespace {

bool NearlyEqual(const double lhs, const double rhs, const double tolerance) {
  return std::abs(lhs - rhs) <= tolerance*std::max(std::abs(rhs), 1.0);
}

}  // namespace

int main() {
  using solar_top_face_opacity::ArithmeticFaceGhost;
  using solar_top_face_opacity::HalfCellAverage;

  assert(HalfCellAverage(4.0, 4.0) == 4.0);
  assert(ArithmeticFaceGhost(4.0, 4.0, 1.0e-30) == 4.0);

  const double outer = 8.0;
  const double inward = 2.0;
  const double slope = std::log(outer/inward);
  const double analytical = 2.0*outer*(std::exp(0.5*slope) - 1.0)/slope;
  assert(NearlyEqual(HalfCellAverage(outer, inward), analytical, 1.0e-14));
  const double ghost = ArithmeticFaceGhost(outer, inward, 1.0e-30);
  assert(NearlyEqual(0.5*(outer + ghost), analytical, 1.0e-14));

  const double near_constant = HalfCellAverage(1.0 + 1.0e-14, 1.0);
  assert(std::isfinite(near_constant));
  assert(NearlyEqual(near_constant, 1.0 + 1.25e-14, 1.0e-14));

  assert(ArithmeticFaceGhost(1.0e-20, 1.0e20, 1.0e-12) == 1.0e-12);
  assert(ArithmeticFaceGhost(-1.0, 1.0, 1.0e-12) == 1.0e-12);
  assert(ArithmeticFaceGhost(std::numeric_limits<double>::infinity(), 1.0,
                             1.0e-12) == 1.0e-12);
  return 0;
}
