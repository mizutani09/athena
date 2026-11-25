#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/pgen/scf_relaxation.hpp"

namespace {

bool Close(double actual, double expected, double tolerance = 1.0e-14) {
  const double scale = std::max(1.0, std::abs(expected));
  if (std::abs(actual-expected) <= tolerance*scale) return true;
  std::cerr << "actual=" << actual << " expected=" << expected << '\n';
  return false;
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && Close(scf_relaxation::TaperStrength(0.0, 2.0, 6.0), 1.0);
  ok = ok && Close(scf_relaxation::TaperStrength(4.0, 2.0, 6.0), 0.5);
  ok = ok && Close(scf_relaxation::TaperStrength(6.0, 2.0, 6.0), 0.0);
  ok = ok && Close(scf_relaxation::TaperStrength(20.0, -1.0, -1.0), 1.0);

  const auto unchanged = scf_relaxation::DampTowardCorotation(
      4.0, -2.0, 3.0, 2.0, -5.0, 0.25, 1.0);
  ok = ok && Close(unchanged.x, 4.0) && Close(unchanged.y, -2.0)
      && Close(unchanged.z, 3.0);

  const auto target = scf_relaxation::DampTowardCorotation(
      4.0, -2.0, 3.0, 2.0, -5.0, 0.25, 0.0);
  ok = ok && Close(target.x, 1.25) && Close(target.y, 0.5)
      && Close(target.z, 0.0);

  const auto half = scf_relaxation::DampTowardCorotation(
      4.0, -2.0, 3.0, 2.0, -5.0, 0.25, 0.5);
  ok = ok && Close(half.x-target.x, 0.5*(4.0-target.x));
  ok = ok && Close(half.y-target.y, 0.5*(-2.0-target.y));
  ok = ok && Close(half.z-target.z, 1.5);

  if (!ok) return EXIT_FAILURE;
  std::cout << "scf relaxation checks passed\n";
  return EXIT_SUCCESS;
}
