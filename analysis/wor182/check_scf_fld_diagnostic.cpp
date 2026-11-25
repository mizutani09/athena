#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "../../src/pgen/scf_fld_diagnostic.hpp"

namespace {

bool Close(double actual, double expected, double tolerance = 1.0e-14) {
  const double scale = std::max(1.0, std::abs(expected));
  if (std::abs(actual-expected) <= tolerance*scale) return true;
  std::cerr << "actual=" << actual << " expected=" << expected << '\n';
  return false;
}

double Heating(double center, double xm, double xp, double ym, double yp,
               double zm, double zp, double diffusion = 1.0) {
  return scf_fld_diagnostic::DiffusionHeating(
      center, xm, xp, ym, yp, zm, zp,
      diffusion, diffusion, diffusion, diffusion, diffusion, diffusion,
      1.0, 1.0, 1.0);
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && Close(Heating(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0), 0.0);
  ok = ok && Close(Heating(0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0), 0.0);
  ok = ok && Close(Heating(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0), 4.0);

  // Two cells sharing the only nonzero face exchange equal and opposite
  // energy when their external faces have zero gradient.
  const double left = Heating(1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0);
  const double right = Heating(3.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0);
  ok = ok && Close(left, 2.0);
  ok = ok && Close(right, -2.0);
  ok = ok && Close(left+right, 0.0);

  const double epsilon = std::numeric_limits<double>::epsilon();
  ok = ok && Close(scf_fld_diagnostic::ResolvableThermalDifference(
      1.0, 1.0-epsilon), 0.0);
  ok = ok && Close(scf_fld_diagnostic::ResolvableThermalDifference(
      1.0, 1.0-64.0*epsilon), 64.0*epsilon);

  if (!ok) return EXIT_FAILURE;
  std::cout << "scf FLD diagnostic checks passed\n";
  return EXIT_SUCCESS;
}
