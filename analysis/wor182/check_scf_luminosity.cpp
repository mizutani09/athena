#include <algorithm>
#include <cmath>
#include <iostream>

#include "../../src/pgen/scf_luminosity.hpp"

int main() {
  constexpr double radius = 2.75;
  constexpr int intervals = 1000000;
  const double dr = radius/intervals;
  double radial_integral = 0.0;
  for (int i = 0; i <= intervals; ++i) {
    const double r = i*dr;
    const double integrand = 4.0*std::acos(-1.0)*r*r
        *scf_luminosity::Weight(r, radius);
    const double coefficient = (i == 0 || i == intervals) ? 1.0
        : (i % 2 == 0 ? 2.0 : 4.0);
    radial_integral += coefficient*integrand;
  }
  radial_integral *= dr/3.0;
  const double exact = scf_luminosity::VolumeNormalization(radius);
  const double relative_error = std::abs(radial_integral-exact)/exact;
  if (scf_luminosity::Weight(radius, radius) != 0.0
      || scf_luminosity::Weight(2.0*radius, radius) != 0.0
      || relative_error > 1.0e-12) {
    std::cerr << "luminosity-kernel check failed: relative_error="
              << relative_error << '\n';
    return 1;
  }
  std::cout << "luminosity-kernel check passed: relative_error="
            << relative_error << '\n';
  return 0;
}
