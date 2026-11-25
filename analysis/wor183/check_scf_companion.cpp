#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/pgen/scf_companion.hpp"

namespace {

bool Close(double actual, double expected, double tolerance) {
  const double scale = std::max(1.0, std::abs(expected));
  if (std::abs(actual-expected) <= tolerance*scale) return true;
  std::cerr << "actual=" << actual << " expected=" << expected
            << " tolerance=" << tolerance*scale << '\n';
  return false;
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && Close(scf_companion::Ramp(1.0, 1.0, 2.0), 0.0, 1.0e-14);
  ok = ok && Close(scf_companion::Ramp(2.0, 1.0, 2.0), 0.5, 1.0e-14);
  ok = ok && Close(scf_companion::Ramp(3.0, 1.0, 2.0), 1.0, 1.0e-14);

  double cx = 0.0, cy = 0.0, cz = 0.0;
  scf_companion::CircularPosition(3.0, 1.0, 10.0, 0.25*M_PI,
                                  0.0, &cx, &cy, &cz);
  ok = ok && Close(cx, 0.0, 1.0e-14);
  ok = ok && Close(cy, 10.0, 1.0e-14);
  ok = ok && Close(cz, 0.0, 1.0e-14);

  const double gm = 3.0;
  const double support = 1.0;
  scf_companion::CircularPosition(0.0, 0.0, 10.0, 0.0,
                                  0.0, &cx, &cy, &cz);
  double ax = 0.0, ay = 0.0, az = 0.0;
  scf_companion::Acceleration(0.0, 0.0, 0.0, cx, cy, cz, gm, support,
                              true, &ax, &ay, &az);
  ok = ok && Close(ax, 0.0, 1.0e-14);
  ok = ok && Close(ay, 0.0, 1.0e-14);
  ok = ok && Close(az, 0.0, 1.0e-14);

  const double x = 2.0;
  const double y = -1.5;
  const double z = 0.75;
  scf_companion::Acceleration(x, y, z, cx, cy, cz, gm, support,
                              true, &ax, &ay, &az);
  const double epsilon = 1.0e-5;
  const auto potential = [=](double px, double py, double pz) {
    return scf_companion::Potential(px, py, pz, cx, cy, cz, gm, support,
                                    true);
  };
  const double ax_fd = -(potential(x+epsilon,y,z)-potential(x-epsilon,y,z))
      /(2.0*epsilon);
  const double ay_fd = -(potential(x,y+epsilon,z)-potential(x,y-epsilon,z))
      /(2.0*epsilon);
  const double az_fd = -(potential(x,y,z+epsilon)-potential(x,y,z-epsilon))
      /(2.0*epsilon);
  ok = ok && Close(ax, ax_fd, 2.0e-9);
  ok = ok && Close(ay, ay_fd, 2.0e-9);
  ok = ok && Close(az, az_fd, 2.0e-9);

  if (!ok) return EXIT_FAILURE;
  std::cout << "SCF companion forcing checks passed\n";
  return EXIT_SUCCESS;
}
