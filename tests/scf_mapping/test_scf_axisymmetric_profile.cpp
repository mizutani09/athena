#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../../src/pgen/scf_axisymmetric_profile.hpp"

namespace {
bool Close(double left, double right, double relative) {
  const double scale = std::max(std::abs(left), std::abs(right));
  return std::abs(left-right) <= relative*std::max(scale, 1.0e-300);
}
}

int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("usage: test_scf_profile PROFILE.csv");
  scf_map::AxisymmetricProfile profile;
  profile.Load(argv[1], 0.6353437771420501);
  assert(profile.radial_points() == 2048);
  assert(profile.angular_points() == 32);

  const double radius = 2.5e13;
  const scf_map::State north = profile.Interpolate(1.8e13, radius);
  const scf_map::State south = profile.Interpolate(1.8e13, -radius);
  assert(north.density > 0.0);
  assert(north.pressure > 0.0);
  assert(north.temperature > 0.0);
  assert(Close(north.density, south.density, 2.0e-14));
  assert(Close(north.pressure, south.pressure, 2.0e-14));

  const double omega = 4.0e-8;
  const double rho_atm = 1.0e-14;
  const double p_atm = 1.0e-2;
  const scf_map::Conserved point = scf_map::Sample(
      profile, 2.0e13, -1.0e13, 1.0e13, omega, 5.0/3.0,
      rho_atm, p_atm, false);
  assert(point.stellar_density > 0.0);
  assert(Close(point.momentum_x/point.density, 4.0e5, 2.0e-14));
  assert(Close(point.momentum_y/point.density, 8.0e5, 2.0e-14));
  assert(point.momentum_z == 0.0);
  assert(point.total_energy > point.pressure/(5.0/3.0-1.0));

  const scf_map::Conserved outside = scf_map::Sample(
      profile, 2.0*profile.maximum_radius(), 0.0, 0.0,
      omega, 5.0/3.0, rho_atm, p_atm, false);
  assert(outside.density == rho_atm);
  assert(outside.atmosphere_density == rho_atm);
  assert(outside.stellar_density == 0.0);
  assert(outside.momentum_x == 0.0 && outside.momentum_y == 0.0);

  const double dx = 1.0e12;
  const scf_map::Conserved average = scf_map::CellAverage(
      profile, 2.0e13-dx, 2.0e13+dx,
      -1.0e13-dx, -1.0e13+dx,
      1.0e13-dx, 1.0e13+dx, 3,
      omega, 5.0/3.0, rho_atm, p_atm, false);
  assert(average.density > 0.0);
  assert(average.radiative_density >= 0.0);
  assert(average.radiative_density <= average.stellar_density);
  assert(Close(average.density,
               average.stellar_density+average.atmosphere_density, 1.0e-14));

  std::cout << "radial_points=" << profile.radial_points()
            << " angular_points=" << profile.angular_points()
            << " sample_temperature_K=" << north.temperature << '\n';
  return 0;
}
