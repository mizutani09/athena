#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "../../src/pgen/scf_axisymmetric_profile.hpp"
#include "../../src/pgen/scf_core_softening.hpp"

namespace {

bool Close(double left, double right, double relative) {
  const double scale = std::max(std::abs(left), std::abs(right));
  return std::abs(left-right) <= relative*std::max(scale, 1.0e-300);
}

}  // namespace

int main() {
  const double gm = 6.67430e-8*1.1340371006082187e34;
  const double support = 3.5414971588025128e11;
  assert(Close(scf_core::Potential(0.0, gm, support),
               -2.8*gm/support, 2.0e-15));
  assert(scf_core::Acceleration(0.0, gm, support) == 0.0);
  assert(Close(scf_core::Potential(support, gm, support),
               -gm/support, 2.0e-15));
  assert(Close(scf_core::Acceleration(support, gm, support),
               gm/(support*support), 2.0e-15));
  assert(Close(scf_core::Potential(2.0*support, gm, support),
               -gm/(2.0*support), 2.0e-15));

  for (int index = 1; index <= 200; ++index) {
    const double radius = support*std::pow(10.0, -3.0+3.5*index/200.0);
    const double step = std::max(1.0e-5*radius, 1.0e-7*support);
    const double derivative = (scf_core::Potential(radius+step, gm, support)
                              -scf_core::Potential(radius-step, gm, support))
        /(2.0*step);
    assert(Close(derivative,
                 scf_core::Acceleration(radius, gm, support), 2.0e-5));
  }

  scf_map::AtmosphereModel atmosphere = {};
  atmosphere.density_reference = 1.0e-14;
  atmosphere.pressure_reference = 1.30865571e-2;
  atmosphere.density_floor = 1.0e-20;
  atmosphere.pressure_floor = 1.0e-8;
  atmosphere.sound_speed_squared = atmosphere.pressure_reference
      /atmosphere.density_reference;
  atmosphere.reference_radius = 6.679783093541112e13;
  atmosphere.monopole_gm = 6.67430e-8
      *(3.041623627857687e34+1.1340371006082187e34);
  atmosphere.isothermal_monopole = true;
  const double radius = 7.0e13;
  const double step = 1.0e8;
  double rho_minus, pressure_minus, rho_plus, pressure_plus;
  scf_map::AtmosphereState(atmosphere, radius-step, 0.0, 0.0,
                           &rho_minus, &pressure_minus);
  scf_map::AtmosphereState(atmosphere, radius+step, 0.0, 0.0,
                           &rho_plus, &pressure_plus);
  const double pressure_gradient = (pressure_plus-pressure_minus)/(2.0*step);
  const double density = 0.5*(rho_minus+rho_plus);
  const double acceleration = atmosphere.monopole_gm/(radius*radius);
  assert(Close(-pressure_gradient/density, acceleration, 2.0e-8));
  assert(rho_plus < rho_minus);

  std::cout << "core_support_cm=" << support
            << " atmosphere_hse_relative_error="
            << std::abs(-pressure_gradient/density/acceleration-1.0) << '\n';
  return 0;
}
