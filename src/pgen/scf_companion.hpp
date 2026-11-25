#ifndef ATHENA_PGEN_SCF_COMPANION_HPP_
#define ATHENA_PGEN_SCF_COMPANION_HPP_

#include <algorithm>
#include <cmath>

#include "scf_core_softening.hpp"

// Prescribed circular companion used for the first RLOF-onset gates.  The
// coordinates remain primary-centred.  The optional indirect term subtracts
// the acceleration of that origin toward the companion, following the frame
// convention in ce_main.cpp.
namespace scf_companion {

template <typename T>
inline T Ramp(T time, T start, T duration) {
  if (time <= start) return T(0);
  if (duration <= T(0)) return T(1);
  const T s = std::max(T(0), std::min(T(1), (time-start)/duration));
  return s*s*(T(3)-T(2)*s);
}

template <typename T>
inline void CircularPosition(T time, T start, T separation, T omega,
                             T phase, T* x, T* y, T* z) {
  const T elapsed = std::max(T(0), time-start);
  const T angle = phase+omega*elapsed;
  *x = separation*std::cos(angle);
  *y = separation*std::sin(angle);
  *z = T(0);
}

template <typename T>
inline void Acceleration(T x, T y, T z, T companion_x, T companion_y,
                         T companion_z, T gm, T support_radius,
                         bool include_indirect, T* ax, T* ay, T* az) {
  const T dx = x-companion_x;
  const T dy = y-companion_y;
  const T dz = z-companion_z;
  const T radius = std::sqrt(dx*dx+dy*dy+dz*dz);
  const T direct = scf_core::Acceleration(radius, gm, support_radius);
  const T inverse_radius = radius > T(0) ? T(1)/radius : T(0);
  *ax = -direct*dx*inverse_radius;
  *ay = -direct*dy*inverse_radius;
  *az = -direct*dz*inverse_radius;
  if (!include_indirect) return;

  const T separation = std::sqrt(companion_x*companion_x
      +companion_y*companion_y+companion_z*companion_z);
  const T origin_acceleration = scf_core::Acceleration(
      separation, gm, support_radius);
  const T inverse_separation = separation > T(0) ? T(1)/separation : T(0);
  *ax -= origin_acceleration*companion_x*inverse_separation;
  *ay -= origin_acceleration*companion_y*inverse_separation;
  *az -= origin_acceleration*companion_z*inverse_separation;
}

template <typename T>
inline T Potential(T x, T y, T z, T companion_x, T companion_y,
                   T companion_z, T gm, T support_radius,
                   bool include_indirect) {
  const T dx = x-companion_x;
  const T dy = y-companion_y;
  const T dz = z-companion_z;
  const T radius = std::sqrt(dx*dx+dy*dy+dz*dz);
  T potential = scf_core::Potential(radius, gm, support_radius);
  if (!include_indirect) return potential;

  const T separation = std::sqrt(companion_x*companion_x
      +companion_y*companion_y+companion_z*companion_z);
  const T origin_acceleration = scf_core::Acceleration(
      separation, gm, support_radius);
  const T inverse_separation = separation > T(0) ? T(1)/separation : T(0);
  potential += origin_acceleration*inverse_separation
      *(companion_x*x+companion_y*y+companion_z*z);
  return potential;
}

}  // namespace scf_companion

#endif  // ATHENA_PGEN_SCF_COMPANION_HPP_
