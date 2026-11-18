#ifndef ATHENA_PGEN_SCF_CORE_SOFTENING_HPP_
#define ATHENA_PGEN_SCF_CORE_SOFTENING_HPP_

#include <cmath>

// Compact-support cubic spline used by BSCF-rsg at commit fe3f380.  The
// support radius h, rather than a Plummer epsilon, is the input convention.
namespace scf_core {

template <typename T>
inline T PotentialShape(T u_in) {
  using std::abs;
  const T u = abs(u_in);
  if (u < T(0.5)) {
    const T u2 = u*u;
    const T u4 = u2*u2;
    const T u5 = u4*u;
    return -T(14)/T(5)+T(16)*u2/T(3)-T(48)*u4/T(5)+T(32)*u5/T(5);
  }
  if (u < T(1)) {
    const T u2 = u*u;
    const T u3 = u2*u;
    const T u4 = u3*u;
    const T u5 = u4*u;
    return -T(16)/T(5)+T(1)/(T(15)*u)+T(32)*u2/T(3)-T(16)*u3
        +T(48)*u4/T(5)-T(32)*u5/T(15);
  }
  return -T(1)/u;
}

template <typename T>
inline T MassShape(T u_in) {
  using std::abs;
  const T u = abs(u_in);
  if (u < T(0.5)) {
    const T u2 = u*u;
    const T u3 = u2*u;
    const T u5 = u3*u2;
    const T u6 = u5*u;
    return T(32)*u3/T(3)-T(192)*u5/T(5)+T(32)*u6;
  }
  if (u < T(1)) {
    const T u2 = u*u;
    const T u3 = u2*u;
    const T u4 = u3*u;
    const T u5 = u4*u;
    const T u6 = u5*u;
    return -T(1)/T(15)+T(64)*u3/T(3)-T(48)*u4
        +T(192)*u5/T(5)-T(32)*u6/T(3);
  }
  return T(1);
}

template <typename T>
inline T AccelerationShape(T u_in) {
  using std::abs;
  const T u = abs(u_in);
  if (u < T(0.5)) {
    const T u2 = u*u;
    const T u3 = u2*u;
    const T u4 = u3*u;
    return T(32)*u/T(3)-T(192)*u3/T(5)+T(32)*u4;
  }
  return MassShape(u)/(u*u);
}

template <typename T>
inline T Potential(T radius, T gm, T support_radius) {
  return gm/support_radius*PotentialShape(radius/support_radius);
}

template <typename T>
inline T Acceleration(T radius, T gm, T support_radius) {
  return gm/(support_radius*support_radius)
      *AccelerationShape(radius/support_radius);
}

}  // namespace scf_core

#endif  // ATHENA_PGEN_SCF_CORE_SOFTENING_HPP_
