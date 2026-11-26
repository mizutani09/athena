#ifndef ATHENA_PGEN_SCF_LUMINOSITY_HPP_
#define ATHENA_PGEN_SCF_LUMINOSITY_HPP_

// Smooth, compact spherical deposition kernel for the unresolved core
// luminosity.  The normalization is the exact three-dimensional integral of
// (1-r^2/R^2)^2 over r<R, so the parameter remains a physical luminosity and
// does not depend on resolution apart from cell-quadrature error.
namespace scf_luminosity {

template <typename T>
inline T Weight(T radius, T support_radius) {
  if (radius >= support_radius) return T(0);
  const T u = radius/support_radius;
  const T one_minus_u2 = T(1)-u*u;
  return one_minus_u2*one_minus_u2;
}

template <typename T>
inline T VolumeNormalization(T support_radius) {
  // 4*pi*R^3*integral_0^1 u^2(1-u^2)^2 du = 32*pi*R^3/105.
  const T pi = T(3.141592653589793238462643383279502884L);
  return T(32)*pi*support_radius*support_radius*support_radius/T(105);
}

}  // namespace scf_luminosity

#endif  // ATHENA_PGEN_SCF_LUMINOSITY_HPP_
