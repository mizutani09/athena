#ifndef ATHENA_PGEN_SCF_RELAXATION_HPP_
#define ATHENA_PGEN_SCF_RELAXATION_HPP_

#include <algorithm>

namespace scf_relaxation {

template <typename T>
struct Velocity {
  T x;
  T y;
  T z;
};

// Linearly taper the damping coefficient from unity to zero.  Negative times
// disable the taper and retain full strength for callers that need it.
template <typename T>
inline T TaperStrength(T time, T taper_start, T taper_end) {
  if (taper_start < T(0) || taper_end < T(0)) return T(1);
  if (time <= taper_start) return T(1);
  if (time >= taper_end) return T(0);
  return (taper_end-time)/(taper_end-taper_start);
}

// Smoothly select the numerical core.  A negative core radius disables the
// selector.  taper_width=0 gives a sharp boundary; otherwise a cubic
// smoothstep avoids a discontinuous damping rate at either edge.
template <typename T>
inline T CoreWeight(T radius, T core_radius, T taper_width) {
  if (core_radius < T(0)) return T(0);
  if (radius <= core_radius) return T(1);
  if (taper_width <= T(0) || radius >= core_radius+taper_width) return T(0);
  const T x = (radius-core_radius)/taper_width;
  return T(1)-x*x*(T(3)-T(2)*x);
}

// Retain a core-only damping floor as the global temporal relaxation is
// tapered away.  This lets an excised/softened numerical core remain quiet
// without suppressing physical envelope convection.
template <typename T>
inline T CombinedStrength(T temporal_strength, T core_weight,
                          T core_strength) {
  temporal_strength = std::max(T(0), std::min(T(1), temporal_strength));
  core_weight = std::max(T(0), std::min(T(1), core_weight));
  core_strength = std::max(T(0), std::min(T(1), core_strength));
  return temporal_strength+(T(1)-temporal_strength)*core_weight*core_strength;
}

// Dampen only the perturbation relative to rigid corotation.  factor=1 is the
// identity and factor=0 returns the target velocity exactly.
template <typename T>
inline Velocity<T> DampTowardCorotation(T vx, T vy, T vz, T x, T y,
                                        T omega, T factor) {
  factor = std::max(T(0), std::min(T(1), factor));
  const T target_x = -omega*y;
  const T target_y = omega*x;
  return {target_x+factor*(vx-target_x),
          target_y+factor*(vy-target_y), factor*vz};
}

}  // namespace scf_relaxation

#endif  // ATHENA_PGEN_SCF_RELAXATION_HPP_
