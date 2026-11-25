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
