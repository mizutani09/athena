#ifndef ATHENA_PGEN_SCF_FLD_DIAGNOSTIC_HPP_
#define ATHENA_PGEN_SCF_FLD_DIAGNOSTIC_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

// Read-only reconstruction of the cell-centred radiation heating associated
// with the same six lagged FLD face coefficients used by NR-FLD.  Positive
// values mean that diffusion adds radiation energy to the cell.
namespace scf_fld_diagnostic {

template <typename T>
inline T DiffusionHeating(T center, T xm, T xp, T ym, T yp, T zm, T zp,
                          T d_xm, T d_xp, T d_ym, T d_yp, T d_zm, T d_zp,
                          T dx, T dy, T dz) {
  const T flux_xm = -d_xm*(center-xm)/dx;
  const T flux_xp = -d_xp*(xp-center)/dx;
  const T flux_ym = -d_ym*(center-ym)/dy;
  const T flux_yp = -d_yp*(yp-center)/dy;
  const T flux_zm = -d_zm*(center-zm)/dz;
  const T flux_zp = -d_zp*(zp-center)/dz;
  return -((flux_xp-flux_xm)/dx
           +(flux_yp-flux_ym)/dy+(flux_zp-flux_zm)/dz);
}

// Match NRFLD's treatment of the irreducible subtraction error between a*T^4
// and Er.  Without this guard, c*sigma amplifies adjacent-floating-point LTE
// states into a diagnostic source that the nonlinear operator intentionally
// treats as zero.
template <typename T>
inline T ResolvableThermalDifference(T equilibrium_rad, T radiation_energy) {
  const T difference = equilibrium_rad-radiation_energy;
  const T scale = std::max(std::abs(equilibrium_rad),
                           std::abs(radiation_energy));
  const T roundoff = T(32)*std::numeric_limits<T>::epsilon()*scale;
  return (std::abs(difference) <= roundoff) ? T(0) : difference;
}

}  // namespace scf_fld_diagnostic

#endif  // ATHENA_PGEN_SCF_FLD_DIAGNOSTIC_HPP_
