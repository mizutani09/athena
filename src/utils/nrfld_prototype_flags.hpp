#ifndef UTILS_NRFLD_PROTOTYPE_FLAGS_HPP_
#define UTILS_NRFLD_PROTOTYPE_FLAGS_HPP_

#include <cstdlib>
#include <cstring>

namespace NRFLDPrototype {
inline bool Flag(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool Enabled() {
  return Flag("ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE");
}

inline bool SharedCoeff() {
  return Enabled() && Flag("ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF");
}

inline bool TangentialReconstruction() {
  return Enabled() && Flag("ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION");
}

inline bool VariableLimiterTrial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER");
}

inline bool CoupledTrial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_COUPLED");
}

inline bool RegridTrial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_REGRID");
}

inline bool DifferenceJv() {
  const char *value = std::getenv("ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV");
  return value == nullptr || std::strcmp(value, "0") != 0;
}
}  // namespace NRFLDPrototype

#endif  // UTILS_NRFLD_PROTOTYPE_FLAGS_HPP_
