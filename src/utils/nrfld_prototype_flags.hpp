#ifndef UTILS_NRFLD_PROTOTYPE_FLAGS_HPP_
#define UTILS_NRFLD_PROTOTYPE_FLAGS_HPP_

#include <cstdlib>
#include <cstring>

namespace NRFLDPrototype {
struct DefaultMode {
  bool enabled = false;
  bool test_c = false;
  bool mach5 = false;
  bool fixed_limiter = false;
};

inline DefaultMode &Mode() {
  static DefaultMode mode;
  return mode;
}

inline bool Flag(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool FlagOrDefault(const char *name, bool fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::strcmp(value, "1") == 0;
}

inline void ConfigureDefault(bool enabled, bool test_c, bool mach5,
                             bool fixed_limiter) {
  DefaultMode &mode = Mode();
  mode.enabled = enabled;
  mode.test_c = test_c;
  mode.mach5 = mach5;
  mode.fixed_limiter = fixed_limiter;
}

inline bool Legacy() {
  return Flag("ATHENA_NRFLD_LEGACY_INTERFACE");
}

inline bool DefaultEnabled() {
  return Mode().enabled && !Legacy();
}

inline bool Enabled() {
  return !Legacy() &&
      (Flag("ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE") || DefaultEnabled());
}

inline bool SharedCoeff() {
  return Enabled() && FlagOrDefault("ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF",
                                    DefaultEnabled());
}

inline bool CenteredFaceTrial() {
  return SharedCoeff() && Mode().test_c
      && Flag("ATHENA_NRFLD_CENTERED_FACE_TRIAL");
}

inline bool TangentialReconstruction() {
  return Enabled() && FlagOrDefault(
      "ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION", DefaultEnabled());
}

inline bool SeriesResistance() {
  return Enabled() && FlagOrDefault("ATHENA_NRFLD_EXPERIMENTAL_SERIES_RESISTANCE",
      DefaultEnabled() && Mode().test_c && Mode().fixed_limiter);
}

inline bool VariableLimiterTrial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER") || DefaultEnabled();
}

inline bool CoupledTrial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_COUPLED") || DefaultEnabled();
}

inline bool RegridTrial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_REGRID") || DefaultEnabled();
}

inline bool Mach5Trial() {
  return Flag("ATHENA_NRFLD_EXPERIMENTAL_MACH5") ||
      (DefaultEnabled() && Mode().mach5);
}

inline bool DifferenceJv() {
  return FlagOrDefault("ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV",
                       !DefaultEnabled());
}
}  // namespace NRFLDPrototype

#endif  // UTILS_NRFLD_PROTOTYPE_FLAGS_HPP_
