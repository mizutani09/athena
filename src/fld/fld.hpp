#ifndef FLD_FLD_HPP_
#define FLD_FLD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fld.hpp
//  \brief defines the fld class

// C headers

// C++ headers
#include <algorithm>
#include <cstdint>  // std::int64_t
#include <cstdio> // std::size_t
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "../utils/interp_table.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;
class ParameterInput;
class Coordinates;
class FLDBoundaryValues;


namespace RadFLD {
  // constexpr int NTEMP=2, NMATRIX=15, NCOEFF=8, NOPACITY=2;
  constexpr int NOPACITY=2;
//   enum VarIndex {GAS=0, RAD=1};
  enum OpacityIndex {SIGMA_P=0, SIGMA_R=1};
  constexpr int NRAD_FACE_STATE = 3;
  enum RadiationFaceIndex {ERAD=0, LAMBDA=1, ARAD=2};

  // The radiation-pressure force has exactly one owner in a coupled update:
  // either the gas Riemann flux or the explicit source term.  The mode is
  // resolved once by FLD and shared by both HLLC variants and the source.
  enum class PressureCouplingMode { kOff, kSource, kFlux };

  inline const char *PressureCouplingModeName(const PressureCouplingMode mode) {
    switch (mode) {
      case PressureCouplingMode::kSource: return "source";
      case PressureCouplingMode::kFlux: return "flux";
      case PressureCouplingMode::kOff: return "off";
    }
    return "invalid";
  }

  // Levermore-Pomraning FLD closure shared by every radiation operator.
  struct ClosureValues {
    Real lambda;
    Real lambda_r;
    Real chi;
    // This is lambda/sigma.  Evaluating it directly is singular for a
    // transparent face, while lambda_r*E/|grad E| has a finite streaming
    // limit.
    Real lambda_over_opacity;
    bool valid;
  };

  inline bool IsValidClosureGradient(const Real gradient) {
    // +Inf is the well-defined streaming limit.  NaN and negative gradients
    // indicate corrupted state or arithmetic upstream of the closure.
    return gradient >= 0.0 && !std::isnan(gradient);
  }

  inline bool IsValidClosureInput(const Real opacity, const Real erad,
                                  const Real gradient) {
    return std::isfinite(opacity) && opacity >= 0.0
        && std::isfinite(erad) && erad >= 0.0
        && IsValidClosureGradient(gradient);
  }

  inline Real InvalidClosureValue() {
    return std::numeric_limits<Real>::quiet_NaN();
  }

  inline Real FluxLimiter(const Real r, const bool fixed) {
    if (fixed) return ONE_3RD;
    if (std::isnan(r) || r < 0.0) return InvalidClosureValue();
    if (std::isinf(r)) return 0.0;
    if (r <= 1.0) return (2.0 + r)/(6.0 + 3.0*r + r*r);

    // Multiply numerator and denominator by 1/r^2.  The original
    // expression overflows at r ~ sqrt(max(Real)) even though the closure
    // itself is already close to its streaming limit.
    const Real inv_r = 1.0/r;
    const Real inv_r2 = inv_r*inv_r;
    return (inv_r + 2.0*inv_r2)/(1.0 + 3.0*inv_r + 6.0*inv_r2);
  }

  inline Real FluxLimiterTimesR(const Real r, const bool fixed) {
    if (fixed) {
      if (std::isnan(r) || r < 0.0) return InvalidClosureValue();
      if (std::isinf(r)) return std::numeric_limits<Real>::max();
      const Real maximum = std::numeric_limits<Real>::max();
      return r > maximum/3.0 ? maximum : r/3.0;
    }
    if (std::isnan(r) || r < 0.0) return InvalidClosureValue();
    if (std::isinf(r)) return 1.0;
    if (r <= 1.0) return r*FluxLimiter(r, false);

    const Real inv_r = 1.0/r;
    const Real inv_r2 = inv_r*inv_r;
    return (1.0 + 2.0*inv_r)/(1.0 + 3.0*inv_r + 6.0*inv_r2);
  }

  inline Real EddingtonFactor(const Real r, const bool fixed) {
    if (fixed) return ONE_3RD;
    const Real lambda = FluxLimiter(r, false);
    const Real lambda_r = FluxLimiterTimesR(r, false);
    return lambda + lambda_r*lambda_r;
  }

  inline Real SaturatingNonnegativeQuotient(const Real numerator,
                                            const Real denominator) {
    const long double value = static_cast<long double>(numerator)
                            / static_cast<long double>(denominator);
    const long double maximum =
        static_cast<long double>(std::numeric_limits<Real>::max());
    return static_cast<Real>(value >= maximum ? maximum : value);
  }

  inline ClosureValues EvaluateClosure(const Real gradient, const Real opacity,
                                       const Real erad, const bool fixed) {
    const Real invalid = InvalidClosureValue();
    ClosureValues result{invalid, invalid, invalid, invalid, false};
    if (!IsValidClosureInput(opacity, erad, gradient)) return result;

    // A fixed limiter retains lambda=1/3 by definition.  At zero opacity a
    // nonzero gradient would require an infinite diffusion coefficient, so
    // that combination is outside the finite-coefficient fixed-limiter
    // contract and must be diagnosed by the caller.
    if (fixed && opacity == 0.0) return result;

    Real r = 0.0;
    if (gradient > 0.0 && opacity > 0.0 && erad > 0.0) {
      // long double avoids an intermediate underflow/overflow when the three
      // finite inputs span much of the Real range.  Conversion back to Real
      // intentionally maps an out-of-range ratio to the corresponding
      // streaming/diffusion limit.
      const long double ratio = static_cast<long double>(gradient)
                              / static_cast<long double>(opacity)
                              / static_cast<long double>(erad);
      r = static_cast<Real>(ratio);
    } else if (gradient > 0.0 && (opacity == 0.0 || erad == 0.0)) {
      r = std::numeric_limits<Real>::infinity();
    }

    result.lambda = FluxLimiter(r, fixed);
    result.lambda_r = fixed
        ? (std::isinf(r) ? std::numeric_limits<Real>::max()
                         : SaturatingNonnegativeQuotient(r, static_cast<Real>(3.0)))
        : FluxLimiterTimesR(r, false);
    result.chi = fixed ? ONE_3RD : result.lambda + result.lambda_r*result.lambda_r;
    if (gradient == 0.0 || fixed) {
      result.lambda_over_opacity = (opacity == 0.0)
          ? 0.0 : SaturatingNonnegativeQuotient(result.lambda, opacity);
    } else if (std::isinf(gradient)) {
      result.lambda_over_opacity = 0.0;
    } else {
      // lambda/sigma = (lambda R)*E/|grad E|.  This remains finite when
      // sigma is zero and avoids the 0*Inf product in the streaming limit.
      const long double value = static_cast<long double>(result.lambda_r)
                              * static_cast<long double>(erad)
                              / static_cast<long double>(gradient);
      const long double maximum =
          static_cast<long double>(std::numeric_limits<Real>::max());
      result.lambda_over_opacity = static_cast<Real>(value >= maximum ? maximum : value);
    }
    result.valid = std::isfinite(result.lambda)
                && std::isfinite(result.lambda_r)
                && std::isfinite(result.chi)
                && std::isfinite(result.lambda_over_opacity);
    return result;
  }

  inline Real FaceOpacity(const Real left, const Real right,
                          const Real inverse_width) {
    if (!std::isfinite(left) || left < 0.0 || !std::isfinite(right)
        || right < 0.0 || !std::isfinite(inverse_width)
        || inverse_width <= 0.0) {
      return InvalidClosureValue();
    }

    // Preserve the Howell-Greenough cap, but avoid both 0/0 and overflow in
    // the arithmetic/harmonic means.  For zero opacity on both sides the
    // harmonic mean is explicitly zero.
    const Real arithmetic = 0.5*left + 0.5*right;
    const Real larger = std::max(left, right);
    const Real smaller = std::min(left, right);
    const Real harmonic = (larger == 0.0)
        ? 0.0 : smaller*(2.0/(1.0 + smaller/larger));
    const Real cap = 2.0*TWO_3RD*inverse_width;
    return std::min(arithmetic, std::max(harmonic, cap));
  }
}

class FLD {
  public:
  FLD(MeshBlock *pmb, ParameterInput *pin);
  ~FLD();

  MeshBlock* pmy_block;

  AthenaArray<Real> u_gas;
  AthenaArray<Real> u_rad, u_rad1, u_rad2;  // (no more than MAX_NREGISTER allowed)
  AthenaArray<Real> u_rad0, u_rad_fl_div;  // rkl2 STS memory registers;
  AthenaArray<Real> u_rad_flux[3];  // face-averaged flux vector
  // Reconstructed radiation energy and scalar FLD closure on hydro faces.  The
  // first index is RadFLD::RadiationFaceIndex.  HLLC-FLD consumes exactly
  // these states, so radiation and gas use the same reconstruction order.
  AthenaArray<Real> rad_face_l[3], rad_face_r[3];
  // Godunov radiation energy selected by the hydro Riemann solver.  Explicit
  // mixed-frame terms use this state together with Hydro::vf.
  AthenaArray<Real> rad_face_g[3];
  AthenaArray<Real> coarse_u_rad;
  int refinement_idx{-1}; // for r

  AthenaArray<Real> sigma_p, sigma_r;
  AthenaArray<Real> empty_flux[3];
  // bool output_defect;
  // bool calc_in_temp;
  bool is_couple;
  bool only_rad;
  bool cut_diff;
  bool include_radiation_force;
  RadFLD::PressureCouplingMode pressure_coupling_mode;
  bool fixed_flux_limiter;
  bool fixed_u_rad;
  // Explicit O(v/c) mixed-frame energy exchange.
  bool include_mixed_frame_terms;
  // Radiation-energy transport and explicit mixed-frame source are separate
  // switches.  Both are resolved once at startup; the legacy environment
  // variable only overrides the transport switch when the input omits it.
  bool mixed_frame_transport;
  // Optional hydrodynamic diode used by an outflow-only physical upper
  // boundary.  It is consumed only by the LHLLC-FLD Riemann solver.
  bool hydro_top_outflow_diode;
  bool marshak_top_boundary;
  Real marshak_top_alpha;
  Real marshak_top_erad_ext;
  AthenaArray<Real> marshak_dface;

  // Problem generators may keep inexpensive diagnostic counters here.  The
  // storage belongs to the MeshBlock's FLD object, so block-parallel boundary
  // callbacks never update shared namespace state.  Diagnostics are not
  // restart state: a reconstructed block starts a new one-shot interval.
  std::vector<std::uint64_t> user_diagnostic_counters;

  // for interaction with Hydro
  void LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &fld_u_gas);
  void UpdateHydroVariables(AthenaArray<Real> &w,
                            AthenaArray<Real> &hydro_u,
                            const AthenaArray<Real> &fld_u_rad,
                            const AthenaArray<Real> &fld_u_gas);

  // for advection of radiation energy
  CellCenteredBoundaryVariable u_rad_fldbvar;
  void CalculateFluxes(AthenaArray<Real> &u, const int order);
  void CalculateRadiationFaceStates(const int order);
  Real RadiationSoundSpeedSquared(int k, int j, int i, int dir) const;
  void AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out);
  void AddExplicitSourceTerms(const Real dt, const AthenaArray<Real> &prim,
                              AthenaArray<Real> &hydro_u);

  // Function in problem generators to update opacity
  void EnrollOpacityFunction(FLDOpacityFunc MyOpacityFunction);
  // Apply the enrolled callback and then enforce the FLD opacity contract.
  void UpdateOpacity(MeshBlock *pmb, AthenaArray<Real> &u_rad_fld,
                     AthenaArray<Real> &prim);

  // Function for Newton Raphson solver
  void CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
                       const AthenaArray<Real> &u_old, const AthenaArray<Real> &coeff,
                       Real dt, bool th);

  // constants
  Real a_r, c_ph, const_opacity;


 private:
  int refinement_idx_;

  // scratch space used to compute fluxes
  // 2D scratch arrays
  AthenaArray<Real> rad_state_cc_, rad_statel_, rad_stater_, rad_statelb_;
  // 1D scratch arrays
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_;
  AthenaArray<Real> dflx_;
  FLDOpacityFunc opacity_function_{nullptr};
  void EmitForceDiagnostic(const Real dt, const AthenaArray<Real> &prim,
                           const AthenaArray<Real> &hydro_u);
  bool opacity_contract_diagnostic_printed_{false};
  bool force_diagnostic_printed_{false};
  int force_diagnostic_verbosity_{0};
  int force_diagnostic_rank_{0};
  int force_diagnostic_gid_{0};

};

#endif // FLD_FLD_HPP_
