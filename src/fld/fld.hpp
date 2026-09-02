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
#include <cstdint>  // std::int64_t
#include <cstdio> // std::size_t
#include <iostream>
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

  // Levermore-Pomraning FLD closure shared by every radiation operator.
  inline Real FluxLimiter(const Real r, const bool fixed) {
    return fixed ? ONE_3RD : (2.0 + r)/(6.0 + 3.0*r + r*r);
  }

  inline Real EddingtonFactor(const Real r, const bool fixed) {
    if (fixed) return ONE_3RD;
    const Real lambda = FluxLimiter(r, false);
    return lambda + (lambda*r)*(lambda*r);
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
  bool fixed_flux_limiter;
  bool fixed_u_rad;
  // Explicit O(v/c) mixed-frame energy exchange.
  bool include_mixed_frame_terms;
  // Optional hydrodynamic diode used by an outflow-only physical upper
  // boundary.  It is consumed only by the LHLLC-FLD Riemann solver.
  bool hydro_top_outflow_diode;
  bool marshak_top_boundary;
  Real marshak_top_alpha;
  Real marshak_top_erad_ext;
  AthenaArray<Real> marshak_dface;

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
  FLDOpacityFunc UpdateOpacity;

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

};

#endif // FLD_FLD_HPP_
