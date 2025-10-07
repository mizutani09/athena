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
class UserOpacityTable;


namespace RadFLD2 {
  constexpr int NTEMP=2, NMATRIX=15, NCOEFF=8, NOPACITY=2;
//   enum VarIndex {GAS=0, RAD=1};
  enum OpacityIndex {SIGMA_P=0, SIGMA_R=1};
}

class FLD2 {
  public:
  FLD2(MeshBlock *pmb, ParameterInput *pin);
  ~FLD2();

  MeshBlock* pmy_block;

  AthenaArray<Real> u_gas;
  AthenaArray<Real> u_rad, u_rad1, u_rad2;  // (no more than MAX_NREGISTER allowed)
  AthenaArray<Real> u_rad0, u_rad_fl_div;  // rkl2 STS memory registers;
  AthenaArray<Real> u_rad_flux[3];  // face-averaged flux vector
  AthenaArray<Real> coarse_u_rad;
  int refinement_idx{-1}; // for r

  AthenaArray<Real> sigma_p, sigma_r;
  AthenaArray<Real> empty_flux[3];
  // bool output_defect;
  // bool calc_in_temp;
  bool is_couple;
  bool only_rad;
  bool cut_diff;
  bool cut_Pnablav;
  bool fixed_flux_limitter;

  // for interaction with Hydro
  void LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u);
  void UpdateHydroVariables(AthenaArray<Real> &w,
       AthenaArray<Real> &hydro_u, const AthenaArray<Real> &fld_u);

  // for advection of radiation energy
  CellCenteredBoundaryVariable u_rad_fldbvar;
  void CalculateFluxes(AthenaArray<Real> &u, const int order);
  void AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out);

  // Function in problem generators to update opacity
  void EnrollOpacityFunction(FLDOpacityFunc MyOpacityFunction);
  FLDOpacityFunc UpdateOpacity;
  UserOpacityTable *pUserOpacityTable;

  // constants
  Real a_r, c_ph, const_opacity;


 private:
  int refinement_idx_;

  // scratch space used to compute fluxes
  // 2D scratch arrays
  AthenaArray<Real> u_radl_, u_radr_, u_radlb_;
  // 1D scratch arrays
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_;
  AthenaArray<Real> dflx_;

  void ComputeUpwindFlux(const int k, const int j, const int il,
                         const int iu, // CoordinateDirection dir,
                         AthenaArray<Real> &u_radl, AthenaArray<Real> &u_radr,
                         AthenaArray<Real> &mass_flx,
                         AthenaArray<Real> &flx_out);
};


// class UserOpacityTable : public InterpTable2D {
//  public:
//   UserOpacityTable(ParameterInput *pin);
//   ~UserOpacityTable();

//   // Methods for opacity interpolation
//   Real GetOpacity(int var_index, Real x2, Real x1); // Generic interface (x2=pressure, x1=temperature)

//   bool use_tables; // Flag to indicate if tables are used
//   // Data members for opacity table properties
//   Real tempMin, tempMax;    // Temperature limits
//   Real pressureMin, pressureMax;  // Pressure limits
//   int nTemp, nPressure, nVar;    // Table dimensions
//   AthenaArray<Real> OpacityTables;  // Tables for each variable
// };


#endif // FLD_FLD_HPP_
