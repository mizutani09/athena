#ifndef RAD_FLD_RAD_FLD_HPP_
#define RAD_FLD_RAD_FLD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rad_fld.hpp
//! \brief defines MGFLD class which implements data and functions for the implicit
//!        FLD solver

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/cc/bvals_cc.hpp"

class MeshBlock;
class ParameterInput;
class Coordinates;
class FLDBoundaryValues;
class MGFLD;
class MGFLDDriver;

namespace RadFLD {
  constexpr int NTEMP=2, NMATRIX=15, NCOEFF=8;//, NADV=2;
  enum TempTndex {GAS=0, RAD=1};
  enum CoeffIndex {DXM=0, DXP=1, DYM=2, DYP=3, DZM=4, DZP=5, DSIGMAP=6, DCOUPLE=7};
  enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,
                    CPRR=7, CPRRS=8, CPRG=9, CPRC=10, CPRCS=11, CPGR=12, CPGG=13, CPGC=14};
                    // CMM=7, CMP=8, CPM=9,
                    // CPP=10, MCM=11, MCP=12, PCM=13, PCP=14, MMC=15, MPC=16, PMC=17, PPC=18};
}

//! \class FLD
//! \brief gravitational potential data and functions

class FLD {
 public:
  FLD(MeshBlock *pmb, ParameterInput *pin);
  ~FLD();

  MeshBlock* pmy_block;
  MGFLD *pmg;
  // AthenaArray<Real> ecr, source, zeta, coeff; //!
  AthenaArray<Real> source, coeff;
  AthenaArray<Real> u, coarse_u;

  AthenaArray<Real> r, r1, r2;  // (no more than MAX_NREGISTER allowed)
  AthenaArray<Real> r0, r_fl_div;  // rkl2 STS memory registers;
  AthenaArray<Real> r_flux[3];  // face-averaged flux vector
  AthenaArray<Real> coarse_r;
  int refinement_idx{-1}; // for r

  AthenaArray<Real> sigma_r;
  AthenaArray<Real> empty_flux[3];
  AthenaArray<Real> def;   // defect from the Multigrid solver
  bool output_defect;
  bool calc_in_temp;
  bool is_couple;
  bool only_rad;
  bool cut_diff;

  void LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u);
  void UpdateRadiationEnergy(AthenaArray<Real> &u, const AthenaArray<Real> &r);
  void CalculateCoefficients(const AthenaArray<Real> &w,
                             const AthenaArray<Real> &u);
  void UpdateHydroVariables(AthenaArray<Real> &w,
       AthenaArray<Real> &hydro_u, const AthenaArray<Real> &fld_u);
  // Real CalculateSigmaR(const Real den, const Real egas);
  CellCenteredBoundaryVariable mgfldbvar;
  CellCenteredBoundaryVariable rfldbvar;
  void CalculateFluxes(AthenaArray<Real> &u, const int order);
  void LoadRadEnergyforFlux(AthenaArray<Real> &u, AthenaArray<Real> &r);

  friend class MGFLDDriver;

  // Function in problem generators to update opacity
  void EnrollOpacityFunction(FLDOpacityFunc MyOpacityFunction);
  FLDOpacityFunc UpdateOpacity;

  void AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out);
  void CheckFLD(const AthenaArray<Real> &r);

 private:
  int refinement_idx_;

  // scratch space used to compute fluxes
  // 2D scratch arrays
  AthenaArray<Real> rl_, rr_, rlb_;
  // 1D scratch arrays
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_;
  AthenaArray<Real> dflx_;

  void ComputeUpwindFlux(const int k, const int j, const int il,
                         const int iu, // CoordinateDirection dir,
                         AthenaArray<Real> &rl, AthenaArray<Real> &rr,
                         AthenaArray<Real> &mass_flx,
                         AthenaArray<Real> &flx_out);
};

#endif // RAD_FLD_RAD_FLD_HPP_
