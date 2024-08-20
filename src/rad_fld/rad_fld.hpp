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
  constexpr int NTEMP=2, NMATRIX=13, NCOEFF=11;
  enum TempTndex {GAS=0, RAD=1};
  enum CoeffIndex {DXM=0, DXP=1, DYM=2, DYP=3, DZM=4, DZP=5, DSIGMAP=6, DCOUPLE=7, DEGAS=8,
                   DCPH=9, DAR=10};
  enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,
                    CPRR=7, CPRG=8, CPRC=9, CPGR=10, CPGG=11, CPGC=12};
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
  AthenaArray<Real> empty_flux[3];
  AthenaArray<Real> def;   // defect from the Multigrid solver
  bool output_defect;
  bool calc_in_temp;
  bool is_couple;
  bool only_rad;
  Real c_ph, a_r, const_opacity;

  void InitFLDConstants(ParameterInput *pin);
  void LoadHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u);
  void CalculateCoefficients(const AthenaArray<Real> &w,
                             const AthenaArray<Real> &u);
  void UpdateHydroVariables(AthenaArray<Real> &w, const AthenaArray<Real> &u);
  CellCenteredBoundaryVariable rfldbvar;

  friend class MGFLDDriver;

 private:
  int refinement_idx_;
};

#endif // RAD_FLD_RAD_FLD_HPP_
