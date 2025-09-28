#ifndef RAD_FLD_NEWTON_RAPHSON_HPP_
#define RAD_FLD_NEWTON_RAPHSON_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson.hpp
//  \brief defines the Newton-Raphson solver class

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
#include "../bvals/bvals_interfaces.hpp"
#include "../bvals/cc/mg/bvals_mg.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
// #include "../task_list/mg_task_list.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class Mesh;
class MeshBlock;
class ParameterInput;
class Coordinates;
class linearMG;
class linearMGDriver;

namespace NewtonRaphsonFLD {
  constexpr int NTEMP=2, NMATRIX=15, NCOEFF=9, NOPACITY=2;
  enum TempIndex {GAS=0, RAD=1};
  enum CoeffIndex {DXM=0, DXP=1, DYM=2, DYP=3, DZM=4, DZP=5};
  enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,
                    CPRR=7, CPRRS=8, CPRG=9, CPRC=10, CPRCS=11, CPGR=12, CPGG=13, CPGC=14};
                    // CMM=7, CMP=8, CPM=9,
                    // CPP=10, MCM=11, MCP=12, PCM=13, PCP=14, MMC=15, MPC=16, PMC=17, PPC=18};
  enum OpacityIndex {SIGMA_P=0, SIGMA_R=1};
}

class NewtonRaphson {
 public:
  NewtonRaphson(MeshBlock *pmb, ParameterInput *pin);
  ~NewtonRaphson();
  void Solve(int stage, Real dt);
  void CalculateCoefficients(AthenaArray<Real> &u, Real dt);
  linearMG *plinmg;

  CellCenteredBoundaryVariable nrmgfldbvar;

  AthenaArray<Real> u; // radiation energy density
  AthenaArray<Real> coeff; // coefficients
  AthenaArray<Real> rhs; // right-hand side

 private:
  MeshBlock* pmy_block;
  int max_iter_;
};

#endif // RAD_FLD_NEWTON_RAPHSON_HPP_
