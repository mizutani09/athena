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
#include "../hydro/hydro.hpp"
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
  constexpr int NTEMP=2, NMATRIX=15, NCOEFF=8, NOPACITY=2;
  // enum TempIndex {GAS=0, RAD=1};
  enum TempIndex {RAD=0, GAS=1}; // caution!!
  enum CoeffIndex {DCCF=0, DCCS=1, DXM=2, DXP=3, DYM=4, DYP=5, DZM=6, DZP=7};
  enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,};
  enum OpacityIndex {SIGMA_P=0, SIGMA_R=1};
}

class NewtonRaphson {
 public:
  // NewtonRaphson(MeshBlock *pmb, ParameterInput *pin);
  NewtonRaphson(Mesh *pm, ParameterInput *pin);
  ~NewtonRaphson();
  void Solve(int stage, Real dt);
  void CalculateCoefficients(const AthenaArray<Real> &u_work,
                             const AthenaArray<Real> &u_pre,
                             const AthenaArray<Real> &w_hydro, Real dt);
  void UpdateHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u,
                             const AthenaArray<Real> &u_fld);
  void UpdateRadEnergy(AthenaArray<Real> &u_work, const AthenaArray<Real> &delta_u);

  friend class linearMG;

  linearMG *plinmg;
  linearMGDriver *plinmgdriver;

  CellCenteredBoundaryVariable nrmgfldbvar;

  AthenaArray<Real> u_pre; // previous radiation energy density
  AthenaArray<Real> u_work; // radiation energy density in work
  AthenaArray<Real> delta_u; // correction of radiation energy density
  AthenaArray<Real> coeff; // coefficients
  AthenaArray<Real> rhs; // right-hand side
  bool only_rad = false;

  AthenaArray<Real> sigma_p, sigma_r;

 private:
  MeshBlock* pmy_block;
  int max_iter_;
};

#endif // RAD_FLD_NEWTON_RAPHSON_HPP_
