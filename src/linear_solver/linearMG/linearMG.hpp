#ifndef LINEAR_SOLVER_LINEARMG_LINEARMG_HPP_
#define LINEAR_SOLVER_LINEARMG_LINEARMG_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linearMG.hpp
//  \brief defines the linear Multigrid base class

// C headers

// C++ headers
#include <cstdint>  // std::int64_t
#include <cstdio> // std::size_t
#include <iostream>
#include <unordered_map>
#include <vector>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../bvals/bvals_interfaces.hpp"
#include "../../globals.hpp"
#include "../../mesh/mesh.hpp"
// #include "../linear_solver.hpp"
#include "../../multigrid/multigrid.hpp"
#include "../../newton_raphson/Newton_Raphson.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;
class ParameterInput;
class Coordinates;
class Multigrid;
class NewtonRaphson;
class LinearMGBoundaryTaskList;

class linearMG: public Multigrid {
 public:
  linearMG(linearMGDriver *pmd, MeshBlock *pmb, ParameterInput *pin, NewtonRaphson *pnr);
  // linearMG(MeshBlock *pmb, ParameterInput *pin);
  ~linearMG();

  void Smooth(AthenaArray<Real> &dst, const AthenaArray<Real> &src,
              const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix, int rlev,
              int il, int iu, int jl, int ju, int kl, int ku, int color, bool th) final;
  void CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
                const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
                const AthenaArray<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
                int kl, int ku, bool th) final;
  void CalculateFASRHS(AthenaArray<Real> &def, const AthenaArray<Real> &src,
                const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
                int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) final;
  void CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &u,
                const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
                int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) final;

  friend class linearMGDriver;
  friend class NewtonRaphson;

 private:
  // linearMGDriver *pmd_;
  NewtonRaphson *pnr_;
  Real omega_;
  int fsmoother_;

};

class linearMGDriver: public MultigridDriver {
 public:
  linearMGDriver(Mesh *pm, ParameterInput *pin, NewtonRaphsonDriver *pnrd);
  ~linearMGDriver();
  void Solve(int stage, Real dt = 0.0) final;
  void ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst,
                 AthenaArray<Real> &cbuf, const AthenaArray<bool> &ncoarse) final;

  friend class linearMG;
  friend class NewtonRaphsonDriver;

  // LinearSolver *plinsolver = nullptr;

 private:
  LinearMGBoundaryTaskList *linmgtlist_;
  NewtonRaphsonDriver *pnrd_;
  Real omega_;
  int fsmoother_;
  bool fsteady_;
};


// class linearMGAdapter : public LinearSolver {
// public:
//   linearMGAdapter() = default;
//   explicit linearMGAdapter(linearMGDriver *pmd, MeshBlock *pmb, ParameterInput *pin) : pmg_(new linearMG(pmd, pmb, pin)) {}

//   ~linearMGAdapter() override = default;

//   // void set_backend(linearMG* plinmg) { plinmg_ = plinmg; }

//   // void set_operator(MatVec A, std::size_t n) override {
//   //   assert(plinmg_ && "linearMGAdapter: backend MG is null");
//   //   plinmg_->Setup(A, n);
//   // }
//   // void set_tolerance(double rel, double abs) override {
//   //   assert(plinmg_);
//   //   plinmg_->SetTol(rel, abs);
//   // }
//   // void set_maxiter(int it) override {
//   //   assert(plinmg_);
//   //   plinmg_->SetMaxIter(it);
//   // }

// private:
//   linearMG* pmg_ = nullptr;
// };


// class linearMGDriverAdapter : public LinearSolverDriver {
// public:
//   linearMGDriverAdapter() = default;
//   explicit linearMGDriverAdapter(Mesh *pm, ParameterInput *pin) {
//     pmd_ = new linearMGDriver(pm, pin);
//   }

//   void Solve(int stage, Real time) override final {
//     // assert(pmd_ && "linearMGDriverAdapter: backend MG driver is null");
//     pmd_->Solve(stage, time);
//   }

//   // linear solverたちの関数をそのまま呼び出せるようにする。
//   // つまり本来はplinsolver->pmd->Solve()と呼ぶところを、
//   // plinsolver->Solve()で済むようにする。

//   linearMGDriver* pmd_ = nullptr;
// private:
// };


#endif // LINEAR_SOLVER_LINEARMG_LINEARMG_HPP_
