#ifndef LINEAR_SOLVER_LINEAR_SOLVER_HPP_
#define LINEAR_SOLVER_LINEAR_SOLVER_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_solver.hpp
//  \brief defines common objects in the linear solver

// C headers

// C++ headers
// #include <cstdint>  // std::int64_t
// #include <cstdio> // std::size_t
#include <iostream>
#include <cstddef>
// #include <unordered_map>
// #include <vector>

// Athena++ headers
// #include "../athena.hpp"
// #include "../athena_arrays.hpp"
// #include "../bvals/bvals_interfaces.hpp"
// #include "../globals.hpp"
// #include "../mesh/mesh.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;
class ParameterInput;
class Coordinates;
class LinearSolverBoundaryValues;
class linearMG;
class linearMGDriver;

namespace linearSolver {
  constexpr int NMATRIX=7, NCOEFF=14;
  enum CoeffIndex {
    // to be multiplied by dt/dx^2
    DCCF=0,
    DXMF=1,
    DXPF=2,
    DYMF=3,
    DYPF=4,
    DZMF=5,
    DZPF=6,
    // to be used directly
    DCCS=7,
    DXMS=8,
    DXPS=9,
    DYMS=10,
    DYPS=11,
    DZMS=12,
    DZPS=13
  };
  enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,};
  struct SolveStatus { bool ok; int iters; double relres; };
}

// class LinearSolver {
//  public:
//   // LinearSolver() = default;
//   virtual ~LinearSolver() = default;

//   MeshBlock* pmy_block;
//   linearMG *pmg;

//   // virtual void Solve(int step, Real dt = 0.0) = 0;
//   // LinearSolver *plinsolver = nullptr;
//   linearMGDriver *plinsolver_ = nullptr; // caution!

//   friend class NewtonRaphsonDriver;
//   friend class NewtonRaphson;
//   friend class linearMGDriver;
//   friend class linearMG;

//   inline LinearSolver() {
//     pmy_block = nullptr;
//     pmg = nullptr;
//     if (NRMGFLD_ENABLED) plinsolver_ = new linearMGDriver(nullptr, nullptr); // caution!
//     else {
//       std::stringstream msg;
//       msg << "### FATAL ERROR in LinearSolver::LinearSolver" << std::endl
//           << "linearMG must be enabled." << std::endl;
//       ATHENA_ERROR(msg);
//       return;
//     }
//   }

//   inline void Solve(int stage, Real dt) {
//     if (plinsolver_ == nullptr) {
//       std::stringstream msg;
//       msg << "### FATAL ERROR in LinearSolver::Solve" << std::endl
//           << "plinsolver_ is not allocated." << std::endl;
//       ATHENA_ERROR(msg);
//       return;
//     }
//     plinsolver_->Solve(stage, dt);
//   }
// };


struct SolveStatus { bool ok; int iters; double relres; };

class LinearSolver {
public:
  virtual ~LinearSolver() = default;
  // virtual void set_operator(/*Ax型*/) = 0;
  // virtual void set_tolerance(double rel, double abs) = 0;
  // virtual void set_maxiter(std::size_t it) = 0;
};

class LinearSolverDriver {
public:
  virtual ~LinearSolverDriver() = default;
  virtual void Solve(int stage, Real time) = 0;

};

#endif // LINEAR_SOLVER_LINEAR_SOLVER_HPP_