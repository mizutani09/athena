#ifndef LINEAR_SOLVER_LINEAR_SOLVER_HPP_
#define LINEAR_SOLVER_LINEAR_SOLVER_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_solver.hpp
//! \brief Defines coefficient and matrix indices used by the FLD linear solver.

namespace linearSolver {
constexpr int NMATRIX = 7;
constexpr int GALERKIN_NMATRIX = 27;
constexpr int NCOEFF = 14;

enum CoeffIndex {
  // Diffusion coefficients multiplied by dt/dx^2.
  DCCF = 0,
  DXMF = 1,
  DXPF = 2,
  DYMF = 3,
  DYPF = 4,
  DZMF = 5,
  DZPF = 6,
  // Local coupling and q*v.grad(E) coefficients used directly.
  DCCS = 7,
  DXMS = 8,
  DXPS = 9,
  DYMS = 10,
  DYPS = 11,
  DZMS = 12,
  DZPS = 13
};

enum MatrixIndex {
  CCC = 0,
  CCM = 1,
  CCP = 2,
  CMC = 3,
  CPC = 4,
  MCC = 5,
  PCC = 6
};

// The production rediscretized and constant-transfer Galerkin operators use
// the first seven entries.  The remaining edge/corner offsets are retained
// solely for the explicit galerkin_27_experimental path.
constexpr int matrix_offset_k[GALERKIN_NMATRIX] = {
   0, 0, 0, 0, 0,-1, 1,
  -1,-1,-1,-1,-1,-1,-1,-1,
   0, 0, 0, 0,
   1, 1, 1, 1, 1, 1, 1, 1
};
constexpr int matrix_offset_j[GALERKIN_NMATRIX] = {
   0, 0, 0,-1, 1, 0, 0,
  -1,-1,-1, 0, 0, 1, 1, 1,
  -1,-1, 1, 1,
  -1,-1,-1, 0, 0, 1, 1, 1
};
constexpr int matrix_offset_i[GALERKIN_NMATRIX] = {
   0,-1, 1, 0, 0, 0, 0,
  -1, 0, 1,-1, 1,-1, 0, 1,
  -1, 1,-1, 1,
  -1, 0, 1,-1, 1,-1, 0, 1
};

inline int MatrixIndexFromOffset(int dk, int dj, int di) {
  for (int n = 0; n < GALERKIN_NMATRIX; ++n) {
    if (matrix_offset_k[n] == dk && matrix_offset_j[n] == dj
        && matrix_offset_i[n] == di) return n;
  }
  return -1;
}
}  // namespace linearSolver

#endif  // LINEAR_SOLVER_LINEAR_SOLVER_HPP_
