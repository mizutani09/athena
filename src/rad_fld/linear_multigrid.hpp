#ifndef RAD_FLD_LINEAR_MULTIGRID_HPP
#define RAD_FLD_LINEAR_MULTIGRID_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_multigrid.hpp
//! \brief defines linearMG and linearMGDriver classes

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../multigrid/multigrid.hpp"
#include "Newton_Raphson.hpp"

class MeshBlock;
class ParameterInput;
class Coordinates;
class Multigrid;
class FLDBoundaryTaskList;


//! \class linearMG
//! \brief Multigrid FLD solver for each block

class linearMG : public Multigrid {
 public:
  linearMG(linearMGDriver *pmd, MeshBlock *pmb, ParameterInput *pin);
  ~linearMG();

  // void AddFLDSource(const AthenaArray<Real> &src, int ngh, Real dt);

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

  MeshBlock* pmy_block;

  AthenaArray<Real> u, A, B, C, D, RHS;

 private:
  Real omega_;
  int fsmoother_;
};


//! \class linearMGDriver
//! \brief linear Multigrid solver

class linearMGDriver : public MultigridDriver {
 public:
  linearMGDriver(Mesh *pm, ParameterInput *pin);
  ~linearMGDriver();
  void Solve(int stage, Real dt = 0.0) final;
  void ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst,
                 AthenaArray<Real> &cbuf, const AthenaArray<bool> &ncoarse) final;
  friend class linearMG;

 private:
  FLDBoundaryTaskList *fldtlist_;
  Real omega_;
  int fsmoother_;
  bool fsteady_;
};

#endif // RAD_FLD_LINEAR_MULTIGRID_HPP
