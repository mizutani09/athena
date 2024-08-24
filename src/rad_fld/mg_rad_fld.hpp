#ifndef RAD_FLD_MG_RAD_FLD_HPP_
#define RAD_FLD_MG_RAD_FLD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_rad_fld.hpp
//! \brief defines MGFLD class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../multigrid/multigrid.hpp"
#include "rad_fld.hpp"

class MeshBlock;
class ParameterInput;
class Coordinates;
class Multigrid;
class FLDBoundaryTaskList;


//! \class MGFLD
//! \brief Multigrid FLD solver for each block

class MGFLD : public Multigrid {
 public:
  MGFLD(MGFLDDriver *pmd, MeshBlock *pmb);
  ~MGFLD();

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
  void CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &coeff,
                int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) final;

  friend class MGFLDDriver;

 private:
  Real omega_;
  int fsmoother_;
};


//! \class MGFLDDriver
//! \brief Multigrid FLD solver

class MGFLDDriver : public MultigridDriver {
 public:
  MGFLDDriver(Mesh *pm, ParameterInput *pin);
  ~MGFLDDriver();
  void Solve(int stage, Real dt = 0.0) final;
  void ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst,
                 AthenaArray<Real> &cbuf, const AthenaArray<bool> &ncoarse) final;
  friend class MGFLD;

 private:
  FLDBoundaryTaskList *fldtlist_;
  Real omega_;
  int fsmoother_;
  bool fsteady_;
  Real dt_;
};

#endif // RAD_FLD_MG_RAD_FLD_HPP_
