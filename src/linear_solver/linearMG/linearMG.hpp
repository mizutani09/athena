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
  NewtonRaphson *pnr_;
  Real omega_;
  int fsmoother_;

};

class linearMGDriver: public MultigridDriver {
 public:
  linearMGDriver(Mesh *pm, ParameterInput *pin, NewtonRaphsonDriver *pnrd);
  ~linearMGDriver();
  void Solve(int stage, Real dt = 0.0) final;
  void SolveCoarsestGrid() final;
  void BeginTimeStep() { coefficient_hierarchy_cached_ = false; }
  void SetSmoothingOnly(bool value) { smoothing_only_ = value; }
  bool GetSmoothingOnly() const { return smoothing_only_; }
  void SetCoarseCorrectionScale(Real value) { coarse_corr_scale_ = value; }
  Real GetCoarseCorrectionScale() const { return coarse_corr_scale_; }
  void ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst,
                 AthenaArray<Real> &cbuf, const AthenaArray<bool> &ncoarse) final;

  friend class linearMG;
  friend class NewtonRaphsonDriver;

 private:
  bool EnforcePeriodicBoundaryConsistency() const final { return false; }
  void RestrictCoefficientCell(AthenaArray<Real> &dst,
                               const AthenaArray<Real> &src,
                               int ck, int cj, int ci,
                               int fk, int fj, int fi) final;
  void ReportDivergentCycle(Real old_defect, Real new_defect) final;
  bool RecoverDivergentCycle(Real old_defect, Real new_defect,
                             Real &recovered_defect) final;
  void ScaleFinestCorrection(Real ratio);
  void PrintCoefficientDiagnostics();

  LinearMGBoundaryTaskList *linmgtlist_;
  NewtonRaphsonDriver *pnrd_;
  Real omega_;
  int fsmoother_;
  bool fsteady_;
  bool cache_coefficient_hierarchy_;
  bool coefficient_hierarchy_cached_;
  int npostsolve_smooth_;
  int nsmoothing_only_sweeps_;
  bool direct_coarse_solve_;
  bool coarse_diagnostics_;
  int coarse_direct_max_cells_;
  int coarse_diagnostic_limit_;
  int coarse_diagnostic_count_;
  bool coefficient_diagnostics_;
  int coefficient_diagnostic_limit_;
  int coefficient_diagnostic_count_;
  bool vcycle_diagnostics_;
  bool vcycle_scale_diagnostics_;
  int vcycle_scale_levels_;
  bool vcycle_damping_;
  int vcycle_damping_levels_;
  Real vcycle_damping_max_factor_;
  int vcycle_diagnostic_limit_;
  int vcycle_diagnostic_count_;
};


#endif // LINEAR_SOLVER_LINEARMG_LINEARMG_HPP_
