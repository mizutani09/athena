//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linearMG.cpp
//! \brief create linear multigrid solver for general equations

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../fld/fld.hpp"
#include "../../globals.hpp"
#include "../../hydro/hydro.hpp"
#include "../../linear_solver/linear_solver.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../task_list/linmg_task_list.hpp"
#include "linearMG.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;

namespace {
  AthenaArray<Real> *temp; // temporary data for the Jacobi iteration

Real RestrictFaceX1Minus(const AthenaArray<Real> &src, int v, int fi, int fj, int fk) {
  return 0.25*(src(v, fk,   fj,   fi) + src(v, fk,   fj+1, fi)
             + src(v, fk+1, fj,   fi) + src(v, fk+1, fj+1, fi));
}

Real RestrictFaceX1Plus(const AthenaArray<Real> &src, int v, int fi, int fj, int fk) {
  return 0.25*(src(v, fk,   fj,   fi+1) + src(v, fk,   fj+1, fi+1)
             + src(v, fk+1, fj,   fi+1) + src(v, fk+1, fj+1, fi+1));
}

Real RestrictFaceX2Minus(const AthenaArray<Real> &src, int v, int fi, int fj, int fk) {
  return 0.25*(src(v, fk,   fj, fi) + src(v, fk,   fj, fi+1)
             + src(v, fk+1, fj, fi) + src(v, fk+1, fj, fi+1));
}

Real RestrictFaceX2Plus(const AthenaArray<Real> &src, int v, int fi, int fj, int fk) {
  return 0.25*(src(v, fk,   fj+1, fi) + src(v, fk,   fj+1, fi+1)
             + src(v, fk+1, fj+1, fi) + src(v, fk+1, fj+1, fi+1));
}

Real RestrictFaceX3Minus(const AthenaArray<Real> &src, int v, int fi, int fj, int fk) {
  return 0.25*(src(v, fk, fj,   fi) + src(v, fk, fj,   fi+1)
             + src(v, fk, fj+1, fi) + src(v, fk, fj+1, fi+1));
}

Real RestrictFaceX3Plus(const AthenaArray<Real> &src, int v, int fi, int fj, int fk) {
  return 0.25*(src(v, fk+1, fj,   fi) + src(v, fk+1, fj,   fi+1)
             + src(v, fk+1, fj+1, fi) + src(v, fk+1, fj+1, fi+1));
}

int LinearMGMatrixCount(ParameterInput *pin) {
  const std::string coarse_operator = pin->GetOrAddString(
      "nrfld", "coarse_operator", "rediscretized");
  // The production Galerkin path uses volume-average restriction and
  // piecewise-constant prolongation.  R A_h P then preserves the seven-point
  // support exactly.  Keep the wider implementation available only behind an
  // explicitly experimental input value.
  return coarse_operator == "galerkin_27_experimental"
      ? linearSolver::GALERKIN_NMATRIX : linearSolver::NMATRIX;
}

inline Real ApplyMatrixRow(const AthenaArray<Real> &matrix,
                           const AthenaArray<Real> &u,
                           int k, int j, int i, int nmatrix) {
  Real value = 0.0;
  for (int n = 0; n < nmatrix; ++n) {
    value += matrix(n,k,j,i)*u(k + linearSolver::matrix_offset_k[n],
                               j + linearSolver::matrix_offset_j[n],
                               i + linearSolver::matrix_offset_i[n]);
  }
  return value;
}

inline Real ApplyMatrixOffDiagonal(const AthenaArray<Real> &matrix,
                                   const AthenaArray<Real> &u,
                                   int k, int j, int i, int nmatrix) {
  Real value = 0.0;
  for (int n = 1; n < nmatrix; ++n) {
    value += matrix(n,k,j,i)*u(k + linearSolver::matrix_offset_k[n],
                               j + linearSolver::matrix_offset_j[n],
                               i + linearSolver::matrix_offset_i[n]);
  }
  return value;
}
}

//----------------------------------------------------------------------------------------
//! \fn linearMGDriver::linearMGFLDDriver(Mesh *pm, ParameterInput *pin, NewtonRaphsonDriver *pnrd)
//! \brief linearMGDriver constructor

linearMGDriver::linearMGDriver(Mesh *pm, ParameterInput *pin, NewtonRaphsonDriver *pnrd)
    : MultigridDriver(pm, pm->LinearMGBoundaryFunction_,
                          pm->LinearMGCoeffBoundaryFunction_,
                          nullptr,
                          nullptr,
                          1, linearSolver::NCOEFF, LinearMGMatrixCount(pin)),
      pnrd_(pnrd), cache_coefficient_hierarchy_(true),
      coefficient_hierarchy_cached_(false), npostsolve_smooth_(0),
      nsmoothing_only_sweeps_(256), direct_coarse_solve_(true),
      coarse_diagnostics_(false), coarse_direct_max_cells_(64),
      coarse_diagnostic_limit_(4), coarse_diagnostic_count_(0),
      coefficient_diagnostics_(false), coefficient_diagnostic_limit_(1),
      coefficient_diagnostic_count_(0), uniform_pcg_(false),
      uniform_pcg_threshold_(1.0e-10), uniform_pcg_max_iterations_(10000),
      fgmres_(false), fgmres_threshold_(1.0e-8), fgmres_restart_(20),
      fgmres_max_iterations_(200), fgmres_mg_vcycles_(1),
      galerkin_coarse_operator_(false), experimental_27_point_galerkin_(false) {
  eps_ = pin->GetOrAddReal("nrfld", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("nrfld", "niteration", -1);
  max_vcycles_ = pin->GetOrAddInteger("nrfld", "linear_max_vcycles", 100);
  divergence_patience_ = pin->GetOrAddInteger(
      "nrfld", "linear_divergence_patience", 1);
  max_transient_growth_ = pin->GetOrAddReal(
      "nrfld", "linear_max_transient_growth", 2.0);
  ffas_ = pin->GetOrAddBoolean("nrfld", "fas", ffas_);
  omega_ = pin->GetOrAddReal("nrfld", "omega", 1.0);
  fsteady_ = pin->GetOrAddBoolean("nrfld", "steady", false);
  npresmooth_ = pin->GetOrAddReal("nrfld", "npresmooth", 2);
  npostsmooth_ = pin->GetOrAddReal("nrfld", "npostsmooth", 2);
  fshowdef_ = pin->GetOrAddBoolean("nrfld", "show_defect", fshowdef_);
  stage_diagnostics_ = pin->GetOrAddInteger(
      "nrfld", "diagnostic_verbosity", 0) >= 1;
  smoothing_only_ = pin->GetOrAddBoolean("nrfld", "smoothing_only", false);
  residual_minimizing_prolongation_ = pin->GetOrAddBoolean(
      "nrfld", "residual_minimizing_prolongation", false);
  uniform_pcg_ = pin->GetOrAddBoolean("nrfld", "uniform_pcg", false);
  uniform_pcg_threshold_ = pin->GetOrAddReal(
      "nrfld", "uniform_pcg_threshold", 1.0e-10);
  uniform_pcg_max_iterations_ = pin->GetOrAddInteger(
      "nrfld", "uniform_pcg_max_iterations", 10000);
  std::string linear_solver = pin->GetOrAddString("nrfld", "linear_solver", "mg");
  std::transform(linear_solver.begin(), linear_solver.end(), linear_solver.begin(),
                 ::tolower);
  if (uniform_pcg_ && linear_solver != "mg" && linear_solver != "uniform_pcg") {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "nrfld/uniform_pcg=true conflicts with nrfld/linear_solver="
        << linear_solver << "." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (uniform_pcg_) linear_solver = "uniform_pcg";
  if (linear_solver == "mg") {
    uniform_pcg_ = false;
  } else if (linear_solver == "uniform_pcg") {
    uniform_pcg_ = true;
  } else if (linear_solver == "fgmres") {
    fgmres_ = true;
    uniform_pcg_ = false;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "nrfld/linear_solver must be mg, fgmres, or uniform_pcg." << std::endl;
    ATHENA_ERROR(msg);
  }
  fgmres_threshold_ = pin->GetOrAddReal("nrfld", "fgmres_threshold", 1.0e-8);
  fgmres_restart_ = pin->GetOrAddInteger("nrfld", "fgmres_restart", 20);
  fgmres_max_iterations_ = pin->GetOrAddInteger(
      "nrfld", "fgmres_max_iterations", 200);
  fgmres_mg_vcycles_ = pin->GetOrAddInteger("nrfld", "fgmres_mg_vcycles", 1);
  const std::string coarse_operator = pin->GetString("nrfld", "coarse_operator");
  if (coarse_operator == "rediscretized") {
    galerkin_coarse_operator_ = false;
  } else if (coarse_operator == "galerkin") {
    galerkin_coarse_operator_ = true;
  } else if (coarse_operator == "galerkin_27_experimental") {
    galerkin_coarse_operator_ = true;
    experimental_27_point_galerkin_ = true;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "nrfld/coarse_operator must be rediscretized, galerkin, or "
        << "galerkin_27_experimental." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!(uniform_pcg_threshold_ > 0.0)
      || !std::isfinite(uniform_pcg_threshold_)
      || uniform_pcg_max_iterations_ <= 0
      || !(fgmres_threshold_ > 0.0) || !std::isfinite(fgmres_threshold_)
      || fgmres_restart_ <= 0 || fgmres_max_iterations_ <= 0
      || fgmres_mg_vcycles_ <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "Krylov thresholds must be finite and positive, and iteration, "
        << "restart, and MG V-cycle counts must be positive." << std::endl;
    ATHENA_ERROR(msg);
  }
  cache_coefficient_hierarchy_ =
      pin->GetOrAddBoolean("nrfld", "cache_coefficient_hierarchy", true);
  relative_defect_ = true;
  coarse_corr_scale_ = pin->GetOrAddReal("nrfld", "coarse_correction_scale", 1.0);
  stage_diagnostic_reference_scale_ = coarse_corr_scale_;
  npostsolve_smooth_ = pin->GetOrAddInteger(
      "nrfld", "post_mg_smooth", pm->multilevel ? 16 : 0);
  nsmoothing_only_sweeps_ = pin->GetOrAddInteger(
      "nrfld", "smoothing_only_sweeps", 256);
  const std::string coarse_solver = pin->GetOrAddString(
      "nrfld", "coarse_solver", "direct");
  if (coarse_solver == "direct") {
    direct_coarse_solve_ = true;
  } else if (coarse_solver == "legacy") {
    direct_coarse_solve_ = false;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "nrfld/coarse_solver must be direct or legacy." << std::endl;
    ATHENA_ERROR(msg);
  }
  coarse_direct_max_cells_ = pin->GetOrAddInteger(
      "nrfld", "coarse_direct_max_cells", 64);
  coarse_diagnostics_ = pin->GetOrAddBoolean(
      "nrfld", "coarse_diagnostics", false);
  coarse_diagnostic_limit_ = pin->GetOrAddInteger(
      "nrfld", "coarse_diagnostic_limit", 4);
  coefficient_diagnostics_ = pin->GetOrAddBoolean(
      "nrfld", "coefficient_diagnostics", false);
  handoff_diagnostics_ = pin->GetOrAddBoolean("nrfld", "handoff_diagnostics", false);
  handoff_diagnostic_limit_ = pin->GetOrAddInteger("nrfld", "handoff_diagnostic_limit", 4);
  if (handoff_diagnostic_limit_ < 0 || (handoff_diagnostics_ && Globals::nranks != 1)) {
    std::stringstream msg;
    msg << "handoff_diagnostics requires one MPI rank and a non-negative limit.";
    ATHENA_ERROR(msg);
  }
  coefficient_diagnostic_limit_ = pin->GetOrAddInteger(
      "nrfld", "coefficient_diagnostic_limit", 1);
  if (coarse_direct_max_cells_ <= 0 || coarse_diagnostic_limit_ < 0
      || coefficient_diagnostic_limit_ < 0 || divergence_patience_ <= 0
      || max_transient_growth_ < 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "coarse_direct_max_cells and linear_divergence_patience must be positive, "
        << "diagnostic limits must be non-negative, and "
        << "linear_max_transient_growth must be >= 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (nsmoothing_only_sweeps_ <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "The nrfld/smoothing_only_sweeps parameter must be positive." << std::endl;
    ATHENA_ERROR(msg);
  }
  std::string smoother = pin->GetOrAddString("nrfld", "smoother", "jacobi-rb");
  symmetric_rb_ = pin->GetOrAddBoolean("nrfld", "symmetric_rb_sweeps", false);
  if (smoother == "jacobi-rb") {
    fsmoother_ = 1;
    redblack_ = true;
  } else if (smoother == "jacobi-double") {
    fsmoother_ = 0;
    redblack_ = true;
  } else { // jacobi
    fsmoother_ = 0;
    redblack_ = false;
  }
  std::string prol = pin->GetOrAddString("nrfld", "prolongation", "trilinear");
  if (prol == "tricubic")
    fprolongation_ = 1;
  else if (prol == "constant" || prol == "inject")
    fprolongation_ = 2;
  if (galerkin_coarse_operator_ && !experimental_27_point_galerkin_
      && fprolongation_ != 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "nrfld/coarse_operator=galerkin requires "
        << "nrfld/prolongation=constant.  The existing volume-average restriction "
        << "is the scaled transpose of constant prolongation; trilinear/tricubic "
        << "would form a non-variational transfer pair and is not stable for the "
        << "strong-diffusion RSG operator." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (experimental_27_point_galerkin_ && fprolongation_ == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "nrfld/coarse_operator=galerkin_27_experimental supports trilinear "
        << "or constant prolongation, not tricubic.  Its retained RAP builder "
        << "contains trilinear weights only." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (galerkin_coarse_operator_ && pm->multilevel) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "The Galerkin hierarchy currently supports uniform MeshBlock "
        << "topology only; AMR/SMR remains outside WOR-53." << std::endl;
    ATHENA_ERROR(msg);
  }

  std::string m = pin->GetOrAddString("nrfld", "mgmode", "none");
  std::transform(m.begin(), m.end(), m.begin(), ::tolower);
  if (m == "fmg") {
    mode_ = 0;
  } else if (m == "mgi") {
    mode_ = 1; // Iterative
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGFLDDriver::linearMGFLDDriver" << std::endl
        << "The \"mgmode\" parameter in the <nrfld> block is invalid." << std::endl
        << "FMG: Full Multigrid + Multigrid iteration (default)" << std::endl
        << "MGI: Multigrid Iteration" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (eps_ < 0.0 && niter_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGFLDDriver::linearMGFLDDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <nrfld> block." << std::endl
        << "Set \"threshold = 0.0\" without \"niteration\" for automatic "
        << "convergence control," << std::endl
        << "or set \"threshold = 0.0\" together with \"niteration\" to use "
        << "a fixed number of V-cycles." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (max_vcycles_ <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGFLDDriver::linearMGFLDDriver" << std::endl
        << "The nrfld/linear_max_vcycles parameter must be positive." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (Globals::my_rank == 0) {
    std::cout << "[LINEAR_MG_CONFIG] threshold=" << eps_
              << " max_vcycles=" << max_vcycles_
              << " divergence_patience=" << divergence_patience_
              << " max_transient_growth=" << max_transient_growth_
              << " solver=" << linear_solver
              << " coarse_operator=" << coarse_operator << std::endl;
    if (experimental_27_point_galerkin_) {
      std::cout << "[LINEAR_MG_EXPERIMENTAL] 27-point RAP storage/transfer path; "
                << "not the validated RSG production configuration." << std::endl;
    }
  }
  // The linear solve acts on Newton corrections, so its physical boundary
  // condition need not be the same type as the full-state hydro/NR boundary.
  // In particular, a prescribed FLD flux has a homogeneous zero-gradient
  // correction.  Honor the dedicated input values instead of silently
  // replacing every non-periodic boundary with a zero-value correction.
  mg_mesh_bcs_[inner_x1] = GetMGBoundaryFlag(
      pin->GetString("mesh", "ix1_bc") == "periodic" ? "periodic" :
      pin->GetOrAddString("nrfld", "linearMG_ix1_bc", "zerofixed"));
  mg_mesh_bcs_[outer_x1] = GetMGBoundaryFlag(
      pin->GetString("mesh", "ox1_bc") == "periodic" ? "periodic" :
      pin->GetOrAddString("nrfld", "linearMG_ox1_bc", "zerofixed"));
  mg_mesh_bcs_[inner_x2] = GetMGBoundaryFlag(
      pin->GetString("mesh", "ix2_bc") == "periodic" ? "periodic" :
      pin->GetOrAddString("nrfld", "linearMG_ix2_bc", "zerofixed"));
  mg_mesh_bcs_[outer_x2] = GetMGBoundaryFlag(
      pin->GetString("mesh", "ox2_bc") == "periodic" ? "periodic" :
      pin->GetOrAddString("nrfld", "linearMG_ox2_bc", "zerofixed"));
  mg_mesh_bcs_[inner_x3] = GetMGBoundaryFlag(
      pin->GetString("mesh", "ix3_bc") == "periodic" ? "periodic" :
      pin->GetOrAddString("nrfld", "linearMG_ix3_bc", "zerofixed"));
  mg_mesh_bcs_[outer_x3] = GetMGBoundaryFlag(
      pin->GetString("mesh", "ox3_bc") == "periodic" ? "periodic" :
      pin->GetOrAddString("nrfld", "linearMG_ox3_bc", "zerofixed"));

  bool has_physical_boundary = false;
  for (int face = 0; face < 6; ++face) {
    const BoundaryFlag flag = mg_mesh_bcs_[face];
    has_physical_boundary |= flag != BoundaryFlag::block
                          && flag != BoundaryFlag::periodic;
  }
  const bool galerkin_zero_physical_boundaries = pin->GetOrAddBoolean(
      "nrfld", "galerkin_zero_physical_boundaries", false);
  if (galerkin_coarse_operator_ && has_physical_boundary
      && !galerkin_zero_physical_boundaries) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "The current Galerkin boundary elimination requires zero correction "
        << "values in physical ghost-cell centres.  Set "
        << "nrfld/galerkin_zero_physical_boundaries=true only when every enrolled "
        << "non-periodic linear-MG boundary callback enforces that condition."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  CheckBoundaryFunctions();
  fsubtract_average_ = false; // override the subtract average flag

  mgtlist_ = new MultigridTaskList(this);

  // Allocate the root multigrid
  mgroot_ = new linearMG(this, nullptr, pin, nullptr);

  linmgtlist_ = new LinearMGBoundaryTaskList(pin, pm);

  int nth = 1;
#ifdef OPENMP_PARALLEL
  nth = omp_get_max_threads();
#endif
  temp = new AthenaArray<Real>[nth];
  int nx = std::max(pmy_mesh_->block_size.nx1, pmy_mesh_->nrbx1) + 2*mgroot_->ngh_;
  int ny = std::max(pmy_mesh_->block_size.nx2, pmy_mesh_->nrbx2) + 2*mgroot_->ngh_;
  int nz = std::max(pmy_mesh_->block_size.nx3, pmy_mesh_->nrbx3) + 2*mgroot_->ngh_;
  for (int n = 0; n < nth; ++n)
    temp[n].NewAthenaArray(nz, ny, nx);
}


//----------------------------------------------------------------------------------------
//! \brief Restrict the FLD stencil coefficients while preserving their geometry

void linearMGDriver::RestrictCoefficientCell(AthenaArray<Real> &dst,
                                             const AthenaArray<Real> &src,
                                             int ck, int cj, int ci,
                                             int fk, int fj, int fi) {
  const Real dxm_f = RestrictFaceX1Minus(src, linearSolver::DXMF, fi, fj, fk);
  const Real dxp_f = RestrictFaceX1Plus(src, linearSolver::DXPF, fi, fj, fk);
  const Real dym_f = RestrictFaceX2Minus(src, linearSolver::DYMF, fi, fj, fk);
  const Real dyp_f = RestrictFaceX2Plus(src, linearSolver::DYPF, fi, fj, fk);
  const Real dzm_f = RestrictFaceX3Minus(src, linearSolver::DZMF, fi, fj, fk);
  const Real dzp_f = RestrictFaceX3Plus(src, linearSolver::DZPF, fi, fj, fk);

  dst(linearSolver::DXMF, ck, cj, ci) = dxm_f;
  dst(linearSolver::DXPF, ck, cj, ci) = dxp_f;
  dst(linearSolver::DYMF, ck, cj, ci) = dym_f;
  dst(linearSolver::DYPF, ck, cj, ci) = dyp_f;
  dst(linearSolver::DZMF, ck, cj, ci) = dzm_f;
  dst(linearSolver::DZPF, ck, cj, ci) = dzp_f;
  // The diffusion off-diagonals are negative. Rebuild the diagonal so each
  // coarse diffusion row retains a zero row sum.
  dst(linearSolver::DCCF, ck, cj, ci) =
      -(dxm_f + dxp_f + dym_f + dyp_f + dzm_f + dzp_f);

  // These off-diagonals discretize q*v.grad(E) and scale as 1/dx.  A coarse
  // cell has twice the spacing, so its restricted coefficient is halved.
  dst(linearSolver::DXMS, ck, cj, ci) =
      0.5*RestrictFaceX1Minus(src, linearSolver::DXMS, fi, fj, fk);
  dst(linearSolver::DXPS, ck, cj, ci) =
      0.5*RestrictFaceX1Plus(src, linearSolver::DXPS, fi, fj, fk);
  dst(linearSolver::DYMS, ck, cj, ci) =
      0.5*RestrictFaceX2Minus(src, linearSolver::DYMS, fi, fj, fk);
  dst(linearSolver::DYPS, ck, cj, ci) =
      0.5*RestrictFaceX2Plus(src, linearSolver::DYPS, fi, fj, fk);
  dst(linearSolver::DZMS, ck, cj, ci) =
      0.5*RestrictFaceX3Minus(src, linearSolver::DZMS, fi, fj, fk);
  dst(linearSolver::DZPS, ck, cj, ci) =
      0.5*RestrictFaceX3Plus(src, linearSolver::DZPS, fi, fj, fk);

  // DCCS contains both the local matter-radiation Schur diagonal and the
  // center part of q*v.grad(E).  The latter is minus the sum of the six *S
  // off-diagonals, and must scale as 1/dx just like those off-diagonals.
  // Restricting the combined DCCS as a cell scalar would leave its drift
  // contribution unscaled while halving the face contributions above.  That
  // destroys the drift row sum and creates a spurious coarse-grid reaction
  // term which compounds on deep AMR hierarchies.  Separate the local part
  // on the fine grid, restrict it, then rebuild the coarse center drift from
  // the restricted faces.
  Real local_diagonal = RestrictOne(src, linearSolver::DCCS, fi, fj, fk);
  local_diagonal += RestrictOne(src, linearSolver::DXMS, fi, fj, fk)
                  + RestrictOne(src, linearSolver::DXPS, fi, fj, fk)
                  + RestrictOne(src, linearSolver::DYMS, fi, fj, fk)
                  + RestrictOne(src, linearSolver::DYPS, fi, fj, fk)
                  + RestrictOne(src, linearSolver::DZMS, fi, fj, fk)
                  + RestrictOne(src, linearSolver::DZPS, fi, fj, fk);
  const Real coarse_drift_offdiag =
      dst(linearSolver::DXMS, ck, cj, ci) + dst(linearSolver::DXPS, ck, cj, ci)
    + dst(linearSolver::DYMS, ck, cj, ci) + dst(linearSolver::DYPS, ck, cj, ci)
    + dst(linearSolver::DZMS, ck, cj, ci) + dst(linearSolver::DZPS, ck, cj, ci);
  dst(linearSolver::DCCS, ck, cj, ci) = local_diagonal - coarse_drift_offdiag;
}


//----------------------------------------------------------------------------------------
//! \fn void linearMGDriver::SolveCoarsestGrid()
//! \brief Solve the small root-grid system directly, including boundary aliases.
//!
//! A one-cell periodic diffusion grid is a particularly important case.  All six
//! stencil neighbors refer to the center unknown, so the diffusion row sum cancels.
//! Treating those neighbors as lagged Jacobi values instead gives a convergence factor
//! arbitrarily close to one in diffusion-dominated NR-FLD systems.  Probing the actual
//! boundary-aware operator also handles two-cell periodic aliases without special cases.
void linearMGDriver::SolveCoarsestGrid() {
  if (!direct_coarse_solve_ || ffas_) {
    MultigridDriver::SolveCoarsestGrid();
    return;
  }

  linearMG *root = static_cast<linearMG*>(mgroot_);
  const int lev = root->current_level_;
  const int ll = root->nlevel_ - 1 - lev;
  const int nx = root->size_.nx1 >> ll;
  const int ny = root->size_.nx2 >> ll;
  const int nz = root->size_.nx3 >> ll;
  const int ncells = nx*ny*nz;
  if (ncells > coarse_direct_max_cells_) {
    MultigridDriver::SolveCoarsestGrid();
    return;
  }

  const int ngh = root->ngh_;
  AthenaArray<Real> &u = root->u_[lev];
  const AthenaArray<Real> &src = root->src_[lev];
  const AthenaArray<Real> &matrix = root->matrix_[lev];
  std::vector<Real> a(static_cast<std::size_t>(ncells)*ncells, 0.0);
  std::vector<Real> affine(ncells), original(static_cast<std::size_t>(nvar_)*ncells);
  std::vector<Real> rhs(ncells), x(ncells);

  auto index = [nx, ny](int k, int j, int i) {
    return (k*ny + j)*nx + i;
  };
  auto apply_row = [&](int k, int j, int i, int v) {
    const int kk = ngh + k, jj = ngh + j, ii = ngh + i;
    Real value = 0.0;
    for (int n = 0; n < nmatrix_; ++n) {
      value += matrix(n,kk,jj,ii)*u(v,kk + linearSolver::matrix_offset_k[n],
                                      jj + linearSolver::matrix_offset_j[n],
                                      ii + linearSolver::matrix_offset_i[n]);
    }
    return value;
  };

  for (int v = 0; v < nvar_; ++v)
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          original[static_cast<std::size_t>(v)*ncells + index(k,j,i)]
              = u(v,ngh+k,ngh+j,ngh+i);

  // Probe columns of the true coarse operator.  Subtract the zero-state response so
  // that affine physical boundary data are moved to the right-hand side.  This also
  // handles multiple periodic ghosts identifying the same unknown on one- and
  // two-cell grids without special cases.
  u.ZeroClear();
  root->pmgbval->ApplyPhysicalBoundaries(0, false);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        affine[index(k,j,i)] = apply_row(k,j,i,0);
  for (int col = 0; col < ncells; ++col) {
    u.ZeroClear();
    const int ci = col % nx;
    const int cj = (col/nx) % ny;
    const int ck = col/(nx*ny);
    u(0,ngh+ck,ngh+cj,ngh+ci) = 1.0;
    root->pmgbval->ApplyPhysicalBoundaries(0, false);
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          a[static_cast<std::size_t>(index(k,j,i))*ncells + col]
              = apply_row(k,j,i,0) - affine[index(k,j,i)];
  }

  const bool print_diag = coarse_diagnostics_
                       && coarse_diagnostic_count_ < coarse_diagnostic_limit_
                       && Globals::my_rank == 0;
  if (print_diag) {
    Real amin = std::numeric_limits<Real>::max();
    Real amax = 0.0;
    for (int row = 0; row < ncells; ++row) {
      const Real diagonal = std::abs(a[static_cast<std::size_t>(row)*ncells + row]);
      amin = std::min(amin, diagonal);
      amax = std::max(amax, diagonal);
    }
    std::cout << "[NRFLD coarse] cells=" << nx << "x" << ny << "x" << nz
              << " direct=1 diag_min=" << amin << " diag_max=" << amax;
    if (ncells == 1) {
      const int q = ngh;
      const Real split_diag = matrix(linearSolver::CCC,q,q,q);
      const Real effective_diag = a[0];
      const Real jacobi_factor = std::abs(1.0 - omega_*effective_diag/split_diag);
      std::cout << " split_diag=" << split_diag
                << " effective_diag=" << effective_diag
                << " predicted_legacy_factor=" << jacobi_factor;
    }
    std::cout << std::endl;
    ++coarse_diagnostic_count_;
  }

  // Partial-pivoted Gaussian elimination.  The matrix is shared by all scalar
  // components, so factor and solve a copy for each component; ncells is deliberately
  // capped at a small value and this path is negligible compared with a fine-grid V-cycle.
  for (int v = 0; v < nvar_; ++v) {
    std::vector<Real> work = a;
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          rhs[index(k,j,i)] = src(v,ngh+k,ngh+j,ngh+i) - affine[index(k,j,i)];

    for (int p = 0; p < ncells; ++p) {
      int pivot = p;
      Real pivot_abs = std::abs(work[static_cast<std::size_t>(p)*ncells + p]);
      for (int row = p + 1; row < ncells; ++row) {
        const Real candidate = std::abs(work[static_cast<std::size_t>(row)*ncells + p]);
        if (candidate > pivot_abs) {
          pivot = row;
          pivot_abs = candidate;
        }
      }
      Real row_scale = 0.0;
      for (int col = p; col < ncells; ++col)
        row_scale = std::max(row_scale,
            std::abs(work[static_cast<std::size_t>(pivot)*ncells + col]));
      if (!std::isfinite(pivot_abs) || !(row_scale > 0.0)
          || !(pivot_abs > std::numeric_limits<Real>::epsilon()*row_scale)) {
        if (Globals::my_rank == 0)
          std::cout << "### Warning in linearMGDriver::SolveCoarsestGrid\n"
                    << "Direct coarse matrix is singular; using legacy smoother."
                    << std::endl;
        for (int vv = 0; vv < nvar_; ++vv)
          for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
              for (int i = 0; i < nx; ++i)
                u(vv,ngh+k,ngh+j,ngh+i) =
                    original[static_cast<std::size_t>(vv)*ncells + index(k,j,i)];
        root->pmgbval->ApplyPhysicalBoundaries(0, false);
        MultigridDriver::SolveCoarsestGrid();
        return;
      }
      if (pivot != p) {
        for (int col = p; col < ncells; ++col)
          std::swap(work[static_cast<std::size_t>(p)*ncells + col],
                    work[static_cast<std::size_t>(pivot)*ncells + col]);
        std::swap(rhs[p], rhs[pivot]);
      }
      const Real diag = work[static_cast<std::size_t>(p)*ncells + p];
      for (int row = p + 1; row < ncells; ++row) {
        const Real factor = work[static_cast<std::size_t>(row)*ncells + p]/diag;
        work[static_cast<std::size_t>(row)*ncells + p] = 0.0;
        for (int col = p + 1; col < ncells; ++col)
          work[static_cast<std::size_t>(row)*ncells + col]
              -= factor*work[static_cast<std::size_t>(p)*ncells + col];
        rhs[row] -= factor*rhs[p];
      }
    }
    for (int row = ncells - 1; row >= 0; --row) {
      Real value = rhs[row];
      for (int col = row + 1; col < ncells; ++col)
        value -= work[static_cast<std::size_t>(row)*ncells + col]*x[col];
      x[row] = value/work[static_cast<std::size_t>(row)*ncells + row];
    }
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
          u(v,ngh+k,ngh+j,ngh+i) = x[index(k,j,i)];
  }
  root->pmgbval->ApplyPhysicalBoundaries(0, false);
}


//----------------------------------------------------------------------------------------
//! \fn linearMGDriver::~linearMGDriver()
//! \brief linearMGDriver destructor

linearMGDriver::~linearMGDriver() {
  const bool trace_dtor = (std::getenv("ATHENA_TRACE_DTOR") != nullptr);
  if (trace_dtor) std::cout << "[DTOR] linearMGDriver delete linmgtlist_" << std::endl;
  delete linmgtlist_;
  if (trace_dtor) std::cout << "[DTOR] linearMGDriver delete mgroot_" << std::endl;
  delete mgroot_;
  if (trace_dtor) std::cout << "[DTOR] linearMGDriver delete mgtlist_" << std::endl;
  delete mgtlist_;
  if (trace_dtor) std::cout << "[DTOR] linearMGDriver delete[] temp" << std::endl;
  delete [] temp;
}

void linearMGDriver::SynchronizeCorrectionBoundaries(int stage) {
  linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
}


//----------------------------------------------------------------------------------------
//! \brief Report coefficient contrast and coarse-operator consistency by MG level.

void linearMGDriver::PrintCoefficientDiagnostics() {
  auto report_level = [&](const char *kind, int lev, bool root_grid) {
    Real vmin[9] = {std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max()};
    Real vmax[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0};
    Real sums[4] = {0.0, 0.0, 0.0, 0.0};
    long long invalid_rows[2] = {0, 0};
    int nx = 0, ny = 0, nz = 0;
    if (!root_grid) {
      const int ll = nmblevel_ - 1 - lev;
      nx = pmy_mesh_->block_size.nx1 >> ll;
      ny = pmy_mesh_->block_size.nx2 >> ll;
      nz = pmy_mesh_->block_size.nx3 >> ll;
    }

    auto scan = [&](linearMG *pmg) {
      const int ll = pmg->nlevel_ - 1 - lev;
      nx = pmg->size_.nx1 >> ll;
      ny = pmg->size_.nx2 >> ll;
      nz = pmg->size_.nx3 >> ll;
      const int is = pmg->ngh_, js = pmg->ngh_, ks = pmg->ngh_;
      const int ie = is + nx - 1, je = js + ny - 1, ke = ks + nz - 1;
      const AthenaArray<Real> &c = pmg->coeff_[lev];
      const AthenaArray<Real> &a = pmg->matrix_[lev];
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            for (int n = 0; n < ncoeff_; ++n) {
              const Real value = c(n,k,j,i);
              sums[0] += std::abs(value);
              sums[1] += value*value;
            }
            for (int n = 0; n < 7; ++n) {
              const Real value = a(n,k,j,i);
              sums[2] += std::abs(value);
              sums[3] += value*value;
            }
            const Real dccs = c(linearSolver::DCCS,k,j,i);
            const Real diag = a(linearSolver::CCC,k,j,i);
            const Real diffusion_diag = std::abs(diag - dccs);
            const Real reaction_fraction = std::abs(dccs)
                                         / std::max(std::abs(diag), TINY_NUMBER);
            const Real rowsum = diag
                + a(linearSolver::CCM,k,j,i) + a(linearSolver::CCP,k,j,i)
                + a(linearSolver::CMC,k,j,i) + a(linearSolver::CPC,k,j,i)
                + a(linearSolver::MCC,k,j,i) + a(linearSolver::PCC,k,j,i);
            const Real expected_rowsum = dccs
                + c(linearSolver::DXMS,k,j,i) + c(linearSolver::DXPS,k,j,i)
                + c(linearSolver::DYMS,k,j,i) + c(linearSolver::DYPS,k,j,i)
                + c(linearSolver::DZMS,k,j,i) + c(linearSolver::DZPS,k,j,i);
            const Real local_schur_diagonal = expected_rowsum;
            const Real drift_center = dccs - local_schur_diagonal;
            const Real drift_fraction = std::abs(drift_center)
                                      / std::max(std::abs(diag), TINY_NUMBER);
            const Real row_error = std::abs(rowsum - expected_rowsum)
                                 / std::max(std::abs(diag), TINY_NUMBER);
            const Real offdiag_abs =
                std::abs(a(linearSolver::CCM,k,j,i))
              + std::abs(a(linearSolver::CCP,k,j,i))
              + std::abs(a(linearSolver::CMC,k,j,i))
              + std::abs(a(linearSolver::CPC,k,j,i))
              + std::abs(a(linearSolver::MCC,k,j,i))
              + std::abs(a(linearSolver::PCC,k,j,i));
            const Real dominance = (diag - offdiag_abs)
                                 / std::max(std::abs(diag), TINY_NUMBER);
            vmin[0] = std::min(vmin[0], std::abs(dccs));
            vmin[1] = std::min(vmin[1], diffusion_diag);
            vmin[3] = std::min(vmin[3], std::abs(diag));
            vmin[4] = std::min(vmin[4], reaction_fraction);
            vmin[5] = std::min(vmin[5], dccs);
            vmin[6] = std::min(vmin[6], diag);
            vmin[7] = std::min(vmin[7], dominance);
            vmin[8] = std::min(vmin[8], local_schur_diagonal);
            vmax[0] = std::max(vmax[0], std::abs(dccs));
            vmax[1] = std::max(vmax[1], diffusion_diag);
            vmax[3] = std::max(vmax[3], std::abs(diag));
            vmax[4] = std::max(vmax[4], reaction_fraction);
            vmax[5] = std::max(vmax[5], row_error);
            vmax[7] = std::max(vmax[7], dccs);
            vmax[8] = std::max(vmax[8], diag);
            vmax[9] = std::max(vmax[9], local_schur_diagonal);
            vmax[10] = std::max(vmax[10], drift_fraction);
            if (!(diag > 0.0) || !std::isfinite(diag)) ++invalid_rows[0];

            const int face_ids[6] = {linearSolver::CCM, linearSolver::CCP,
                                     linearSolver::CMC, linearSolver::CPC,
                                     linearSolver::MCC, linearSolver::PCC};
            for (int n = 0; n < 6; ++n) {
              const Real face = std::abs(a(face_ids[n],k,j,i));
              if (face > 0.0) vmin[2] = std::min(vmin[2], face);
              vmax[2] = std::max(vmax[2], face);
              if (a(face_ids[n],k,j,i) > 0.0
                  || !std::isfinite(a(face_ids[n],k,j,i))) ++invalid_rows[1];
            }
            auto update_asymmetry = [&](Real lhs, Real rhs) {
              vmax[6] = std::max(vmax[6], std::abs(lhs-rhs)
                  /std::max({std::abs(lhs), std::abs(rhs), TINY_NUMBER}));
            };
            if (i < ie)
              update_asymmetry(a(linearSolver::CCP,k,j,i),
                               a(linearSolver::CCM,k,j,i+1));
            if (j < je)
              update_asymmetry(a(linearSolver::CPC,k,j,i),
                               a(linearSolver::CMC,k,j+1,i));
            if (k < ke)
              update_asymmetry(a(linearSolver::PCC,k,j,i),
                               a(linearSolver::MCC,k+1,j,i));
          }
        }
      }
    };

    if (root_grid) {
      scan(static_cast<linearMG*>(mgroot_));
    } else {
      for (Multigrid *base : vmg_) scan(static_cast<linearMG*>(base));
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, vmin, 9, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, vmax, 11, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, invalid_rows, 2, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if (Globals::my_rank == 0) {
      const Real face_contrast = vmax[2]/std::max(vmin[2], TINY_NUMBER);
      std::cout << "[NRFLD coefficients] solve=" << coefficient_diagnostic_count_
                << " kind=" << kind << " level=" << lev
                << " cells=" << nx << "x" << ny << "x" << nz
                << " DCCS=[" << vmin[0] << "," << vmax[0] << "]"
                << " diffusion_diag=[" << vmin[1] << "," << vmax[1] << "]"
                << " face_abs=[" << vmin[2] << "," << vmax[2] << "]"
                << " face_contrast=" << face_contrast
                << " diag=[" << vmin[3] << "," << vmax[3] << "]"
                << " reaction_fraction=[" << vmin[4] << "," << vmax[4] << "]"
                << " signed_DCCS=[" << vmin[5] << "," << vmax[7] << "]"
                << " signed_diag=[" << vmin[6] << "," << vmax[8] << "]"
                << " local_schur=[" << vmin[8] << "," << vmax[9] << "]"
                << " drift_fraction_max=" << vmax[10]
                << " dominance_min=" << vmin[7]
                << " nonpositive_diag=" << invalid_rows[0]
                << " positive_or_nonfinite_offdiag=" << invalid_rows[1]
                << " row_error_max=" << vmax[5]
                << " face_asymmetry_max=" << vmax[6] << std::endl;
      std::cout << "[NRFLD coefficient fingerprint] solve="
                << coefficient_diagnostic_count_
                << " kind=" << kind << " level=" << lev
                << " coeff_l1=" << sums[0]
                << " coeff_l2=" << std::sqrt(sums[1])
                << " matrix_l1=" << sums[2]
                << " matrix_l2=" << std::sqrt(sums[3]) << std::endl;
    }
  };

  for (int lev = nmblevel_ - 1; lev >= 0; --lev)
    report_level("meshblock", lev, false);
  linearMG *root = static_cast<linearMG*>(mgroot_);
  for (int lev = root->nlevel_ - 1; lev >= 0; --lev)
    report_level("root", lev, true);
}


//----------------------------------------------------------------------------------------
//! \fn linearMG::linearMG(linearMGDriver *pmd, MeshBlock *pmb, ParameterInput *pin, NewtonRaphson *pnr)
//! \brief linearMG constructor

linearMG::linearMG(linearMGDriver *pmd, MeshBlock *pmb, ParameterInput *pin, NewtonRaphson *pnr)
  : Multigrid(pmd, pmb, 1),
  // pmd_(pmd),
  omega_(pmd->omega_), fsmoother_(pmd->fsmoother_),
  pnr_(pnr) {
  btype = btypef = BoundaryQuantity::mg;
  pmgbval = new MGBoundaryValues(this, mg_block_bcs_);
}


//----------------------------------------------------------------------------------------
//! \fn linearMG::~linearMG()
//! \brief linearMG deconstructor

linearMG::~linearMG() {
  const bool trace_dtor = (std::getenv("ATHENA_TRACE_DTOR") != nullptr);
  if (trace_dtor) {
    std::cout << "[DTOR] linearMG block="
              << (pmy_block_ ? pmy_block_->gid : -1)
              << " delete pmgbval ptr=" << pmgbval << std::endl;
  }
  delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void linearMGDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

//----------------------------------------------------------------------------------------
//! \brief Reference PCG solve for a single uniform MeshBlock.
//!
//! This bypasses the coarse hierarchy to distinguish a fine-grid Schur/Jacobian error
//! from a multigrid-preconditioner failure in strongly diffusive uniform-grid problems.
void linearMGDriver::SolveUniformPCG(int stage) {
  if (nranks_ != 1 || pmy_mesh_->nbtotal != 1 || pmy_mesh_->nblocal != 1
      || pmy_mesh_->multilevel || vmg_.size() != 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::SolveUniformPCG" << std::endl
        << "nrfld/uniform_pcg requires one MPI rank, one MeshBlock, and no AMR/SMR."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  linearMG *pmg = static_cast<linearMG*>(vmg_.front());
  NewtonRaphson *pnr = pmg->pnr_;
  MeshBlock *pmb = pnr->pmy_block_;
  if (pmb->prfld->MixedFrameCoupled()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::SolveUniformPCG" << std::endl
        << "nrfld/uniform_pcg requires mixed-frame coupling to be off." << std::endl;
    ATHENA_ERROR(msg);
  }

  AthenaArray<Real> &delta = pnr->delta_u_;
  const AthenaArray<Real> &src = pnr->src_;
  const AthenaArray<Real> &coeff = pnr->coeff_;
  AthenaArray<Real> x(delta), r(delta), z(delta), p(delta), ap(delta);
  x.ZeroClear();
  r.ZeroClear();
  z.ZeroClear();
  p.ZeroClear();
  ap.ZeroClear();

  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;
  const Real fac = dt_/SQR(pmb->pcoord->dx1f(is));
  long double source2 = 0.0L;
  long double rz = 0.0L;
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        const Real diag = fac*coeff(linearSolver::DCCF,k,j,i)
                        + coeff(linearSolver::DCCS,k,j,i);
        if (!(diag > 0.0) || !std::isfinite(diag)) {
          std::stringstream msg;
          msg << "### FATAL ERROR in linearMGDriver::SolveUniformPCG" << std::endl
              << "Nonpositive or nonfinite diagonal at (k,j,i)=("
              << k << "," << j << "," << i << ")." << std::endl;
          ATHENA_ERROR(msg);
        }
        r(k,j,i) = src(k,j,i);
        z(k,j,i) = r(k,j,i)/diag;
        p(k,j,i) = z(k,j,i);
        source2 += static_cast<long double>(src(k,j,i))*src(k,j,i);
        rz += static_cast<long double>(r(k,j,i))*z(k,j,i);
      }
  const long double source_norm = std::sqrt(
      std::max(source2, std::numeric_limits<long double>::min()));
  Real relative_residual = static_cast<Real>(std::sqrt(source2)/source_norm);
  int iterations = 0;
  bool breakdown = false;
  while (relative_residual > uniform_pcg_threshold_
         && iterations < uniform_pcg_max_iterations_) {
    delta = p;
    linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
    long double pap = 0.0L;
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i) {
          Real value = (fac*coeff(linearSolver::DCCF,k,j,i)
                       + coeff(linearSolver::DCCS,k,j,i))*delta(k,j,i);
          value += (fac*coeff(linearSolver::DXMF,k,j,i)
                    + coeff(linearSolver::DXMS,k,j,i))*delta(k,j,i-1);
          value += (fac*coeff(linearSolver::DXPF,k,j,i)
                    + coeff(linearSolver::DXPS,k,j,i))*delta(k,j,i+1);
          value += (fac*coeff(linearSolver::DYMF,k,j,i)
                    + coeff(linearSolver::DYMS,k,j,i))*delta(k,j-1,i);
          value += (fac*coeff(linearSolver::DYPF,k,j,i)
                    + coeff(linearSolver::DYPS,k,j,i))*delta(k,j+1,i);
          value += (fac*coeff(linearSolver::DZMF,k,j,i)
                    + coeff(linearSolver::DZMS,k,j,i))*delta(k-1,j,i);
          value += (fac*coeff(linearSolver::DZPF,k,j,i)
                    + coeff(linearSolver::DZPS,k,j,i))*delta(k+1,j,i);
          ap(k,j,i) = value;
          pap += static_cast<long double>(p(k,j,i))*value;
        }
    if (!(pap > 0.0L) || !std::isfinite(static_cast<Real>(pap))
        || !std::isfinite(static_cast<Real>(rz))) {
      breakdown = true;
      break;
    }
    const Real alpha = static_cast<Real>(rz/pap);
    long double residual2 = 0.0L;
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i) {
          x(k,j,i) += alpha*p(k,j,i);
          r(k,j,i) -= alpha*ap(k,j,i);
          residual2 += static_cast<long double>(r(k,j,i))*r(k,j,i);
        }
    relative_residual = static_cast<Real>(std::sqrt(residual2)/source_norm);
    ++iterations;
    if (relative_residual <= uniform_pcg_threshold_) break;

    long double rz_new = 0.0L;
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i) {
          const Real diag = fac*coeff(linearSolver::DCCF,k,j,i)
                          + coeff(linearSolver::DCCS,k,j,i);
          z(k,j,i) = r(k,j,i)/diag;
          rz_new += static_cast<long double>(r(k,j,i))*z(k,j,i);
        }
    if (!(rz_new >= 0.0L) || !std::isfinite(static_cast<Real>(rz_new))) {
      breakdown = true;
      break;
    }
    const Real beta = static_cast<Real>(rz_new/rz);
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
          p(k,j,i) = z(k,j,i) + beta*p(k,j,i);
    rz = rz_new;
  }

  delta = x;
  linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
  if (Globals::my_rank == 0) {
    std::cout << "[LINEAR_PCG] iterations=" << iterations
              << " relative_residual=" << relative_residual
              << " target=" << uniform_pcg_threshold_
              << " breakdown=" << breakdown << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Build the exact R A_h P hierarchy for a uniform MeshBlock topology.
//!
//! Restriction is the existing eight-cell volume average and prolongation is piecewise
//! constant, its scaled transpose, in the production path.  A seven-point fine operator
//! therefore remains exactly seven-point on every coarse level.  The former wider RAP
//! builder is retained for explicit galerkin_27_experimental use with trilinear transfer.
void linearMGDriver::BuildGalerkinHierarchy() {
  if (!galerkin_coarse_operator_) return;

  auto floor_half = [](int q) { return q >= 0 ? q/2 : -((-q + 1)/2); };
  auto build_level = [&](linearMG *owner, const AthenaArray<Real> &fine,
                         AthenaArray<Real> &coarse,
                         int fnx, int fny, int fnz, int ngh) {
    const int cnx = fnx/2, cny = fny/2, cnz = fnz/2;
    for (int ck = 0; ck < cnz; ++ck) {
      const int k = ngh + ck;
      for (int cj = 0; cj < cny; ++cj) {
        const int j = ngh + cj;
        for (int ci = 0; ci < cnx; ++ci) {
          const int i = ngh + ci;
          for (int n = 0; n < nmatrix_; ++n)
            coarse(n,k,j,i) = 0.0;

          for (int ok = 0; ok < 2; ++ok) {
            const int fk = ngh + 2*ck + ok;
            for (int oj = 0; oj < 2; ++oj) {
              const int fj = ngh + 2*cj + oj;
              for (int oi = 0; oi < 2; ++oi) {
                const int fi = ngh + 2*ci + oi;
                for (int s = 0; s < nmatrix_; ++s) {
                  const Real a = fine(s,fk,fj,fi);
                  if (a == 0.0) continue;
                  const int gk = 2*ck + ok + linearSolver::matrix_offset_k[s];
                  const int gj = 2*cj + oj + linearSolver::matrix_offset_j[s];
                  const int gi = 2*ci + oi + linearSolver::matrix_offset_i[s];
                  auto physical = [&](BoundaryFace face) {
                    const BoundaryFlag flag = owner->mg_block_bcs_[face];
                    return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic;
                  };
                  // RSG correction boundaries prescribe zero ghost-centre values.  Such
                  // fine-grid columns are not coarse unknowns and must be eliminated from
                  // R A_h P instead of trilinearly extrapolated beyond the domain.
                  if ((gi < 0 && physical(inner_x1))
                      || (gi >= fnx && physical(outer_x1))
                      || (gj < 0 && physical(inner_x2))
                      || (gj >= fny && physical(outer_x2))
                      || (gk < 0 && physical(inner_x3))
                      || (gk >= fnz && physical(outer_x3))) continue;
                  const int pk = floor_half(gk), pj = floor_half(gj), pi = floor_half(gi);
                  const int nk = pk + ((gk - 2*pk) == 0 ? -1 : 1);
                  const int nj = pj + ((gj - 2*pj) == 0 ? -1 : 1);
                  const int ni = pi + ((gi - 2*pi) == 0 ? -1 : 1);
                  const int dk[2] = {pk, nk};
                  const int dj[2] = {pj, nj};
                  const int di[2] = {pi, ni};
                  const bool constant = fprolongation_ == 2;
                  const Real weight[2] = {constant ? 1.0 : 0.75, 0.25};
                  const int nweight = constant ? 1 : 2;
                  for (int wk = 0; wk < nweight; ++wk)
                    for (int wj = 0; wj < nweight; ++wj)
                      for (int wi = 0; wi < nweight; ++wi) {
                        const int sok = dk[wk] - ck;
                        const int soj = dj[wj] - cj;
                        const int soi = di[wi] - ci;
                        const int index = linearSolver::MatrixIndexFromOffset(
                            sok, soj, soi);
                        if (index < 0) {
                          std::stringstream msg;
                          msg << "### FATAL ERROR in BuildGalerkinHierarchy" << std::endl
                              << "RAP generated support outside the 27-point stencil."
                              << std::endl;
                          ATHENA_ERROR(msg);
                        }
                        if (!experimental_27_point_galerkin_
                            && index >= linearSolver::NMATRIX) {
                          std::stringstream msg;
                          msg << "### FATAL ERROR in BuildGalerkinHierarchy" << std::endl
                              << "The production constant-transfer RAP generated an "
                              << "edge/corner coupling.  Its exact support must remain "
                              << "seven-point; use galerkin_27_experimental only for an "
                              << "intentional wider-transfer experiment." << std::endl;
                          ATHENA_ERROR(msg);
                        }
                        coarse(index,k,j,i) += 0.125*a
                            *weight[wk]*weight[wj]*weight[wi];
                      }
                }
              }
            }
          }
        }
      }
    }
  };

  auto build_object = [&](linearMG *pmg) {
    for (int lev = pmg->nlevel_ - 1; lev > 0; --lev) {
      const int shift = pmg->nlevel_ - 1 - lev;
      build_level(pmg, pmg->matrix_[lev], pmg->matrix_[lev-1],
                  pmg->size_.nx1 >> shift, pmg->size_.nx2 >> shift,
                  pmg->size_.nx3 >> shift, pmg->ngh_);
    }
  };

#pragma omp parallel for num_threads(nthreads_)
  for (auto itr = vmg_.begin(); itr < vmg_.end(); ++itr)
    build_object(static_cast<linearMG*>(*itr));

  // Each uniform MeshBlock contributes its one-cell coarsest Galerkin row to
  // the finest root-grid level.  Summation is safe because exactly one rank
  // owns each gid and all other entries are initialized to zero.
  std::vector<Real> rows(static_cast<std::size_t>(nbtotal_)
                         *nmatrix_, 0.0);
  for (auto *base : vmg_) {
    linearMG *pmg = static_cast<linearMG*>(base);
    const int ngh = pmg->ngh_;
    const int gid = pmg->pmy_block_->gid;
    for (int n = 0; n < nmatrix_; ++n)
      rows[static_cast<std::size_t>(gid)*nmatrix_ + n]
          = pmg->matrix_[0](n,ngh,ngh,ngh);
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, rows.data(),
                static_cast<int>(rows.size()), MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  linearMG *root = static_cast<linearMG*>(mgroot_);
  const int root_finest = root->nlevel_ - 1;
  const int rngh = root->ngh_;
  for (int gid = 0; gid < nbtotal_; ++gid) {
    const LogicalLocation &loc = pmy_mesh_->loclist[gid];
    const int i = rngh + static_cast<int>(loc.lx1);
    const int j = rngh + static_cast<int>(loc.lx2);
    const int k = rngh + static_cast<int>(loc.lx3);
    for (int n = 0; n < nmatrix_; ++n)
      root->matrix_[root_finest](n,k,j,i) =
          rows[static_cast<std::size_t>(gid)*nmatrix_ + n];
  }
  build_object(root);
  if (Globals::my_rank == 0 && fshowdef_) {
    std::cout << "[LINEAR_MG_GALERKIN] rebuilt " << nmatrix_
              << "-point RAP hierarchy"
              << (experimental_27_point_galerkin_ ? " (experimental)" : "")
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Restarted right-preconditioned FGMRES on the distributed fine-grid operator.
//!
//! Fine-grid matvecs use the same MeshBlock correction boundary exchange as NRFLD.  One
//! or more existing V-cycles form a flexible right preconditioner, so a nonstationary or
//! nonsymmetric MG response is permitted.
void linearMGDriver::SolveFGMRES(int stage) {
  using BlockVector = std::vector<AthenaArray<Real>>;
  auto make_vector = [&]() {
    BlockVector value;
    value.reserve(vmg_.size());
    for (auto *base : vmg_) {
      linearMG *pmg = static_cast<linearMG*>(base);
      value.emplace_back(pmg->pnr_->delta_u_);
      value.back().ZeroClear();
    }
    return value;
  };
  auto copy = [&](BlockVector &dst, const BlockVector &src) {
    for (std::size_t b = 0; b < dst.size(); ++b) dst[b] = src[b];
  };
  auto scale = [&](BlockVector &x, Real a) {
    for (std::size_t b = 0; b < x.size(); ++b) {
      MeshBlock *pmb = static_cast<linearMG*>(vmg_[b])->pmy_block_;
      for (int k = pmb->ks; k <= pmb->ke; ++k)
        for (int j = pmb->js; j <= pmb->je; ++j)
          for (int i = pmb->is; i <= pmb->ie; ++i) x[b](k,j,i) *= a;
    }
  };
  auto axpy = [&](BlockVector &y, Real a, const BlockVector &x) {
    for (std::size_t b = 0; b < y.size(); ++b) {
      MeshBlock *pmb = static_cast<linearMG*>(vmg_[b])->pmy_block_;
      for (int k = pmb->ks; k <= pmb->ke; ++k)
        for (int j = pmb->js; j <= pmb->je; ++j)
          for (int i = pmb->is; i <= pmb->ie; ++i)
            y[b](k,j,i) += a*x[b](k,j,i);
    }
  };
  auto dot = [&](const BlockVector &x, const BlockVector &y) {
    long double sum = 0.0L;
    for (std::size_t b = 0; b < x.size(); ++b) {
      MeshBlock *pmb = static_cast<linearMG*>(vmg_[b])->pmy_block_;
      for (int k = pmb->ks; k <= pmb->ke; ++k)
        for (int j = pmb->js; j <= pmb->je; ++j)
          for (int i = pmb->is; i <= pmb->ie; ++i)
            sum += static_cast<long double>(x[b](k,j,i))*y[b](k,j,i);
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    return sum;
  };
  auto norm = [&](const BlockVector &x) {
    return std::sqrt(std::max(dot(x,x), 0.0L));
  };
  auto load_delta = [&](const BlockVector &x) {
    for (std::size_t b = 0; b < x.size(); ++b) {
      linearMG *pmg = static_cast<linearMG*>(vmg_[b]);
      MeshBlock *pmb = pmg->pmy_block_;
      AthenaArray<Real> &delta = pmg->pnr_->delta_u_;
      delta.ZeroClear();
      for (int k = pmb->ks; k <= pmb->ke; ++k)
        for (int j = pmb->js; j <= pmb->je; ++j)
          for (int i = pmb->is; i <= pmb->ie; ++i) delta(k,j,i) = x[b](k,j,i);
    }
    linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
  };
  auto apply_operator = [&](const BlockVector &x, BlockVector &y) {
    load_delta(x);
    for (std::size_t b = 0; b < y.size(); ++b) {
      linearMG *pmg = static_cast<linearMG*>(vmg_[b]);
      MeshBlock *pmb = pmg->pmy_block_;
      const AthenaArray<Real> &delta = pmg->pnr_->delta_u_;
      const AthenaArray<Real> &coeff = pmg->pnr_->coeff_;
      const Real fac = dt_/SQR(pmb->pcoord->dx1f(pmb->is));
      y[b].ZeroClear();
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        for (int j = pmb->js; j <= pmb->je; ++j) {
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            Real value = (fac*coeff(linearSolver::DCCF,k,j,i)
                         + coeff(linearSolver::DCCS,k,j,i))*delta(k,j,i);
            value += (fac*coeff(linearSolver::DXMF,k,j,i)
                     + coeff(linearSolver::DXMS,k,j,i))*delta(k,j,i-1);
            value += (fac*coeff(linearSolver::DXPF,k,j,i)
                     + coeff(linearSolver::DXPS,k,j,i))*delta(k,j,i+1);
            value += (fac*coeff(linearSolver::DYMF,k,j,i)
                     + coeff(linearSolver::DYMS,k,j,i))*delta(k,j-1,i);
            value += (fac*coeff(linearSolver::DYPF,k,j,i)
                     + coeff(linearSolver::DYPS,k,j,i))*delta(k,j+1,i);
            value += (fac*coeff(linearSolver::DZMF,k,j,i)
                     + coeff(linearSolver::DZMS,k,j,i))*delta(k-1,j,i);
            value += (fac*coeff(linearSolver::DZPF,k,j,i)
                     + coeff(linearSolver::DZPS,k,j,i))*delta(k+1,j,i);
            y[b](k,j,i) = value;
          }
        }
      }
    }
  };
  auto apply_preconditioner = [&](const BlockVector &rhs, BlockVector &z) {
    current_level_ = fmglevel_ = ntotallevel_ - 1;
    for (std::size_t b = 0; b < rhs.size(); ++b) {
      linearMG *pmg = static_cast<linearMG*>(vmg_[b]);
      pmg->current_level_ = pmg->nlevel_ - 1;
      pmg->u_[pmg->current_level_].ZeroClear();
      pmg->src_[pmg->current_level_].ZeroClear();
      MeshBlock *pmb = pmg->pmy_block_;
      const int ngh = pmg->ngh_;
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        const int mk = k - pmb->ks + ngh;
        for (int j = pmb->js; j <= pmb->je; ++j) {
          const int mj = j - pmb->js + ngh;
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            const int mi = i - pmb->is + ngh;
            pmg->src_[pmg->current_level_](0,mk,mj,mi) = rhs[b](k,j,i);
          }
        }
      }
    }
    for (int cycle = 0; cycle < fgmres_mg_vcycles_; ++cycle)
      SolveVCycle(npresmooth_, npostsmooth_);
    for (std::size_t b = 0; b < z.size(); ++b) {
      linearMG *pmg = static_cast<linearMG*>(vmg_[b]);
      MeshBlock *pmb = pmg->pmy_block_;
      const int ngh = pmg->ngh_;
      z[b].ZeroClear();
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        const int mk = k - pmb->ks + ngh;
        for (int j = pmb->js; j <= pmb->je; ++j) {
          const int mj = j - pmb->js + ngh;
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            const int mi = i - pmb->is + ngh;
            z[b](k,j,i) = pmg->u_[pmg->current_level_](0,mk,mj,mi);
          }
        }
      }
    }
  };

  BlockVector b = make_vector(), x = make_vector(), r = make_vector(), ax = make_vector();
  for (std::size_t q = 0; q < b.size(); ++q) {
    linearMG *pmg = static_cast<linearMG*>(vmg_[q]);
    MeshBlock *pmb = pmg->pmy_block_;
    for (int k = pmb->ks; k <= pmb->ke; ++k)
      for (int j = pmb->js; j <= pmb->je; ++j)
        for (int i = pmb->is; i <= pmb->ie; ++i)
          b[q](k,j,i) = pmg->pnr_->src_(k,j,i);
  }
  copy(r,b);
  const long double bnorm = std::max(norm(b),
      static_cast<long double>(std::numeric_limits<Real>::min()));
  long double beta = norm(r);
  Real relative_residual = static_cast<Real>(beta/bnorm);
  int total_iterations = 0;
  bool breakdown = false;

  while (relative_residual > fgmres_threshold_
         && total_iterations < fgmres_max_iterations_) {
    const int width = std::min(fgmres_restart_,
                               fgmres_max_iterations_ - total_iterations);
    std::vector<BlockVector> v, z;
    v.reserve(width + 1);
    z.reserve(width);
    for (int n = 0; n <= width; ++n) v.emplace_back(make_vector());
    for (int n = 0; n < width; ++n) z.emplace_back(make_vector());
    copy(v[0], r);
    scale(v[0], static_cast<Real>(1.0L/beta));
    std::vector<Real> h(static_cast<std::size_t>(width + 1)*width, 0.0);
    std::vector<Real> cs(width, 0.0), sn(width, 0.0), g(width + 1, 0.0);
    g[0] = static_cast<Real>(beta);
    int used = 0;
    for (int j = 0; j < width; ++j) {
      apply_preconditioner(v[j], z[j]);
      apply_operator(z[j], v[j+1]);
      for (int i = 0; i <= j; ++i) {
        h[static_cast<std::size_t>(i)*width + j] =
            static_cast<Real>(dot(v[j+1], v[i]));
        axpy(v[j+1], -h[static_cast<std::size_t>(i)*width + j], v[i]);
      }
      // A second MGS pass prevents loss of orthogonality for the 1e11-scale
      // coefficient contrasts present in the RSG Schur operator.
      for (int i = 0; i <= j; ++i) {
        const Real correction = static_cast<Real>(dot(v[j+1], v[i]));
        h[static_cast<std::size_t>(i)*width + j] += correction;
        axpy(v[j+1], -correction, v[i]);
      }
      const long double hnext = norm(v[j+1]);
      h[static_cast<std::size_t>(j + 1)*width + j] = static_cast<Real>(hnext);
      if (hnext > std::numeric_limits<Real>::epsilon()*bnorm)
        scale(v[j+1], static_cast<Real>(1.0L/hnext));

      for (int i = 0; i < j; ++i) {
        const Real hij = h[static_cast<std::size_t>(i)*width + j];
        const Real hip = h[static_cast<std::size_t>(i + 1)*width + j];
        h[static_cast<std::size_t>(i)*width + j] = cs[i]*hij + sn[i]*hip;
        h[static_cast<std::size_t>(i + 1)*width + j] = -sn[i]*hij + cs[i]*hip;
      }
      const Real hjj = h[static_cast<std::size_t>(j)*width + j];
      const Real hjp = h[static_cast<std::size_t>(j + 1)*width + j];
      const Real rho = std::hypot(hjj, hjp);
      if (!(rho > 0.0) || !std::isfinite(rho)) {
        breakdown = true;
        break;
      }
      cs[j] = hjj/rho;
      sn[j] = hjp/rho;
      h[static_cast<std::size_t>(j)*width + j] = rho;
      h[static_cast<std::size_t>(j + 1)*width + j] = 0.0;
      g[j+1] = -sn[j]*g[j];
      g[j] = cs[j]*g[j];
      used = j + 1;
      ++total_iterations;
      relative_residual = std::abs(g[j+1])/static_cast<Real>(bnorm);
      if (relative_residual <= fgmres_threshold_
          || hnext <= std::numeric_limits<Real>::epsilon()*bnorm) break;
    }
    if (used == 0) {
      breakdown = true;
      break;
    }
    std::vector<Real> y(used, 0.0);
    for (int row = used - 1; row >= 0; --row) {
      Real value = g[row];
      for (int col = row + 1; col < used; ++col)
        value -= h[static_cast<std::size_t>(row)*width + col]*y[col];
      const Real diag = h[static_cast<std::size_t>(row)*width + row];
      if (!(std::abs(diag) > std::numeric_limits<Real>::min())) {
        breakdown = true;
        break;
      }
      y[row] = value/diag;
    }
    for (int j = 0; j < used; ++j) axpy(x, y[j], z[j]);
    apply_operator(x, ax);
    copy(r,b);
    axpy(r, -1.0, ax);
    beta = norm(r);
    relative_residual = static_cast<Real>(beta/bnorm);
    if (breakdown) break;
  }
  load_delta(x);
  if (Globals::my_rank == 0) {
    std::cout << "[LINEAR_FGMRES] iterations=" << total_iterations
              << " relative_residual=" << relative_residual
              << " target=" << fgmres_threshold_
              << " restart=" << fgmres_restart_
              << " mg_vcycles=" << fgmres_mg_vcycles_
              << " breakdown=" << breakdown << std::endl;
  }
  if (breakdown || !(relative_residual <= fgmres_threshold_)
      || !std::isfinite(relative_residual)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::SolveFGMRES" << std::endl
        << "FGMRES did not reach its requested relative residual: iterations="
        << total_iterations << ", relative_residual=" << relative_residual
        << ", target=" << fgmres_threshold_ << ", breakdown=" << breakdown
        << "." << std::endl;
    ATHENA_ERROR(msg);
  }
}

void linearMGDriver::Solve(int stage, Real dt) {
  // std::cout << "In linearMGDriver::Solve()" << std::endl;
  dt_ = dt;
  // Construct the linearMG array
  vmg_.clear();
  for (int i = 0; i < pmy_mesh_->nblocal; ++i)
    vmg_.push_back(pmy_mesh_->my_blocks(i)->pnr->plmg_);
  // The implicit q*v.grad(Er) Schur coefficients vary with the local
  // matter/radiation Jacobian, so the full nonsymmetric matrix must be
  // refreshed at each Newton iteration.
  const FLD *mixed_fld = vmg_.empty() ? nullptr
      : static_cast<linearMG*>(vmg_.front())->pmy_block_->prfld;
  const bool mixed_implicit = mixed_fld != nullptr
      && mixed_fld->MixedFrameCoupled();

  // load the source
#pragma omp parallel for num_threads(nthreads_)
  for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
    linearMG *pmg = static_cast<linearMG*>(*itr);
    // assume all the data are located on the same node
    // FLD *prfld = pmg->pmy_block_->prfld;
    NewtonRaphson *pnr = pmg->pmy_block_->pnr;
    pmg->LoadSource(pnr->src_, 0, NGHOST, 1.0);
    pmg->LoadFinestData(pnr->delta_u_, 0, NGHOST); // caution! should be zero
    if (mixed_implicit || !cache_coefficient_hierarchy_ || !coefficient_hierarchy_cached_
        || mode_ == 0 || pmy_mesh_->amr_updated) {
      pmg->LoadCoefficients(pnr->coeff_, NGHOST);
    } else {
      // In NRFLD the face diffusion coefficients and DCCF are fixed during a
      // hydro step.  Only the local Schur-complement diagonal changes.
      pmg->LoadCoefficient(pnr->coeff_, linearSolver::DCCS, NGHOST);
    }
    // pmg->AddFLDSource(prfld->source, NGHOST, dt_);
  }

  // std::cout << "Source loaded to linearMG. Start to solve... at " << Globals::my_rank << std::endl;

  const bool rebuild_coefficients = mixed_implicit || !cache_coefficient_hierarchy_
                                  || !coefficient_hierarchy_cached_
                                  || mode_ == 0 || pmy_mesh_->amr_updated;
  if (rebuild_coefficients) {
    SetupMultigrid(false);
    coefficient_hierarchy_cached_ = true;
  } else {
    SetupMultigrid(true);
    SetupCoefficient(linearSolver::DCCS);
    CalculateMatrixAll();
  }
  BuildGalerkinHierarchy();
  if (coefficient_diagnostics_
      && coefficient_diagnostic_count_ < coefficient_diagnostic_limit_) {
    PrintCoefficientDiagnostics();
    ++coefficient_diagnostic_count_;
  }
  // std::cout << "setup done at " << Globals::my_rank << std::endl;
  const bool used_krylov = uniform_pcg_ || fgmres_;
  if (uniform_pcg_) {
    SolveUniformPCG(stage);
  } else if (fgmres_) {
    SolveFGMRES(stage);
  } else if (smoothing_only_) {
    // The NR globalization fallback must not use any coarse-grid correction.
    // Start from a neutral correction and solve with the same fine-grid
    // boundary operator that is used when the Newton residual is evaluated.
#pragma omp parallel for num_threads(nthreads_)
    for (auto itr = vmg_.begin(); itr < vmg_.end(); ++itr) {
      linearMG *pmg = static_cast<linearMG*>(*itr);
      pmg->pnr_->delta_u_.ZeroClear();
    }
  } else if (mode_ == 0) {
    SolveFMGCycle();
  } else {
    // Interpret threshold == 0.0 with niteration >= 0 as a fixed-count solve.
    // Several NRFLD inputs rely on this combination to cap the linear solve
    // without setting a positive absolute residual tolerance.
    if (eps_ > 0.0 || (eps_ == 0.0 && niter_ < 0)) {
      // std::cout << "linearMG solve with threshold " << eps_ << " at " << Globals::my_rank << std::endl;
      SolveIterative();
      // std::cout << "linearMG solve with threshold " << eps_ << " finished at " << Globals::my_rank << std::endl;
    } else {
      SolveIterativeFixedTimes();
    }
  }
  // std::cout << "linearMG solve finished at " << Globals::my_rank << std::endl;

  // Return the result
  if (!used_krylov && handoff_diagnostics_
      && handoff_diagnostic_count_ < handoff_diagnostic_limit_)
    PrintHandoffDiagnostic("mg_end", true);
  if (!used_krylov) {
#pragma omp parallel for num_threads(nthreads_)
    for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
      linearMG *pmg = static_cast<linearMG*>(*itr);
      // FLD *prfld = pmg->pmy_block_->prfld;
      NewtonRaphson *pnr = pmg->pnr_;
      // Hydro *phydro = pmg->pmy_block_->phydro;
      pmg->RetrieveResult(pnr->delta_u_, 0, NGHOST);
      // if (pnr->output_defect)
      //   pmg->RetrieveDefect(pnr->def_, 0, NGHOST);
    }
  }

  // std::cout << "Result retrieved from linearMG at " << Globals::my_rank << std::endl;
  if (handoff_diagnostics_ && handoff_diagnostic_count_ < handoff_diagnostic_limit_)
    PrintHandoffDiagnostic("retrieved", false);
  linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
  if (handoff_diagnostics_ && handoff_diagnostic_count_ < handoff_diagnostic_limit_)
    PrintHandoffDiagnostic("boundary", false);
  // MG and MeshRefinement use slightly different coarse/fine ghost
  // interpolation.  Polish the returned correction with the exact
  // MeshBlock boundary operator used by NRFLD, so the Newton step satisfies
  // the same discrete Jacobian that will be used to evaluate the residual.
  const int fine_sweeps = smoothing_only_
                                           ? std::max(npostsolve_smooth_,
                                                      nsmoothing_only_sweeps_)
                                           : npostsolve_smooth_;
  for (int sweep = 0; sweep < fine_sweeps; ++sweep) {
#pragma omp parallel for num_threads(nthreads_)
    for (auto itr = vmg_.begin(); itr < vmg_.end(); ++itr) {
      linearMG *pmg = static_cast<linearMG*>(*itr);
      NewtonRaphson *pnr = pmg->pnr_;
      MeshBlock *pmb = pmg->pmy_block_;
      AthenaArray<Real> &work = pmg->iteration_backup_;
      const AthenaArray<Real> &delta = pnr->delta_u_;
      const AthenaArray<Real> &coeff = pnr->coeff_;
      const AthenaArray<Real> &src = pnr->src_;
      const Real fac = dt_/SQR(pmb->pcoord->dx1f(pmb->is));
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        const int mk = k - pmb->ks + pmg->ngh_;
        for (int j = pmb->js; j <= pmb->je; ++j) {
          const int mj = j - pmb->js + pmg->ngh_;
#pragma omp simd
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            const int mi = i - pmb->is + pmg->ngh_;
            const Real diag = fac*coeff(linearSolver::DCCF,k,j,i)
                            + coeff(linearSolver::DCCS,k,j,i);
            Real offdiag = (fac*coeff(linearSolver::DXMF,k,j,i)
                              + coeff(linearSolver::DXMS,k,j,i))*delta(k,j,i-1);
            offdiag += (fac*coeff(linearSolver::DXPF,k,j,i)
                              + coeff(linearSolver::DXPS,k,j,i))*delta(k,j,i+1);
            offdiag += (fac*coeff(linearSolver::DYMF,k,j,i)
                              + coeff(linearSolver::DYMS,k,j,i))*delta(k,j-1,i);
            offdiag += (fac*coeff(linearSolver::DYPF,k,j,i)
                              + coeff(linearSolver::DYPS,k,j,i))*delta(k,j+1,i);
            offdiag += (fac*coeff(linearSolver::DZMF,k,j,i)
                              + coeff(linearSolver::DZMS,k,j,i))*delta(k-1,j,i);
            offdiag += (fac*coeff(linearSolver::DZPF,k,j,i)
                              + coeff(linearSolver::DZPS,k,j,i))*delta(k+1,j,i);
            work(0,mk,mj,mi) = (src(k,j,i) - offdiag)/diag;
          }
        }
      }
    }
#pragma omp parallel for num_threads(nthreads_)
    for (auto itr = vmg_.begin(); itr < vmg_.end(); ++itr) {
      linearMG *pmg = static_cast<linearMG*>(*itr);
      NewtonRaphson *pnr = pmg->pnr_;
      MeshBlock *pmb = pmg->pmy_block_;
      const AthenaArray<Real> &work = pmg->iteration_backup_;
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        const int mk = k - pmb->ks + pmg->ngh_;
        for (int j = pmb->js; j <= pmb->je; ++j) {
          const int mj = j - pmb->js + pmg->ngh_;
#pragma omp simd
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            const int mi = i - pmb->is + pmg->ngh_;
            pnr->delta_u_(k,j,i) = work(0,mk,mj,mi);
          }
        }
      }
    }
    linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
  }
  // std::cout << "linearMG boundary conditions applied." << std::endl;
  if (handoff_diagnostics_ && handoff_diagnostic_count_ < handoff_diagnostic_limit_)
    PrintHandoffDiagnostic("fine_sweeps", false);
  ++handoff_diagnostic_count_;
// #pragma omp parallel for num_threads(nthreads_)
//   for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
//     linearMG *pmg = static_cast<linearMG*>(*itr);
//     FLD *prfld = pmg->pmy_block_->prfld;
//     Hydro *phydro = pmg->pmy_block_->phydro;
//     // if (!prfld->only_rad)
//     //   prfld->UpdateHydroVariables(phydro->w, phydro->u, prfld->u);
//   }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void linearMG::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
//!            const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix, int rlev,
//!            int il, int iu, int jl, int ju, int kl, int ku, int color, bool th)
//! \brief Implementation of the Red-Black Gauss-Seidel Smoother
//!        rlev = relative level from the finest level of this Multigrid block

void linearMG::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
         const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix, int rlev,
         int il, int iu, int jl, int ju, int kl, int ku, int color, bool th) {
  color ^= pmy_driver_->coffset_;
  if (fsmoother_ == 1) { // jacobi-rb
    if (th == true && (ku-kl) >=  minth_) {
      AthenaArray<Real> &work = temp[0];
#pragma omp parallel num_threads(pmy_driver_->nthreads_)
      {
#pragma omp for
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            int c = (color + k + j) & 1;
#pragma ivdep
            for (int i=il+c; i<=iu; i+=2) {
              const Real M = ApplyMatrixOffDiagonal(
                  matrix, u, k, j, i, pmy_driver_->nmatrix_);
              work(k,j,i) = (src(k,j,i) - M) / matrix(linearSolver::CCC,k,j,i);
            }
          }
        }
#pragma omp for
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            int c = (color + k + j) & 1;
#pragma ivdep
            for (int i=il+c; i<=iu; i+=2) {
              u(k,j,i) += omega_ * (work(k,j,i) - u(k,j,i));
            }
          }
        }
      }
    } else {
      int t = 0;
#ifdef OPENMP_PARALLEL
      t = omp_get_thread_num();
#endif
      AthenaArray<Real> &work = temp[t];
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          int c = (color + k + j) & 1;
#pragma ivdep
          for (int i=il+c; i<=iu; i+=2) {
            const Real M = ApplyMatrixOffDiagonal(
                matrix, u, k, j, i, pmy_driver_->nmatrix_);
            work(k,j,i) = (src(k,j,i) - M) / matrix(linearSolver::CCC,k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          int c = (color + k + j) & 1;
#pragma ivdep
          for (int i=il+c; i<=iu; i+=2) {
            u(k,j,i) += omega_ * (work(k,j,i) - u(k,j,i));
          }
        }
      }
    }
  } else { // jacobi
    if (th == true && (ku-kl) >=  minth_) {
      AthenaArray<Real> &work = temp[0];
#pragma omp parallel num_threads(pmy_driver_->nthreads_)
      {
#pragma omp for
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
#pragma ivdep
            for (int i=il; i<=iu; i++) {
              const Real M = ApplyMatrixOffDiagonal(
                  matrix, u, k, j, i, pmy_driver_->nmatrix_);
              work(k,j,i) = (src(k,j,i) - M) / matrix(linearSolver::CCC,k,j,i);
            }
          }
        }
#pragma omp for
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
#pragma ivdep
            for (int i=il; i<=iu; i++)
              u(k,j,i) += omega_ * (work(k,j,i) - u(k,j,i));
          }
        }
      }
    } else {
      int t = 0;
#ifdef OPENMP_PARALLEL
      t = omp_get_thread_num();
#endif
      AthenaArray<Real> &work = temp[t];
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
#pragma ivdep
          for (int i=il; i<=iu; i++) {
            const Real M = ApplyMatrixOffDiagonal(
                matrix, u, k, j, i, pmy_driver_->nmatrix_);
            work(k,j,i) = (src(k,j,i) - M) / matrix(linearSolver::CCC,k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
#pragma ivdep
          for (int i=il; i<=iu; i++)
            u(k,j,i) += omega_ * (work(k,j,i) - u(k,j,i));
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void linearMG::CalculateDefect(AthenaArray<Real> &def,
//!            const AthenaArray<Real> &u, const AthenaArray<Real> &src,
//!            const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
//!            int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th)
//! \brief Implementation of the Defect calculation
//!        rlev = relative level from the finest level of this Multigrid block

void linearMG::CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
                    const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
                    const AthenaArray<Real> &matrix, int rlev, int il, int iu,
                    int jl, int ju, int kl, int ku, bool th) {
  // std::cout << "In linearMGFLD::CalculateDefect" << std::endl;
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);

#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        const Real M = ApplyMatrixRow(
            matrix, u, k, j, i, pmy_driver_->nmatrix_);
        // Multigrid coarse-grid correction must use the raw residual.
        // Normalizing here distorts the restricted error equation.
        def(k,j,i) = src(k,j,i) - M;
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void linearMG::CalculateFASRHS(AthenaArray<Real> &src,
//!            const AthenaArray<Real> &u, const AthenaArray<Real> &coeff,
//!            const AthenaArray<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
//!            int kl, int ku, bool th)
//! \brief Implementation of the RHS calculation for FAS
//!        rlev = relative level from the finest level of this Multigrid block

void linearMG::CalculateFASRHS(AthenaArray<Real> &src, const AthenaArray<Real> &u,
                    const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
                    int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) {
  // std::cout << "In linearMGFLD::CalculateFASRHS" << std::endl;
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);
#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        const Real M = ApplyMatrixRow(
            matrix, u, k, j, i, pmy_driver_->nmatrix_);
        src(k,j,i) += M;
      }
    }
  }

  return;
}

//caution, just copied from mg_gravity.cpp
//----------------------------------------------------------------------------------------
//! \fn void linearMGDriver::ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst,
//!                           AthenaArray<Real> &cbuf, const AthenaArray<bool> &ncoarse)
//! \brief prolongate octet boundaries using the flux conservation formula

void linearMGDriver::ProlongateOctetBoundariesFluxCons(AthenaArray<Real> &dst,
                      AthenaArray<Real> &cbuf, const AthenaArray<bool> &ncoarse) {
  // std::cout << "In linearMGFLDDriver::ProlongateOctetBoundariesFluxCons" << std::endl;
  constexpr Real ot = 1.0/3.0;
  const int ngh = mgroot_->ngh_;
  const AthenaArray<Real> &u = dst;
  const int ci = ngh, cj = ngh, ck = ngh, l = ngh, r = ngh + 1;

  // x1face
  for (int ox1=-1; ox1<=1; ox1+=2) {
    if (ncoarse(1, 1, ox1+1)) {
      int i, fi, fig;
      if (ox1 > 0) i = ngh + 1, fi = ngh + 1, fig = ngh + 2;
      else         i = ngh - 1, fi = ngh,     fig = ngh - 1;
      Real ccval = cbuf(ck, cj, i);
      Real gx2c = 0.125*(cbuf(ck, cj+1, i) - cbuf(ck, cj-1, i));
      Real gx3c = 0.125*(cbuf(ck+1, cj, i) - cbuf(ck-1, cj, i));
      dst(l, l, fig) = ot*(2.0*(ccval - gx2c - gx3c) + u(l, l, fi));
      dst(l, r, fig) = ot*(2.0*(ccval + gx2c - gx3c) + u(l, r, fi));
      dst(r, l, fig) = ot*(2.0*(ccval - gx2c + gx3c) + u(r, l, fi));
      dst(r, r, fig) = ot*(2.0*(ccval + gx2c + gx3c) + u(r, r, fi));
    }
  }

  // x2face
  for (int ox2=-1; ox2<=1; ox2+=2) {
    if (ncoarse(1, ox2+1, 1)) {
      int j, fj, fjg;
      if (ox2 > 0) j = ngh + 1, fj = ngh + 1, fjg = ngh + 2;
      else         j = ngh - 1, fj = ngh,     fjg = ngh - 1;
      Real ccval = cbuf(ck, j, ci);
      Real gx1c = 0.125*(cbuf(ck, j, ci+1) - cbuf(ck, j, ci-1));
      Real gx3c = 0.125*(cbuf(ck+1, j, ci) - cbuf(ck-1, j, ci));
      dst(l, fjg, l) = ot*(2.0*(ccval - gx1c - gx3c) + u(l, fj, l));
      dst(l, fjg, r) = ot*(2.0*(ccval + gx1c - gx3c) + u(l, fj, r));
      dst(r, fjg, l) = ot*(2.0*(ccval - gx1c + gx3c) + u(r, fj, l));
      dst(r, fjg, r) = ot*(2.0*(ccval + gx1c + gx3c) + u(r, fj, r));
    }
  }

  // x3face
  for (int ox3=-1; ox3<=1; ox3+=2) {
    if (ncoarse(ox3+1, 1, 1)) {
      int k, fk, fkg;
      if (ox3 > 0) k = ngh + 1, fk = ngh + 1, fkg = ngh + 2;
      else         k = ngh - 1, fk = ngh,     fkg = ngh - 1;
      Real ccval = cbuf(k, cj, ci);
      Real gx1c = 0.125*(cbuf(k, cj, ci+1) - cbuf(k, cj, ci-1));
      Real gx2c = 0.125*(cbuf(k, cj+1, ci) - cbuf(k, cj-1, ci));
      dst(fkg, l, l) = ot*(2.0*(ccval - gx1c - gx2c) + u(fk, l, l));
      dst(fkg, l, r) = ot*(2.0*(ccval + gx1c - gx2c) + u(fk, l, r));
      dst(fkg, r, l) = ot*(2.0*(ccval - gx1c + gx2c) + u(fk, r, l));
      dst(fkg, r, r) = ot*(2.0*(ccval + gx1c + gx2c) + u(fk, r, r));
    }
  }

  return;
}




//----------------------------------------------------------------------------------------
//! \fn void linearMG::CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &u,
//!                 const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
//!                 int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th)
//! \brief calculate Matrix element for FLD
//!        rlev = relative level from the finest level of this Multigrid block

void linearMG::CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &u,
                     const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
                     int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) {
  Real dx, dt = pmy_driver_->dt_;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real fac = dt/SQR(dx);
#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        for (int n = linearSolver::NMATRIX; n < pmy_driver_->nmatrix_; ++n)
          matrix(n,k,j,i) = 0.0;
        // center
        matrix(linearSolver::CCC,k,j,i) = fac*coeff(linearSolver::DCCF,k,j,i)
                                            + coeff(linearSolver::DCCS,k,j,i);
        // face
        matrix(linearSolver::CCM,k,j,i) = fac*coeff(linearSolver::DXMF,k,j,i)
                                            + coeff(linearSolver::DXMS,k,j,i);
        matrix(linearSolver::CCP,k,j,i) = fac*coeff(linearSolver::DXPF,k,j,i)
                                            + coeff(linearSolver::DXPS,k,j,i);
        matrix(linearSolver::CMC,k,j,i) = fac*coeff(linearSolver::DYMF,k,j,i)
                                            + coeff(linearSolver::DYMS,k,j,i);
        matrix(linearSolver::CPC,k,j,i) = fac*coeff(linearSolver::DYPF,k,j,i)
                                            + coeff(linearSolver::DYPS,k,j,i);
        matrix(linearSolver::MCC,k,j,i) = fac*coeff(linearSolver::DZMF,k,j,i)
                                            + coeff(linearSolver::DZMS,k,j,i);
        matrix(linearSolver::PCC,k,j,i) = fac*coeff(linearSolver::DZPF,k,j,i)
                                            + coeff(linearSolver::DZPS,k,j,i);
      }
    }
  }
  return;
}
