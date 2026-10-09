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
#include "../../globals.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../linear_solver/linearMG/linearMG.hpp"
#include "../../parameter_input.hpp"
#include "../../task_list/linmg_task_list.hpp"
#include "linearMG.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;

namespace {
  AthenaArray<Real> *temp; // temporary data for the Jacobi iteration
}

//----------------------------------------------------------------------------------------
//! \fn linearMGDriver::linearMGFLDDriver(Mesh *pm, ParameterInput *pin, NewtonRaphsonDriver *pnrd)
//! \brief linearMGDriver constructor

linearMGDriver::linearMGDriver(Mesh *pm, ParameterInput *pin, NewtonRaphsonDriver *pnrd)
    : MultigridDriver(pm, pm->LinearMGBoundaryFunction_,
                          pm->LinearMGCoeffBoundaryFunction_,
                          nullptr,
                          nullptr,
                          1, linearSolver::NCOEFF, linearSolver::NMATRIX),
      pnrd_(pnrd), cache_coefficient_hierarchy_(true),
      coefficient_hierarchy_cached_(false), npostsolve_smooth_(0),
      nsmoothing_only_sweeps_(256), direct_coarse_solve_(true),
      coarse_diagnostics_(false), coarse_direct_max_cells_(64),
      coarse_diagnostic_limit_(4), coarse_diagnostic_count_(0),
      coefficient_diagnostics_(false), coefficient_diagnostic_limit_(1),
      coefficient_diagnostic_count_(0) {
  eps_ = pin->GetOrAddReal("nrfld", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("nrfld", "niteration", -1);
  ffas_ = pin->GetOrAddBoolean("nrfld", "fas", ffas_);
  omega_ = pin->GetOrAddReal("nrfld", "omega", 1.0);
  fsteady_ = pin->GetOrAddBoolean("nrfld", "steady", false);
  npresmooth_ = pin->GetOrAddReal("nrfld", "npresmooth", 2);
  npostsmooth_ = pin->GetOrAddReal("nrfld", "npostsmooth", 2);
  fshowdef_ = pin->GetOrAddBoolean("nrfld", "show_defect", fshowdef_);
  smoothing_only_ = pin->GetOrAddBoolean("nrfld", "smoothing_only", false);
  cache_coefficient_hierarchy_ =
      pin->GetOrAddBoolean("nrfld", "cache_coefficient_hierarchy", true);
  relative_defect_ = true;
  coarse_corr_scale_ = pin->GetOrAddReal("nrfld", "coarse_correction_scale", 1.0);
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
  coefficient_diagnostic_limit_ = pin->GetOrAddInteger(
      "nrfld", "coefficient_diagnostic_limit", 1);
  if (coarse_direct_max_cells_ <= 0 || coarse_diagnostic_limit_ < 0
      || coefficient_diagnostic_limit_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGDriver::linearMGDriver" << std::endl
        << "coarse_direct_max_cells must be positive, and diagnostic limits "
        << "must be non-negative." << std::endl;
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
//   matrixmode_ = 1;
  matrixmode_ = 0; // caution!
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

  // mg_mesh_bcs_[inner_x1] = GetMGBoundaryFlag("zerofixed");
  // mg_mesh_bcs_[outer_x1] = GetMGBoundaryFlag("zerofixed");
  // mg_mesh_bcs_[inner_x2] = GetMGBoundaryFlag("zerofixed");
  // mg_mesh_bcs_[outer_x2] = GetMGBoundaryFlag("zerofixed");
  // mg_mesh_bcs_[inner_x3] = GetMGBoundaryFlag("zerofixed");
  // mg_mesh_bcs_[outer_x3] = GetMGBoundaryFlag("zerofixed");
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
    return matrix(linearSolver::CCC,kk,jj,ii)*u(v,kk,jj,ii)
         + matrix(linearSolver::CCM,kk,jj,ii)*u(v,kk,jj,ii-1)
         + matrix(linearSolver::CCP,kk,jj,ii)*u(v,kk,jj,ii+1)
         + matrix(linearSolver::CMC,kk,jj,ii)*u(v,kk,jj-1,ii)
         + matrix(linearSolver::CPC,kk,jj,ii)*u(v,kk,jj+1,ii)
         + matrix(linearSolver::MCC,kk,jj,ii)*u(v,kk-1,jj,ii)
         + matrix(linearSolver::PCC,kk,jj,ii)*u(v,kk+1,jj,ii);
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


//----------------------------------------------------------------------------------------
//! \brief Report coefficient contrast and coarse-operator consistency by MG level.

void linearMGDriver::PrintCoefficientDiagnostics() {
  auto report_level = [&](const char *kind, int lev, bool root_grid) {
    Real vmin[8] = {std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max(),
                    std::numeric_limits<Real>::max()};
    Real vmax[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
            vmax[0] = std::max(vmax[0], std::abs(dccs));
            vmax[1] = std::max(vmax[1], diffusion_diag);
            vmax[3] = std::max(vmax[3], std::abs(diag));
            vmax[4] = std::max(vmax[4], reaction_fraction);
            vmax[5] = std::max(vmax[5], row_error);
            vmax[7] = std::max(vmax[7], dccs);
            vmax[8] = std::max(vmax[8], diag);
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
    MPI_Allreduce(MPI_IN_PLACE, vmin, 8, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, vmax, 9, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, invalid_rows, 2, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if (Globals::my_rank == 0) {
      const Real face_contrast = vmax[2]/std::max(vmin[2], TINY_NUMBER);
      std::cout << "[NRFLD coefficients] kind=" << kind << " level=" << lev
                << " cells=" << nx << "x" << ny << "x" << nz
                << " DCCS=[" << vmin[0] << "," << vmax[0] << "]"
                << " diffusion_diag=[" << vmin[1] << "," << vmax[1] << "]"
                << " face_abs=[" << vmin[2] << "," << vmax[2] << "]"
                << " face_contrast=" << face_contrast
                << " diag=[" << vmin[3] << "," << vmax[3] << "]"
                << " reaction_fraction=[" << vmin[4] << "," << vmax[4] << "]"
                << " signed_DCCS=[" << vmin[5] << "," << vmax[7] << "]"
                << " signed_diag=[" << vmin[6] << "," << vmax[8] << "]"
                << " dominance_min=" << vmin[7]
                << " nonpositive_diag=" << invalid_rows[0]
                << " positive_or_nonfinite_offdiag=" << invalid_rows[1]
                << " row_error_max=" << vmax[5]
                << " face_asymmetry_max=" << vmax[6] << std::endl;
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

void linearMGDriver::Solve(int stage, Real dt) {
  // std::cout << "In linearMGDriver::Solve()" << std::endl;
  dt_ = dt;
  // Construct the linearMG array
  vmg_.clear();
  for (int i = 0; i < pmy_mesh_->nblocal; ++i)
    vmg_.push_back(pmy_mesh_->my_blocks(i)->pnr->plmg_);

  // load the source
#pragma omp parallel for num_threads(nthreads_)
  for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
    linearMG *pmg = static_cast<linearMG*>(*itr);
    // assume all the data are located on the same node
    // FLD *prfld = pmg->pmy_block_->prfld;
    NewtonRaphson *pnr = pmg->pmy_block_->pnr;
    pmg->LoadSource(pnr->src_, 0, NGHOST, 1.0);
    pmg->LoadFinestData(pnr->delta_u_, 0, NGHOST); // caution! should be zero
    if (!cache_coefficient_hierarchy_ || !coefficient_hierarchy_cached_
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

  const bool rebuild_coefficients = !cache_coefficient_hierarchy_
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
  if (coefficient_diagnostics_
      && coefficient_diagnostic_count_ < coefficient_diagnostic_limit_) {
    PrintCoefficientDiagnostics();
    ++coefficient_diagnostic_count_;
  }
  // std::cout << "setup done at " << Globals::my_rank << std::endl;
  if (smoothing_only_) {
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

  // std::cout << "Result retrieved from linearMG at " << Globals::my_rank << std::endl;
  linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
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

// void linearMG::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
void linearMG::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
         const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix, int rlev,
         int il, int iu, int jl, int ju, int kl, int ku, int color, bool th) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real dx2 = SQR(dx);
  Real isix = omega_/6.0;
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
              Real M = matrix(linearSolver::CCM,k,j,i)*u(k,j,i-1)+matrix(linearSolver::CCP,k,j,i)*u(k,j,i+1)
                     + matrix(linearSolver::CMC,k,j,i)*u(k,j-1,i)+matrix(linearSolver::CPC,k,j,i)*u(k,j+1,i)
                     + matrix(linearSolver::MCC,k,j,i)*u(k-1,j,i)+matrix(linearSolver::PCC,k,j,i)*u(k+1,j,i);
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
            Real M = matrix(linearSolver::CCM,k,j,i)*u(k,j,i-1)+matrix(linearSolver::CCP,k,j,i)*u(k,j,i+1)
                    + matrix(linearSolver::CMC,k,j,i)*u(k,j-1,i)+matrix(linearSolver::CPC,k,j,i)*u(k,j+1,i)
                    + matrix(linearSolver::MCC,k,j,i)*u(k-1,j,i)+matrix(linearSolver::PCC,k,j,i)*u(k+1,j,i);
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
      // std::cout << "rlev " << rlev << " il " << il << " iu " << iu << " jl " << jl << " ju " << ju << " kl " << kl << " ku " << ku << std::endl;
      // // std::cout << "CPRR " <<matrix(linearSolver::CPRR,1,1,1) << " CPRG " << matrix(linearSolver::CPRG,1,1,1) << " CPGR " <<matrix(linearSolver::CPGR,1,1,1) << " CPGG " << matrix(linearSolver::CPGG,1,1,1) << " CPGC " << matrix(linearSolver::CPGC,1,1,1) << " CPRC " << matrix(linearSolver::CPRC,1,1,1)<< std::endl;
      // // std::cout << "RSRC " << src(linearSolver::RAD,1,1,1) << " MGSRC " << matrix(linearSolver::CPRG,1,1,1)/matrix(linearSolver::CPGG,1,1,1)*src(linearSolver::GAS,1,1,1) << " MGCG " << matrix(linearSolver::CPRG,1,1,1)/matrix(linearSolver::CPGG,1,1,1)*matrix(linearSolver::CPGC,1,1,1) << " CPRC " <<matrix(linearSolver::CPRC,1,1,1)<< " CPRCS " << matrix(linearSolver::CPRCS,1,1,1) << std::endl;
      // // std::cout << src(linearSolver::RAD,1,1,1)-matrix(linearSolver::CPRG,1,1,1)/matrix(linearSolver::CPGG,1,1,1)*(src(linearSolver::GAS,1,1,1)-matrix(linearSolver::CPGC,1,1,1))-matrix(linearSolver::CPRC,1,1,1)<< std::endl;
      // // std::cout << "RAD " << u(linearSolver::RAD,1,1,1) << " GAS " << u(linearSolver::GAS,1,1,1) << " GSRC " <<src(linearSolver::GAS,1,1,1) << std::endl;
      // std::cout << "CCM = " << matrix(linearSolver::CCM,1,1,1) << ", CCP = " << matrix(linearSolver::CCP,1,1,1)
      //           << ", CMC = " << matrix(linearSolver::CMC,1,1,1) << ", CPC = " << matrix(linearSolver::CPC,1,1,1)
      //           << ", MCC = " << matrix(linearSolver::MCC,1,1,1) << ", PCC = " << matrix(linearSolver::PCC,1,1,1)
      //           << ", CCC = " << matrix(linearSolver::CCC,1,1,1) << std::endl;
      // std::cout << "src_ = " << src(1,1,1) << std::endl;
      // std::cout << "work_ = " << work(1,1,1) << std::endl;
      // std::cout << "u_ = " << u(1,1,1) << std::endl;
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
              // Real M = matrix(linearSolver::CCM,k,j,i)*u(k,j,i-1)   + matrix(linearSolver::CCP,k,j,i)*u(k,j,i+1)
              //        + matrix(linearSolver::CMC,k,j,i)*u(k,j-1,i)   + matrix(linearSolver::CPC,k,j,i)*u(k,j+1,i)
              //        + matrix(linearSolver::MCC,k,j,i)*u(k-1,j,i)   + matrix(linearSolver::PCC,k,j,i)*u(k+1,j,i)
              //        + matrix(linearSolver::CMM,k,j,i)*u(k,j-1,i-1) + matrix(linearSolver::CMP,k,j,i)*u(k,j-1,i+1)
              //        + matrix(linearSolver::CPM,k,j,i)*u(k,j+1,i-1) + matrix(linearSolver::CPP,k,j,i)*u(k,j+1,i+1)
              //        + matrix(linearSolver::MCM,k,j,i)*u(k-1,j,i-1) + matrix(linearSolver::MCP,k,j,i)*u(k-1,j,i+1)
              //        + matrix(linearSolver::PCM,k,j,i)*u(k+1,j,i-1) + matrix(linearSolver::PCP,k,j,i)*u(k+1,j,i+1)
              //        + matrix(linearSolver::MMC,k,j,i)*u(k-1,j-1,i) + matrix(linearSolver::MPC,k,j,i)*u(k-1,j+1,i)
              //        + matrix(linearSolver::PMC,k,j,i)*u(k+1,j-1,i) + matrix(linearSolver::PPC,k,j,i)*u(k+1,j+1,i);
              //   work(k,j,i) = (src(k,j,i) - M) / matrix(linearSolver::CCC,k,j,i);
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
            Real M = matrix(linearSolver::CCM,k,j,i)*u(k,j,i-1)+matrix(linearSolver::CCP,k,j,i)*u(k,j,i+1)
                   + matrix(linearSolver::CMC,k,j,i)*u(k,j-1,i)+matrix(linearSolver::CPC,k,j,i)*u(k,j+1,i)
                   + matrix(linearSolver::MCC,k,j,i)*u(k-1,j,i)+matrix(linearSolver::PCC,k,j,i)*u(k+1,j,i);
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
        Real M = matrix(linearSolver::CCC,k,j,i)*u(k,j,i)
               + matrix(linearSolver::CCM,k,j,i)*u(k,j,i-1)+matrix(linearSolver::CCP,k,j,i)*u(k,j,i+1)
               + matrix(linearSolver::CMC,k,j,i)*u(k,j-1,i)+matrix(linearSolver::CPC,k,j,i)*u(k,j+1,i)
               + matrix(linearSolver::MCC,k,j,i)*u(k-1,j,i)+matrix(linearSolver::PCC,k,j,i)*u(k+1,j,i);
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
        Real M = matrix(linearSolver::CCC,k,j,i)*u(k,j,i)
               + matrix(linearSolver::CCM,k,j,i)*u(k,j,i-1)+matrix(linearSolver::CCP,k,j,i)*u(k,j,i+1)
               + matrix(linearSolver::CMC,k,j,i)*u(k,j-1,i)+matrix(linearSolver::CPC,k,j,i)*u(k,j+1,i)
               + matrix(linearSolver::MCC,k,j,i)*u(k-1,j,i)+matrix(linearSolver::PCC,k,j,i)*u(k+1,j,i);
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
  Real idx = 1.0/dx;
  Real fac = dt/SQR(dx), efac = 0.125*fac;
#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
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

  // // output for 1,1,1
  // std::cout << "matrix CCC = " << matrix(linearSolver::CCC,1,1,1) << std::endl
  //           << "       CCM = " << matrix(linearSolver::CCM,1,1,1)
  //           << ", CCP = " << matrix(linearSolver::CCP,1,1,1) << std::endl
  //           << "       CMC = " << matrix(linearSolver::CMC,1,1,1)
  //           << ", CPC = " << matrix(linearSolver::CPC,1,1,1) << std::endl
  //           << "       MCC = " << matrix(linearSolver::MCC,1,1,1)
  //           << ", PCC = " << matrix(linearSolver::PCC,1,1,1) << std::endl;
  return;
}
