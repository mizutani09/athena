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
#include <iostream>
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
      pnrd_(pnrd) {
  eps_ = pin->GetOrAddReal("nrfld", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("nrfld", "niteration", -1);
  ffas_ = pin->GetOrAddBoolean("nrfld", "fas", ffas_);
  omega_ = pin->GetOrAddReal("nrfld", "omega", 1.0);
  fsteady_ = pin->GetOrAddBoolean("nrfld", "steady", false);
  npresmooth_ = pin->GetOrAddReal("nrfld", "npresmooth", 2);
  npostsmooth_ = pin->GetOrAddReal("nrfld", "npostsmooth", 2);
  fshowdef_ = pin->GetOrAddBoolean("nrfld", "show_defect", fshowdef_);
  smoothing_only_ = pin->GetOrAddBoolean("nrfld", "smoothing_only", false);
  coarse_corr_scale_ = pin->GetOrAddReal("nrfld", "coarse_correction_scale", 1.0);
  std::string smoother = pin->GetOrAddString("nrfld", "smoother", "jacobi-rb");
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
  mg_mesh_bcs_[inner_x1] = GetMGBoundaryFlag(pin->GetString("mesh", "ix1_bc") == "periodic" ?
                                              "periodic" : "zerofixed");
  mg_mesh_bcs_[outer_x1] = GetMGBoundaryFlag(pin->GetString("mesh", "ox1_bc") == "periodic" ?
                                              "periodic" : "zerofixed");
  mg_mesh_bcs_[inner_x2] = GetMGBoundaryFlag(pin->GetString("mesh", "ix2_bc") == "periodic" ?
                                              "periodic" : "zerofixed");
  mg_mesh_bcs_[outer_x2] = GetMGBoundaryFlag(pin->GetString("mesh", "ox2_bc") == "periodic" ?
                                              "periodic" : "zerofixed");
  mg_mesh_bcs_[inner_x3] = GetMGBoundaryFlag(pin->GetString("mesh", "ix3_bc") == "periodic" ?
                                              "periodic" : "zerofixed");
  mg_mesh_bcs_[outer_x3] = GetMGBoundaryFlag(pin->GetString("mesh", "ox3_bc") == "periodic" ?
                                              "periodic" : "zerofixed");

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
    // FLD2 *prfld = pmg->pmy_block_->prfld2;
    NewtonRaphson *pnr = pmg->pmy_block_->pnr;
    pmg->LoadSource(pnr->src_, 0, NGHOST, 1.0);
    pmg->LoadFinestData(pnr->delta_u_, 0, NGHOST); // caution! should be zero
    pmg->LoadCoefficients(pnr->coeff_, NGHOST);
    // pmg->AddFLDSource(prfld->source, NGHOST, dt_);
  }

  // std::cout << "Source loaded to linearMG. Start to solve... at " << Globals::my_rank << std::endl;

  SetupMultigrid(false);
  // std::cout << "setup done at " << Globals::my_rank << std::endl;
  if (mode_ == 0) {
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
    // FLD2 *prfld = pmg->pmy_block_->prfld2;
    NewtonRaphson *pnr = pmg->pnr_;
    // Hydro *phydro = pmg->pmy_block_->phydro;
    pmg->RetrieveResult(pnr->delta_u_, 0, NGHOST);
    // if (pnr->output_defect)
    //   pmg->RetrieveDefect(pnr->def_, 0, NGHOST);
  }

  // std::cout << "Result retrieved from linearMG at " << Globals::my_rank << std::endl;
  linmgtlist_->DoTaskListOneStage(pmy_mesh_, stage);
  // std::cout << "linearMG boundary conditions applied." << std::endl;
// #pragma omp parallel for num_threads(nthreads_)
//   for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
//     linearMG *pmg = static_cast<linearMG*>(*itr);
//     FLD2 *prfld = pmg->pmy_block_->prfld2;
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
