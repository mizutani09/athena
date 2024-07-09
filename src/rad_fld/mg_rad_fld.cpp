//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_rad_fld.cpp
//! \brief create multigrid solver for FLD

// C headers

// C++ headers
#include <algorithm>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../task_list/rad_fld_task_list.hpp"
#include "mg_rad_fld.hpp"
#include "rad_fld.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;

namespace {
  AthenaArray<Real> *temp; // temporary data for the Jacobi iteration
}

//----------------------------------------------------------------------------------------
//! \fn MGFLDDriver::MGFLDDriver(Mesh *pm, ParameterInput *pin)
//! \brief MGFLDDriver constructor

MGFLDDriver::MGFLDDriver(Mesh *pm, ParameterInput *pin)
    : MultigridDriver(pm, pm->MGFLDBoundaryFunction_,
                          pm->MGFLDCoeffBoundaryFunction_,
                          pm->MGFLDSourceMaskFunction_,
                          pm->MGFLDCoeffMaskFunction_,
                          1, RadFLD::NCOEFF, RadFLD::NMATRIX) {
  eps_ = pin->GetOrAddReal("mgfld", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("mgfld", "niteration", -1);
  ffas_ = pin->GetOrAddBoolean("mgfld", "fas", ffas_);
  fsubtract_average_ = false;
  omega_ = pin->GetOrAddReal("mgfld", "omega", 1.0);
  fsteady_ = pin->GetOrAddBoolean("mgfld", "steady", false);
  npresmooth_ = pin->GetOrAddReal("mgfld", "npresmooth", 2);
  npostsmooth_ = pin->GetOrAddReal("mgfld", "npostsmooth", 2);
  fshowdef_ = pin->GetOrAddBoolean("mgfld", "show_defect", fshowdef_);
  std::string smoother = pin->GetOrAddString("mgfld", "smoother", "jacobi-rb");
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
  std::string prol = pin->GetOrAddString("mgfld", "prolongation", "trilinear");
  if (prol == "tricubic")
    fprolongation_ = 1;

  std::string m = pin->GetOrAddString("mgfld", "mgmode", "none");
  std::transform(m.begin(), m.end(), m.begin(), ::tolower);
  if (m == "fmg") {
    mode_ = 0;
  } else if (m == "mgi") {
    mode_ = 1; // Iterative
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in MGFLDDriver::MGFLDDriver" << std::endl
        << "The \"mgmode\" parameter in the <mgfld> block is invalid." << std::endl
        << "FMG: Full Multigrid + Multigrid iteration (default)" << std::endl
        << "MGI: Multigrid Iteration" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (eps_ < 0.0 && niter_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MGFLDDriver::MGFLDDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <mgfld> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
    ATHENA_ERROR(msg);
  }
  mg_mesh_bcs_[inner_x1] =
              GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ix1_bc", "none"));
  mg_mesh_bcs_[outer_x1] =
              GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ox1_bc", "none"));
  mg_mesh_bcs_[inner_x2] =
              GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ix2_bc", "none"));
  mg_mesh_bcs_[outer_x2] =
              GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ox2_bc", "none"));
  mg_mesh_bcs_[inner_x3] =
              GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ix3_bc", "none"));
  mg_mesh_bcs_[outer_x3] =
              GetMGBoundaryFlag(pin->GetOrAddString("mgfld", "ox3_bc", "none"));
  CheckBoundaryFunctions();

  mgtlist_ = new MultigridTaskList(this);

  // Allocate the root multigrid
  mgroot_ = new MGFLD(this, nullptr);

  fldtlist_ = new FLDBoundaryTaskList(pin, pm);

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
//! \fn MGFLDDriver::~MGFLDDriver()
//! \brief MGFLDDriver destructor

MGFLDDriver::~MGFLDDriver() {
  delete fldtlist_;
  delete mgroot_;
  delete mgtlist_;
  delete [] temp;
}


//----------------------------------------------------------------------------------------
//! \fn MGFLD::MGFLD(MGFLDDriver *pmd, MeshBlock *pmb)
//! \brief MGFLD constructor

MGFLD::MGFLD(MGFLDDriver *pmd, MeshBlock *pmb)
  : Multigrid(pmd, pmb, 1), omega_(pmd->omega_), fsmoother_(pmd->fsmoother_) {
  btype = btypef = BoundaryQuantity::mg;
  pmgbval = new MGBoundaryValues(this, mg_block_bcs_);
}


//----------------------------------------------------------------------------------------
//! \fn MGFLD::~MGFLD()
//! \brief MGFLD deconstructor

MGFLD::~MGFLD() {
  delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGFLDDriver::AddFLDSource(const AthenaArray<Real> &src,
//!                                           int ngh, Real dt)
//! \brief Add the FLD source term

void MGFLD::AddFLDSource(const AthenaArray<Real> &src, int ngh, Real dt) {
  AthenaArray<Real> &dst=src_[nlevel_-1];
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+size_.nx1-1, je=js+size_.nx2-1, ke=ks+size_.nx3-1;
  if (!(static_cast<MGFLDDriver*>(pmy_driver_)->fsteady_)) {
    for (int mk=ks; mk<=ke; ++mk) {
      int k = mk - ks + ngh;
      for (int mj=js; mj<=je; ++mj) {
        int j = mj - js + ngh;
#pragma omp simd
        for (int mi=is; mi<=ie; ++mi) {
          int i = mi - is + ngh;
          dst(mk,mj,mi) += dt * src(k,j,i);
        }
      }
    }
  } else {
    for (int mk=ks; mk<=ke; ++mk) {
      int k = mk - ks + ngh;
      for (int mj=js; mj<=je; ++mj) {
        int j = mj - js + ngh;
#pragma omp simd
        for (int mi=is; mi<=ie; ++mi) {
          int i = mi - is + ngh;
          dst(mk,mj,mi) = src(k,j,i);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGFLDDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

void MGFLDDriver::Solve(int stage, Real dt) {
  // Construct the Multigrid array
  vmg_.clear();
  for (int i = 0; i < pmy_mesh_->nblocal; ++i)
    vmg_.push_back(pmy_mesh_->my_blocks(i)->pfld->pmg);

  // load the source
#pragma omp parallel for num_threads(nthreads_)
  for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
    MGFLD *pmg = static_cast<MGFLD*>(*itr);
    // assume all the data are located on the same node
    FLD *pfld = pmg->pmy_block_->pfld;
    Hydro *phydro = pmg->pmy_block_->phydro;
    Field *pfield = pmg->pmy_block_->pfield;
    pfld->CalculateCoefficients(phydro->w, pfield->bcc);
    if (!fsteady_)
      pmg->LoadSource(pfld->ecr, 0, NGHOST, 1.0); //!
    if (mode_ == 1) // load the current data as the initial guess
      pmg->LoadFinestData(pfld->ecr, 0, NGHOST); //!
    pmg->LoadCoefficients(pfld->coeff, NGHOST);
    pmg->AddFLDSource(pfld->source, NGHOST, dt);
  }

  if (dt > 0.0 || fsteady_) {
    SetupMultigrid(dt, false);
    if (mode_ == 0) {
      SolveFMGCycle();
    } else {
      if (eps_ >= 0.0)
        SolveIterative();
      else
        SolveIterativeFixedTimes();
    }
  } else { // just copy trivial solution and set boundaries
    SetupMultigrid(dt, true);
    if (mode_ != 1) {
#pragma omp parallel for num_threads(nthreads_)
      for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
        MGFLD *pmg = static_cast<MGFLD*>(*itr);
        AthenaArray<Real> &ecr = pmg->GetCurrentData(); //!
        AthenaArray<Real> &ecr0 = pmg->GetCurrentSource(); //!
        ecr = ecr0; //!
      }
    }
    mgtlist_->SetMGTaskListBoundaryCommunication();
    mgtlist_->DoTaskListOneStage(this);
  }

  // Return the result
#pragma omp parallel for num_threads(nthreads_)
  for (auto itr = vmg_.begin(); itr < vmg_.end(); itr++) {
    Multigrid *pmg = *itr;
    FLD *pfld = pmg->pmy_block_->pfld;
    pmg->RetrieveResult(pfld->ecr, 0, NGHOST); //!
    if(pfld->output_defect)
      pmg->RetrieveDefect(pfld->def, 0, NGHOST);
  }

  fldtlist_->DoTaskListOneStage(pmy_mesh_, stage);

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MGFLD::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
//!            const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix, int rlev,
//!            int il, int iu, int jl, int ju, int kl, int ku, int color, bool th)
//! \brief Implementation of the Red-Black Gauss-Seidel Smoother
//!        rlev = relative level from the finest level of this Multigrid block

void MGFLD::Smooth(AthenaArray<Real> &u, const AthenaArray<Real> &src,
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
              Real M = matrix(RadFLD::CCM,k,j,i)*u(k,j,i-1)   + matrix(RadFLD::CCP,k,j,i)*u(k,j,i+1)
                     + matrix(RadFLD::CMC,k,j,i)*u(k,j-1,i)   + matrix(RadFLD::CPC,k,j,i)*u(k,j+1,i)
                     + matrix(RadFLD::MCC,k,j,i)*u(k-1,j,i)   + matrix(RadFLD::PCC,k,j,i)*u(k+1,j,i)
                     + matrix(RadFLD::CMM,k,j,i)*u(k,j-1,i-1) + matrix(RadFLD::CMP,k,j,i)*u(k,j-1,i+1)
                     + matrix(RadFLD::CPM,k,j,i)*u(k,j+1,i-1) + matrix(RadFLD::CPP,k,j,i)*u(k,j+1,i+1)
                     + matrix(RadFLD::MCM,k,j,i)*u(k-1,j,i-1) + matrix(RadFLD::MCP,k,j,i)*u(k-1,j,i+1)
                     + matrix(RadFLD::PCM,k,j,i)*u(k+1,j,i-1) + matrix(RadFLD::PCP,k,j,i)*u(k+1,j,i+1)
                     + matrix(RadFLD::MMC,k,j,i)*u(k-1,j-1,i) + matrix(RadFLD::MPC,k,j,i)*u(k-1,j+1,i)
                     + matrix(RadFLD::PMC,k,j,i)*u(k+1,j-1,i) + matrix(RadFLD::PPC,k,j,i)*u(k+1,j+1,i);
                work(k,j,i) = (src(k,j,i) - M) / matrix(RadFLD::CCC,k,j,i);
            }
          }
        }
#pragma omp for
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            int c = (color + k + j) & 1;
#pragma ivdep
            for (int i=il+c; i<=iu; i+=2)
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
          int c = (color + k + j) & 1;
#pragma ivdep
          for (int i=il+c; i<=iu; i+=2) {
            Real M = matrix(RadFLD::CCM,k,j,i)*u(k,j,i-1)   + matrix(RadFLD::CCP,k,j,i)*u(k,j,i+1)
                   + matrix(RadFLD::CMC,k,j,i)*u(k,j-1,i)   + matrix(RadFLD::CPC,k,j,i)*u(k,j+1,i)
                   + matrix(RadFLD::MCC,k,j,i)*u(k-1,j,i)   + matrix(RadFLD::PCC,k,j,i)*u(k+1,j,i)
                   + matrix(RadFLD::CMM,k,j,i)*u(k,j-1,i-1) + matrix(RadFLD::CMP,k,j,i)*u(k,j-1,i+1)
                   + matrix(RadFLD::CPM,k,j,i)*u(k,j+1,i-1) + matrix(RadFLD::CPP,k,j,i)*u(k,j+1,i+1)
                   + matrix(RadFLD::MCM,k,j,i)*u(k-1,j,i-1) + matrix(RadFLD::MCP,k,j,i)*u(k-1,j,i+1)
                   + matrix(RadFLD::PCM,k,j,i)*u(k+1,j,i-1) + matrix(RadFLD::PCP,k,j,i)*u(k+1,j,i+1)
                   + matrix(RadFLD::MMC,k,j,i)*u(k-1,j-1,i) + matrix(RadFLD::MPC,k,j,i)*u(k-1,j+1,i)
                   + matrix(RadFLD::PMC,k,j,i)*u(k+1,j-1,i) + matrix(RadFLD::PPC,k,j,i)*u(k+1,j+1,i);
            work(k,j,i) = (src(k,j,i) - M) / matrix(RadFLD::CCC,k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          int c = (color + k + j) & 1;
#pragma ivdep
          for (int i=il+c; i<=iu; i+=2)
            u(k,j,i) += omega_ * (work(k,j,i) - u(k,j,i));
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
              Real M = matrix(RadFLD::CCM,k,j,i)*u(k,j,i-1)   + matrix(RadFLD::CCP,k,j,i)*u(k,j,i+1)
                     + matrix(RadFLD::CMC,k,j,i)*u(k,j-1,i)   + matrix(RadFLD::CPC,k,j,i)*u(k,j+1,i)
                     + matrix(RadFLD::MCC,k,j,i)*u(k-1,j,i)   + matrix(RadFLD::PCC,k,j,i)*u(k+1,j,i)
                     + matrix(RadFLD::CMM,k,j,i)*u(k,j-1,i-1) + matrix(RadFLD::CMP,k,j,i)*u(k,j-1,i+1)
                     + matrix(RadFLD::CPM,k,j,i)*u(k,j+1,i-1) + matrix(RadFLD::CPP,k,j,i)*u(k,j+1,i+1)
                     + matrix(RadFLD::MCM,k,j,i)*u(k-1,j,i-1) + matrix(RadFLD::MCP,k,j,i)*u(k-1,j,i+1)
                     + matrix(RadFLD::PCM,k,j,i)*u(k+1,j,i-1) + matrix(RadFLD::PCP,k,j,i)*u(k+1,j,i+1)
                     + matrix(RadFLD::MMC,k,j,i)*u(k-1,j-1,i) + matrix(RadFLD::MPC,k,j,i)*u(k-1,j+1,i)
                     + matrix(RadFLD::PMC,k,j,i)*u(k+1,j-1,i) + matrix(RadFLD::PPC,k,j,i)*u(k+1,j+1,i);
                work(k,j,i) = (src(k,j,i) - M) / matrix(RadFLD::CCC,k,j,i);
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
            Real M = matrix(RadFLD::CCM,k,j,i)*u(k,j,i-1)   + matrix(RadFLD::CCP,k,j,i)*u(k,j,i+1)
                   + matrix(RadFLD::CMC,k,j,i)*u(k,j-1,i)   + matrix(RadFLD::CPC,k,j,i)*u(k,j+1,i)
                   + matrix(RadFLD::MCC,k,j,i)*u(k-1,j,i)   + matrix(RadFLD::PCC,k,j,i)*u(k+1,j,i)
                   + matrix(RadFLD::CMM,k,j,i)*u(k,j-1,i-1) + matrix(RadFLD::CMP,k,j,i)*u(k,j-1,i+1)
                   + matrix(RadFLD::CPM,k,j,i)*u(k,j+1,i-1) + matrix(RadFLD::CPP,k,j,i)*u(k,j+1,i+1)
                   + matrix(RadFLD::MCM,k,j,i)*u(k-1,j,i-1) + matrix(RadFLD::MCP,k,j,i)*u(k-1,j,i+1)
                   + matrix(RadFLD::PCM,k,j,i)*u(k+1,j,i-1) + matrix(RadFLD::PCP,k,j,i)*u(k+1,j,i+1)
                   + matrix(RadFLD::MMC,k,j,i)*u(k-1,j-1,i) + matrix(RadFLD::MPC,k,j,i)*u(k-1,j+1,i)
                   + matrix(RadFLD::PMC,k,j,i)*u(k+1,j-1,i) + matrix(RadFLD::PPC,k,j,i)*u(k+1,j+1,i);
            work(k,j,i) = (src(k,j,i) - M) / matrix(RadFLD::CCC,k,j,i);
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
//! \fn void MGFLD::CalculateDefect(AthenaArray<Real> &def,
//!            const AthenaArray<Real> &u, const AthenaArray<Real> &src,
//!            const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
//!            int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th)
//! \brief Implementation of the Defect calculation
//!        rlev = relative level from the finest level of this Multigrid block

void MGFLD::CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
                    const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
                    const AthenaArray<Real> &matrix, int rlev, int il, int iu,
                    int jl, int ju, int kl, int ku, bool th) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);

#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        Real M = matrix(RadFLD::CCC,k,j,i)*u(k,j,i)
               + matrix(RadFLD::CCM,k,j,i)*u(k,j,i-1)   + matrix(RadFLD::CCP,k,j,i)*u(k,j,i+1)
               + matrix(RadFLD::CMC,k,j,i)*u(k,j-1,i)   + matrix(RadFLD::CPC,k,j,i)*u(k,j+1,i)
               + matrix(RadFLD::MCC,k,j,i)*u(k-1,j,i)   + matrix(RadFLD::PCC,k,j,i)*u(k+1,j,i)
               + matrix(RadFLD::CMM,k,j,i)*u(k,j-1,i-1) + matrix(RadFLD::CMP,k,j,i)*u(k,j-1,i+1)
               + matrix(RadFLD::CPM,k,j,i)*u(k,j+1,i-1) + matrix(RadFLD::CPP,k,j,i)*u(k,j+1,i+1)
               + matrix(RadFLD::MCM,k,j,i)*u(k-1,j,i-1) + matrix(RadFLD::MCP,k,j,i)*u(k-1,j,i+1)
               + matrix(RadFLD::PCM,k,j,i)*u(k+1,j,i-1) + matrix(RadFLD::PCP,k,j,i)*u(k+1,j,i+1)
               + matrix(RadFLD::MMC,k,j,i)*u(k-1,j-1,i) + matrix(RadFLD::MPC,k,j,i)*u(k-1,j+1,i)
               + matrix(RadFLD::PMC,k,j,i)*u(k+1,j-1,i) + matrix(RadFLD::PPC,k,j,i)*u(k+1,j+1,i);
        def(k,j,i) = src(k,j,i) - M;
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MGFLD::CalculateFASRHS(AthenaArray<Real> &src,
//!            const AthenaArray<Real> &u, const AthenaArray<Real> &coeff,
//!            const AthenaArray<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
//!            int kl, int ku, bool th)
//! \brief Implementation of the RHS calculation for FAS
//!        rlev = relative level from the finest level of this Multigrid block

void MGFLD::CalculateFASRHS(AthenaArray<Real> &src, const AthenaArray<Real> &u,
                    const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
                    int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  Real idx2 = 1.0/SQR(dx);
#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        Real M = matrix(RadFLD::CCC,k,j,i)*u(k,j,i)
               + matrix(RadFLD::CCM,k,j,i)*u(k,j,i-1)   + matrix(RadFLD::CCP,k,j,i)*u(k,j,i+1)
               + matrix(RadFLD::CMC,k,j,i)*u(k,j-1,i)   + matrix(RadFLD::CPC,k,j,i)*u(k,j+1,i)
               + matrix(RadFLD::MCC,k,j,i)*u(k-1,j,i)   + matrix(RadFLD::PCC,k,j,i)*u(k+1,j,i)
               + matrix(RadFLD::CMM,k,j,i)*u(k,j-1,i-1) + matrix(RadFLD::CMP,k,j,i)*u(k,j-1,i+1)
               + matrix(RadFLD::CPM,k,j,i)*u(k,j+1,i-1) + matrix(RadFLD::CPP,k,j,i)*u(k,j+1,i+1)
               + matrix(RadFLD::MCM,k,j,i)*u(k-1,j,i-1) + matrix(RadFLD::MCP,k,j,i)*u(k-1,j,i+1)
               + matrix(RadFLD::PCM,k,j,i)*u(k+1,j,i-1) + matrix(RadFLD::PCP,k,j,i)*u(k+1,j,i+1)
               + matrix(RadFLD::MMC,k,j,i)*u(k-1,j-1,i) + matrix(RadFLD::MPC,k,j,i)*u(k-1,j+1,i)
               + matrix(RadFLD::PMC,k,j,i)*u(k+1,j-1,i) + matrix(RadFLD::PPC,k,j,i)*u(k+1,j+1,i);
        src(k,j,i) += M;
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MGFLD::CalculateMatrix(AthenaArray<Real> &matrix,
//!                         const AthenaArray<Real> &coeff, int rlev, Real dt,
//!                         int il, int iu, int jl, int ju, int kl, int ku, bool th)
//! \brief calculate Matrix element for cosmic ray transport
//!        rlev = relative level from the finest level of this Multigrid block

void MGFLD::CalculateMatrix(AthenaArray<Real> &matrix,
                             const AthenaArray<Real> &coeff, Real dt, int rlev,
                             int il, int iu, int jl, int ju, int kl, int ku, bool th) {
  Real dx;
  if (rlev <= 0) dx = rdx_*static_cast<Real>(1<<(-rlev));
  else           dx = rdx_/static_cast<Real>(1<<rlev);
  if (!(static_cast<MGFLDDriver*>(pmy_driver_)->fsteady_)) { // time-dependent
    Real fac = dt/SQR(dx), efac = 0.125*fac;
#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
#pragma omp simd
        for (int i=il; i<=iu; i++) {
          // center
          matrix(RadFLD::CCC,k,j,i) = 1.0 + dt*coeff(RadFLD::NLAMBDA,k,j,i) + 0.5*fac*(
                              2.0*coeff(RadFLD::DXX,k,j,i)+coeff(RadFLD::DXX,k,j,i+1)+coeff(RadFLD::DXX,k,j,i-1)
                            + 2.0*coeff(RadFLD::DYY,k,j,i)+coeff(RadFLD::DYY,k,j+1,i)+coeff(RadFLD::DYY,k,j-1,i)
                            + 2.0*coeff(RadFLD::DZZ,k,j,i)+coeff(RadFLD::DZZ,k+1,j,i)+coeff(RadFLD::DZZ,k-1,j,i));
          // face
          matrix(RadFLD::CCM,k,j,i) = fac*(-0.5*(coeff(RadFLD::DXX,k,j,i) + coeff(RadFLD::DXX,k,j,i-1))
                                +0.125*((coeff(RadFLD::DYX,k,j+1,i)-coeff(RadFLD::DYX,k,j-1,i))
                                       +(coeff(RadFLD::DZX,k+1,j,i)-coeff(RadFLD::DZX,k-1,j,i))));
          matrix(RadFLD::CCP,k,j,i) = fac*(-0.5*(coeff(RadFLD::DXX,k,j,i+1)+coeff(RadFLD::DXX,k,j,i))
                                -0.125*((coeff(RadFLD::DYX,k,j+1,i)-coeff(RadFLD::DYX,k,j-1,i))
                                       +(coeff(RadFLD::DZX,k+1,j,i)-coeff(RadFLD::DZX,k-1,j,i))));
          matrix(RadFLD::CMC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DYY,k,j,i) + coeff(RadFLD::DYY,k,j-1,i))
                                +0.125*((coeff(RadFLD::DXY,k,j,i+1)-coeff(RadFLD::DXY,k,j,i-1))
                                       +(coeff(RadFLD::DZY,k+1,j,i)-coeff(RadFLD::DZY,k-1,j,i))));
          matrix(RadFLD::CPC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DYY,k,j+1,i)+coeff(RadFLD::DYY,k,j,i))
                                -0.125*((coeff(RadFLD::DXY,k,j,i+1)-coeff(RadFLD::DXY,k,j,i-1))
                                       +(coeff(RadFLD::DZY,k+1,j,i)-coeff(RadFLD::DZY,k-1,j,i))));
          matrix(RadFLD::MCC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DZZ,k,j,i) + coeff(RadFLD::DZZ,k-1,j,i))
                                +0.125*((coeff(RadFLD::DYZ,k,j+1,i)-coeff(RadFLD::DYZ,k,j-1,i))
                                       +(coeff(RadFLD::DXZ,k,j,i+1)-coeff(RadFLD::DXZ,k,j,i-1))));
          matrix(RadFLD::PCC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DZZ,k+1,j,i)+coeff(RadFLD::DZZ,k,j,i))
                                -0.125*((coeff(RadFLD::DYZ,k,j+1,i)-coeff(RadFLD::DYZ,k,j-1,i))
                                       +(coeff(RadFLD::DXZ,k,j,i+1)-coeff(RadFLD::DXZ,k,j,i-1))));
          // edge
          matrix(RadFLD::CMM,k,j,i) = -efac*(coeff(RadFLD::DXY,k,j,i) + coeff(RadFLD::DXY,k,j,i-1)
                                    +coeff(RadFLD::DYX,k,j,i) + coeff(RadFLD::DYX,k,j-1,i));
          matrix(RadFLD::CMP,k,j,i) =  efac*(coeff(RadFLD::DXY,k,j,i+1)+coeff(RadFLD::DXY,k,j,i)
                                    +coeff(RadFLD::DYX,k,j,i) + coeff(RadFLD::DYX,k,j-1,i));
          matrix(RadFLD::CPM,k,j,i) =  efac*(coeff(RadFLD::DXY,k,j,i) + coeff(RadFLD::DXY,k,j,i-1)
                                    +coeff(RadFLD::DYX,k,j+1,i)+coeff(RadFLD::DYX,k,j,i));
          matrix(RadFLD::CPP,k,j,i) = -efac*(coeff(RadFLD::DXY,k,j,i+1)+coeff(RadFLD::DXY,k,j,i)
                                    +coeff(RadFLD::DYX,k,j+1,i)+coeff(RadFLD::DYX,k,j,i));
          matrix(RadFLD::MCM,k,j,i) = -efac*(coeff(RadFLD::DZX,k,j,i) + coeff(RadFLD::DZX,k-1,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i) + coeff(RadFLD::DXZ,k,j,i-1));
          matrix(RadFLD::MCP,k,j,i) =  efac*(coeff(RadFLD::DZX,k,j,i) + coeff(RadFLD::DZX,k-1,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i+1)+coeff(RadFLD::DXZ,k,j,i));
          matrix(RadFLD::PCM,k,j,i) =  efac*(coeff(RadFLD::DZX,k+1,j,i)+coeff(RadFLD::DZX,k,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i) + coeff(RadFLD::DXZ,k,j,i-1));
          matrix(RadFLD::PCP,k,j,i) = -efac*(coeff(RadFLD::DZX,k+1,j,i)+coeff(RadFLD::DZX,k,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i+1)+coeff(RadFLD::DXZ,k,j,i));
          matrix(RadFLD::MMC,k,j,i) = -efac*(coeff(RadFLD::DYZ,k,j,i) + coeff(RadFLD::DYZ,k,j-1,i)
                                    +coeff(RadFLD::DZY,k,j,i) + coeff(RadFLD::DZY,k-1,j,i));
          matrix(RadFLD::MPC,k,j,i) =  efac*(coeff(RadFLD::DYZ,k,j+1,i)+coeff(RadFLD::DYZ,k,j,i)
                                    +coeff(RadFLD::DZY,k,j,i) + coeff(RadFLD::DZY,k-1,j,i));
          matrix(RadFLD::PMC,k,j,i) =  efac*(coeff(RadFLD::DYZ,k,j,i) + coeff(RadFLD::DYZ,k,j-1,i)
                                    +coeff(RadFLD::DZY,k+1,j,i)+coeff(RadFLD::DZY,k,j,i));
          matrix(RadFLD::PPC,k,j,i) = -efac*(coeff(RadFLD::DYZ,k,j+1,i)+coeff(RadFLD::DYZ,k,j,i)
                                    +coeff(RadFLD::DZY,k+1,j,i)+coeff(RadFLD::DZY,k,j,i));
        }
      }
    }
  } else { // steady state
    Real fac = 1.0/SQR(dx), efac = 0.125*fac;
#pragma omp parallel for num_threads(pmy_driver_->nthreads_) if (th && (ku-kl) >= minth_)
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
#pragma omp simd
        for (int i=il; i<=iu; i++) {
          // center
          matrix(RadFLD::CCC,k,j,i) = coeff(RadFLD::NLAMBDA,k,j,i) + 0.5*fac*(
                              2.0*coeff(RadFLD::DXX,k,j,i)+coeff(RadFLD::DXX,k,j,i+1)+coeff(RadFLD::DXX,k,j,i-1)
                            + 2.0*coeff(RadFLD::DYY,k,j,i)+coeff(RadFLD::DYY,k,j+1,i)+coeff(RadFLD::DYY,k,j-1,i)
                            + 2.0*coeff(RadFLD::DZZ,k,j,i)+coeff(RadFLD::DZZ,k+1,j,i)+coeff(RadFLD::DZZ,k-1,j,i));
          // face
          matrix(RadFLD::CCM,k,j,i) = fac*(-0.5*(coeff(RadFLD::DXX,k,j,i) + coeff(RadFLD::DXX,k,j,i-1))
                                +0.125*((coeff(RadFLD::DYX,k,j+1,i)-coeff(RadFLD::DYX,k,j-1,i))
                                       +(coeff(RadFLD::DZX,k+1,j,i)-coeff(RadFLD::DZX,k-1,j,i))));
          matrix(RadFLD::CCP,k,j,i) = fac*(-0.5*(coeff(RadFLD::DXX,k,j,i+1)+coeff(RadFLD::DXX,k,j,i))
                                -0.125*((coeff(RadFLD::DYX,k,j+1,i)-coeff(RadFLD::DYX,k,j-1,i))
                                       +(coeff(RadFLD::DZX,k+1,j,i)-coeff(RadFLD::DZX,k-1,j,i))));
          matrix(RadFLD::CMC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DYY,k,j,i) + coeff(RadFLD::DYY,k,j-1,i))
                                +0.125*((coeff(RadFLD::DXY,k,j,i+1)-coeff(RadFLD::DXY,k,j,i-1))
                                       +(coeff(RadFLD::DZY,k+1,j,i)-coeff(RadFLD::DZY,k-1,j,i))));
          matrix(RadFLD::CPC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DYY,k,j+1,i)+coeff(RadFLD::DYY,k,j,i))
                                -0.125*((coeff(RadFLD::DXY,k,j,i+1)-coeff(RadFLD::DXY,k,j,i-1))
                                       +(coeff(RadFLD::DZY,k+1,j,i)-coeff(RadFLD::DZY,k-1,j,i))));
          matrix(RadFLD::MCC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DZZ,k,j,i) + coeff(RadFLD::DZZ,k-1,j,i))
                                +0.125*((coeff(RadFLD::DYZ,k,j+1,i)-coeff(RadFLD::DYZ,k,j-1,i))
                                       +(coeff(RadFLD::DXZ,k,j,i+1)-coeff(RadFLD::DXZ,k,j,i-1))));
          matrix(RadFLD::PCC,k,j,i) = fac*(-0.5*(coeff(RadFLD::DZZ,k+1,j,i)+coeff(RadFLD::DZZ,k,j,i))
                                -0.125*((coeff(RadFLD::DYZ,k,j+1,i)-coeff(RadFLD::DYZ,k,j-1,i))
                                       +(coeff(RadFLD::DXZ,k,j,i+1)-coeff(RadFLD::DXZ,k,j,i-1))));
          // edge
          matrix(RadFLD::CMM,k,j,i) = -efac*(coeff(RadFLD::DXY,k,j,i) + coeff(RadFLD::DXY,k,j,i-1)
                                    +coeff(RadFLD::DYX,k,j,i) + coeff(RadFLD::DYX,k,j-1,i));
          matrix(RadFLD::CMP,k,j,i) =  efac*(coeff(RadFLD::DXY,k,j,i+1)+coeff(RadFLD::DXY,k,j,i)
                                    +coeff(RadFLD::DYX,k,j,i) + coeff(RadFLD::DYX,k,j-1,i));
          matrix(RadFLD::CPM,k,j,i) =  efac*(coeff(RadFLD::DXY,k,j,i) + coeff(RadFLD::DXY,k,j,i-1)
                                    +coeff(RadFLD::DYX,k,j+1,i)+coeff(RadFLD::DYX,k,j,i));
          matrix(RadFLD::CPP,k,j,i) = -efac*(coeff(RadFLD::DXY,k,j,i+1)+coeff(RadFLD::DXY,k,j,i)
                                    +coeff(RadFLD::DYX,k,j+1,i)+coeff(RadFLD::DYX,k,j,i));
          matrix(RadFLD::MCM,k,j,i) = -efac*(coeff(RadFLD::DZX,k,j,i) + coeff(RadFLD::DZX,k-1,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i) + coeff(RadFLD::DXZ,k,j,i-1));
          matrix(RadFLD::MCP,k,j,i) =  efac*(coeff(RadFLD::DZX,k,j,i) + coeff(RadFLD::DZX,k-1,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i+1)+coeff(RadFLD::DXZ,k,j,i));
          matrix(RadFLD::PCM,k,j,i) =  efac*(coeff(RadFLD::DZX,k+1,j,i)+coeff(RadFLD::DZX,k,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i) + coeff(RadFLD::DXZ,k,j,i-1));
          matrix(RadFLD::PCP,k,j,i) = -efac*(coeff(RadFLD::DZX,k+1,j,i)+coeff(RadFLD::DZX,k,j,i)
                                    +coeff(RadFLD::DXZ,k,j,i+1)+coeff(RadFLD::DXZ,k,j,i));
          matrix(RadFLD::MMC,k,j,i) = -efac*(coeff(RadFLD::DYZ,k,j,i) + coeff(RadFLD::DYZ,k,j-1,i)
                                    +coeff(RadFLD::DZY,k,j,i) + coeff(RadFLD::DZY,k-1,j,i));
          matrix(RadFLD::MPC,k,j,i) =  efac*(coeff(RadFLD::DYZ,k,j+1,i)+coeff(RadFLD::DYZ,k,j,i)
                                    +coeff(RadFLD::DZY,k,j,i) + coeff(RadFLD::DZY,k-1,j,i));
          matrix(RadFLD::PMC,k,j,i) =  efac*(coeff(RadFLD::DYZ,k,j,i) + coeff(RadFLD::DYZ,k,j-1,i)
                                    +coeff(RadFLD::DZY,k+1,j,i)+coeff(RadFLD::DZY,k,j,i));
          matrix(RadFLD::PPC,k,j,i) = -efac*(coeff(RadFLD::DYZ,k,j+1,i)+coeff(RadFLD::DYZ,k,j,i)
                                    +coeff(RadFLD::DZY,k+1,j,i)+coeff(RadFLD::DZY,k,j,i));
        }
      }
    }
  }
  return;
}
