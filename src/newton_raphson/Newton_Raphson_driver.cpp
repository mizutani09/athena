//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson_driver.cpp
//! \brief implementation of functions in class NewtonRaphsonDriver

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdlib>    // abs
#include <limits>
#include <iomanip>    // setprecision
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/nr/bvals_nr.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "Newton_Raphson.hpp"
#include "../linear_solver/linearMG/linearMG.hpp"
#include "../linear_solver/linear_solver.hpp"
#include "../task_list/nr_task_list.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

namespace {

void PrintMaxDefectStencil(NewtonRaphson *pnr, int k, int j, int i, Real signed_defect) {
  MeshBlock *pmb = pnr->pmy_block_;
  const int is = pmb->is;
  const int ie = pmb->ie;
  const int js = pmb->js;
  const int je = pmb->je;
  const int ks = pmb->ks;
  const int ke = pmb->ke;
  const Real dx = pmb->pcoord->dx1f(i);
  const Real idx2 = 1.0/(dx*dx);

  auto print_axis_triplet = [&](const char *label, int km, int jm, int im,
                                int kc, int jc, int ic,
                                int kp, int jp, int ip) {
    std::cout << "      " << label
              << " u=(" << pnr->u_(km, jm, im) << ", "
              << pnr->u_(kc, jc, ic) << ", "
              << pnr->u_(kp, jp, ip) << ")"
              << " def=(" << pnr->def_(0, km, jm, im) << ", "
              << pnr->def_(0, kc, jc, ic) << ", "
              << pnr->def_(0, kp, jp, ip) << ")"
              << std::endl;
  };

  std::cout << "      max_defect_stencil rank=" << Globals::my_rank
            << " gid=" << pmb->gid
            << " signed_def=" << signed_defect
            << " block_i=[" << is << "," << ie << "]"
            << " block_j=[" << js << "," << je << "]"
            << " block_k=[" << ks << "," << ke << "]"
            << " on_edge=(" << (i == is || i == ie)
            << "," << (j == js || j == je)
            << "," << (k == ks || k == ke) << ")"
            << std::endl;

  print_axis_triplet("x-neigh", k, j, i - 1, k, j, i, k, j, i + 1);
  print_axis_triplet("y-neigh", k, j - 1, i, k, j, i, k, j + 1, i);
  print_axis_triplet("z-neigh", k - 1, j, i, k, j, i, k + 1, j, i);

  const Real diff_xm = -pnr->coeff_(linearSolver::DXMF, k, j, i)
      * (pnr->u_(k, j, i - 1) - pnr->u_(k, j, i)) * idx2;
  const Real diff_xp = -pnr->coeff_(linearSolver::DXPF, k, j, i)
      * (pnr->u_(k, j, i + 1) - pnr->u_(k, j, i)) * idx2;
  const Real diff_ym = -pnr->coeff_(linearSolver::DYMF, k, j, i)
      * (pnr->u_(k, j - 1, i) - pnr->u_(k, j, i)) * idx2;
  const Real diff_yp = -pnr->coeff_(linearSolver::DYPF, k, j, i)
      * (pnr->u_(k, j + 1, i) - pnr->u_(k, j, i)) * idx2;
  const Real diff_zm = -pnr->coeff_(linearSolver::DZMF, k, j, i)
      * (pnr->u_(k - 1, j, i) - pnr->u_(k, j, i)) * idx2;
  const Real diff_zp = -pnr->coeff_(linearSolver::DZPF, k, j, i)
      * (pnr->u_(k + 1, j, i) - pnr->u_(k, j, i)) * idx2;
  const Real diff_sum = diff_xm + diff_xp + diff_ym + diff_yp + diff_zm + diff_zp;
  std::cout << "      diff_contrib"
            << " xm=" << diff_xm
            << " xp=" << diff_xp
            << " ym=" << diff_ym
            << " yp=" << diff_yp
            << " zm=" << diff_zm
            << " zp=" << diff_zp
            << " sum=" << diff_sum
            << std::endl;

  std::cout << "      coeff center DCCF=" << pnr->coeff_(linearSolver::DCCF, k, j, i)
            << " DCCS=" << pnr->coeff_(linearSolver::DCCS, k, j, i)
            << " DXMF=" << pnr->coeff_(linearSolver::DXMF, k, j, i)
            << " DXPF=" << pnr->coeff_(linearSolver::DXPF, k, j, i)
            << " DYMF=" << pnr->coeff_(linearSolver::DYMF, k, j, i)
            << " DYPF=" << pnr->coeff_(linearSolver::DYPF, k, j, i)
            << " DZMF=" << pnr->coeff_(linearSolver::DZMF, k, j, i)
            << " DZPF=" << pnr->coeff_(linearSolver::DZPF, k, j, i)
            << std::endl;
  std::cout << "      src center=" << pnr->src_(k, j, i)
            << " uold center=" << pnr->uold_(k, j, i)
            << " u center=" << pnr->u_(k, j, i)
            << " x=(" << pmb->pcoord->x1v(i) << ","
            << pmb->pcoord->x2v(j) << ","
            << pmb->pcoord->x3v(k) << ")"
            << std::endl;
  pnr->PrintCellPhysicsDebug(k, j, i);
}

}  // namespace

// constructor, initializes data structures and parameters

NewtonRaphsonDriver::NewtonRaphsonDriver(Mesh *pm,
                 int invar, int ncoeff, int nmatrix) :
    nranks_(Globals::nranks),
    nthreads_(pm->num_mesh_threads_),
    nbtotal_(pm->nbtotal),
    nvar_(invar), ncoeff_(ncoeff), nmatrix_(nmatrix),
    // matrixmode_(0), // 0: fixed, 1: update after every V-cycle
    // nrbx1_(pm->nrbx1), nrbx2_(pm->nrbx2), nrbx3_(pm->nrbx3),
    pmy_mesh_(pm),
    needinit_(true), fshowdef_(false), use_mg_smoothing_fallback_(false),
    eps_(-1.0), dt_(0.0), step_scale_(1.0),
    backtrack_factor_(0.5), min_step_scale_(0.05), niter_(-1), max_backtrack_(0)
    // nb_rank_(0)
    {
  std::cout << std::scientific << std::setprecision(15);

  if (pmy_mesh_->mesh_size.nx2==1 || pmy_mesh_->mesh_size.nx3==1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NewtonRaphsonDriver::NewtonRaphsonDriver" << std::endl
        << "Currently the Newton-Raphson solver works only in 3D." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  if ( !(pmy_mesh_->use_uniform_meshgen_fn_[X1DIR])
    || !(pmy_mesh_->use_uniform_meshgen_fn_[X2DIR])
    || !(pmy_mesh_->use_uniform_meshgen_fn_[X3DIR])) {
    std::stringstream msg;
    msg << "### FATAL ERROR in NewtonRaphsonDriver::NewtonRaphsonDriver" << std::endl
        << "Non-uniform mesh spacing is not supported." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }

  // NRBoundaryFunction_[BoundaryFace::inner_x1] = pmy_mesh_->NRBoundaryFunc_[BoundaryFace::inner_x1];
  for (int i = 0; i < 6; ++i) {
    nr_mesh_bcs_[i] = pmy_mesh_->mesh_bcs[i];
    NRBoundaryFunction_[i] = pmy_mesh_->NRBoundaryFunc_[i];
  }

  // ranklist_  = new int[nbtotal_];
  // int nv = std::max(nvar_*2, ncoeff_);
  // rootbuf_ = new Real[nbtotal_*nv];
  // for (int n = 0; n < nbtotal_; ++n)
  //   ranklist_[n] = pmy_mesh_->ranklist[n];
  // nslist_  = new int[nranks_];
  // nblist_  = new int[nranks_];
  // nvlist_  = new int[nranks_];
  // nvslist_ = new int[nranks_];
  // nvlisti_  = new int[nranks_];
  // nvslisti_ = new int[nranks_];
  // if (ncoeff_ > 0) {
  //   nclist_  = new int[nranks_];
  //   ncslist_ = new int[nranks_];
  // }


#ifdef MPI_PARALLEL
  MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_NEWTON_RAPHSON);
  nr_phys_id_ = pmy_mesh_->ReserveTagPhysIDs(1);
#endif
  nrtlist_coeff_ = new NewtonRaphsonTaskList(this, NewtonRaphsonTaskList::Mode::coeff);
  nrtlist_post_ = new NewtonRaphsonTaskList(this, NewtonRaphsonTaskList::Mode::post);

//   if (maxreflevel_ > 0) { // SMR / AMR
//     octets_ = new std::vector<MGOctet>[maxreflevel_];
//     octetmap_ = new std::unordered_map<LogicalLocation, int,
//                                        LogicalLocationHash>[maxreflevel_];
//     octetbflag_ = new std::vector<bool>[maxreflevel_];
//     noctets_ = new int[maxreflevel_]();
//     pmaxnoct_ = new int[maxreflevel_]();

//     int nth = 1;
// #ifdef OPENMP_PARALLEL
//     nth = omp_get_max_threads();
// #endif
//     cbuf_ = new AthenaArray<Real>[nth];
//     cbufold_ = new AthenaArray<Real>[nth];
//     ncoarse_ = new AthenaArray<bool>[nth];
//     nv = std::max(nvar_, ncoeff_);
//     for (int n = 0; n < nth; ++n) {
//       cbuf_[n].NewAthenaArray(nv,3,3,3);
//       cbufold_[n].NewAthenaArray(nv,3,3,3);
//       ncoarse_[n].NewAthenaArray(3,3,3);
//     }
//   }

  // if (NRMGFLD_ENABLED) {
  //   plmgd_ = new linearMGDriver(pm, pin);
  // } else {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in NewtonRaphsonDriver::NewtonRaphsonDriver" << std::endl
  //       << "Failed to allocate linear solver" << std::endl;
  //   ATHENA_ERROR(msg);
  // }
}

//! destructor

NewtonRaphsonDriver::~NewtonRaphsonDriver() {
  // delete [] ranklist_;
  // delete [] nslist_;
  // delete [] nblist_;
  // delete [] nvlist_;
  // delete [] nvslist_;
  // delete [] nvlisti_;
  // delete [] nvslisti_;
  // delete [] rootbuf_;
  // if (ncoeff_ > 0) {
  //   delete [] nclist_;
  //   delete [] ncslist_;
  // }
#ifdef MPI_PARALLEL
  MPI_Comm_free(&MPI_COMM_NEWTON_RAPHSON);
#endif
  delete nrtlist_coeff_;
  delete nrtlist_post_;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::CheckBoundaryFunctions()
//  \brief check boundary functions and set some internal flags.

void NewtonRaphsonDriver::CheckBoundaryFunctions() {
  switch(nr_mesh_bcs_[BoundaryFace::inner_x1]) {
    case BoundaryFlag::user:
      if (NRBoundaryFunction_[BoundaryFace::inner_x1] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x1 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::outflow:
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(nr_mesh_bcs_[BoundaryFace::outer_x1]) {
    case BoundaryFlag::user:
      if (NRBoundaryFunction_[BoundaryFace::outer_x1] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x1 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::outflow:
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(nr_mesh_bcs_[BoundaryFace::inner_x2]) {
    case BoundaryFlag::user:
      if (NRBoundaryFunction_[BoundaryFace::inner_x2] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x2 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::outflow:
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(nr_mesh_bcs_[BoundaryFace::outer_x2]) {
    case BoundaryFlag::user:
      if (NRBoundaryFunction_[BoundaryFace::outer_x2] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x2 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::outflow:
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(nr_mesh_bcs_[BoundaryFace::inner_x3]) {
    case BoundaryFlag::user:
      if (NRBoundaryFunction_[BoundaryFace::inner_x3] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x3 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::outflow:
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(nr_mesh_bcs_[BoundaryFace::outer_x3]) {
    case BoundaryFlag::user:
      if (NRBoundaryFunction_[BoundaryFace::outer_x3] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x3 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::outflow:
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }

  // check periodic boundary conditions
  for (int i = 0; i < 6; ++i) {
    if (pmy_mesh_->mesh_bcs[i] == BoundaryFlag::periodic
     || nr_mesh_bcs_[i] == BoundaryFlag::periodic) {
      if (pmy_mesh_->mesh_bcs[i] != nr_mesh_bcs_[i]) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "When periodic boundary condition is set either for" << std::endl
            << "Multigrid or for the main part, both must be periodic." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  return;
}


void NewtonRaphsonDriver::Solve_general(int stage, Real dt) {
  // std::cout << "In NewtonRaphsonDriver::Solve_general" << std::endl;
  stage_ = stage;
  dt_ = dt;
  // Construct the NewtonRaphson array
  vnr_.clear();
  for (int i = 0; i < pmy_mesh_->nblocal; ++i)
    vnr_.push_back(pmy_mesh_->my_blocks(i)->pnr);
  plmgd_->BeginTimeStep();


  // data load
#pragma omp parallel for num_threads(nthreads_)
  for (int b = 0; b < static_cast<int>(vnr_.size()); ++b) {
    NewtonRaphson *pnr = vnr_[b];
    MeshBlock *pmb = pnr->pmy_block_;
    pnr->LoadVariables();
  }

  // calc coefficients for initial setup
  // std::cout << "Number of NewtonRaphson objects: " << vnr_.size() << std::endl;
#pragma omp parallel for num_threads(nthreads_)
  for (int b = 0; b < static_cast<int>(vnr_.size()); ++b) {
    NewtonRaphson *pnr = vnr_[b];
    MeshBlock *pmb = pnr->pmy_block_;
    pnr->CalculateCoefficientsOnce(pnr->u_, pmb->phydro->w,
                                  pnr->def_coeff_, pnr->derivetive_);
    pnr->CalculateCoefficients(pnr->uold_, pnr->u_,
      pnr->def_coeff_, pnr->coeff_, pnr->derivetive_, pnr->src_,
      dt_);
  }

  int n = 0;
  Real def = 0.0, defmax = 0.0;
  CalculateDefectNorms(def, defmax);

  // std::cout << "epsilon for Newton-Raphson: " << eps_ << std::endl;

  if (fshowdef_ && Globals::my_rank == 0)
    std::cout << "initial defect " << def << " max " << defmax << std::endl;
  while (def > eps_) {
    // if (matrixmode_ == 1)
    //   CalculateMatrix();
    Real olddef = def, oldmax = defmax;
    Real trial_scale = 1.0;
    bool accepted = false;
    const bool base_smoothing_only = plmgd_->GetSmoothingOnly();
    const Real base_coarse_corr_scale = plmgd_->GetCoarseCorrectionScale();
    bool use_smoothing_retry = false;
    Real coarse_corr_trial_scale = base_coarse_corr_scale;
    int nback = 0;
    int ncoarse_retry = 0;

    while (true) {
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->StoreIterate();
      }

      step_scale_ = trial_scale;
      plmgd_->SetSmoothingOnly(base_smoothing_only || use_smoothing_retry);
      plmgd_->SetCoarseCorrectionScale(coarse_corr_trial_scale);
      SolveOneCycle();

      def = 0.0, defmax = 0.0;
      CalculateDefectNorms(def, defmax);

      if (fshowdef_ && Globals::my_rank == 0) {
        const Real conv = (olddef > 0.0 ? def/olddef : 0.0);
        const Real convmax = (oldmax > 0.0 ? defmax/oldmax : 0.0);
        std::cout << "[debug in NR] niter " << n << " step_scale " << step_scale_
                  << " def " << def << " convergence factor " << conv
                  << " defmax  " << defmax << " cf " << convmax << std::endl;
      }
      if (fshowdef_) {
        Real local_absmax = -1.0;
        Real local_signed = 0.0;
        Real local_x1 = 0.0, local_x2 = 0.0, local_x3 = 0.0;
        int local_gid = -1, local_i = -1, local_j = -1, local_k = -1;
        for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
          NewtonRaphson *pnr = *itr;
          MeshBlock *pmb = pnr->pmy_block_;
          for (int k = pmb->ks; k <= pmb->ke; ++k) {
            for (int j = pmb->js; j <= pmb->je; ++j) {
              for (int i = pmb->is; i <= pmb->ie; ++i) {
                const Real val = pnr->def_(0, k, j, i);
                const Real aval = std::abs(val);
                if (aval > local_absmax) {
                  local_absmax = aval;
                  local_signed = val;
                  local_gid = pmb->gid;
                  local_i = i;
                  local_j = j;
                  local_k = k;
                  local_x1 = pmb->pcoord->x1v(i);
                  local_x2 = pmb->pcoord->x2v(j);
                  local_x3 = pmb->pcoord->x3v(k);
                }
              }
            }
          }
        }

        Real global_absmax = local_absmax;
        int owner_rank = (local_absmax >= 0.0 ? Globals::my_rank : nranks_);
#ifdef MPI_PARALLEL
        MPI_Allreduce(MPI_IN_PLACE, &global_absmax, 1, MPI_ATHENA_REAL, MPI_MAX,
                      MPI_COMM_NEWTON_RAPHSON);
        const Real tol = std::max(static_cast<Real>(1.0e-14),
                                  static_cast<Real>(1.0e-12)*global_absmax);
        if (std::abs(local_absmax - global_absmax) > tol) owner_rank = nranks_;
        MPI_Allreduce(MPI_IN_PLACE, &owner_rank, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_NEWTON_RAPHSON);
#endif
        Real max_info[8] = {local_signed, static_cast<Real>(local_gid),
                            static_cast<Real>(local_i), static_cast<Real>(local_j),
                            static_cast<Real>(local_k), local_x1, local_x2, local_x3};
#ifdef MPI_PARALLEL
        MPI_Bcast(max_info, 8, MPI_ATHENA_REAL, owner_rank, MPI_COMM_NEWTON_RAPHSON);
#endif
        if (Globals::my_rank == 0) {
          std::cout << "    max_defect_loc gid=" << static_cast<int>(max_info[1])
                    << " (k,j,i)=(" << static_cast<int>(max_info[4]) << ","
                    << static_cast<int>(max_info[3]) << ","
                    << static_cast<int>(max_info[2]) << ")"
                    << " x=(" << max_info[5] << "," << max_info[6] << ","
                    << max_info[7] << ")"
                    << " signed=" << max_info[0]
                    << " abs=" << global_absmax << std::endl;
        }
        if (Globals::my_rank == owner_rank) {
          for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
            NewtonRaphson *pnr = *itr;
            if (pnr->pmy_block_->gid == static_cast<int>(max_info[1])) {
              PrintMaxDefectStencil(
                  pnr, static_cast<int>(max_info[4]), static_cast<int>(max_info[3]),
                  static_cast<int>(max_info[2]), max_info[0]);
              break;
            }
          }
        }
      }

      if (std::isfinite(def) && def <= olddef) {
        accepted = true;
        break;
      }

      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->RestoreIterate();
      }

      // Coarse-correction damping is a safeguard for coarse/fine MG
      // coupling.  On a uniform mesh it only repeats the same Newton solve.
      if (pmy_mesh_->multilevel
          && !base_smoothing_only && !use_smoothing_retry
          && ncoarse_retry < mg_coarse_retry_max_
          && coarse_corr_trial_scale*mg_coarse_retry_factor_
                 >= mg_coarse_retry_min_scale_) {
        coarse_corr_trial_scale *= mg_coarse_retry_factor_;
        ++ncoarse_retry;
        if (fshowdef_ && Globals::my_rank == 0) {
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "Retrying Newton-Raphson iterate with damped coarse correction: "
                    << "previous defect norm = " << olddef
                    << ", rejected defect norm = " << def
                    << ", step scale = " << trial_scale
                    << ", coarse_correction_scale = " << coarse_corr_trial_scale
                    << ", niter = " << n << "." << std::endl;
        }
        continue;
      }

      // The fine-grid-only fallback corrects an SMR/AMR transfer mismatch;
      // it is redundant (and expensive) when no refinement levels exist.
      if (pmy_mesh_->multilevel && use_mg_smoothing_fallback_
          && !base_smoothing_only && !use_smoothing_retry) {
        use_smoothing_retry = true;
        if (fshowdef_ && Globals::my_rank == 0) {
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "Retrying Newton-Raphson iterate with smoothing-only linear MG: "
                    << "previous defect norm = " << olddef
                    << ", rejected defect norm = " << def
                    << ", step scale = " << trial_scale
                    << ", niter = " << n << "." << std::endl;
        }
        continue;
      }

      if (nback >= max_backtrack_ ||
          trial_scale*backtrack_factor_ < min_step_scale_) {
        if (fshowdef_ && Globals::my_rank == 0) {
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "Rejecting Newton-Raphson iterate after backtracking attempts: "
                    << "defect norm = " << def
                    << ", previous defect norm = " << olddef
                    << ", last step scale = " << trial_scale
                    << ", niter = " << n << "." << std::endl;
        }
        def = olddef;
        defmax = oldmax;
        break;
      }

      trial_scale *= backtrack_factor_;
      ++nback;
      if (fshowdef_ && Globals::my_rank == 0) {
        std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                  << "Backtracking Newton-Raphson iterate: new step scale = "
                  << trial_scale << ", previous defect norm = " << olddef
                  << ", rejected defect norm = " << def
                  << ", niter = " << n << "." << std::endl;
      }
    }
    step_scale_ = 1.0;
    plmgd_->SetSmoothingOnly(base_smoothing_only);
    plmgd_->SetCoarseCorrectionScale(base_coarse_corr_scale);

    if (!accepted) break;

    if (pmy_mesh_->ncycle == 0 && dt_ == 0.0) break; // only for the first time: caution! ncycle=0 is also used after the calculation started (but dt > 0.0).
    if (!std::isfinite(def)) {
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->RestoreIterate();
      }
      if (fshowdef_ && Globals::my_rank == 0)
        std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                  << "Rolling back Newton-Raphson iterate: defect norm = " << def
                  << ", previous defect norm = " << olddef
                  << ", convergence factor = " << def/olddef
                  << ", and niter = " << n << "." << std::endl;
      def = olddef;
      defmax = oldmax;
      break;
    }
    if (def/olddef > 0.9) {
      if (n > 1 && eps_ == 0.0) break;
      if (fshowdef_ && Globals::my_rank == 0)
        std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                  << "Slow Newton-Raphson convergence : defect norm = " << def
                  << ", convergence factor = " << def/olddef << "." << std::endl;
      if (n > 1 && def/olddef > 1.0) {
        if (fshowdef_ && Globals::my_rank == 0)
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "NewtonRaphson is diverging: defect norm = " << def
                    << ", convergence factor = " << def/olddef << ", and niter = " << n << "." << std::endl;
        break;
      }
      if (n > 1 && std::abs(def - olddef) < 1e-12) {
        if (fshowdef_ && Globals::my_rank == 0)
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "NewtonRaphson is not converging: defect norm = " << def
                    << ", convergence factor = " << def/olddef << ", and niter = " << n << "." << std::endl;
        break;
      }
    }
    if (niter_ != -1 && n > niter_) {
      if (Globals::my_rank == 0) {
        std::cout
            << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
            << "Aborting because the # iterations is too large, n > " << niter_ << "." << std::endl
            << "Check the solution as it may not be accurate enough." << std::endl;
      }
      break;
    }
    n++;
  }

  // return the results to hydro variables
  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    pnr->UpdateHydroVariables();
  }
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SolveOneCycle()
//! \brief Solve one cycle of NewtonRaphson

void NewtonRaphsonDriver::SolveOneCycle() {
  nrtlist_coeff_->DoTaskListOneStage(stage_);
  plmgd_->Solve(stage_, dt_);
  nrtlist_post_->DoTaskListOneStage(stage_);
  return;
}

// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphsonDriver::SolveIterative()
// //  \brief Solve iteratively until the convergence is achieved

// void NewtonRaphsonDriver::SolveIterative() {
//   int n = 0;
//   Real def = 0.0, defmax = 0.0;
//   for (int v = 0; v < nvar_; ++v) {
//     def += CalculateDefectNorm(NRNormType::l2, v);
// //    defmax = std::max(defmax, CalculateDefectNorm(NRNormType::max, v));
//   }
// //  if (Globals::my_rank == 0)
// //    std::cout << "initial defect " << def << " max " << defmax << std::endl;
//   while (def > eps_) {
//     SolveOneCycle();
//     if (matrixmode_ == 1)
//       CalculateMatrix();
//     Real olddef = def, oldmax = defmax;
//     def = 0.0, defmax = 0.0;
//     for (int v = 0; v < nvar_; ++v) {
//       def += CalculateDefectNorm(NRNormType::l2, v);
// //      defmax = std::max(defmax, CalculateDefectNorm(NRNormType::max, v));
//     }
//    if (Globals::my_rank == 0)
//      std::cout << "[debug] niter " << n << " def " << def << " convergence factor "
//                << def/olddef<< " defmax  "<< defmax << " cf "
//                <<  defmax/oldmax << std::endl;
//     if (def/olddef > 0.9) {
//       if (eps_ == 0.0) break;
//       if (Globals::my_rank == 0)
//         std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
//                   << "Slow multigrid convergence : defect norm = " << def
//                   << ", convergence factor = " << def/olddef << "." << std::endl;
//       if (def/olddef > 1.0) {
//         if (Globals::my_rank == 0)
//           std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
//                     << "NewtonRaphson is diverging: defect norm = " << def
//                     << ", convergence factor = " << def/olddef << ", and niter = " << n << "." << std::endl;
//         break;
//       }
//     }
//     // if (n > 100) {
//     if (n > 30) {
//       if (Globals::my_rank == 0) {
//         std::cout
//             << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
//             << "Aborting because the # iterations is too large, n > 30." << std::endl
//             << "Check the solution as it may not be accurate enough." << std::endl;
//       }
//       break;
//     }
//     n++;
//   }
//   // if (fsubtract_average_)
//   //   SubtractAverage(NRVariable::u);
//   return;
// }


// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphsonDriver::SolveIterativeFixedTimes()
// //  \brief Solve iteratively niter_ times

// void NewtonRaphsonDriver::SolveIterativeFixedTimes() {
//   for (int n = 0; n < niter_; ++n) {
//     SolveOneCycle();
//     if (matrixmode_ == 1)
//       CalculateMatrix();
//   }
//   // if (fsubtract_average_)
//   //   SubtractAverage(NRVariable::u);
//   Real def = 0.0;
//   for (int v = 0; v < nvar_; ++v)
//     def += CalculateDefectNorm(NRNormType::l2, v);
//   if (fshowdef_ && Globals::my_rank == 0)
//     std::cout << "NewtonRaphson defect L2-norm : " << def << std::endl;

//   return;
// }


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::CalculateDefectNorms(Real &l2_norm,
//!                                                     Real &max_norm)
//! \brief calculate each block defect once and accumulate both convergence norms

void NewtonRaphsonDriver::CalculateDefectNorms(Real &l2_norm, Real &max_norm) {
  const int nblock = static_cast<int>(vnr_.size());
  std::vector<Real> block_l2(nblock*nvar_, 0.0);
  std::vector<Real> block_max(nblock*nvar_, 0.0);

#pragma omp parallel for num_threads(nthreads_)
  for (int b = 0; b < nblock; ++b) {
    NewtonRaphson *pnr = vnr_[b];
    pnr->CalculateDefectBlock();
    for (int v = 0; v < nvar_; ++v) {
      pnr->CalculateDefectNorms(v, block_l2[b*nvar_ + v],
                                block_max[b*nvar_ + v]);
    }
  }

  l2_norm = 0.0;
  max_norm = 0.0;
  const Real vol = (pmy_mesh_->mesh_size.x1max-pmy_mesh_->mesh_size.x1min)
                 * (pmy_mesh_->mesh_size.x2max-pmy_mesh_->mesh_size.x2min)
                 * (pmy_mesh_->mesh_size.x3max-pmy_mesh_->mesh_size.x3min);
  for (int v = 0; v < nvar_; ++v) {
    Real sum = 0.0;
    Real maximum = 0.0;
    for (int b = 0; b < nblock; ++b) {
      sum += block_l2[b*nvar_ + v];
      maximum = std::max(maximum, block_max[b*nvar_ + v]);
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_NEWTON_RAPHSON);
    MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_NEWTON_RAPHSON);
#endif
    l2_norm += std::sqrt(sum/vol);
    max_norm = std::max(max_norm, maximum);
  }
}

// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphsonDriver::CalculateMatrix()
// //! \brief Calculate Matrix elements

// void NewtonRaphsonDriver::CalculateMatrix() {
//   if (nmatrix_ == 0)
//     return;
//   // RestrictInitialData();
//   // if (current_level_ >= nrootlevel_ + nreflevel_ - 1) {
// #pragma omp parallel for num_threads(nthreads_)
//     for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
//       NewtonRaphson *pnr = *itr;
//       pnr->CalculateMatrixBlock();
//     }
//   // }
//   return;
// }
