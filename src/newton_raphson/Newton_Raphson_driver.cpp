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

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

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
    needinit_(true), fshowdef_(false),
    eps_(-1.0), dt_(0.0), niter_(-1)
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
}


void NewtonRaphsonDriver::Solve_general(int stage, Real dt) {
  // std::cout << "In NewtonRaphsonDriver::Solve_general" << std::endl;
  stage_ = stage;
  dt_ = dt;
  // Construct the NewtonRaphson array
  vnr_.clear();
  for (int i = 0; i < pmy_mesh_->nblocal; ++i)
    vnr_.push_back(pmy_mesh_->my_blocks(i)->pnr);


  // data load
  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    MeshBlock *pmb = pnr->pmy_block_;
    pnr->LoadVariables();
  }

  // calc coefficients for initial setup
  // std::cout << "Number of NewtonRaphson objects: " << vnr_.size() << std::endl;
  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    MeshBlock *pmb = pnr->pmy_block_;
    pnr->CalculateCoefficientsOnce(pnr->u_, pmb->phydro->w,
                                  pnr->def_coeff_, pnr->derivetive_);
    pnr->CalculateCoefficients(pnr->uold_, pnr->u_,
      pnr->def_coeff_, pnr->coeff_, pnr->derivetive_, pnr->src_,
      dt_);
  }

  int n = 0;
  Real def = 0.0, defmax = 0.0;
  for (int v = 0; v < nvar_; ++v) {
    def += CalculateDefectNorm(NRNormType::l2, v);
  //  defmax = std::max(defmax, CalculateDefectNorm(NRNormType::max, v));
  }

  // std::cout << "epsilon for Newton-Raphson: " << eps_ << std::endl;

//  if (Globals::my_rank == 0)
//    std::cout << "initial defect " << def << " max " << defmax << std::endl;
  while (def > eps_) {
    SolveOneCycle();
    // if (matrixmode_ == 1)
    //   CalculateMatrix();
    Real olddef = def, oldmax = defmax;
    def = 0.0, defmax = 0.0;
    for (int v = 0; v < nvar_; ++v) {
      def += CalculateDefectNorm(NRNormType::l2, v);
//      defmax = std::max(defmax, CalculateDefectNorm(NRNormType::max, v));
    }
    if (Globals::my_rank == 0)
      std::cout << "[debug in NR] niter " << n << " def " << def << " convergence factor "
                << def/olddef<< " defmax  "<< defmax << " cf "
                <<  defmax/oldmax << std::endl;
    if (pmy_mesh_->ncycle == 0) break;
    if (def/olddef > 0.9) {
      if (n > 1 && eps_ == 0.0) break;
      if (Globals::my_rank == 0)
        std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                  << "Slow Newton-Raphson convergence : defect norm = " << def
                  << ", convergence factor = " << def/olddef << "." << std::endl;
      if (n > 1 && def/olddef > 1.0) {
        if (Globals::my_rank == 0)
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "NewtonRaphson is diverging: defect norm = " << def
                    << ", convergence factor = " << def/olddef << ", and niter = " << n << "." << std::endl;
        break;
      }
      if (n > 1 && std::abs(def - olddef) < 1e-12) {
        if (Globals::my_rank == 0)
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "NewtonRaphson is not converging: defect norm = " << def
                    << ", convergence factor = " << def/olddef << ", and niter = " << n << "." << std::endl;
        break;
      }
    }
    // if (n > 100) {
    if (n > 100) {
      if (Globals::my_rank == 0) {
        std::cout
            << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
            << "Aborting because the # iterations is too large, n > 30." << std::endl
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
  // calc coefficients
  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    pnr->CalculateCoefficients(pnr->uold_, pnr->u_,
      pnr->def_coeff_, pnr->coeff_,
      pnr->derivetive_, pnr->src_, dt_);
  }

  // print for debug
  if (fshowdef_) {
    NewtonRaphson *pnr = *(vnr_.begin());
    MeshBlock *pmb = pnr->pmy_block_;
    int is = pmb->is, ie = pmb->ie;
    int js = pmb->js, je = pmb->je;
    int ks = pmb->ks, ke = pmb->ke;
    int i = (is + ie) / 2;
    int j = (js + je) / 2;
    int k = (ks + ke) / 2;
    std::cout << "At (" << k << "," << j << "," << i << "):" << std::endl;
    std::cout <<"delta_u_ before MG at " << Globals::my_rank << ": ";
    for (int n = 0; n < nvar_; n++)
      std::cout << pnr->delta_u_(n,k,j,i) << " ";
    std::cout << std::endl;
  }

  // call linear solver (should be replaced general solver)
  plmgd_->Solve(stage_, dt_);

  // print for debug
  if (fshowdef_) {
    NewtonRaphson *pnr = *(vnr_.begin());
    MeshBlock *pmb = pnr->pmy_block_;
    int is = pmb->is, ie = pmb->ie;
    int js = pmb->js, je = pmb->je;
    int ks = pmb->ks, ke = pmb->ke;
    int i = (is + ie) / 2;
    int j = (js + je) / 2;
    int k = (ks + ke) / 2;
    std::cout << "At (" << k << "," << j << "," << i << "):" << std::endl;
    std::cout <<"delta_u_ after MG: ";
    std::cout << pnr->delta_u_(k,j,i) << " ";
    std::cout << std::endl;
  }

  // std::cout << "NewtonRaphson correction retrieved from linear solver at "
  //           << Globals::my_rank << std::endl;

  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    pnr->AddDifference(pnr->u_,
                       pnr->delta_u_,
                       pnr->derivetive_);
  }

  // std::cout << "NewtonRaphson update applied at " << Globals::my_rank << std::endl;

  // std::cout << "size of vnr_: " << vnr_.size() << std::endl;
  // for boundary values
  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    // std::cout << (itr - vnr_.begin()) << std::endl;
    // pnr->nrbvar.StartReceiving(BoundaryCommSubset::newton_raphson);
    pnr->nrbvar.StartReceiving(BoundaryCommSubset::all);
    // std::cout << "NewtonRaphson boundary buffers sent at " << Globals::my_rank << std::endl;
    
    pnr->nrbvar.SendBoundaryBuffers();
    // std::cout << "NewtonRaphson boundary buffers sent at " << Globals::my_rank << std::endl;
    // bool received = pnr->nrbvar.ReceiveBoundaryBuffers();
    // if (!received) {
    //   std::stringstream msg;
    //   msg << "### FATAL ERROR in NewtonRaphsonDriver::SolveOneCycle" << std::endl
    //       << "Failed to receive NewtonRaphson boundary buffers." << std::endl;
    //   ATHENA_ERROR(msg);
    // } else {
    //   std::cout << "NewtonRaphson boundary buffers received at " << Globals::my_rank << std::endl;
    // }

    pnr->nrbvar.ReceiveAndSetBoundariesWithWait();

  }


  // for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
  //   NewtonRaphson *pnr = *itr;
  //   pnr->nrbvar.SetBoundaries();
  // }

  // std::cout << "NewtonRaphson boundary values set at " << Globals::my_rank << std::endl;
  // std::cout << "pmy_mesh_->multilevel: " << pmy_mesh_->multilevel << std::endl;
  if (pmy_mesh_->multilevel) {
    for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
      NewtonRaphson *pnr = *itr;
      MeshBlock *pmb = pnr->pmy_block_;
      pmb->pbval->ProlongateBoundaries(pmy_mesh_->time, dt_, pmb->pbval->bvars_main_int);
    }
  }

  // std::cout << "NewtonRaphson prolongation done at " << Globals::my_rank << std::endl;

  // for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
  //   NewtonRaphson *pnr = *itr;
  //   MeshBlock *pmb = pnr->pmy_block_;
  //   pnr->nrbvar.var_cc = &(pnr->u_);
  //   pmb->pbval->ApplyPhysicalBoundaries(pmy_mesh_->time, dt_, pmb->pbval->bvars_main_int);
  // }
  
  // std::cout << "NewtonRaphson physical boundaries applied at " << Globals::my_rank << std::endl;

  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    // pnr->nrbvar.ClearBoundary(BoundaryCommSubset::newton_raphson);
    pnr->nrbvar.ClearBoundary(BoundaryCommSubset::all);
  }

  // std::cout << "NewtonRaphson boundary buffers cleared at " << Globals::my_rank << std::endl;

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
//! \fn Real NewtonRaphsonDriver::CalculateDefectNorm(NRNormType nrm, int n)
//! \brief calculate the defect norm

Real NewtonRaphsonDriver::CalculateDefectNorm(NRNormType nrm, int n) {
  Real norm=0.0;

  if (nrm == NRNormType::max) {
#pragma omp parallel for reduction(max : norm) num_threads(nthreads_)
    for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
      NewtonRaphson *pnr = *itr;
      norm = std::max(norm, pnr->CalculateDefectNorm(nrm, n));
    }
  } else {
#pragma omp parallel for reduction(+ : norm) num_threads(nthreads_)
    for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
      NewtonRaphson *pnr = *itr;
      norm += pnr->CalculateDefectNorm(nrm, n);
    }
  }
#ifdef MPI_PARALLEL
  if (nrm == NRNormType::max)
    MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_NEWTON_RAPHSON);
  else
    MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_NEWTON_RAPHSON);
#endif
  if (nrm != NRNormType::max) {
    Real vol = (pmy_mesh_->mesh_size.x1max-pmy_mesh_->mesh_size.x1min)
             * (pmy_mesh_->mesh_size.x2max-pmy_mesh_->mesh_size.x2min)
             * (pmy_mesh_->mesh_size.x3max-pmy_mesh_->mesh_size.x3min);
    norm /= vol;
  }
  if (nrm == NRNormType::l2)
    norm = std::sqrt(norm);

  return norm;
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
