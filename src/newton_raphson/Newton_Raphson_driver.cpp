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
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "Newton_Raphson.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// constructor, initializes data structures and parameters

NewtonRaphsonDriver::NewtonRaphsonDriver(Mesh *pm,
                 int invar, int ncoeff, int nmatrix) :
    nranks_(Globals::nranks), nthreads_(pm->num_mesh_threads_), nbtotal_(pm->nbtotal),
    nvar_(invar), ncoeff_(ncoeff), nmatrix_(nmatrix),
    // matrixmode_(0), // 0: fixed, 1: update after every V-cycle
    nrbx1_(pm->nrbx1), nrbx2_(pm->nrbx2), nrbx3_(pm->nrbx3), srcmask_(NRSourceMask),
    coeffmask_(NRCoeffMask), pmy_mesh_(pm),
    needinit_(true), fshowdef_(false),
    eps_(-1.0), dt_(0.0), niter_(-1),
    nb_rank_(0) {
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


  ranklist_  = new int[nbtotal_];
  int nv = std::max(nvar_*2, ncoeff_);
  rootbuf_ = new Real[nbtotal_*nv];
  for (int n = 0; n < nbtotal_; ++n)
    ranklist_[n] = pmy_mesh_->ranklist[n];
  nslist_  = new int[nranks_];
  nblist_  = new int[nranks_];
  nvlist_  = new int[nranks_];
  nvslist_ = new int[nranks_];
  nvlisti_  = new int[nranks_];
  nvslisti_ = new int[nranks_];
  if (ncoeff_ > 0) {
    nclist_  = new int[nranks_];
    ncslist_ = new int[nranks_];
  }


#ifdef MPI_PARALLEL
  MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_NEWTON_RAPHSON);
  nr_phys_id_ = pmy_mesh_->ReserveTagPhysIDs(1);
#endif

  if (maxreflevel_ > 0) { // SMR / AMR
    octets_ = new std::vector<MGOctet>[maxreflevel_];
    octetmap_ = new std::unordered_map<LogicalLocation, int,
                                       LogicalLocationHash>[maxreflevel_];
    octetbflag_ = new std::vector<bool>[maxreflevel_];
    noctets_ = new int[maxreflevel_]();
    pmaxnoct_ = new int[maxreflevel_]();

    int nth = 1;
#ifdef OPENMP_PARALLEL
    nth = omp_get_max_threads();
#endif
    cbuf_ = new AthenaArray<Real>[nth];
    cbufold_ = new AthenaArray<Real>[nth];
    ncoarse_ = new AthenaArray<bool>[nth];
    nv = std::max(nvar_, ncoeff_);
    for (int n = 0; n < nth; ++n) {
      cbuf_[n].NewAthenaArray(nv,3,3,3);
      cbufold_[n].NewAthenaArray(nv,3,3,3);
      ncoarse_[n].NewAthenaArray(3,3,3);
    }
  }
}

//! destructor

NewtonRaphsonDriver::~NewtonRaphsonDriver() {
  delete [] ranklist_;
  delete [] nslist_;
  delete [] nblist_;
  delete [] nvlist_;
  delete [] nvslist_;
  delete [] nvlisti_;
  delete [] nvslisti_;
  delete [] rootbuf_;
  if (ncoeff_ > 0) {
    delete [] nclist_;
    delete [] ncslist_;
  }
  if (maxreflevel_ > 0) {
    delete [] octets_;
    delete [] octetmap_;
    delete [] octetbflag_;
    delete [] noctets_;
    delete [] pmaxnoct_;
    delete [] cbuf_;
    delete [] cbufold_;
    delete [] ncoarse_;
  }
  if (mporder_ > 0)
    delete [] mpcoeff_;
#ifdef MPI_PARALLEL
  MPI_Comm_free(&MPI_COMM_NEWTON_RAPHSON);
#endif
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::CheckBoundaryFunctions()
//  \brief check boundary functions and set some internal flags.

void NewtonRaphsonDriver::CheckBoundaryFunctions() {
  fsubtract_average_ = true;
  switch(mg_mesh_bcs_[BoundaryFace::inner_x1]) {
    case BoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::inner_x1] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x1 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::mg_zerograd:
      break;
    case BoundaryFlag::mg_zerofixed:
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::mg_multipole:
      mporder_ = 0;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_mesh_bcs_[BoundaryFace::outer_x1]) {
    case BoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::outer_x1] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x1 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::mg_zerograd:
      break;
    case BoundaryFlag::mg_zerofixed:
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::mg_multipole:
      mporder_ = 0;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_mesh_bcs_[BoundaryFace::inner_x2]) {
    case BoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::inner_x2] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x2 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::mg_zerograd:
      break;
    case BoundaryFlag::mg_zerofixed:
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::mg_multipole:
      mporder_ = 0;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_mesh_bcs_[BoundaryFace::outer_x2]) {
    case BoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::outer_x2] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x2 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::mg_zerograd:
      break;
    case BoundaryFlag::mg_zerofixed:
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::mg_multipole:
      mporder_ = 0;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_mesh_bcs_[BoundaryFace::inner_x3]) {
    case BoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::inner_x3] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "inner_x3 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::mg_zerograd:
      break;
    case BoundaryFlag::mg_zerofixed:
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::mg_multipole:
      mporder_ = 0;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  switch(mg_mesh_bcs_[BoundaryFace::outer_x3]) {
    case BoundaryFlag::user:
      if (MGBoundaryFunction_[BoundaryFace::outer_x3] == nullptr) {
        std::stringstream msg;
        msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
            << "A user-defined boundary condition is specified for " << std::endl
            << "outer_x3 but no function is enrolled." << std::endl;
        ATHENA_ERROR(msg);
      }
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::periodic:
    case BoundaryFlag::mg_zerograd:
      break;
    case BoundaryFlag::mg_zerofixed:
      fsubtract_average_ = false;
      break;
    case BoundaryFlag::mg_multipole:
      mporder_ = 0;
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in MGGravityDriver::CheckBoundaryFunctions" << std::endl
          << "Invalid or no boundary type is specified." << std::endl;
      ATHENA_ERROR(msg);
      break;
  }

  // check periodic boundary conditions
  for (int i = 0; i < 6; ++i) {
    if (pmy_mesh_->mesh_bcs[i] == BoundaryFlag::periodic
     || mg_mesh_bcs_[i] == BoundaryFlag::periodic) {
      if (pmy_mesh_->mesh_bcs[i] != mg_mesh_bcs_[i]) {
        std::stringstream msg;
        msg << "### FATAL ERROR in NewtonRaphsonDriver::CheckBoundaryFunctions" << std::endl
            << "When periodic boundary condition is set either for" << std::endl
            << "NewtonRaphson or for the main part, both must be periodic." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  if (mporder_ >= 0) {
    ffas_ = true;
    fsubtract_average_ = false;
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SetupNewtonRaphson(bool ftrivial)
//  \brief initialize the source assuming that the source terms are already loaded

void NewtonRaphsonDriver::SetupNewtonRaphson(bool ftrivial) {
  locrootlevel_ = pmy_mesh_->root_level;
  nrootlevel_ = mgroot_->GetNumberOfLevels();
  nmblevel_ = vnr_[0]->GetNumberOfLevels();
  nreflevel_ = pmy_mesh_->current_level - locrootlevel_;
  ntotallevel_ = nrootlevel_ + nmblevel_ + nreflevel_ - 1;
  fmglevel_ = current_level_ = ntotallevel_ - 1;
  os_ = mgroot_->ngh_;
  oe_ = os_+1;
  const int ncoct = 2 + 2*mgroot_->ngh_, nccoct = 1 + 2*mgroot_->ngh_;

  if (pmy_mesh_->amr_updated)
    needinit_ = true;

  // note: the level of an Octet is one level lower than the data stored there
  if (nreflevel_ > 0 && needinit_) {
    for (int l = 0; l < nreflevel_; ++l) { // clear old data
      octetmap_[l].clear();
      pmaxnoct_[l] = std::max(pmaxnoct_[l], noctets_[l]);
      noctets_[l] = 0;
    }
    pmy_mesh_->tree.CountMGOctets(noctets_);
    for (int l = 0; l < nreflevel_; ++l) { // increase the octet array size if needed
      if (pmaxnoct_[l] < noctets_[l]) {
        octets_[l].resize(noctets_[l]);
        octetmap_[l].reserve(noctets_[l]);
        octetbflag_[l].resize(noctets_[l]);
      }
      for (int o = pmaxnoct_[l]; o < noctets_[l]; ++o)
        octets_[l][o].Allocate(nvar_, ncoct, nccoct, ncoeff_, nmatrix_);
      noctets_[l] = 0;
    }
    pmy_mesh_->tree.GetMGOctetList(octets_, octetmap_, noctets_);
  }

  if (needinit_) {
    // reallocate buffers if needed
    if (nbtotal_ != pmy_mesh_->nbtotal) {
      if (nbtotal_ < pmy_mesh_->nbtotal) {
        delete [] ranklist_;
        delete [] rootbuf_;
        ranklist_ = new int[pmy_mesh_->nbtotal];
        int nv = std::max(nvar_*2, ncoeff_);
        rootbuf_ = new Real[pmy_mesh_->nbtotal*nv];
      }
      nbtotal_ = pmy_mesh_->nbtotal;
    }
    nb_rank_ = pmy_mesh_->nblist[0];
    for (int n = 1; n < nranks_; ++n) {
      if (nb_rank_ != pmy_mesh_->nblist[n]) {
        nb_rank_ = 0;
        break;
      }
    }

    // Setting up the MPI information
    // *** this part should be modified when dedicate processes are allocated ***
    // *** we also need to construct another neighbor list for NewtonRaphson ***

    // assume the same parallelization as hydro
    for (int n = 0; n < nbtotal_; ++n)
      ranklist_[n] = pmy_mesh_->ranklist[n];
    for (int n = 0; n < nranks_; ++n) {
      nslist_[n]  = pmy_mesh_->nslist[n];
      nblist_[n]  = pmy_mesh_->nblist[n];
      nvslist_[n] = nslist_[n]*nvar_*2;
      nvlist_[n]  = nblist_[n]*nvar_*2;
      nvslisti_[n] = nslist_[n]*nvar_;
      nvlisti_[n]  = nblist_[n]*nvar_;
    }
    if (ncoeff_ > 0) {
      for (int n = 0; n < nranks_; ++n) {
        nclist_[n]  = nblist_[n]*ncoeff_;
        ncslist_[n] = nslist_[n]*ncoeff_;
      }
    }
    for (NewtonRaphson* pnr : vnr_) {
      pnr->pnrbval->SearchAndSetNeighbors(pmy_mesh_->tree, ranklist_, nslist_);
      pnr->pnrbval->bcolor_ = 0;
    }
    if (nreflevel_ > 0)
      CalculateOctetCoordinates();
    needinit_ = false;
  }

  if (!ftrivial) {
    SetupCoefficients();
    CalculateMatrixAll();
  }

  return;
}



//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SetupCoefficients()
//! \brief Setup coefficients

void NewtonRaphsonDriver::SetupCoefficients() {
  if (ncoeff_ == 0)
    return;
#pragma omp parallel for num_threads(nthreads_)
  for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
    NewtonRaphson *pnr = *itr;
    pnr->RestrictCoefficients();
  }
  TransferFromBlocksToRoot(false);
  TransferCoefficientFromBlocksToRoot();

  // Block boundaries
#pragma omp parallel num_threads(nthreads_)
  {
    for (int lev = nmblevel_ - 2; lev >= 1; lev--) {
#pragma omp for nowait
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->current_level_ = lev;
        pnr->pnrbval->StartReceivingNewtonRaphson(BoundaryQuantity::mg_coeff, false);
      }
#pragma omp for nowait
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->pnrbval->SendNewtonRaphsonBoundaryBuffers(BoundaryQuantity::mg_coeff, false);
      }
#pragma omp for nowait
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->pnrbval->ReceiveNewtonRaphsonCoefficientBoundaryBuffers();
      }
#pragma omp for nowait
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->pnrbval->ClearBoundaryNewtonRaphson(BoundaryQuantity::mg_coeff);
      }
      if (nreflevel_ > 0) {
#pragma omp for nowait
        for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
          NewtonRaphson *pnr = *itr;
          pnr->pnrbval->ProlongateNewtonRaphsonBoundaries(false, true);
        }
      }
#pragma omp for nowait
      for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
        NewtonRaphson *pnr = *itr;
        pnr->pnrbval->ApplyPhysicalBoundaries(0, true);
      }
    }
#pragma omp for nowait
    for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
      NewtonRaphson *pnr = *itr;
      pnr->current_level_ = nmblevel_ - 1;
    }
  }
  for (int lev = 0; lev < nreflevel_; lev++) { // octets
    current_level_ = nrootlevel_ + lev;
    SetBoundariesOctets(false, false, true);
  }
  for (int lev = nrootlevel_ - 1; lev >= 0; lev--) {
    mgroot_->current_level_ = lev;
    mgroot_->pnrbval->ApplyPhysicalBoundaries(0, true);
  }
  mgroot_->current_level_ = nrootlevel_ - 1;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SolveVCycle(int npresmooth, int npostsmooth)
//! \brief Solve the V-cycle starting from the current level

void NewtonRaphsonDriver::SolveVCycle(int npresmooth, int npostsmooth) {
  int startlevel=current_level_;
  // std::cout << "In SolveVCycle: startlevel " << startlevel << std::endl;
  coffset_ ^= 1;
  while (current_level_ > 0)
    OneStepToCoarser(npresmooth);
  SolveCoarsestGrid();
  while (current_level_ < startlevel) {
    OneStepToFiner(npostsmooth);
    Real def = 0.0;
    for (int v = 0; v < nvar_; ++v)
      def += CalculateDefectNorm(MGNormType::l2, v);
    // std::cout << "NewtonRaphson defect after post smooth : " << def << std::endl;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SolveFMGCycle()
//! \brief Solve the FMG Cycle using the V(1,1) or F(0,1) cycle

void NewtonRaphsonDriver::SolveFMGCycle() {
  // std::cout << "ntotallevel_ " << ntotallevel_ << std::endl;
  Real def = 0.0;
  for (int v = 0; v < nvar_; ++v)
    def += CalculateDefectNorm(MGNormType::l2, v);
  if (Globals::my_rank == 0 && fshowdef_)
    std::cout << "Before FMG defect L2-norm : " << def << std::endl;

  for (fmglevel_ = 0; fmglevel_ < ntotallevel_; fmglevel_++) {
    SolveVCycle(npresmooth_, npostsmooth_);
    if (fmglevel_ != ntotallevel_-1) {
      FMGProlongate();
    }
    if (matrixmode_ == 1)
      CalculateMatrixAll();
  }
  // std::cout << "fmglevel_ " << fmglevel_ << std::endl;
  def = 0.0;
  for (int v = 0; v < nvar_; ++v)
    def += CalculateDefectNorm(MGNormType::l2, v);
  if (Globals::my_rank == 0 && fshowdef_)
    std::cout << "After FMG defect L2-norm : " << def << std::endl;
  fmglevel_ = ntotallevel_ - 1;
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  if (eps_ >= 0.0)
    SolveIterative();
  else
    SolveIterativeFixedTimes();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SolveIterative()
//  \brief Solve iteratively until the convergence is achieved

void NewtonRaphsonDriver::SolveIterative() {
  int n = 0;
  Real def = 0.0, defmax = 0.0;
  for (int v = 0; v < nvar_; ++v) {
    def += CalculateDefectNorm(NRNormType::l2, v);
//    defmax = std::max(defmax, CalculateDefectNorm(NRNormType::max, v));
  }
//  if (Globals::my_rank == 0)
//    std::cout << "initial defect " << def << " max " << defmax << std::endl;
  while (def > eps_) {
    SolveVCycle(npresmooth_, npostsmooth_);
    if (matrixmode_ == 1)
      CalculateMatrixAll();
    Real olddef = def, oldmax = defmax;
    def = 0.0, defmax = 0.0;
    for (int v = 0; v < nvar_; ++v) {
      def += CalculateDefectNorm(NRNormType::l2, v);
//      defmax = std::max(defmax, CalculateDefectNorm(NRNormType::max, v));
    }
   if (Globals::my_rank == 0)
     std::cout << "[debug] niter " << n << " def " << def << " convergence factor "
               << def/olddef<< " defmax  "<< defmax << " cf "
               <<  defmax/oldmax << std::endl;
    if (def/olddef > 0.9) {
      if (eps_ == 0.0) break;
      if (Globals::my_rank == 0)
        std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                  << "Slow multigrid convergence : defect norm = " << def
                  << ", convergence factor = " << def/olddef << "." << std::endl;
      if (def/olddef > 1.0) {
        if (Globals::my_rank == 0)
          std::cout << "### Warning in NewtonRaphsonDriver::SolveIterative" << std::endl
                    << "NewtonRaphson is diverging: defect norm = " << def
                    << ", convergence factor = " << def/olddef << ", and niter = " << n << "." << std::endl;
        break;
      }
    }
    // if (n > 100) {
    if (n > 30) {
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
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::SolveIterativeFixedTimes()
//  \brief Solve iteratively niter_ times

void NewtonRaphsonDriver::SolveIterativeFixedTimes() {
  for (int n = 0; n < niter_; ++n) {
    SolveVCycle(npresmooth_, npostsmooth_);
    if (matrixmode_ == 1)
      CalculateMatrixAll();
  }
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  Real def = 0.0;
  for (int v = 0; v < nvar_; ++v)
    def += CalculateDefectNorm(MGNormType::l2, v);
  if (fshowdef_ && Globals::my_rank == 0)
    std::cout << "NewtonRaphson defect L2-norm : " << def << std::endl;

  return;
}


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

//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphsonDriver::CalculateMatrixAll()
//! \brief Calculate Matrix elements for levels higher than the current level

void NewtonRaphsonDriver::CalculateMatrixAll() {
  if (nmatrix_ == 0)
    return;
  RestrictInitialData();
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) {
#pragma omp parallel for num_threads(nthreads_)
    for (auto itr = vnr_.begin(); itr < vnr_.end(); itr++) {
      NewtonRaphson *pnr = *itr;
      pnr->CalculateMatrixBlockAll();
    }
  }
  if (current_level_ >= nrootlevel_ - 1 && nreflevel_ > 0) {
    const int &ngh = mgroot_->ngh_;
    for (int l = current_level_ - (nrootlevel_ - 1); l >= 0; --l) {  // fine to coarse
#pragma omp parallel for num_threads(nthreads_)
      for (int o = 0; o < noctets_[l]; ++o) {
        MGOctet &oct = octets_[l][o];
        mgroot_->CalculateMatrix(oct.matrix, oct.u, oct.src, oct.coeff, l+1,
                          os_, oe_, os_, oe_, os_, oe_, false);
      }
    }
  }
  mgroot_->CalculateMatrixBlockAll();

  return;
}
