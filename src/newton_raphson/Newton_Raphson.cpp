//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson.cpp
//! \brief implementation of the functions commonly used in NewtonRaphson

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>    // memset, memcpy
#include <iostream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "Newton_Raphson.hpp"

//----------------------------------------------------------------------------------------
//! \fn NewtonRaphson::NewtonRaphson(NewtonRaphsonDriver *pmd, MeshBlock *pmb, int nghost)
//  \brief NewtonRaphson constructor

NewtonRaphson::NewtonRaphson(NewtonRaphsonDriver *pmd, MeshBlock *pmb, int nghost,
                             int nderivetive, int ndef_coeff) :
  pmy_driver_(pmd), pmy_block_(pmb),
  ngh_(nghost), nvar_(pmd->nvar_),
  u_(nvar_, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  flux{ {nvar_, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
        {nvar_, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
          (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
          AthenaArray<Real>::DataStatus::empty)},
        {nvar_, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
          (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
          AthenaArray<Real>::DataStatus::empty)}
  },
  coarse_u_(nvar_, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                AthenaArray<Real>::DataStatus::empty)),
  nrbvar(pmb, &u_, &coarse_u_, flux),
  delta_u_(nvar_, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  coarse_delta_u_(nvar_, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                AthenaArray<Real>::DataStatus::empty)),
  empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
  delta_bvar(pmb, &delta_u_, &coarse_delta_u_, empty_flux, false), //!
  derivetive_(nderivetive, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  def_coeff_(ndef_coeff, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  output_defect(true), // caution!
  ncoeff_(pmd->ncoeff_), nmatrix_(pmd->nmatrix_), defscale_(1.0) {
  if (pmy_block_ != nullptr) {
    loc_ = pmy_block_->loc;
    size_ = pmy_block_->block_size;
    if (size_.nx1 != size_.nx2 || size_.nx1 != size_.nx3) {
      std::stringstream msg;
      msg << "### FATAL ERROR in NewtonRaphson::NewtonRaphson" << std::endl
          << "The NewtonRaphson solver requires logically cubic MeshBlock." << std::endl;
      ATHENA_ERROR(msg);
      return;
    }
    for (int i = 0; i < 6; ++i) {
      if (pmy_block_->pbval->block_bcs[i] == BoundaryFlag::block)
        nr_block_bcs_[i] = BoundaryFlag::block;
      else
        nr_block_bcs_[i] = pmy_driver_->nr_mesh_bcs_[i];
    }
  } else {
    loc_.lx1 = loc_.lx2 = loc_.lx3 = 0;
    loc_.level = 0;
    size_ = pmy_driver_->pmy_mesh_->mesh_size;
    size_.nx1 = pmy_driver_->nrbx1_;
    size_.nx2 = pmy_driver_->nrbx2_;
    size_.nx3 = pmy_driver_->nrbx3_;
    for (int i = 0; i < 6; ++i)
      nr_block_bcs_[i] = pmy_driver_->nr_mesh_bcs_[i];
  }
  rdx_ = (size_.x1max-size_.x1min)/static_cast<Real>(size_.nx1);
  rdy_ = (size_.x2max-size_.x2min)/static_cast<Real>(size_.nx2);
  rdz_ = (size_.x3max-size_.x3min)/static_cast<Real>(size_.nx3);

  src_.NewAthenaArray(nvar_,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  def_.NewAthenaArray(nvar_,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  if (pmy_block_ == nullptr)
    uold_.NewAthenaArray(nvar_,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  else
    uold_.NewAthenaArray(nvar_,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  coeff_.NewAthenaArray(ncoeff_,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  matrix_.NewAthenaArray(nmatrix_,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh *pm = pmy_block_->pmy_mesh;

  pmb->RegisterMeshBlockData(u_);

  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block_->pmr->AddToRefinement(&u_, &coarse_u_);
  }

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pmb->pmy_mesh->multilevel) {
    refinement_idx = pmy_block_->pmr->AddToRefinement(&delta_u_, &coarse_delta_u_);
  }


  // enroll NRBoundaryVariable object
  nrbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&nrbvar);
  pmb->pbval->bvars_main_int.push_back(&nrbvar);

  // Enroll CellCenteredBoundaryVariable object for linear solver
  delta_bvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&delta_bvar);
  pmb->pbval->prfldbvar = &delta_bvar;
}


//----------------------------------------------------------------------------------------
//! \fn NewtonRaphson::~NewtonRaphson
//! \brief NewtonRaphson destroctor

NewtonRaphson::~NewtonRaphson() {
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphson::RetrieveResult(AthenaArray<Real> &dst, int ns, int ngh)
//! \brief Set the result, including the ghost zone

void NewtonRaphson::RetrieveResult(AthenaArray<Real> &dst, int ns, int ngh) {
  const AthenaArray<Real> &src=u_;
  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;
  for (int v=0; v<nvar_; ++v) {
    int ndst=ns+v;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dst(ndst,k,j,i)=src(v,k,j,i);
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphson::RetrieveDefect(AthenaArray<Real> &dst, int ns, int ngh)
//! \brief Set the defect, including the ghost zone

void NewtonRaphson::RetrieveDefect(AthenaArray<Real> &dst, int ns, int ngh) {
  const AthenaArray<Real> &src=def_;
  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;
  for (int v=0; v<nvar_; ++v) {
    int ndst=ns+v;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dst(ndst,k,j,i)=src(v,k,j,i)*defscale_;
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphson::ZeroClearData()
//! \brief Clear the data array with zero

void NewtonRaphson::ZeroClearData() {
  u_.ZeroClear();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphson::CalculateDefectBlock()
//! \brief calculate the residual

void NewtonRaphson::CalculateDefectBlock() {
  int th = false;
#ifdef OPENMP_PARALLEL
  if (pmy_block_ == nullptr)
    th = true;
#endif
  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;

  CalculateDefect(def_, u_, uold_,
                  coeff_, def_coeff_,
                  th);

  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real NewtonRaphson::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the residual norm

Real NewtonRaphson::CalculateDefectNorm(NRNormType nrm, int n) {
  AthenaArray<Real> &def=def_;
  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;
  Real dx=rdx_, dy=rdy_, dz=rdz_;

  CalculateDefect(def_, u_, uold_,
                  coeff_, def_coeff_,
                  false);

  Real norm=0.0;
  if (nrm == NRNormType::max) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd reduction(max: norm)
        for (int i=is; i<=ie; ++i)
          norm = std::max(norm, std::abs(def(n,k,j,i)));
      }
    }
    return norm;
  } else if (nrm == NRNormType::l1) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd reduction(+: norm)
        for (int i=is; i<=ie; ++i)
          norm += std::abs(def(n,k,j,i));
      }
    }
  } else { // L2 norm
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd reduction(+: norm)
        for (int i=is; i<=ie; ++i)
          norm += SQR(def(n,k,j,i));
      }
    }
  }
  return norm*dx*dy*dz*defscale_;
}


// //----------------------------------------------------------------------------------------
// //! \fn Real NewtonRaphson::CalculateTotal(NRVariable type, int n)
// //! \brief calculate the sum of the array (type: 0=src, 1=u)

// Real NewtonRaphson::CalculateTotal(NRVariable type, int n) {
//   AthenaArray<Real> &src =
//                     (type == NRVariable::src) ? src_ : u_;
//   Real s=0.0;
//   int is = pmy_block_->is;
//   int ie = pmy_block_->ie;
//   int js = pmy_block_->js;
//   int je = pmy_block_->je;
//   int ks = pmy_block_->ks;
//   int ke = pmy_block_->ke;
//   int is, ie, js, je, ks, ke;
//   Real dx=rdx_, dy=rdy_, dz=rdz_;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
// #pragma omp simd reduction(+: s)
//       for (int i=is; i<=ie; ++i)
//         s+=src(n,k,j,i);
//     }
//   }
//   return s*dx*dy*dz;
// }


// //----------------------------------------------------------------------------------------
// //! \fn Real NewtonRaphson::SubtractAverage(MGVariable type, int v, Real ave)
// //! \brief subtract the average value (type: 0=src, 1=u)

// void NewtonRaphson::SubtractAverage(MGVariable type, int n, Real ave) {
//   AthenaArray<Real> &dst = (type == MGVariable::src) ? src_[nlevel_-1] : u_[nlevel_-1];
//   int is, ie, js, je, ks, ke;
//   is=js=ks=0;
//   ie=is+size_.nx1+1, je=js+size_.nx2+1, ke=ks+size_.nx3+1;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
// #pragma omp simd
//       for (int i=is; i<=ie; ++i)
//         dst(n,k,j,i)-=ave;
//     }
//   }

//   return;
// }


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphson::StoreOldData()
//! \brief store the old u data in the uold array

void NewtonRaphson::StoreOldData() {
  memcpy(uold_.data(), u_.data(),
         u_.GetSizeInBytes());
  return;
}
