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

NewtonRaphson::NewtonRaphson(NewtonRaphsonDriver *pmd, MeshBlock *pmb, int nghost) :
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

  // enroll NRBoundaryVariable object
  nrbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&nrbvar);
  pmb->pbval->bvars_main_int.push_back(&nrbvar);
}


//----------------------------------------------------------------------------------------
//! \fn NewtonRaphson::~NewtonRaphson
//! \brief NewtonRaphson destroctor

NewtonRaphson::~NewtonRaphson() {
}


// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphson::LoadFinestData(const AthenaArray<Real> &src, int ns, int ngh)
// //! \brief Fill the inital guess in the active zone of the finest level

// void NewtonRaphson::LoadFinestData(const AthenaArray<Real> &src, int ns, int ngh) {
//   AthenaArray<Real> &dst=u_;
//   int is = pmy_block_->is;
//   int ie = pmy_block_->ie;
//   int js = pmy_block_->js;
//   int je = pmy_block_->je;
//   int ks = pmy_block_->ks;
//   int ke = pmy_block_->ke;
//   for (int v=0; v<nvar_; ++v) {
//     int nsrc=ns+v;
//     for (int k=ks; k<=ke; ++k) {
//       for (int j=js; j<=je; ++j) {
// #pragma omp simd
//         for (int i=is; i<=ie; ++i) {
//           dst(v,k,j,i)=src(nsrc,k,j,i);
//         }
//       }
//     }
//   }
//   return;
// }


// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphson::LoadSource(const AthenaArray<Real> &src, int ns, int ngh,
// //!                                Real fac)
// //! \brief Fill the source in the active zone of the finest level

// void NewtonRaphson::LoadSource(const AthenaArray<Real> &src, int ns, int ngh, Real fac) {
//   AthenaArray<Real> &dst=src_;
//   int is = pmy_block_->is;
//   int ie = pmy_block_->ie;
//   int js = pmy_block_->js;
//   int je = pmy_block_->je;
//   int ks = pmy_block_->ks;
//   int ke = pmy_block_->ke;
//   if (fac == 1.0) {
//     for (int v=0; v<nvar_; ++v) {
//       int nsrc=ns+v;
//       for (int k=ks; k<=ke; ++k) {
//         for (int j=js; j<=je; ++j) {
// #pragma omp simd
//           for (int i=is; i<=ie; ++i) {
//             dst(v,k,j,i)=src(nsrc,k,j,i);
//           }
//         }
//       }
//     }
//   } else {
//     for (int v=0; v<nvar_; ++v) {
//       int nsrc=ns+v;
//       for (int k=ks; k<=ke; ++k) {
//         for (int j=js; j<=je; ++j) {
// #pragma omp simd
//           for (int i=is; i<=ie; ++i) {
//             dst(v,k,j,i)=src(nsrc,k,j,i)*fac;
//           }
//         }
//       }
//     }
//   }
//   return;
// }


// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphson::LoadCoefficients(const AthenaArray<Real> &coeff, int ngh)
// //! \brief Load coefficients of the diffusion and source terms

// void NewtonRaphson::LoadCoefficients(const AthenaArray<Real> &coeff, int ngh) {
//   AthenaArray<Real> &cm=coeff_;
//   int is = pmy_block_->is;
//   int ie = pmy_block_->ie;
//   int js = pmy_block_->js;
//   int je = pmy_block_->je;
//   int ks = pmy_block_->ks;
//   int ke = pmy_block_->ke;
//   for (int v = 0; v < ncoeff_; ++v) {
//     for (int k=ks; k<=ke; ++k) {
//       for (int j=js; j<=je; ++j) {
// #pragma omp simd
//         for (int i=is; i<=ie; ++i) {
//           cm(v,k,j,i) = coeff(v,k,j,i);
//         }
//       }
//     }
//   }
//   return;
// }


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

  CalculateDefect(def_, u_, src_,
                  coeff_, matrix_,
                  is, ie, js, je, ks, ke, th);

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NewtonRaphson::CalculateMatrixBlock()
//  \brief calculate matrix elements

void NewtonRaphson::CalculateMatrixBlock() {
  int is = pmy_block_->is;
  int ie = pmy_block_->ie;
  int js = pmy_block_->js;
  int je = pmy_block_->je;
  int ks = pmy_block_->ks;
  int ke = pmy_block_->ke;

  CalculateMatrix(matrix_, u_, src_, coeff_,
                  is, ie, js, je, ks, ke, false);

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
  Real dx=rdx_, dy=rdy_, dz=rdz_; //??

  CalculateDefect(def_, u_, src_,
                  coeff_, matrix_,
                  is, ie, js, je, ks, ke, false);

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


// //----------------------------------------------------------------------------------------
// //! \fn void NewtonRaphson::SetData(MGVariable type, int n, int k, int j, int i, Real v)
// //! \brief set a value to a cell on the current level

// void NewtonRaphson::SetData(NRVariable type, int n, int k, int j, int i, Real v) {
//   auto& arr = (type == NRVariable::src) ? src_:
//               (type == NRVariable::u  ) ? u_ : coeff_;

//   const int niTot = arr.GetDim3(), njTot = arr.GetDim2(), nkTot = arr.GetDim1();
//   const int niInt = size_.nx1, njInt = size_.nx2, nkInt = size_.nx3;
//   const int ngh   = ngh_;

//   int ni = arr.GetDim3();
//   int nj = arr.GetDim2();
//   int nk = arr.GetDim1();
//   int nn = arr.GetDim4();  // optional
//   if ((n < 0) || (n >= nn) ||
//     (ngh_ + i < 0) || (ngh_ + i >= ni) ||
//     (ngh_ + j < 0) || (ngh_ + j >= nj) ||
//     (ngh_ + k < 0) || (ngh_ + k >= nk)) {
//     std::fprintf(stderr,
//       "OOB in SetData(): n=%d i=%d j=%d k=%d (nn=%d ni=%d nj=%d nk=%d ngh=%d)\n",
//       n,i,j,k, nn,ni,nj,nk,ngh_);
//     __builtin_trap();  // or throw
//   }
//   if (type == MGVariable::src)
//     src_[current_level_](n, ngh_+k, ngh_+j, ngh_+i) = v;
//   else if (type == MGVariable::u)
//     u_[current_level_](n, ngh_+k, ngh_+j, ngh_+i) = v;
//   else
//     coeff_[current_level_](n, ngh_+k, ngh_+j, ngh_+i) = v;
//   return;
// }

void NewtonRaphson::AddDifference(AthenaArray<Real> &dst, const AthenaArray<Real> &src,
                   int is, int ie, int js, int je, int ks, int ke) {
  for (int v=0; v<nvar_; ++v) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          dst(v,k,j,i) += (dst(v,k,j,i) - src(v,k,j,i));
        }
      }
    }
  }
  return;
}
