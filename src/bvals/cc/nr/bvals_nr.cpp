//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_nr_hydro.cpp
//! \brief implements boundary functions for NewtonRaphson variables and utilities
//! a derived class of the CellCenteredBoundaryVariable base class.

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../newton_raphson/Newton_Raphson.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../utils/buffer_utils.hpp"
#include "bvals_nr.hpp"

//----------------------------------------------------------------------------------------
//! \fn NRBoundaryVariable::NRBoundaryVariable
//! \brief

NRBoundaryVariable::NRBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var_nr, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux, int num_phys) :
    CellCenteredBoundaryVariable(pmb, var_nr, coarse_var, var_flux, num_phys, true) {
}

//----------------------------------------------------------------------------------------
//! \fn void NRBoundaryVariable::SelectCoarseBuffer(NRBoundaryQuantity type)
//! \brief

void NRBoundaryVariable::SelectCoarseBuffer() {
  if (pmy_mesh_->multilevel) {
    coarse_buf = &(pmy_block_->pnr->coarse_u_);
  }
  return;
}

// //----------------------------------------------------------------------------------------
// //! \fn void NRBoundaryVariable::SwapNRQuantity
// //! \brief
// //! \todo (felker):
// //! * make general (but restricted) setter fns in CellCentered and FaceCentered

// void NRBoundaryVariable::SwapNRQuantity(AthenaArray<Real> &var_nr,
//                                               NRBoundaryQuantity nr_type) {
//   var_cc = &var_nr;
//   SelectCoarseBuffer(nr_type);
//   return;
// }


//----------------------------------------------------------------------------------------
//! \fn void NRBoundaryVariable::SetBoundarySameLevel(Real *buf,
//!                                                      const NeighborBlock& nb)
//! \brief Set nr boundary received from a block on the same level

void NRBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                 const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cc;

  if (nb.ni.ox1 == 0)     si = pmb->is,        ei = pmb->ie;
  else if (nb.ni.ox1 > 0) si = pmb->ie + 1,      ei = pmb->ie + NGHOST;
  else              si = pmb->is - NGHOST, ei = pmb->is - 1;
  if (nb.ni.ox2 == 0)     sj = pmb->js,        ej = pmb->je;
  else if (nb.ni.ox2 > 0) sj = pmb->je + 1,      ej = pmb->je + NGHOST;
  else              sj = pmb->js - NGHOST, ej = pmb->js - 1;
  if (nb.ni.ox3 == 0)     sk = pmb->ks,        ek = pmb->ke;
  else if (nb.ni.ox3 > 0) sk = pmb->ke + 1,      ek = pmb->ke + NGHOST;
  else              sk = pmb->ks - NGHOST, ek = pmb->ks - 1;

  int p = 0;
  BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  return;
}

// //----------------------------------------------------------------------------------------
// //! \fn int NRBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
// //                                                                const NeighborBlock& nb)
// //  \brief Set cell-centered boundary buffers for sending to a block on the same level

// int NRBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) {
//   MeshBlock *pmb = pmy_block_;
//   int si, sj, sk, ei, ej, ek;

//   si = (nb.ni.ox1 > 0) ? (pmb->ie - NGHOST + 1) : pmb->is;
//   ei = (nb.ni.ox1 < 0) ? (pmb->is + NGHOST - 1) : pmb->ie;
//   sj = (nb.ni.ox2 > 0) ? (pmb->je - NGHOST + 1) : pmb->js;
//   ej = (nb.ni.ox2 < 0) ? (pmb->js + NGHOST - 1) : pmb->je;
//   sk = (nb.ni.ox3 > 0) ? (pmb->ke - NGHOST + 1) : pmb->ks;
//   ek = (nb.ni.ox3 < 0) ? (pmb->ks + NGHOST - 1) : pmb->ke;
//   int p = 0;
//   AthenaArray<Real> &var = *var_cc;
//   BufferUtility::PackData(var, buf, sk, ek, nl_, nu_, si, ei, sj, ej, p);

//   return p;
// }
