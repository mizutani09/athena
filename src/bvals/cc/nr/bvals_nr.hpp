#ifndef BVALS_CC_NR_BVALS_NR_HPP_
#define BVALS_CC_NR_BVALS_NR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_nr.hpp
//! \brief

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../bvals_cc.hpp"

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//! \brief

class NRBoundaryVariable : public CellCenteredBoundaryVariable {
 public:
  NRBoundaryVariable(MeshBlock *pmb,
                     AthenaArray<Real> *var_nr, AthenaArray<Real> *coarse_var,
                     AthenaArray<Real> *var_flux, int num_phys);
  virtual ~NRBoundaryVariable() = default;

  // switch between NR class members "u" and "w" (or "u" and "u1", ...)
  // void SwapNRQuantity(AthenaArray<Real> &var_nr);
  void SelectCoarseBuffer();

  // //!@{
  // //! BoundaryPhysics: need to flip sign of velocity vectors for Reflect*()
  // void ReflectInnerX1(Real time, Real dt,
  //                     int il, int jl, int ju, int kl, int ku, int ngh) override;
  // void ReflectOuterX1(Real time, Real dt,
  //                     int iu, int jl, int ju, int kl, int ku, int ngh) override;
  // void ReflectInnerX2(Real time, Real dt,
  //                     int il, int iu, int jl, int kl, int ku, int ngh) override;
  // void ReflectOuterX2(Real time, Real dt,
  //                     int il, int iu, int ju, int kl, int ku, int ngh) override;
  // void ReflectInnerX3(Real time, Real dt,
  //                     int il, int iu, int jl, int ju, int kl, int ngh) override;
  // void ReflectOuterX3(Real time, Real dt,
  //                     int il, int iu, int jl, int ju, int ku, int ngh) override;
  // //!@}

  //protected:
 private:
  void SetBoundarySameLevel(Real *buf, const NeighborBlock& nb) override;
  //! ???
  //! NR is a unique cell-centered variable because of the relationship between
  //! NRBoundaryQuantity::cons u and NRBoundaryQuantity::prim w.
  // int LoadFluxBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) final;
};

#endif // BVALS_CC_NR_BVALS_NR_HPP_
