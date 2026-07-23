#ifndef NR_MULTIGRID_NRFLD_HPP_
#define NR_MULTIGRID_NRFLD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson.hpp
//  \brief defines the Newton-Raphson solver class

// C headers

// C++ headers
#include <cstdint>  // std::int64_t
#include <cstdio> // std::size_t
#include <iostream>
#include <unordered_map>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals_interfaces.hpp"
#include "../bvals/cc/nr/bvals_nr.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../newton_raphson/Newton_Raphson.hpp"
#include "../fld/fld.hpp"
#include "../linear_solver/linearMG/linearMG.hpp"
// #include "../task_list/nr_task_list.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class Mesh;
class MeshBlock;
class ParameterInput;
class Coordinates;
// class linearMG;
// class linearMGDriver;
class FLD;

// enum class NRNormType {max, l1, l2};

namespace NewtonRaphsonFLD {
  constexpr int NNRDIV = 12, NDCOEFF = 2;
  enum DerivativeIndex {
    Fg=0,
    Fr=1,
    dFg_deg=2,
    dFg_dEr=3,
    dFr_deg=4,
    dFr_dEr=5,
    dFr_dEr_xm=6,
    dFr_dEr_xp=7,
    dFr_dEr_ym=8,
    dFr_dEr_yp=9,
    dFr_dEr_zm=10,
    dFr_dEr_zp=11,
  };
  enum CoeffIndex {DRHO=0, DCOUPLE=1};
}

// class NewtonRaphson {
//  public:
//   // NewtonRaphson(MeshBlock *pmb, ParameterInput *pin);
//   NewtonRaphson(Mesh *pm, ParameterInput *pin);
//   ~NewtonRaphson();
//   void Solve(int stage, Real dt);
//   void CalculateCoefficients(const AthenaArray<Real> &u_work,
//                              const AthenaArray<Real> &u_pre,
//                              const AthenaArray<Real> &w_hydro, Real dt);
//   void UpdateHydroVariables(const AthenaArray<Real> &w, AthenaArray<Real> &u,
//                              const AthenaArray<Real> &u_fld);
//   void UpdateRadEnergy(AthenaArray<Real> &u_work, const AthenaArray<Real> &delta_u);
//   Real CalculateDefectNorm(NRNormType nrm, int n);

//   // friend class linearMG;

//   // linearMG *plinmg;
//   // linearMGDriver *plinmgdriver;

//   CellCenteredBoundaryVariable nrmgfldbvar;

//   AthenaArray<Real> u_pre; // previous radiation energy density
//   AthenaArray<Real> u_work; // radiation energy density in work
//   AthenaArray<Real> delta_u; // correction of radiation energy density
//   AthenaArray<Real> coeff; // coefficients
//   AthenaArray<Real> rhs; // right-hand side
//   bool only_rad = false;

//   AthenaArray<Real> sigma_p, sigma_r;

//  private:
//   MeshBlock* pmy_block;
//   int max_iter_;
// };


//! \class NRFLD
//  \brief NewtonRaphson object for FLD

class NRFLD : public NewtonRaphson {
 public:
  NRFLD(MeshBlock *pmb, ParameterInput *pin);
  ~NRFLD();

  FLD *pfld;
  // NRBoundaryValues *pnrbval;
  BoundaryQuantity btype, btypef;

  void LoadVariables() final;
  void UpdateHydroVariables() final;
  void CalculateCoefficientsOnce(const AthenaArray<Real> &u_pre,
                                 const AthenaArray<Real> &w,
                                 AthenaArray<Real> &def_coeff,
                                 AthenaArray<Real> &derivetive) final;
  void CalculateCoefficients(const AthenaArray<Real> &u_rad_old,
                             const AthenaArray<Real> &u_rad_new,
                            //  const AthenaArray<Real> &u_gas_old,
                            //  const AthenaArray<Real> &u_gas_new,
                             const AthenaArray<Real> &def_coeff,
                             AthenaArray<Real> &coeff,
                             AthenaArray<Real> &derivetive,
                             AthenaArray<Real> &src,
                             Real dt) final;
  void ApplyPhysicalBoundary() final;
  void PrintCellPhysicsDebug(int k, int j, int i) final;
  void StoreIterate() final;
  void RestoreIterate() final;

  // void LoadSource(const AthenaArray<Real> &src, int ns, int ngh, Real fac);
  // void LoadCoefficients(const AthenaArray<Real> &coeff, int ngh);
  // void RetrieveResult(AthenaArray<Real> &dst, int ns, int ngh);
  // void RetrieveDefect(AthenaArray<Real> &dst, int ns, int ngh);
  // void ZeroClearData();
  // void SmoothBlock(int color);
  // void CalculateDefectBlock();
  // void CalculateFASRHSBlock();
  // void CalculateMatrixBlockCurrent();
  // void CalculateMatrixBlockAll();
  // Real CalculateDefectNorm(MGNormType nrm, int n);
  // Real CalculateTotal(MGVariable type, int n);
  // void SubtractAverage(MGVariable type, int n, Real ave);
  // void StoreOldData();
  // Real GetCoarsestData(MGVariable type, int n);
  // void SetData(MGVariable type, int n, int k, int j, int i, Real v);

  // physics-dependent virtual functions
  // void Smooth(AthenaArray<Real> &dst, const AthenaArray<Real> &src,
  //                     const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrx,
  //                     int rlev, int il, int iu, int jl, int ju, int kl, int ku,
  //                     int color, bool th) final;
  void CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
                       const AthenaArray<Real> &u_old, const AthenaArray<Real> &coeff,
                       const AthenaArray<Real> &def_coeff,
                       bool th) final;
  // void CalculateFASRHS(AthenaArray<Real> &def, const AthenaArray<Real> &src,
  //                const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
  //                int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) final;
  // void CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &u,
  //              const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
  //              int il, int iu, int jl, int ju, int kl, int ku, bool th) final;

  void AddDifference(AthenaArray<Real> &dst,
                     AthenaArray<Real> &delta,
                     const AthenaArray<Real> &derivetive) final;

  friend class NewtonRaphsonDriver;
  friend class NewtonRaphsonTaskList;
  friend class NRBoundaryValues;
  friend class linearMGDriver;

  // for boundaries
  AthenaArray<Real> coarse_u_;
  AthenaArray<Real> u_flux_;
  int refinement_idx{-1};

 protected:
  NewtonRaphsonDriver *pmy_driver_;
  MeshBlock *pmy_block_;
  int ngh_, nvar_, ncoeff_, nmatrix_;
  LogicalLocation loc_;
  RegionSize size_;
  BoundaryFlag nr_block_bcs_[6];
  Real rdx_, rdy_, rdz_;
  Real defscale_;
  Real max_update_fraction_;
  bool fixed_linear_coefficients_initialized_;
  // AthenaArray<Real> *u_, *def_, *src_, *uold_, *coeff_, *matrix_;
  // AthenaArray<Real> delta_u_;
  AthenaArray<Real> u_gas_, u_gas_iter_backup_;
  AthenaArray<Real> last_delta_rad_;
  // MGCoordinates *coord_, *ccoord_;


 private:
  TaskStates ts_;
  // linearMG *plmg_;
};



//! \class NRFLDDriver
//  \brief NRFLD driver

class NRFLDDriver : public NewtonRaphsonDriver {
 public:
  // NRFLDDriver(Mesh *pm, NRBoundaryFunc *NRBoundary, NRBoundaryFunc *NRCoeffBoundary,
  //                 NRMaskFunc NRSourceMask, NRMaskFunc NRCoeffMask, int invar, int ncoeff,
  //                 int nmatrix);
  // virtual ~NRFLDDriver();
  NRFLDDriver(Mesh *pm, ParameterInput *pin);
  ~NRFLDDriver();

  // void Solve(int step, Real dt = 0.0) final;

  friend class NewtonRaphson;
  friend class NewtonRaphsonTaskList;
  friend class NRBoundaryValues;
  friend class NRFLD;
  friend class linearMGDriver;
  friend class linearMG;

  // linearMGDriver *plmgd;

 protected:
};

#endif // NR_MULTIGRID_NRFLD_HPP_
