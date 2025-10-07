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
class FLD2;

// enum class NRNormType {max, l1, l2};

// namespace NewtonRaphsonFLD {
//   constexpr int NTEMP=2, NMATRIX=15, NCOEFF=8, NOPACITY=2;
//   // enum TempIndex {GAS=0, RAD=1};
//   enum TempIndex {RAD=0, GAS=1}; // caution!!
//   enum CoeffIndex {DCCF=0, DCCS=1, DXM=2, DXP=3, DYM=4, DYP=5, DZM=6, DZP=7};
//   enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,};
//   enum OpacityIndex {SIGMA_P=0, SIGMA_R=1};
// }

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

  FLD2 *pfld;
  // NRBoundaryValues *pnrbval;
  BoundaryQuantity btype, btypef;

  void LoadHydroVariables() final;
  void UpdateHydroVariables() final;
  void CalculateCoefficients(const AthenaArray<Real> &work,
                             const AthenaArray<Real> &pre,
                             const AthenaArray<Real> &w, Real dt) final;

  void LoadSource(const AthenaArray<Real> &src, int ns, int ngh, Real fac);
  void LoadCoefficients(const AthenaArray<Real> &coeff, int ngh);
  void RetrieveResult(AthenaArray<Real> &dst, int ns, int ngh);
  void RetrieveDefect(AthenaArray<Real> &dst, int ns, int ngh);
  void ZeroClearData();
  void SmoothBlock(int color);
  void CalculateDefectBlock();
  void CalculateFASRHSBlock();
  void CalculateMatrixBlockCurrent();
  void CalculateMatrixBlockAll();
  Real CalculateDefectNorm(MGNormType nrm, int n);
  Real CalculateTotal(MGVariable type, int n);
  // void SubtractAverage(MGVariable type, int n, Real ave);
  void StoreOldData();
  // Real GetCoarsestData(MGVariable type, int n);
  void SetData(MGVariable type, int n, int k, int j, int i, Real v);

  // physics-dependent virtual functions
  // void Smooth(AthenaArray<Real> &dst, const AthenaArray<Real> &src,
  //                     const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrx,
  //                     int rlev, int il, int iu, int jl, int ju, int kl, int ku,
  //                     int color, bool th) final;
  void CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
               const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
               const AthenaArray<Real> &matrix, int il, int iu, int jl, int ju,
               int kl, int ku, bool th) final;
  // void CalculateFASRHS(AthenaArray<Real> &def, const AthenaArray<Real> &src,
  //                const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
  //                int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) final;
  void CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &u,
               const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
               int il, int iu, int jl, int ju, int kl, int ku, bool th) final;

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
  LogicalLocation loc_;
  RegionSize size_;
  BoundaryFlag nr_block_bcs_[6];
  int ngh_, nvar_, ncoeff_, nmatrix_;
  Real rdx_, rdy_, rdz_;
  Real defscale_;
  // AthenaArray<Real> *u_, *def_, *src_, *uold_, *coeff_, *matrix_;
  AthenaArray<Real> delta_u_;
  // MGCoordinates *coord_, *ccoord_;


 private:
  TaskStates ts_;
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
  void SetLinearSolver() final;

  friend class NewtonRaphson;
  friend class NewtonRaphsonTaskList;
  friend class NRBoundaryValues;
  friend class NRFLD;
  friend class linearMG;

 protected:
};

#endif // NR_MULTIGRID_NRFLD_HPP_
