#ifndef RAD_FLD_NEWTON_RAPHSON_HPP_
#define RAD_FLD_NEWTON_RAPHSON_HPP_
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

enum class NRNormType {max, l1, l2};

namespace NewtonRaphsonFLD {
  constexpr int NTEMP=2, NMATRIX=15, NCOEFF=8, NOPACITY=2;
  // enum TempIndex {GAS=0, RAD=1};
  enum TempIndex {RAD=0, GAS=1}; // caution!!
  enum CoeffIndex {DCCF=0, DCCS=1, DXM=2, DXP=3, DYM=4, DYP=5, DZM=6, DZP=7};
  enum MatrixIndex {CCC=0, CCM=1, CCP=2, CMC=3, CPC=4, MCC=5, PCC=6,};
  enum OpacityIndex {SIGMA_P=0, SIGMA_R=1};
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


//! \class NewtonRaphson
//  \brief NewtonRaphson object containing each MeshBlock and/or the root block

class NewtonRaphson {
 public:
  NewtonRaphson(NewtonRaphsonDriver *pmd, MeshBlock *pmb, int nghost);
  virtual ~NewtonRaphson();

  NRBoundaryValues *pnrbval;
  BoundaryQuantity btype, btypef;

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
  virtual void Smooth(AthenaArray<Real> &dst, const AthenaArray<Real> &src,
                      const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrx,
                      int rlev, int il, int iu, int jl, int ju, int kl, int ku,
                      int color, bool th) = 0;
  virtual void CalculateDefect(AthenaArray<Real> &def, const AthenaArray<Real> &u,
               const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
               const AthenaArray<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
               int kl, int ku, bool th) = 0;
  virtual void CalculateFASRHS(AthenaArray<Real> &def, const AthenaArray<Real> &src,
                 const AthenaArray<Real> &coeff, const AthenaArray<Real> &matrix,
                 int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) = 0;
  virtual void CalculateMatrix(AthenaArray<Real> &matrix, const AthenaArray<Real> &u,
               const AthenaArray<Real> &src, const AthenaArray<Real> &coeff,
               int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) {}

  friend class NewtonRaphsonDriver;
  friend class NewtonRaphsonTaskList;
  friend class NRBoundaryValues;
  friend class linearMGDriver;

  // for boundaries
  AthenaArray<Real> coarse_u_;
  int refinement_idx{-1};

  NRBoundaryVariable nrbvar;

 protected:
  NewtonRaphsonDriver *pmy_driver_;
  MeshBlock *pmy_block_;
  LogicalLocation loc_;
  RegionSize size_;
  BoundaryFlag nr_block_bcs_[6];
  int ngh_, nvar_, ncoeff_, nmatrix_;
  Real rdx_, rdy_, rdz_;
  Real defscale_;
  AthenaArray<Real> *u_, *def_, *src_, *uold_, *coeff_, *matrix_;
  // MGCoordinates *coord_, *ccoord_;


 private:
  TaskStates ts_;
};



//! \class NewtonRaphsonDriver
//  \brief NewtonRaphson driver

class NewtonRaphsonDriver {
 public:
  NewtonRaphsonDriver(Mesh *pm, NRBoundaryFunc *NRBoundary, NRBoundaryFunc *NRCoeffBoundary,
                  NRMaskFunc NRSourceMask, NRMaskFunc NRCoeffMask, int invar, int ncoeff,
                  int nmatrix);
  virtual ~NewtonRaphsonDriver();

  // pure virtual function
  virtual void Solve(int step, Real dt = 0.0) = 0;

  friend class NewtonRaphson;
  friend class NewtonRaphsonTaskList;
  friend class NRBoundaryValues;
  friend class NRFLD;
  friend class linearMG;

 protected:
  void CheckBoundaryFunctions();
  void SetupNewtonRaphson(bool ftrivial = false);
  void SetupCoefficients();
  void RestrictInitialData();
  void SolveVCycle(int npresmooth, int npostsmooth);
  void SolveFMGCycle();
  void SolveIterative();
  void SolveIterativeFixedTimes();

  Real CalculateDefectNorm(NRNormType nrm, int n);
  void CalculateMatrixAll();

  // small functions
  int GetNumNewtonRaphsons() { return nblist_[Globals::my_rank]; }

  int nranks_, nthreads_, nbtotal_, nvar_, ncoeff_, nmatrix_, mode_, matrixmode_;
  int *nslist_, *nblist_, *nvlist_, *nvslist_, *nvlisti_, *nvslisti_,
                          *nclist_, *ncslist_, *ranklist_;
  int nrbx1_, nrbx2_, nrbx3_;
  BoundaryFlag nr_mesh_bcs_[6];
  NRBoundaryFunc NRBoundaryFunction_[6];
  NRBoundaryFunc NRCoeffBoundaryFunction_[6];
  Mesh *pmy_mesh_;
  std::vector<NewtonRaphson*> vnr_;
  bool needinit_, fshowdef_;
  Real eps_, dt_;
  int niter_;
  int os_, oe_;

  NewtonRaphsonTaskList *nrtlist_;

 private:
  Real *rootbuf_;
  int nb_rank_;
#ifdef MPI_PARALLEL
  MPI_Comm MPI_COMM_NEWTON_RAPHSON;
  int nr_phys_id_;
#endif
};


// NewtonRaphson Boundary functions

// void MGPeriodicInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
//                        int is, int ie, int js, int je, int ks, int ke, int ngh,
//                        const MGCoordinates &coord);
// void MGPeriodicOuterX1(AthenaArray<Real> &dst, Real time, int nvar,
//                        int is, int ie, int js, int je, int ks, int ke, int ngh,
//                        const MGCoordinates &coord);
// void MGPeriodicInnerX2(AthenaArray<Real> &dst, Real time, int nvar,
//                        int is, int ie, int js, int je, int ks, int ke, int ngh,
//                        const MGCoordinates &coord);
// void MGPeriodicOuterX2(AthenaArray<Real> &dst, Real time, int nvar,
//                        int is, int ie, int js, int je, int ks, int ke, int ngh,
//                        const MGCoordinates &coord);
// void MGPeriodicInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
//                        int is, int ie, int js, int je, int ks, int ke, int ngh,
//                        const MGCoordinates &coord);
// void MGPeriodicOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
//                        int is, int ie, int js, int je, int ks, int ke, int ngh,
//                        const MGCoordinates &coord);

// void MGZeroGradientInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
//                            int is, int ie, int js, int je, int ks, int ke, int ngh,
//                            const MGCoordinates &coord);
// void MGZeroGradientOuterX1(AthenaArray<Real> &dst, Real time, int nvar,
//                            int is, int ie, int js, int je, int ks, int ke, int ngh,
//                            const MGCoordinates &coord);
// void MGZeroGradientInnerX2(AthenaArray<Real> &dst, Real time, int nvar,
//                            int is, int ie, int js, int je, int ks, int ke, int ngh,
//                            const MGCoordinates &coord);
// void MGZeroGradientOuterX2(AthenaArray<Real> &dst, Real time, int nvar,
//                            int is, int ie, int js, int je, int ks, int ke, int ngh,
//                            const MGCoordinates &coord);
// void MGZeroGradientInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
//                            int is, int ie, int js, int je, int ks, int ke, int ngh,
//                            const MGCoordinates &coord);
// void MGZeroGradientOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
//                            int is, int ie, int js, int je, int ks, int ke, int ngh,
//                            const MGCoordinates &coord);

// void MGZeroFixedInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh,
//                         const MGCoordinates &coord);
// void MGZeroFixedOuterX1(AthenaArray<Real> &dst, Real time, int nvar,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh,
//                         const MGCoordinates &coord);
// void MGZeroFixedInnerX2(AthenaArray<Real> &dst, Real time, int nvar,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh,
//                         const MGCoordinates &coord);
// void MGZeroFixedOuterX2(AthenaArray<Real> &dst, Real time, int nvar,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh,
//                         const MGCoordinates &coord);
// void MGZeroFixedInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh,
//                         const MGCoordinates &coord);
// void MGZeroFixedOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh,
//                         const MGCoordinates &coord);

#endif // RAD_FLD_NEWTON_RAPHSON_HPP_
