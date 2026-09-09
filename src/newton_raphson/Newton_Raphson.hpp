#ifndef RAD_FLD_NEWTON_RAPHSON_HPP_
#define RAD_FLD_NEWTON_RAPHSON_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson.hpp
//  \brief defines the Newton-Raphson base class

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
#include "../linear_solver/linear_solver.hpp"
#include "../mesh/mesh.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class Mesh;
class MeshBlock;
class ParameterInput;
class Coordinates;
class NewtonRaphsonTaskList;

enum class NRVariable {src, u, coeff};
enum class NRNormType {max, l1, l2};

//! \brief Aggregated normalized residual norms for one Newton state.
struct NewtonResidualNorms {
  Real l2_norm{0.0};
  Real max_norm{0.0};
  Real gas_l2_norm{0.0};
  Real gas_max_norm{0.0};
  Real radiation_l2_norm{0.0};
  Real radiation_max_norm{0.0};
  bool finite{true};
};

//! rief Reason why a Newton solve stopped.
enum class NewtonSolveReason {
  converged,
  fixed_iterations_complete,
  dt_zero_initialization,
  initial_nonfinite,
  final_nonfinite,
  stagnation,
  backtracking_exhausted,
  max_iterations
};

//! rief Result of one nonlinear Newton solve.
struct NewtonSolveResult {
  NewtonSolveReason reason{NewtonSolveReason::initial_nonfinite};
  int iterations{0};
  Real initial_norm{0.0};
  Real initial_max_norm{0.0};
  Real final_norm{0.0};
  Real final_max_norm{0.0};
  Real initial_gas_l2_norm{0.0};
  Real initial_gas_max_norm{0.0};
  Real initial_radiation_l2_norm{0.0};
  Real initial_radiation_max_norm{0.0};
  Real final_gas_l2_norm{0.0};
  Real final_gas_max_norm{0.0};
  Real final_radiation_l2_norm{0.0};
  Real final_radiation_max_norm{0.0};
  bool committed{false};

  bool IsSuccess() const {
    return committed;
  }
};

const char *NewtonSolveReasonName(NewtonSolveReason reason);

// constexpr int minth_ = 8;

// //! \fn inline std::int64_t rotl(std::int64_t i, int s)
// //  \brief left bit rotation function for 64bit integers (unsafe if s > 64)

// inline std::int64_t rotl(std::int64_t i, int s) {
//   return (i << s) | (i >> (64 - s));
// }


// //! \struct LogicalLocationHash
// //  \brief Hash function object for LogicalLocation

// struct LogicalLocationHash {
//  public:
//   std::size_t operator()(const LogicalLocation &l) const {
//     return static_cast<std::size_t>(l.lx1^rotl(l.lx2,21)^rotl(l.lx3,42));
//   }
// };


//! \class NewtonRaphson
//  \brief NewtonRaphson object containing each MeshBlock and/or the root block

class NewtonRaphson {
 public:
  NewtonRaphson(NewtonRaphsonDriver *pmd, MeshBlock *pmb, int nghost,
                int nderivetive, int ndef_coeff);
  virtual ~NewtonRaphson();

  void RetrieveResult(AthenaArray<Real> &dst, int ns, int ngh);
  void RetrieveDefect(AthenaArray<Real> &dst, int ns, int ngh);
  void ZeroClearData();
  void CalculateDefectBlock();
  void CalculateDefectNorms(int n, Real &l2_sum, Real &max_norm) const;
  // Real CalculateTotal(NRVariable type, int n);
//   void SubtractAverage(NRVariable type, int n, Real ave);
  void StoreOldData();
  virtual void AddDifference(AthenaArray<Real> &dst,
                             AthenaArray<Real> &delta,
                             const AthenaArray<Real> &derivetive) = 0;

  // physics-dependent virtual functions
  virtual void LoadVariables() = 0;
  virtual void UpdateHydroVariables() = 0;
  virtual void CalculateCoefficientsOnce(const AthenaArray<Real> &u_pre,
                                         const AthenaArray<Real> &w,
                                         AthenaArray<Real> &def_coeff,
                                         AthenaArray<Real> &derivetive) = 0;
  virtual void CalculateCoefficients(const AthenaArray<Real> &u_rad_old,
                                     const AthenaArray<Real> &u_rad_new,
                                    //  const AthenaArray<Real> &u_gas_old,
                                    //  const AthenaArray<Real> &u_gas_new,
                                     const AthenaArray<Real> &def_coeff,
                                     AthenaArray<Real> &coeff,
                                     AthenaArray<Real> &derivetive,
                                     AthenaArray<Real> &src,
                                     Real dt) = 0;
  virtual void CalculateDefect(AthenaArray<Real> &def,
                               const AthenaArray<Real> &u,
                               const AthenaArray<Real> &u_old,
                               const AthenaArray<Real> &coeff,
                               const AthenaArray<Real> &def_coeff,
                               bool th) = 0;
  virtual void CalculateAdditionalDefectNorms(Real &l2_sum, Real &max_norm,
                                              bool &active, bool &finite) const;
  virtual bool PrimaryDefectIsGas() const { return false; }
  virtual void ApplyPhysicalBoundary() = 0;
  virtual void PrintCellPhysicsDebug(int k, int j, int i) {}
  void CalculateCoefficientsTask(Real dt);
  void ApplyCorrectionTask();
  virtual void StoreIterate();
  virtual void RestoreIterate();
  void StartBoundary();
  void SendBoundary();
  bool ReceiveBoundary();
  void SetBoundary();
  void ProlongateBoundary(Real dt);
  void ClearBoundary();
  friend class NewtonRaphsonTaskList;

  friend class NewtonRaphsonDriver;
  friend class NRBoundaryValues;
  friend class NRFLDDriver;
  friend class linearNRDriver;

  bool output_defect;

  NewtonRaphsonDriver *pmy_driver_;
  MeshBlock *pmy_block_;
  LogicalLocation loc_;
  RegionSize size_;
  int ngh_, nvar_, ncoeff_, nmatrix_;
  
  AthenaArray<Real> u_, u_iter_backup_, def_, src_, uold_, coeff_, matrix_;
  AthenaArray<Real> flux[3];  // face-averaged flux vector
  
  AthenaArray<Real> derivetive_, def_coeff_; // caution! have to be initialized in derived class constructors!!
  
  // storage for SMR/AMR
  AthenaArray<Real> coarse_u_;
  int refinement_idx{-1};
  
  AthenaArray<Real> delta_u_; // for temporary storage of updates
  AthenaArray<Real> coarse_delta_u_;
  AthenaArray<Real> empty_flux[3];
  
  NRBoundaryVariable nrbvar;
  CellCenteredBoundaryVariable delta_bvar;
  BoundaryFlag nr_block_bcs_[6];
  linearMG *plmg_; // to be set in derived class constructors
protected:
  Real rdx_, rdy_, rdz_;
  Real defscale_;


 private:
  int refinement_idx_; // for delta_u_
  TaskStates ts_;
};


//! \class NewtonRaphsonDriver
//  \brief NewtonRaphson driver

class NewtonRaphsonDriver {
 public:
  NewtonRaphsonDriver(Mesh *pm,
                  int invar, int ncoeff, int nmatrix);
  virtual ~NewtonRaphsonDriver();
  void CheckBoundaryFunctions();
  // pure virtual function
  // virtual void Solve(int step, Real dt = 0.0) = 0;

  NewtonSolveResult Solve_general(int step, Real dt = 0.0);
  void HandleSolveResult(const NewtonSolveResult &result) const;

  friend class NewtonRaphson;
  friend class NewtonRaphsonTaskList;
  friend class NRBoundaryValues;
  friend class NRFLD;
  friend class linearMG;

  protected:
  // void CheckBoundaryFunctions();
  // void SetupNewtonRaphson();
  // void SetupCoefficients();
  void SolveOneCycle();
  // void SolveIterative();
  // void SolveIterativeFixedTimes();

  void CalculateDefectNorms(NewtonResidualNorms &norms);
  // void CalculateMatrix();

  // // small functions
  // int GetNumNewtonRaphsons() { return nblist_[Globals::my_rank]; }

  int nranks_, nthreads_, nbtotal_, nvar_, ncoeff_, nmatrix_, mode_, matrixmode_;
  // int *nslist_, *nblist_, *nvlist_, *nvslist_, *nvlisti_, *nvslisti_,
  //                         *nclist_, *ncslist_, *ranklist_;
  int nrbx1_, nrbx2_, nrbx3_;
  BoundaryFlag nr_mesh_bcs_[6];
  NRBoundaryFunc NRBoundaryFunction_[6]; // caution!!
  // NRBoundaryFunc NRCoeffBoundaryFunction_[6];
  Mesh *pmy_mesh_;
  linearMGDriver *plmgd_; // to be set in derived class constructors

  std::vector<NewtonRaphson*> vnr_;
  bool needinit_, fshowdef_, use_mg_smoothing_fallback_;
  Real eps_, dt_;
  Real mg_coarse_retry_factor_, mg_coarse_retry_min_scale_;
  Real step_scale_, backtrack_factor_, min_step_scale_;
  int stage_;
  int niter_, nr_max_iterations_, max_backtrack_, mg_coarse_retry_max_;
  int os_, oe_;
  NewtonRaphsonTaskList *nrtlist_coeff_;
  NewtonRaphsonTaskList *nrtlist_post_;


 private:
  // Real *rootbuf_;
  // int nb_rank_;
#ifdef MPI_PARALLEL
  MPI_Comm MPI_COMM_NEWTON_RAPHSON;
  int nr_phys_id_;
#endif
};

#endif // NEWTON_RAPHSON_NEWTON_RAPHSON_HPP_
