//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_solver.cpp
//! \brief create linear solver object

// C headers

// C++ headers
#include <algorithm>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "linear_solver.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class MeshBlock;

namespace {
  AthenaArray<Real> *temp; // temporary data for the Jacobi iteration
}

//----------------------------------------------------------------------------------------
//! \fn linearMGDriver::linearMGFLDDriver(Mesh *pm, ParameterInput *pin)
//! \brief linearMGDriver constructor

linearMGDriver::linearMGDriver(Mesh *pm, ParameterInput *pin)
    : MultigridDriver(pm, pm->LinearMGBoundaryFunction_,
                          pm->LinearMGCoeffBoundaryFunction_,
                          nullptr,
                          nullptr,
                          1, linearSolver::NCOEFF, linearSolver::NMATRIX) {
  eps_ = pin->GetOrAddReal("mgfld", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("mgfld", "niteration", -1);
  ffas_ = pin->GetOrAddBoolean("mgfld", "fas", ffas_);
  omega_ = pin->GetOrAddReal("mgfld", "omega", 1.0);
  fsteady_ = pin->GetOrAddBoolean("mgfld", "steady", false);
  npresmooth_ = pin->GetOrAddReal("mgfld", "npresmooth", 2);
  npostsmooth_ = pin->GetOrAddReal("mgfld", "npostsmooth", 2);
  fshowdef_ = pin->GetOrAddBoolean("mgfld", "show_defect", fshowdef_);
  std::string smoother = pin->GetOrAddString("mgfld", "smoother", "jacobi-rb");
//   matrixmode_ = 1;
  matrixmode_ = 0; // caution!
  if (smoother == "jacobi-rb") {
    fsmoother_ = 1;
    redblack_ = true;
  } else if (smoother == "jacobi-double") {
    fsmoother_ = 0;
    redblack_ = true;
  } else { // jacobi
    fsmoother_ = 0;
    redblack_ = false;
  }
  std::string prol = pin->GetOrAddString("mgfld", "prolongation", "trilinear");
  if (prol == "tricubic")
    fprolongation_ = 1;

  std::string m = pin->GetOrAddString("mgfld", "mgmode", "none");
  std::transform(m.begin(), m.end(), m.begin(), ::tolower);
  if (m == "fmg") {
    mode_ = 0;
  } else if (m == "mgi") {
    mode_ = 1; // Iterative
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGFLDDriver::linearMGFLDDriver" << std::endl
        << "The \"mgmode\" parameter in the <mgfld> block is invalid." << std::endl
        << "FMG: Full Multigrid + Multigrid iteration (default)" << std::endl
        << "MGI: Multigrid Iteration" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (eps_ < 0.0 && niter_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in linearMGFLDDriver::linearMGFLDDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <mgfld> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
    ATHENA_ERROR(msg);
  }
  mg_mesh_bcs_[inner_x1] = GetMGBoundaryFlag("zero-fixed");
  mg_mesh_bcs_[outer_x1] = GetMGBoundaryFlag("zero-fixed");
  mg_mesh_bcs_[inner_x2] = GetMGBoundaryFlag("zero-fixed");
  mg_mesh_bcs_[outer_x2] = GetMGBoundaryFlag("zero-fixed");
  mg_mesh_bcs_[inner_x3] = GetMGBoundaryFlag("zero-fixed");
  mg_mesh_bcs_[outer_x3] = GetMGBoundaryFlag("zero-fixed");
  CheckBoundaryFunctions();
  fsubtract_average_ = false; // override the subtract average flag

  mgtlist_ = new MultigridTaskList(this);

  // Allocate the root multigrid
  mgroot_ = new linearMG(this, nullptr, pin);

  linmgtlist_ = new LinearMGBoundaryTaskList(pin, pm);

  int nth = 1;
#ifdef OPENMP_PARALLEL
  nth = omp_get_max_threads();
#endif
  temp = new AthenaArray<Real>[nth];
  int nx = std::max(pmy_mesh_->block_size.nx1, pmy_mesh_->nrbx1) + 2*mgroot_->ngh_;
  int ny = std::max(pmy_mesh_->block_size.nx2, pmy_mesh_->nrbx2) + 2*mgroot_->ngh_;
  int nz = std::max(pmy_mesh_->block_size.nx3, pmy_mesh_->nrbx3) + 2*mgroot_->ngh_;
  for (int n = 0; n < nth; ++n)
    temp[n].NewAthenaArray(nz, ny, nx);
}


//----------------------------------------------------------------------------------------
//! \fn linearMGDriver::~linearMGDriver()
//! \brief linearMGDriver destructor

linearMGDriver::~linearMGDriver() {
  delete linmgtlist_;
  delete mgroot_;
  delete mgtlist_;
  delete [] temp;
}
