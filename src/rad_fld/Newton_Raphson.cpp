//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Newton_Raphson.cpp
//! \brief implementation of the functions commonly used in Newton-Raphson

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
#include "linear_multigrid.hpp"

NewtonRaphson::NewtonRaphson(MeshBlock *pmb, ParameterInput *pin)
 : pmy_block(pmb) {
  max_iter_ = pin->GetOrAddInteger("rad_fld", "nr_maxiter", 100);
  plinmg = new linearMG(pmb->pmy_mesh->plinmg, pmb, pin);
}

NewtonRaphson::~NewtonRaphson() {
  delete plinmg;
}

void NewtonRaphson::CalculateCoefficients(AthenaArray<Real> &u, Real dt) {

}

void NewtonRaphson::Solve(int stage, Real dt) {
  // make linear eq. to be solved with linear multigrid
  CalculateCoefficients(u, dt);
  // solve linear eq. with linear multigrid
  plinmg->Solve(stage, dt);
}

