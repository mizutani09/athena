//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_fld_fluxes.cpp
//! \brief Calculate fld fluxes(advection term)
//!        mainly copied from scaler/calculate_scalar_fluxes.cpp

// C headers

// C++ headers
#include <algorithm>   // min,max

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"   // reapply floors to face-centered reconstructed states
#include "../hydro/hydro.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "fld.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void FLD::CalculateFluxes(AthenaArray<Real> &u_rad, const int order)
void FLD::CalculateFluxes(AthenaArray<Real> &u_rad, const int order) {
  // HLLC-FLD writes u_rad_flux from the same Godunov solution as the hydro flux.
  (void)u_rad;
  (void)order;
}
