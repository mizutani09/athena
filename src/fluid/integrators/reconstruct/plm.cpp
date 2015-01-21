//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================

// Primary header
#include "../fluid_integrator.hpp"

// Athena headers
#include "../../../athena.hpp"         // macros, Real
#include "../../../athena_arrays.hpp"  // AthenaArray
#include "../../fluid.hpp"          // Fluid
#include "../../../mesh.hpp"           // Block

//======================================================================================
//! \file plm.cpp
//  \brief  piecewise linear reconstruction
//======================================================================================

//--------------------------------------------------------------------------------------
//! \fn FluidIntegrator::ReconstructionFuncX1()
//  \brief 

void FluidIntegrator::ReconstructionFuncX1(const int n, const int m, const int k,
  const int j, const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  int is = pmy_fluid->pmy_block->is, ie = pmy_fluid->pmy_block->ie;

#pragma simd
  for (int i=is; i<=(ie+1); ++i){
    Real dql = (q(n,k,j,i-1) - q(n,k,j,i-2))/pmy_fluid->pmy_block->dx1v(i-2);
    Real dqc = (q(n,k,j,i  ) - q(n,k,j,i-1))/pmy_fluid->pmy_block->dx1v(i-1);
    Real dqr = (q(n,k,j,i+1) - q(n,k,j,i  ))/pmy_fluid->pmy_block->dx1v(i  );

    // Apply monotonicity constraints, compute ql_(i-1/2)
    Real dq2 = dql*dqc;
    Real dqm = (dq2 > 0.0) ? dqm = dq2/(dql + dqc) : 0.0;

    ql(m,i) = q(n,k,j,i-1) + (pmy_fluid->pmy_block->dx1f(i-1))*dqm;
    
    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    dqm = (dq2 > 0.0) ? dqm = dq2/(dqc + dqr) : 0.0;

    qr(m,i) = q(n,k,j,i) - (pmy_fluid->pmy_block->dx1f(i))*dqm;
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn FluidIntegrator::ReconstructionFuncX2()
//  \brief 

void FluidIntegrator::ReconstructionFuncX2(const int n, const int m, const int k,
  const int j, const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  int is = pmy_fluid->pmy_block->is, ie = pmy_fluid->pmy_block->ie;
  Real dx2jm2i = 1.0/pmy_fluid->pmy_block->dx2v(j-2);
  Real dx2jm1i = 1.0/pmy_fluid->pmy_block->dx2v(j-1);
  Real dx2ji   = 1.0/pmy_fluid->pmy_block->dx2v(j);

#pragma simd
  for (int i=is; i<=ie; ++i){
    Real dql = (q(n,k,j-1,i) - q(n,k,j-2,i))*dx2jm2i;
    Real dqc = (q(n,k,j  ,i) - q(n,k,j-1,i))*dx2jm1i;
    Real dqr = (q(n,k,j+1,i) - q(n,k,j  ,i))*dx2ji;

    // Apply monotonicity constraints, compute ql_(i-1/2)
    Real dq2 = dql*dqc;
    Real dqm = (dq2 > 0.0) ? dqm = dq2/(dql + dqc) : 0.0;

    ql(m,i) = q(n,k,j-1,i) + (pmy_fluid->pmy_block->dx2f(j-1))*dqm;
    
    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    dqm = (dq2 > 0.0) ? dqm = dq2/(dqc + dqr) : 0.0;

    qr(m,i) = q(n,k,j,i) - (pmy_fluid->pmy_block->dx2f(j))*dqm;
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn FluidIntegrator::ReconstructionFuncX3()
//  \brief 

void FluidIntegrator::ReconstructionFuncX3(const int n, const int m, const int k,
  const int j, const AthenaArray<Real> &q, AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  int is = pmy_fluid->pmy_block->is, ie = pmy_fluid->pmy_block->ie;
  Real dx3km2i = 1.0/pmy_fluid->pmy_block->dx3v(k-2);
  Real dx3km1i = 1.0/pmy_fluid->pmy_block->dx3v(k-1);
  Real dx3ki   = 1.0/pmy_fluid->pmy_block->dx3v(k);

#pragma simd
  for (int i=is; i<=ie; ++i){
    Real dql = (q(n,k-1,j,i) - q(n,k-2,j,i))*dx3km2i;
    Real dqc = (q(n,k  ,j,i) - q(n,k-1,j,i))*dx3km1i;
    Real dqr = (q(n,k+1,j,i) - q(n,k  ,j,i))*dx3ki;

    // Apply monotonicity constraints, compute ql_(i-1/2)
    Real dq2 = dql*dqc;
    Real dqm = (dq2 > 0.0) ? dqm = dq2/(dql + dqc) : 0.0;

    ql(m,i) = q(n,k-1,j,i) + (pmy_fluid->pmy_block->dx3f(k-1))*dqm;
    
    // Apply monotonicity constraints, compute qr_(i-1/2)
    dq2 = dqc*dqr;
    dqm = (dq2 > 0.0) ? dqm = dq2/(dqc + dqr) : 0.0;

    qr(m,i) = q(n,k,j,i) - (pmy_fluid->pmy_block->dx3f(k))*dqm;
  }

  return;
}
