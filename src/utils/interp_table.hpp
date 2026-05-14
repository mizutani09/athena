#ifndef UTILS_INTERP_TABLE_HPP_
#define UTILS_INTERP_TABLE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file interp_table.hpp
//! \brief defines class InterpTable2D
//!   Contains functions that implement an intpolated lookup table

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"         // Real
#include "../athena_arrays.hpp"  // AthenaArray

class InterpTable2D {
 public:
  InterpTable2D() = default;
  InterpTable2D(const int nvar, const int nx2, const int nx1);

  void SetSize(const int nvar, const int nx2, const int nx1);
  Real interpolate(int nvar, Real x2, Real x1);
  int nvar();
  AthenaArray<Real> data;
  void SetX1lim(Real x1min, Real x1max);
  void SetX2lim(Real x2min, Real x2max);
  void GetX1lim(Real &x1min, Real &x1max);
  void GetX2lim(Real &x2min, Real &x2max);
  void GetSize(int &nvar, int &nx2, int &nx1);

 private:
  int nvar_;
  int nx1_;
  int nx2_;
  Real x1min_;
  Real x1max_;
  Real x1norm_;
  Real x2min_;
  Real x2max_;
  Real x2norm_;
};

class EosTable {
 public:
  explicit EosTable(ParameterInput *pin);

  InterpTable2D table;
  Real logRhoMin, logRhoMax;
  Real logEgasMin, logEgasMax;
  Real rhoUnit, eUnit, hUnit, tUnit;
  int nRho, nEgas, nVar;
  AthenaArray<Real> EosRatios;
};

class UserTable {
 public:
  explicit UserTable(ParameterInput *pin, std::string tag);
  ~UserTable();
  InterpTable2D table;
  //Real GetRawUserTableData(int kOut, Real var2, Real var1, Real var2Unit, Real var1Unit);
  Real GetRawUserTableData(int kOut, Real x2, Real x1);
  Real X1Min, X1Max;
  Real X2Min, X2Max;
  //Real X1Unit, X2Unit, X3Unit;
  int nX1, nX2, nVar;
  AthenaArray<Real> UserRatios;
};

#endif //UTILS_INTERP_TABLE_HPP_
