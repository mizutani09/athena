#ifndef FLD_OPACITY_TABLE_HPP_
#define FLD_OPACITY_TABLE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.hpp
//  \brief defines the user opacity table helper for FLD problem generators

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../utils/interp_table.hpp"

class ParameterInput;

class UserOpacityTable : public InterpTable2D {
 public:
  enum class X2AxisKind {pressure, density, opal_r};

  explicit UserOpacityTable(ParameterInput *pin);
  ~UserOpacityTable();

  Real GetOpacity(int var_index, Real density, Real temperature);
  Real GetOpacityFromRhoT(int var_index, Real density, Real temperature);
  Real GetOpacityFromPT(int var_index, Real pressure, Real temperature);

  bool use_tables;
  Real tempMin, tempMax;
  Real pressureMin, pressureMax;
  int nTemp, nPressure, nVar;
  AthenaArray<Real> OpacityTables;
  X2AxisKind x2_axis_kind = X2AxisKind::pressure;
  Real mean_molecular_weight = 1.0;
  bool values_are_log10 = false;
};

#endif // FLD_OPACITY_TABLE_HPP_
