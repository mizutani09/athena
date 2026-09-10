#ifndef FLD_OPACITY_TABLE_HPP_
#define FLD_OPACITY_TABLE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.hpp
//  \brief defines the user opacity table helper for FLD problem generators

#include <atomic>
#include <cstdint>
#include <iosfwd>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../utils/interp_table.hpp"

class ParameterInput;

class UserOpacityTable : public InterpTable2D {
 public:
  enum class X2AxisKind {pressure, density, opal_r};
  enum class DomainPolicy {error, clamp, extrapolate};

  struct Diagnostics {
    std::uint64_t out_of_domain = 0;
    std::uint64_t clamped = 0;
    std::uint64_t negative_results = 0;
    std::uint64_t zero_results = 0;
    std::uint64_t nonfinite_results = 0;
  };

  explicit UserOpacityTable(ParameterInput *pin);
  ~UserOpacityTable();

  Real GetOpacity(int var_index, Real density, Real temperature);
  Real GetOpacityFromRhoT(int var_index, Real density, Real temperature);
  Real GetOpacityFromPT(int var_index, Real pressure, Real temperature);
  // Counters are per opacity-field lookup: evaluating both Planck and Rosseland
  // at one cell records two events when both lookups are outside the domain.
  Diagnostics GetLocalDiagnostics() const;
  // Collective when MPI is enabled. Call only from a point reached by every rank.
  void ReportDiagnostics(std::ostream &stream) const;

  bool use_tables;
  Real tempMin, tempMax;
  Real pressureMin, pressureMax;
  int nTemp, nPressure, nVar;
  AthenaArray<Real> OpacityTables;
  X2AxisKind x2_axis_kind = X2AxisKind::pressure;
  DomainPolicy domain_policy = DomainPolicy::extrapolate;
  Real mean_molecular_weight = 1.0;
  // One entry per RadFLD::OpacityIndex. Keeping the value representation per
  // field allows Planck and Rosseland tables to use different encodings.
  bool values_are_log10[2] = {false, false};

 private:
  std::atomic<std::uint64_t> out_of_domain_count_{0};
  std::atomic<std::uint64_t> clamped_count_{0};
  std::atomic<std::uint64_t> negative_result_count_{0};
  std::atomic<std::uint64_t> zero_result_count_{0};
  std::atomic<std::uint64_t> nonfinite_result_count_{0};
};

#endif // FLD_OPACITY_TABLE_HPP_
