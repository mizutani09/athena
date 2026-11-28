#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> Split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

struct Row {
  double radius;
  double mass;
  double density;
  double pressure;
  double binding;
  double radiation_fraction;
  double pressure_scale_height;
  double dynamical_time;
};

std::size_t Column(const std::map<std::string, std::size_t>& columns,
                   const std::string& name) {
  const auto found = columns.find(name);
  if (found == columns.end()) throw std::runtime_error("missing column " + name);
  return found->second;
}

double Value(const std::vector<std::string>& fields, std::size_t column) {
  if (column >= fields.size()) throw std::runtime_error("short CSV row");
  std::size_t consumed = 0;
  const double value = std::stod(fields[column], &consumed);
  if (consumed != fields[column].size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid numeric field " + fields[column]);
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: " << argv[0]
                << " MESA_LAYERS.csv OUTPUT.csv RADIUS_RSUN [...]\n";
      return 2;
    }
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("cannot open input CSV");
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty input CSV");
    const std::vector<std::string> names = Split(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < names.size(); ++i) columns[names[i]] = i;
    const std::size_t radius = Column(columns, "radius_Rsun");
    const std::size_t mass = Column(columns, "mass_Msun");
    const std::size_t density = Column(columns, "rho_g_cm3");
    const std::size_t pressure = Column(columns, "pressure_dyn_cm2");
    const std::size_t binding = Column(columns, "binding_energy_exterior_erg");
    const std::size_t radiation_fraction = Column(columns, "prad_over_ptotal");
    const std::size_t pressure_scale_height =
        Column(columns, "pressure_scale_height_cm");
    const std::size_t dynamical_time = Column(columns, "t_dyn_local_s");
    std::vector<Row> rows;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      const std::vector<std::string> fields = Split(line);
      rows.push_back({Value(fields, radius), Value(fields, mass),
                      Value(fields, density), Value(fields, pressure),
                      Value(fields, binding), Value(fields, radiation_fraction),
                      Value(fields, pressure_scale_height),
                      Value(fields, dynamical_time)});
    }
    if (rows.empty()) throw std::runtime_error("input CSV has no rows");
    std::vector<Row> selected;
    std::vector<double> targets;
    for (int argument = 3; argument < argc; ++argument) {
      const double target = std::stod(argv[argument]);
      if (!(target > 0.0)) throw std::runtime_error("non-positive radius");
      targets.push_back(target);
      selected.push_back(*std::min_element(
          rows.begin(), rows.end(), [=](const Row& left, const Row& right) {
            return std::abs(left.radius-target) < std::abs(right.radius-target);
          }));
    }
    const Row& reference = selected.front();
    constexpr double kL3SpacingRsun = 4.6875;
    constexpr double kL4SpacingRsun = 2.34375;
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot create output CSV");
    output << "target_radius_rsun,mesa_radius_rsun,mass_msun,"
              "additional_cut_mass_msun,density_g_cm3,pressure_dyn_cm2,"
              "radiation_pressure_fraction,pressure_scale_height_cm,"
              "exterior_binding_energy_erg,binding_relative_to_reference,"
              "local_dynamical_time_s,cells_per_support_l3,"
              "cells_per_support_l4,fixed_dx_core_cfl_gain_heuristic\n";
    output << std::scientific << std::setprecision(12);
    for (std::size_t i = 0; i < selected.size(); ++i) {
      const Row& row = selected[i];
      output << targets[i] << ',' << row.radius << ',' << row.mass << ','
             << row.mass-reference.mass << ',' << row.density << ','
             << row.pressure << ',' << row.radiation_fraction << ','
             << row.pressure_scale_height << ',' << row.binding << ','
             << row.binding/reference.binding << ',' << row.dynamical_time << ','
             << row.radius/kL3SpacingRsun << ','
             << row.radius/kL4SpacingRsun << ','
             << std::sqrt(row.radius/reference.radius) << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "audit_core_cut_candidates: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
