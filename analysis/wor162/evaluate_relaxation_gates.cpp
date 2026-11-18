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

struct Table {
  std::vector<std::string> names;
  std::vector<std::vector<double> > rows;
};

std::vector<std::string> Split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

Table ReadCsv(const std::string& path) {
  std::ifstream input(path.c_str());
  if (!input) throw std::runtime_error("cannot open " + path);
  Table table;
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty CSV");
  table.names = Split(line);
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = Split(line);
    if (fields.size() != table.names.size()) {
      throw std::runtime_error("CSV row width does not match header");
    }
    std::vector<double> row;
    for (const std::string& field : fields) {
      std::size_t consumed = 0;
      const double value = std::stod(field, &consumed);
      if (consumed != field.size() || !std::isfinite(value)) {
        throw std::runtime_error("invalid numeric CSV field: " + field);
      }
      row.push_back(value);
    }
    table.rows.push_back(row);
  }
  if (table.rows.empty()) throw std::runtime_error("CSV has no data rows");
  return table;
}

std::size_t Column(const Table& table, const std::string& name) {
  for (std::size_t i = 0; i < table.names.size(); ++i) {
    if (table.names[i] == name) return i;
  }
  throw std::runtime_error("missing CSV column: " + name);
}

double Relative(double value, double reference) {
  if (reference == 0.0) throw std::runtime_error("zero radius reference");
  return (value-reference)/reference;
}

struct Gate {
  std::string name;
  double measured;
  double limit;
};

bool Pass(const Gate& gate) { return gate.measured < gate.limit; }

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr << "usage: " << argv[0]
                << " RELAXATION.csv OUTPUT.json [START_TIME_OVER_TDYN]\n";
      return 2;
    }
    const double start_time = argc == 4 ? std::stod(argv[3]) : 0.0;
    if (start_time < 0.0) throw std::runtime_error("negative start time");

    const Table table = ReadCsv(argv[1]);
    const std::size_t time = Column(table, "time_over_tdyn");
    const std::size_t mass = Column(table, "mass_relative_drift");
    const std::size_t angular_momentum = Column(table, "J_relative_drift");
    const std::size_t energy = Column(table, "mechanical_energy_relative_drift");
    const std::size_t mach = Column(table, "radial_mach_rms");
    const std::size_t radial_ke = Column(table, "radial_ke_over_abs_W");
    const std::size_t nonradial_ke =
        Column(table, "nonradial_perturbation_ke_over_abs_W");
    const std::size_t mean_radius = Column(table, "mean_radius_cm");
    const std::size_t rms_radius = Column(table, "rms_radius_cm");
    const std::size_t virial = Column(table, "virial_relative_residual");

    const double initial_mean_radius = table.rows.front()[mean_radius];
    const double initial_rms_radius = table.rows.front()[rms_radius];
    double maximum_mass = 0.0;
    double maximum_angular_momentum = 0.0;
    double maximum_energy = 0.0;
    double maximum_mach = 0.0;
    double maximum_radial_ke = 0.0;
    double maximum_nonradial_ke = 0.0;
    double maximum_mean_radius = 0.0;
    double maximum_rms_radius = 0.0;
    double maximum_virial = 0.0;
    std::size_t samples = 0;
    for (const std::vector<double>& row : table.rows) {
      if (row[time]+1.0e-15 < start_time) continue;
      maximum_mass = std::max(maximum_mass, std::abs(row[mass]));
      maximum_angular_momentum =
          std::max(maximum_angular_momentum, std::abs(row[angular_momentum]));
      maximum_energy = std::max(maximum_energy, std::abs(row[energy]));
      maximum_mach = std::max(maximum_mach, std::abs(row[mach]));
      maximum_radial_ke = std::max(maximum_radial_ke, std::abs(row[radial_ke]));
      maximum_nonradial_ke =
          std::max(maximum_nonradial_ke, std::abs(row[nonradial_ke]));
      maximum_mean_radius = std::max(maximum_mean_radius,
          std::abs(Relative(row[mean_radius], initial_mean_radius)));
      maximum_rms_radius = std::max(maximum_rms_radius,
          std::abs(Relative(row[rms_radius], initial_rms_radius)));
      maximum_virial = std::max(maximum_virial, std::abs(row[virial]));
      ++samples;
    }
    if (samples == 0) throw std::runtime_error("no samples at or after start time");

    const std::vector<Gate> gates = {
      {"stellar_mass_relative_drift", maximum_mass, 1.0e-4},
      {"angular_momentum_relative_drift", maximum_angular_momentum, 1.0e-3},
      {"mechanical_energy_relative_drift", maximum_energy, 1.0e-2},
      {"radial_mach_rms", maximum_mach, 3.0e-2},
      {"radial_ke_over_abs_W", maximum_radial_ke, 5.0e-4},
      {"nonradial_ke_over_abs_W", maximum_nonradial_ke, 5.0e-4},
      {"mean_radius_relative_drift", maximum_mean_radius, 1.0e-2},
      {"rms_radius_relative_drift", maximum_rms_radius, 1.0e-2},
      {"absolute_virial_relative_residual", maximum_virial, 2.0e-2}
    };
    bool all_pass = true;
    for (const Gate& gate : gates) all_pass = all_pass && Pass(gate);

    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot create output JSON");
    output << std::scientific << std::setprecision(12)
           << "{\n"
           << "  \"start_time_over_tdyn\": " << start_time << ",\n"
           << "  \"final_time_over_tdyn\": " << table.rows.back()[time] << ",\n"
           << "  \"evaluated_samples\": " << samples << ",\n"
           << "  \"all_numeric_gates_pass\": "
           << (all_pass ? "true" : "false") << ",\n"
           << "  \"virial_non_growing_requires_time_series_review\": true,\n"
           << "  \"gates\": {\n";
    for (std::size_t i = 0; i < gates.size(); ++i) {
      const Gate& gate = gates[i];
      output << "    \"" << gate.name << "\": {\"maximum\": "
             << gate.measured << ", \"limit\": " << gate.limit
             << ", \"pass\": " << (Pass(gate) ? "true" : "false") << "}"
             << (i+1 == gates.size() ? "\n" : ",\n");
    }
    output << "  }\n}\n";
    return all_pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "evaluate_relaxation_gates: " << error.what() << '\n';
    return 2;
  }
}
