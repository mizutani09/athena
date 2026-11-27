#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Table {
  std::vector<std::string> names;
  std::vector<std::vector<double> > rows;
};

struct Gate {
  std::string name;
  double maximum;
  double limit;
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
    row.reserve(fields.size());
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
  if (reference == 0.0) throw std::runtime_error("zero reference value");
  return (value-reference)/reference;
}

bool Pass(const Gate& gate) { return gate.maximum <= gate.limit; }

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr << "usage: " << argv[0]
                << " RELAXATION.csv OUTPUT.json [START_TIME_OVER_TDYN]\n";
      return 2;
    }
    const double start_time = argc == 4 ? std::stod(argv[3]) : 0.0;
    if (!(start_time >= 0.0)) throw std::runtime_error("negative start time");

    const Table table = ReadCsv(argv[1]);
    const std::size_t time = Column(table, "time_over_tdyn");
    const std::size_t mass = Column(table, "mass_relative_drift");
    const std::size_t angular_momentum = Column(table, "J_relative_drift");
    const std::size_t total_energy =
        Column(table, "total_energy_with_radiation_relative_drift");
    const std::size_t radial_mach_rms = Column(table, "radial_mach_rms");
    const std::size_t interior_mach =
        Column(table, "radial_mach_max_interior");
    const std::size_t dense_mach =
        Column(table, "radial_mach_max_rho_ge_surface");
    const std::size_t mach_gt_0p1_mass =
        Column(table, "radial_mach_gt_0p1_mass_fraction");
    const std::size_t mach_gt_0p3_mass =
        Column(table, "radial_mach_gt_0p3_mass_fraction");
    const std::size_t radial_ke =
        Column(table, "radial_ke_over_abs_W");
    const std::size_t nonradial_ke =
        Column(table, "nonradial_perturbation_ke_over_abs_W");
    const std::size_t mean_radius = Column(table, "mean_radius_cm");
    const std::size_t rms_radius = Column(table, "rms_radius_cm");
    const std::size_t virial = Column(table, "virial_relative_residual");
    const std::size_t surface_mach =
        Column(table, "radial_mach_max_surface_taper");

    std::size_t reference_row = table.rows.size();
    for (std::size_t i = 0; i < table.rows.size(); ++i) {
      if (table.rows[i][time]+1.0e-15 >= start_time) {
        reference_row = i;
        break;
      }
    }
    if (reference_row == table.rows.size()) {
      throw std::runtime_error("no samples at or after start time");
    }
    const double reference_mean_radius = table.rows[reference_row][mean_radius];
    const double reference_rms_radius = table.rows[reference_row][rms_radius];
    const double reference_virial = table.rows[reference_row][virial];

    std::vector<Gate> gates = {
      {"stellar_mass_relative_drift", 0.0, 1.0e-5},
      {"angular_momentum_relative_drift", 0.0, 1.0e-3},
      {"total_energy_with_radiation_relative_drift", 0.0, 1.0e-2},
      {"radial_mach_rms", 0.0, 5.0e-2},
      {"radial_mach_max_interior", 0.0, 1.0e-1},
      {"radial_mach_max_rho_ge_surface", 0.0, 1.0e-1},
      {"radial_mach_gt_0p1_mass_fraction", 0.0, 1.0e-2},
      {"radial_mach_gt_0p3_mass_fraction", 0.0, 2.0e-3},
      {"radial_ke_over_abs_W", 0.0, 1.0e-4},
      {"nonradial_ke_over_abs_W", 0.0, 1.0e-4},
      {"mean_radius_relative_drift", 0.0, 5.0e-3},
      {"rms_radius_relative_drift", 0.0, 5.0e-3},
      {"virial_residual_change", 0.0, 2.0e-2}
    };
    double maximum_surface_mach = 0.0;
    std::size_t samples = 0;
    for (std::size_t i = reference_row; i < table.rows.size(); ++i) {
      const std::vector<double>& row = table.rows[i];
      gates[0].maximum = std::max(gates[0].maximum, std::abs(row[mass]));
      gates[1].maximum = std::max(
          gates[1].maximum, std::abs(row[angular_momentum]));
      gates[2].maximum = std::max(
          gates[2].maximum, std::abs(row[total_energy]));
      gates[3].maximum = std::max(gates[3].maximum, row[radial_mach_rms]);
      gates[4].maximum = std::max(gates[4].maximum, row[interior_mach]);
      gates[5].maximum = std::max(gates[5].maximum, row[dense_mach]);
      gates[6].maximum = std::max(gates[6].maximum, row[mach_gt_0p1_mass]);
      gates[7].maximum = std::max(gates[7].maximum, row[mach_gt_0p3_mass]);
      gates[8].maximum = std::max(gates[8].maximum, std::abs(row[radial_ke]));
      gates[9].maximum = std::max(
          gates[9].maximum, std::abs(row[nonradial_ke]));
      gates[10].maximum = std::max(gates[10].maximum,
          std::abs(Relative(row[mean_radius], reference_mean_radius)));
      gates[11].maximum = std::max(gates[11].maximum,
          std::abs(Relative(row[rms_radius], reference_rms_radius)));
      gates[12].maximum = std::max(
          gates[12].maximum, std::abs(row[virial]-reference_virial));
      maximum_surface_mach = std::max(maximum_surface_mach, row[surface_mach]);
      ++samples;
    }

    bool all_pass = true;
    for (const Gate& gate : gates) all_pass = all_pass && Pass(gate);
    const std::vector<double>& final = table.rows.back();
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot create output JSON");
    output << std::scientific << std::setprecision(12)
           << "{\n"
           << "  \"start_time_over_tdyn\": "
           << table.rows[reference_row][time] << ",\n"
           << "  \"final_time_over_tdyn\": " << final[time] << ",\n"
           << "  \"evaluated_samples\": " << samples << ",\n"
           << "  \"all_numeric_gates_pass\": "
           << (all_pass ? "true" : "false") << ",\n"
           << "  \"surface_mach_is_diagnostic_only\": true,\n"
           << "  \"maximum_surface_mach\": " << maximum_surface_mach << ",\n"
           << "  \"final_high_mach_mean_radius_cm\": "
           << final[Column(table, "radial_mach_gt_0p1_mean_radius_cm")] << ",\n"
           << "  \"final_high_mach_geometric_mean_density_g_cm3\": "
           << final[Column(table,
                "radial_mach_gt_0p1_geometric_mean_density_g_cm3")] << ",\n"
           << "  \"gates\": {\n";
    for (std::size_t i = 0; i < gates.size(); ++i) {
      const Gate& gate = gates[i];
      output << "    \"" << gate.name << "\": {\"maximum\": "
             << gate.maximum << ", \"limit\": " << gate.limit
             << ", \"pass\": " << (Pass(gate) ? "true" : "false") << "}"
             << (i+1 == gates.size() ? "\n" : ",\n");
    }
    output << "  }\n}\n";
    return all_pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "evaluate_fld_relaxation_gates: " << error.what() << '\n';
    return 2;
  }
}
