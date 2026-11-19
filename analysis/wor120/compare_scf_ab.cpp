#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct History {
  std::vector<std::string> names;
  std::vector<std::vector<double> > rows;
};

History ReadHistory(const std::string& filename) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  History history;
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, 2, "# ") == 0 && line.find("[1]=") != std::string::npos) {
      std::stringstream fields(line.substr(2));
      std::string field;
      while (fields >> field) {
        const std::size_t equal = field.find('=');
        if (equal != std::string::npos) history.names.push_back(field.substr(equal+1));
      }
    } else if (!line.empty() && line[0] != '#') {
      std::stringstream values(line);
      std::vector<double> row;
      double value = 0.0;
      while (values >> value) row.push_back(value);
      if (!row.empty()) {
        if (!history.rows.empty() && row.front() == history.rows.back().front()) {
          history.rows.back() = row;
        } else {
          history.rows.push_back(row);
        }
      }
    }
  }
  if (history.names.empty() || history.rows.empty()) {
    throw std::runtime_error("history file has no header or data: " + filename);
  }
  for (const std::vector<double>& row : history.rows) {
    if (row.size() != history.names.size()) {
      throw std::runtime_error("history row width does not match header: " + filename);
    }
    for (const double value : row) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("history contains a non-finite value: " + filename);
      }
    }
  }
  return history;
}

std::size_t Column(const History& history, const std::string& name) {
  for (std::size_t i = 0; i < history.names.size(); ++i) {
    if (history.names[i] == name) return i;
  }
  throw std::runtime_error("missing history column: " + name);
}

double Value(const History& history, const std::vector<double>& row,
             const std::string& name) {
  return row[Column(history, name)];
}

struct Diagnostics {
  double time;
  double mass;
  double angular_momentum;
  double mechanical_energy;
  double radial_mach_rms;
  double radial_mach_max;
  double virial_residual;
};

Diagnostics GetDiagnostics(const History& history) {
  const std::vector<double>& row = history.rows.back();
  const double mass = Value(history, row, "mapped_stellar_mass_g");
  const double self_virial = Value(history, row, "stellar_self_gravity_virial_erg");
  const double core_virial = Value(history, row, "stellar_core_gravity_virial_erg");
  const double gravity_virial = self_virial+core_virial;
  Diagnostics result;
  result.time = Value(history, row, "time");
  result.mass = mass;
  result.angular_momentum = Value(history, row, "mapped_Jz_g_cm2_s");
  result.mechanical_energy = Value(history, row, "mapped_E_erg")
      +Value(history, row, "self_potential_energy_erg")
      +Value(history, row, "core_potential_energy_erg");
  result.radial_mach_rms = std::sqrt(std::max(0.0,
      Value(history, row, "stellar_radial_mach2_mass_g")/mass));
  result.radial_mach_max = Value(history, row, "stellar_radial_mach_max");
  result.virial_residual = (2.0*Value(history, row, "stellar_kinetic_energy_erg")
      +Value(history, row, "stellar_pressure_virial_erg")+gravity_virial)
      /std::abs(gravity_virial);
  return result;
}

double SymmetricRelativeDifference(double left, double right) {
  const double scale = std::max(std::max(std::abs(left), std::abs(right)),
                                std::numeric_limits<double>::min());
  return (right-left)/scale;
}

void WriteMetric(std::ostream& output, const char* name, double x2, double wz,
                 bool trailing_comma) {
  output << "  \"" << name << "\": {\"rk2_x2\": " << x2
         << ", \"rk4_4wz\": " << wz
         << ", \"symmetric_relative_difference\": "
         << SymmetricRelativeDifference(x2, wz) << "}"
         << (trailing_comma ? "," : "") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: " << argv[0]
                << " RK2_X2.hst RK4_4WZ.hst OUTPUT.json\n";
      return 2;
    }
    const History x2_history = ReadHistory(argv[1]);
    const History wz_history = ReadHistory(argv[2]);
    const Diagnostics x2 = GetDiagnostics(x2_history);
    const Diagnostics wz = GetDiagnostics(wz_history);
    const double time_scale = std::max(std::max(std::abs(x2.time), std::abs(wz.time)),
                                       1.0);
    if (std::abs(x2.time-wz.time) > 1.0e-12*time_scale) {
      throw std::runtime_error("A/B histories end at different times");
    }
    const std::vector<double>& wz_final = wz_history.rows.back();
    const double available = Value(wz_history, wz_final, "wenoz_available_cells");
    const double sensor = Value(wz_history, wz_final, "wenoz_sensor_fallback_cells");
    const double amr = Value(wz_history, wz_final, "wenoz_amr_fallback_cells");
    const double fallback = Value(wz_history, wz_final, "wenoz_plm_fallback_cells");
    const double fallback_mass = Value(
        wz_history, wz_final, "wenoz_plm_fallback_stellar_mass_g");
    if (available <= 0.0 || fallback < 0.0 || fallback > available) {
      throw std::runtime_error("invalid 4wz fallback cell counts");
    }
    if (sensor < 0.0 || amr < 0.0
        || std::abs(sensor+amr-fallback) > 0.5) {
      throw std::runtime_error("4wz fallback reason counts do not sum to total");
    }

    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("cannot create output JSON");
    output << std::scientific << std::setprecision(12)
           << "{\n"
           << "  \"final_time_code\": " << x2.time << ",\n"
           << "  \"wenoz_available_cells\": " << available << ",\n"
           << "  \"wenoz_sensor_fallback_cells\": " << sensor << ",\n"
           << "  \"wenoz_amr_fallback_cells\": " << amr << ",\n"
           << "  \"wenoz_plm_fallback_cells\": " << fallback << ",\n"
           << "  \"wenoz_sensor_fallback_fraction\": " << sensor/available << ",\n"
           << "  \"wenoz_amr_fallback_fraction\": " << amr/available << ",\n"
           << "  \"wenoz_plm_fallback_fraction\": " << fallback/available << ",\n";
    output << "  \"wenoz_plm_fallback_stellar_mass_fraction\": "
           << fallback_mass/wz.mass << ",\n";
    WriteMetric(output, "mapped_stellar_mass_g", x2.mass, wz.mass, true);
    WriteMetric(output, "mapped_Jz_g_cm2_s", x2.angular_momentum,
                wz.angular_momentum, true);
    WriteMetric(output, "mechanical_energy_erg", x2.mechanical_energy,
                wz.mechanical_energy, true);
    WriteMetric(output, "stellar_radial_mach_rms", x2.radial_mach_rms,
                wz.radial_mach_rms, true);
    WriteMetric(output, "stellar_radial_mach_max", x2.radial_mach_max,
                wz.radial_mach_max, true);
    WriteMetric(output, "virial_relative_residual", x2.virial_residual,
                wz.virial_residual, false);
    output << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "compare_scf_ab: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
