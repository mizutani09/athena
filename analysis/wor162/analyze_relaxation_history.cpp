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

constexpr double kDay = 86400.0;
constexpr double kDynamicalTime = 1.03413107e7;

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
    throw std::runtime_error("history file has no header or data");
  }
  for (const std::vector<double>& row : history.rows) {
    if (row.size() != history.names.size()) {
      throw std::runtime_error("history row width does not match header");
    }
    for (const double value : row) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("history contains a non-finite value");
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

std::size_t OptionalColumn(const History& history, const std::string& name) {
  for (std::size_t i = 0; i < history.names.size(); ++i) {
    if (history.names[i] == name) return i;
  }
  return history.names.size();
}

double Relative(double value, double reference) {
  return (value-reference)/reference;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: " << argv[0]
                << " INPUT.hst OUTPUT.csv SUMMARY.json\n";
      return 2;
    }
    const History h = ReadHistory(argv[1]);
    const std::size_t time = Column(h, "time");
    const std::size_t mass = Column(h, "mapped_stellar_mass_g");
    const std::size_t angular_momentum = Column(h, "mapped_Jz_g_cm2_s");
    const std::size_t hydro_energy = Column(h, "mapped_E_erg");
    const std::size_t self_energy = Column(h, "self_potential_energy_erg");
    const std::size_t core_energy = Column(h, "core_potential_energy_erg");
    const std::size_t radial_ke = Column(h, "stellar_radial_ke_erg");
    const std::size_t nonradial_ke = Column(h, "stellar_nonradial_perturbation_ke_erg");
    const std::size_t mach2_mass = Column(h, "stellar_radial_mach2_mass_g");
    const std::size_t mach_max = Column(h, "stellar_radial_mach_max");
    const std::size_t r_mass = Column(h, "stellar_r_mass_g_cm");
    const std::size_t r2_mass = Column(h, "stellar_r2_mass_g_cm2");
    const std::size_t pressure_virial = Column(h, "stellar_pressure_virial_erg");
    const std::size_t self_virial = Column(h, "stellar_self_gravity_virial_erg");
    const std::size_t core_virial = Column(h, "stellar_core_gravity_virial_erg");
    const std::size_t kinetic = Column(h, "stellar_kinetic_energy_erg");
    const std::size_t outside_mass = Column(h, "stellar_mass_outside_initial_surface_g");
    const std::size_t q_mass = Column(h, "stellar_ellipsoidal_radius_mass_g");
    const std::size_t q_max = Column(h, "stellar_ellipsoidal_radius_max");
    const std::size_t wenoz_cells = OptionalColumn(h, "wenoz_available_cells");
    const std::size_t wenoz_sensor = OptionalColumn(h, "wenoz_sensor_fallback_cells");
    const std::size_t wenoz_amr = OptionalColumn(h, "wenoz_amr_fallback_cells");
    const std::size_t wenoz_fallback = OptionalColumn(h, "wenoz_plm_fallback_cells");
    const std::size_t wenoz_fallback_mass = OptionalColumn(
        h, "wenoz_plm_fallback_stellar_mass_g");
    const bool has_wenoz = wenoz_cells < h.names.size()
        && wenoz_fallback < h.names.size();

    const std::vector<double>& initial = h.rows.front();
    const double initial_mechanical = initial[hydro_energy]+initial[self_energy]
        +initial[core_energy];
    const double initial_mean_radius = initial[r_mass]/initial[mass];
    const double initial_rms_radius = std::sqrt(initial[r2_mass]/initial[mass]);
    std::ofstream csv(argv[2]);
    if (!csv) throw std::runtime_error("cannot create CSV output");
    csv << "time_code,time_days,time_over_tdyn,mass_relative_drift,J_relative_drift,"
           "mechanical_energy_relative_drift,radial_mach_rms,radial_mach_max,"
           "radial_ke_over_abs_W,nonradial_perturbation_ke_over_abs_W,"
           "mean_radius_cm,rms_radius_cm,virial_relative_residual,"
           "outside_initial_surface_mass_fraction,mean_ellipsoidal_radius,"
           "max_ellipsoidal_radius,wenoz_available_cells,"
           "wenoz_sensor_fallback_cells,wenoz_amr_fallback_cells,"
           "wenoz_plm_fallback_cells,wenoz_sensor_fallback_fraction,"
           "wenoz_amr_fallback_fraction,wenoz_plm_fallback_fraction,"
           "wenoz_plm_fallback_stellar_mass_fraction\n";
    csv << std::scientific << std::setprecision(12);

    double maximum_mass_drift = 0.0;
    double maximum_j_drift = 0.0;
    double maximum_energy_drift = 0.0;
    double maximum_mach_rms = 0.0;
    double maximum_mach = 0.0;
    double maximum_virial = 0.0;
    double maximum_radial_ke_fraction = 0.0;
    double maximum_nonradial_ke_fraction = 0.0;
    double maximum_mean_radius_drift = 0.0;
    double maximum_rms_radius_drift = 0.0;
    double maximum_wenoz_sensor_fraction = 0.0;
    double maximum_wenoz_amr_fraction = 0.0;
    double maximum_wenoz_fallback_fraction = 0.0;
    double maximum_wenoz_fallback_mass_fraction = 0.0;
    for (const std::vector<double>& row : h.rows) {
      const double mechanical = row[hydro_energy]+row[self_energy]+row[core_energy];
      const double mass_drift = Relative(row[mass], initial[mass]);
      const double j_drift = initial[angular_momentum] != 0.0
          ? Relative(row[angular_momentum], initial[angular_momentum]) : 0.0;
      const double energy_drift = Relative(mechanical, initial_mechanical);
      const double mach_rms = std::sqrt(std::max(0.0, row[mach2_mass]/row[mass]));
      const double gravity_virial = row[self_virial]+row[core_virial];
      const double virial = (2.0*row[kinetic]+row[pressure_virial]+gravity_virial)
          /std::abs(gravity_virial);
      const double binding = std::abs(gravity_virial);
      const double radial_ke_fraction = row[radial_ke]/binding;
      const double nonradial_ke_fraction = row[nonradial_ke]/binding;
      const double mean_radius = row[r_mass]/row[mass];
      const double rms_radius = std::sqrt(row[r2_mass]/row[mass]);
      const double mean_radius_drift = Relative(mean_radius, initial_mean_radius);
      const double rms_radius_drift = Relative(rms_radius, initial_rms_radius);
      const double available_cells = has_wenoz ? row[wenoz_cells] : 0.0;
      const double sensor_cells = has_wenoz && wenoz_sensor < h.names.size()
          ? row[wenoz_sensor] : 0.0;
      const double amr_cells = has_wenoz && wenoz_amr < h.names.size()
          ? row[wenoz_amr] : 0.0;
      const double fallback_cells = has_wenoz ? row[wenoz_fallback] : 0.0;
      const double fallback_fraction = available_cells > 0.0
          ? fallback_cells/available_cells : 0.0;
      const double fallback_mass_fraction = has_wenoz
          && wenoz_fallback_mass < h.names.size() && row[mass] > 0.0
          ? row[wenoz_fallback_mass]/row[mass] : 0.0;
      csv << row[time] << ',' << row[time]*2.19999657e4/kDay << ','
          << row[time]*2.19999657e4/kDynamicalTime << ','
          << mass_drift << ',' << j_drift << ',' << energy_drift << ','
          << mach_rms << ',' << row[mach_max] << ',' << radial_ke_fraction << ','
          << nonradial_ke_fraction << ',' << mean_radius << ','
          << rms_radius << ',' << virial << ','
          << row[outside_mass]/row[mass] << ',' << row[q_mass]/row[mass] << ','
          << row[q_max] << ',' << available_cells << ',' << sensor_cells << ','
          << amr_cells << ',' << fallback_cells << ','
          << (available_cells > 0.0 ? sensor_cells/available_cells : 0.0) << ','
          << (available_cells > 0.0 ? amr_cells/available_cells : 0.0) << ','
          << fallback_fraction << ',' << fallback_mass_fraction << '\n';
      maximum_mass_drift = std::max(maximum_mass_drift, std::abs(mass_drift));
      maximum_j_drift = std::max(maximum_j_drift, std::abs(j_drift));
      maximum_energy_drift = std::max(maximum_energy_drift, std::abs(energy_drift));
      maximum_mach_rms = std::max(maximum_mach_rms, mach_rms);
      maximum_mach = std::max(maximum_mach, row[mach_max]);
      maximum_virial = std::max(maximum_virial, std::abs(virial));
      maximum_radial_ke_fraction = std::max(
          maximum_radial_ke_fraction, radial_ke_fraction);
      maximum_nonradial_ke_fraction = std::max(
          maximum_nonradial_ke_fraction, nonradial_ke_fraction);
      maximum_mean_radius_drift = std::max(
          maximum_mean_radius_drift, std::abs(mean_radius_drift));
      maximum_rms_radius_drift = std::max(
          maximum_rms_radius_drift, std::abs(rms_radius_drift));
      if (available_cells > 0.0) {
        maximum_wenoz_sensor_fraction = std::max(
            maximum_wenoz_sensor_fraction, sensor_cells/available_cells);
        maximum_wenoz_amr_fraction = std::max(
            maximum_wenoz_amr_fraction, amr_cells/available_cells);
      }
      maximum_wenoz_fallback_fraction = std::max(
          maximum_wenoz_fallback_fraction, fallback_fraction);
      maximum_wenoz_fallback_mass_fraction = std::max(
          maximum_wenoz_fallback_mass_fraction, fallback_mass_fraction);
    }

    const std::vector<double>& final = h.rows.back();
    const double final_mechanical = final[hydro_energy]+final[self_energy]
        +final[core_energy];
    const double final_gravity_virial = final[self_virial]+final[core_virial];
    const double final_binding = std::abs(final_gravity_virial);
    const double final_mean_radius = final[r_mass]/final[mass];
    const double final_rms_radius = std::sqrt(final[r2_mass]/final[mass]);
    const double final_available_cells = has_wenoz ? final[wenoz_cells] : 0.0;
    const double final_fallback_cells = has_wenoz ? final[wenoz_fallback] : 0.0;
    const double final_fallback_mass = has_wenoz
        && wenoz_fallback_mass < h.names.size() ? final[wenoz_fallback_mass] : 0.0;
    std::ofstream json(argv[3]);
    if (!json) throw std::runtime_error("cannot create JSON output");
    json << std::scientific << std::setprecision(12)
         << "{\n"
         << "  \"samples\": " << h.rows.size() << ",\n"
         << "  \"final_time_code\": " << h.rows.back()[time] << ",\n"
         << "  \"final_time_over_tdyn\": "
         << h.rows.back()[time]*2.19999657e4/kDynamicalTime << ",\n"
         << "  \"final_mass_relative_drift\": "
         << Relative(final[mass], initial[mass]) << ",\n"
         << "  \"final_J_relative_drift\": "
         << (initial[angular_momentum] != 0.0
             ? Relative(final[angular_momentum], initial[angular_momentum]) : 0.0)
         << ",\n"
         << "  \"final_mechanical_energy_relative_drift\": "
         << Relative(final_mechanical, initial_mechanical) << ",\n"
         << "  \"final_radial_mach_rms\": "
         << std::sqrt(std::max(0.0, final[mach2_mass]/final[mass])) << ",\n"
         << "  \"final_radial_ke_over_abs_W\": "
         << final[radial_ke]/final_binding << ",\n"
         << "  \"final_nonradial_perturbation_ke_over_abs_W\": "
         << final[nonradial_ke]/final_binding << ",\n"
         << "  \"final_mean_radius_relative_drift\": "
         << Relative(final_mean_radius, initial_mean_radius) << ",\n"
         << "  \"final_rms_radius_relative_drift\": "
         << Relative(final_rms_radius, initial_rms_radius) << ",\n"
         << "  \"final_virial_relative_residual\": "
         << (2.0*final[kinetic]+final[pressure_virial]+final_gravity_virial)
             /final_binding << ",\n"
         << "  \"final_wenoz_plm_fallback_fraction\": "
         << (final_available_cells > 0.0
             ? final_fallback_cells/final_available_cells : 0.0) << ",\n"
         << "  \"final_wenoz_plm_fallback_stellar_mass_fraction\": "
         << (final[mass] > 0.0 ? final_fallback_mass/final[mass] : 0.0) << ",\n"
         << "  \"maximum_abs_mass_relative_drift\": " << maximum_mass_drift << ",\n"
         << "  \"maximum_abs_J_relative_drift\": " << maximum_j_drift << ",\n"
         << "  \"maximum_abs_mechanical_energy_relative_drift\": "
         << maximum_energy_drift << ",\n"
         << "  \"maximum_radial_mach_rms\": " << maximum_mach_rms << ",\n"
         << "  \"maximum_radial_mach\": " << maximum_mach << ",\n"
         << "  \"maximum_abs_virial_relative_residual\": " << maximum_virial << ",\n"
         << "  \"maximum_radial_ke_over_abs_W\": "
         << maximum_radial_ke_fraction << ",\n"
         << "  \"maximum_nonradial_perturbation_ke_over_abs_W\": "
         << maximum_nonradial_ke_fraction << ",\n"
         << "  \"maximum_abs_mean_radius_relative_drift\": "
         << maximum_mean_radius_drift << ",\n"
         << "  \"maximum_abs_rms_radius_relative_drift\": "
         << maximum_rms_radius_drift << ",\n"
         << "  \"maximum_wenoz_sensor_fallback_fraction\": "
         << maximum_wenoz_sensor_fraction << ",\n"
         << "  \"maximum_wenoz_amr_fallback_fraction\": "
         << maximum_wenoz_amr_fraction << ",\n"
         << "  \"maximum_wenoz_plm_fallback_fraction\": "
         << maximum_wenoz_fallback_fraction << ",\n"
         << "  \"maximum_wenoz_plm_fallback_stellar_mass_fraction\": "
         << maximum_wenoz_fallback_mass_fraction << "\n"
         << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "analyze_relaxation_history: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
