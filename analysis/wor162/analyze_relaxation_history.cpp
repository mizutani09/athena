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
constexpr double kGravitationalConstant = 6.67430e-8;

struct History {
  std::vector<std::string> names;
  std::vector<std::vector<double> > rows;
};

void ParseHeader(const std::string& line, std::vector<std::string>* names) {
  if (line.compare(0, 2, "# ") != 0 ||
      line.find("[1]=") == std::string::npos) return;
  std::stringstream fields(line.substr(2));
  std::string field;
  while (fields >> field) {
    const std::size_t equal = field.find('=');
    if (equal != std::string::npos) names->push_back(field.substr(equal+1));
  }
}

std::vector<std::string> ReadHeader(const std::string& filename) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open history header source " + filename);
  std::vector<std::string> names;
  std::string line;
  while (std::getline(input, line)) ParseHeader(line, &names);
  if (names.empty()) {
    throw std::runtime_error("history header source has no column header");
  }
  return names;
}

History ReadHistory(const std::string& filename,
                    const std::string& header_filename) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  History history;
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, 2, "# ") == 0) {
      ParseHeader(line, &history.names);
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
  if (history.names.empty() && !header_filename.empty()) {
    history.names = ReadHeader(header_filename);
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

double NormRatio(double error_squared, double norm_squared) {
  return norm_squared > 0.0
      ? std::sqrt(std::max(0.0, error_squared/norm_squared)) : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 5 && argc != 6) {
      std::cerr << "usage: " << argv[0]
                << " INPUT.hst OUTPUT.csv SUMMARY.json [TDYN_SECONDS]"
                   " [HEADER.hst]\n";
      return 2;
    }
    const double dynamical_time = argc >= 5
        ? std::stod(argv[4]) : kDynamicalTime;
    if (!(dynamical_time > 0.0)) {
      throw std::invalid_argument("dynamical time must be positive");
    }
    const History h = ReadHistory(argv[1], argc == 6 ? argv[5] : "");
    const std::size_t time = Column(h, "time");
    const std::size_t mass = Column(h, "mapped_stellar_mass_g");
    const std::size_t angular_momentum = Column(h, "mapped_Jz_g_cm2_s");
    const std::size_t hydro_energy = Column(h, "mapped_E_erg");
    const std::size_t radiation_energy = OptionalColumn(h, "radiation_energy_erg");
    const std::size_t gas_internal_energy = OptionalColumn(
        h, "gas_internal_energy_erg");
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
    const std::size_t mach_max_interior = OptionalColumn(
        h, "stellar_radial_mach_max_interior");
    const std::size_t mach_max_surface_taper = OptionalColumn(
        h, "stellar_radial_mach_max_surface_taper");
    const std::size_t mach_max_dense = OptionalColumn(
        h, "stellar_radial_mach_max_rho_ge_surface");
    const std::size_t mach_max_thin = OptionalColumn(
        h, "stellar_radial_mach_max_rho_lt_surface");
    const std::size_t mach_gt_0p1_mass = OptionalColumn(
        h, "stellar_mass_radial_mach_gt_0p1_g");
    const std::size_t mach_gt_0p1_r_mass = OptionalColumn(
        h, "stellar_r_mass_radial_mach_gt_0p1_g_cm");
    const std::size_t mach_gt_0p1_logrho_mass = OptionalColumn(
        h, "stellar_logrho_mass_radial_mach_gt_0p1_g");
    const std::size_t mach_gt_0p3_mass = OptionalColumn(
        h, "stellar_mass_radial_mach_gt_0p3_g");
    const std::size_t wenoz_cells = OptionalColumn(h, "wenoz_available_cells");
    const std::size_t wenoz_sensor = OptionalColumn(h, "wenoz_sensor_fallback_cells");
    const std::size_t wenoz_amr = OptionalColumn(h, "wenoz_amr_fallback_cells");
    const std::size_t wenoz_fallback = OptionalColumn(h, "wenoz_plm_fallback_cells");
    const std::size_t wenoz_fallback_mass = OptionalColumn(
        h, "wenoz_plm_fallback_stellar_mass_g");
    const std::size_t relaxation_inertial_delta = OptionalColumn(
        h, "relaxation_inertial_ke_delta_erg");
    const std::size_t relaxation_inertial_power = OptionalColumn(
        h, "relaxation_inertial_power_erg_s");
    const std::size_t relaxation_corot_removed = OptionalColumn(
        h, "relaxation_corot_ke_removed_erg");
    const std::size_t relaxation_corot_power = OptionalColumn(
        h, "relaxation_corot_power_erg_s");
    const std::size_t material_force_error = OptionalColumn(h, "force_error2");
    const std::size_t material_force_norm = OptionalColumn(h, "force_norm2");
    const std::size_t full_force_error = OptionalColumn(h, "full_force_error2");
    const std::size_t full_force_norm = OptionalColumn(h, "full_force_norm2");
    const std::size_t core_full_force_error = OptionalColumn(
        h, "core_region_full_force_error2");
    const std::size_t core_full_force_norm = OptionalColumn(
        h, "core_region_full_force_norm2");
    std::vector<std::size_t> meshblocks;
    for (int level = 0; level < 32; ++level) {
      const std::size_t column = OptionalColumn(
          h, "meshblocks_L"+std::to_string(level));
      if (column == h.names.size()) break;
      meshblocks.push_back(column);
    }
    const bool has_wenoz = wenoz_cells < h.names.size()
        && wenoz_fallback < h.names.size();
    const bool has_surface_mach = mach_max_interior < h.names.size()
        && mach_max_surface_taper < h.names.size()
        && mach_max_dense < h.names.size() && mach_max_thin < h.names.size()
        && mach_gt_0p1_mass < h.names.size()
        && mach_gt_0p1_r_mass < h.names.size()
        && mach_gt_0p1_logrho_mass < h.names.size()
        && mach_gt_0p3_mass < h.names.size();
    const bool has_relaxation = relaxation_inertial_delta < h.names.size()
        && relaxation_inertial_power < h.names.size()
        && relaxation_corot_removed < h.names.size()
        && relaxation_corot_power < h.names.size();
    const bool has_meshblocks = !meshblocks.empty();
    const bool has_material_force = material_force_error < h.names.size()
        && material_force_norm < h.names.size();
    const bool has_full_force = full_force_error < h.names.size()
        && full_force_norm < h.names.size();
    const bool has_core_full_force = core_full_force_error < h.names.size()
        && core_full_force_norm < h.names.size();

    const std::vector<double>& initial = h.rows.front();
    const double initial_mechanical = initial[hydro_energy]+initial[self_energy]
        +initial[core_energy];
    const bool has_radiation_energy = radiation_energy < h.names.size();
    const bool has_gas_internal_energy = gas_internal_energy < h.names.size();
    const double initial_radiation = has_radiation_energy
        ? initial[radiation_energy] : 0.0;
    const double initial_total_with_radiation = initial_mechanical+initial_radiation;
    const double initial_mean_radius = initial[r_mass]/initial[mass];
    const double initial_rms_radius = std::sqrt(initial[r2_mass]/initial[mass]);
    std::ofstream csv(argv[2]);
    if (!csv) throw std::runtime_error("cannot create CSV output");
    csv << "time_code,time_days,time_over_tdyn,mass_relative_drift,J_relative_drift,"
           "angular_momentum_g_cm2_s,angular_momentum_breakup_fraction,"
           "mechanical_energy_relative_drift,"
           "total_energy_with_radiation_relative_drift,radiation_energy_erg,"
           "gas_internal_energy_erg,radiation_to_gas_internal_energy,"
           "radial_mach_rms,radial_mach_max,"
           "radial_ke_over_abs_W,nonradial_perturbation_ke_over_abs_W,"
           "mean_radius_cm,rms_radius_cm,virial_relative_residual,"
           "outside_initial_surface_mass_fraction,mean_ellipsoidal_radius,"
           "max_ellipsoidal_radius,radial_mach_max_interior,"
           "radial_mach_max_surface_taper,radial_mach_max_rho_ge_surface,"
           "radial_mach_max_rho_lt_surface,radial_mach_gt_0p1_mass_fraction,"
           "radial_mach_gt_0p1_mean_radius_cm,"
           "radial_mach_gt_0p1_geometric_mean_density_g_cm3,"
           "radial_mach_gt_0p3_mass_fraction,wenoz_available_cells,"
           "wenoz_sensor_fallback_cells,wenoz_amr_fallback_cells,"
           "wenoz_plm_fallback_cells,wenoz_sensor_fallback_fraction,"
           "wenoz_amr_fallback_fraction,wenoz_plm_fallback_fraction,"
           "wenoz_plm_fallback_stellar_mass_fraction,"
           "material_force_l2_relative,full_force_l2_relative,"
           "core_region_full_force_l2_relative";
    for (std::size_t level = 0; level < meshblocks.size(); ++level) {
      csv << ",meshblocks_L" << level;
    }
    csv << ",meshblocks_total,relaxation_inertial_ke_delta_erg,"
           "relaxation_inertial_power_erg_s,"
           "relaxation_corot_ke_removed_erg,"
           "relaxation_corot_power_erg_s\n";
    csv << std::scientific << std::setprecision(12);

    double maximum_mass_drift = 0.0;
    double maximum_j_drift = 0.0;
    double maximum_angular_momentum_breakup_fraction = 0.0;
    double maximum_energy_drift = 0.0;
    double maximum_total_with_radiation_drift = 0.0;
    double maximum_mach_rms = 0.0;
    double maximum_mach = 0.0;
    double maximum_mach_interior = 0.0;
    double maximum_mach_surface_taper = 0.0;
    double maximum_mach_dense = 0.0;
    double maximum_mach_thin = 0.0;
    double maximum_mach_gt_0p1_mass_fraction = 0.0;
    double maximum_mach_gt_0p3_mass_fraction = 0.0;
    double maximum_virial = 0.0;
    double maximum_radial_ke_fraction = 0.0;
    double maximum_nonradial_ke_fraction = 0.0;
    double maximum_mean_radius_drift = 0.0;
    double maximum_rms_radius_drift = 0.0;
    double maximum_wenoz_sensor_fraction = 0.0;
    double maximum_wenoz_amr_fraction = 0.0;
    double maximum_wenoz_fallback_fraction = 0.0;
    double maximum_wenoz_fallback_mass_fraction = 0.0;
    double maximum_meshblocks = 0.0;
    double cumulative_relaxation_inertial_work = 0.0;
    double cumulative_relaxation_corot_removed = 0.0;
    double maximum_abs_relaxation_inertial_power = 0.0;
    double maximum_relaxation_corot_power = 0.0;
    double maximum_material_force_l2 = 0.0;
    double maximum_full_force_l2 = 0.0;
    double maximum_core_full_force_l2 = 0.0;
    for (const std::vector<double>& row : h.rows) {
      const double mechanical = row[hydro_energy]+row[self_energy]+row[core_energy];
      const double mass_drift = Relative(row[mass], initial[mass]);
      const double j_drift = initial[angular_momentum] != 0.0
          ? Relative(row[angular_momentum], initial[angular_momentum]) : 0.0;
      const double energy_drift = Relative(mechanical, initial_mechanical);
      const double radiation = has_radiation_energy ? row[radiation_energy] : 0.0;
      const double gas_internal = has_gas_internal_energy
          ? row[gas_internal_energy] : 0.0;
      const double total_with_radiation_drift = Relative(
          mechanical+radiation, initial_total_with_radiation);
      const double mach_rms = std::sqrt(std::max(0.0, row[mach2_mass]/row[mass]));
      const double gravity_virial = row[self_virial]+row[core_virial];
      const double virial = (2.0*row[kinetic]+row[pressure_virial]+gravity_virial)
          /std::abs(gravity_virial);
      const double binding = std::abs(gravity_virial);
      const double radial_ke_fraction = row[radial_ke]/binding;
      const double nonradial_ke_fraction = row[nonradial_ke]/binding;
      const double mean_radius = row[r_mass]/row[mass];
      const double rms_radius = std::sqrt(row[r2_mass]/row[mass]);
      const double breakup_angular_momentum = row[mass]*std::sqrt(
          kGravitationalConstant*row[mass]*mean_radius);
      const double angular_momentum_breakup_fraction =
          breakup_angular_momentum > 0.0
          ? std::abs(row[angular_momentum])/breakup_angular_momentum : 0.0;
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
      const double surface_mach_interior = has_surface_mach
          ? row[mach_max_interior] : 0.0;
      const double surface_mach_taper = has_surface_mach
          ? row[mach_max_surface_taper] : 0.0;
      const double surface_mach_dense = has_surface_mach
          ? row[mach_max_dense] : 0.0;
      const double surface_mach_thin = has_surface_mach
          ? row[mach_max_thin] : 0.0;
      const double high_mach_mass = has_surface_mach
          ? row[mach_gt_0p1_mass] : 0.0;
      const double high_mach_mass_fraction = row[mass] > 0.0
          ? high_mach_mass/row[mass] : 0.0;
      const double high_mach_mean_radius = high_mach_mass > 0.0
          ? row[mach_gt_0p1_r_mass]/high_mach_mass : 0.0;
      const double high_mach_mean_density = high_mach_mass > 0.0
          ? std::exp(row[mach_gt_0p1_logrho_mass]/high_mach_mass) : 0.0;
      const double very_high_mach_mass_fraction = has_surface_mach
          && row[mass] > 0.0 ? row[mach_gt_0p3_mass]/row[mass] : 0.0;
      double meshblock_total = 0.0;
      if (has_meshblocks) {
        for (const std::size_t column : meshblocks) meshblock_total += row[column];
      }
      const double relaxation_delta = has_relaxation
          ? row[relaxation_inertial_delta] : 0.0;
      const double relaxation_power = has_relaxation
          ? row[relaxation_inertial_power] : 0.0;
      const double corot_removed = has_relaxation
          ? row[relaxation_corot_removed] : 0.0;
      const double corot_power = has_relaxation
          ? row[relaxation_corot_power] : 0.0;
      const double material_force_l2 = has_material_force
          ? NormRatio(row[material_force_error], row[material_force_norm]) : 0.0;
      const double full_force_l2 = has_full_force
          ? NormRatio(row[full_force_error], row[full_force_norm]) : 0.0;
      const double core_full_force_l2 = has_core_full_force
          ? NormRatio(row[core_full_force_error], row[core_full_force_norm]) : 0.0;
      csv << row[time] << ',' << row[time]*2.19999657e4/kDay << ','
          << row[time]*2.19999657e4/dynamical_time << ','
          << mass_drift << ',' << j_drift << ',' << row[angular_momentum] << ','
          << angular_momentum_breakup_fraction << ',' << energy_drift << ','
          << total_with_radiation_drift << ',' << radiation << ','
          << gas_internal << ','
          << (gas_internal != 0.0 ? radiation/gas_internal : 0.0) << ','
          << mach_rms << ',' << row[mach_max] << ',' << radial_ke_fraction << ','
          << nonradial_ke_fraction << ',' << mean_radius << ','
          << rms_radius << ',' << virial << ','
          << row[outside_mass]/row[mass] << ',' << row[q_mass]/row[mass] << ','
          << row[q_max] << ',' << surface_mach_interior << ','
          << surface_mach_taper << ',' << surface_mach_dense << ','
          << surface_mach_thin << ',' << high_mach_mass_fraction << ','
          << high_mach_mean_radius << ',' << high_mach_mean_density << ','
          << very_high_mach_mass_fraction << ',' << available_cells << ','
          << sensor_cells << ','
          << amr_cells << ',' << fallback_cells << ','
          << (available_cells > 0.0 ? sensor_cells/available_cells : 0.0) << ','
          << (available_cells > 0.0 ? amr_cells/available_cells : 0.0) << ','
          << fallback_fraction << ',' << fallback_mass_fraction << ','
          << material_force_l2 << ',' << full_force_l2 << ','
          << core_full_force_l2;
      for (const std::size_t column : meshblocks) {
        csv << ',' << (has_meshblocks ? row[column] : 0.0);
      }
      csv << ',' << meshblock_total << ',' << relaxation_delta << ','
          << relaxation_power << ',' << corot_removed << ',' << corot_power
          << '\n';
      maximum_mass_drift = std::max(maximum_mass_drift, std::abs(mass_drift));
      maximum_j_drift = std::max(maximum_j_drift, std::abs(j_drift));
      maximum_angular_momentum_breakup_fraction = std::max(
          maximum_angular_momentum_breakup_fraction,
          angular_momentum_breakup_fraction);
      maximum_energy_drift = std::max(maximum_energy_drift, std::abs(energy_drift));
      maximum_total_with_radiation_drift = std::max(
          maximum_total_with_radiation_drift,
          std::abs(total_with_radiation_drift));
      maximum_mach_rms = std::max(maximum_mach_rms, mach_rms);
      maximum_mach = std::max(maximum_mach, row[mach_max]);
      maximum_mach_interior = std::max(
          maximum_mach_interior, surface_mach_interior);
      maximum_mach_surface_taper = std::max(
          maximum_mach_surface_taper, surface_mach_taper);
      maximum_mach_dense = std::max(maximum_mach_dense, surface_mach_dense);
      maximum_mach_thin = std::max(maximum_mach_thin, surface_mach_thin);
      maximum_mach_gt_0p1_mass_fraction = std::max(
          maximum_mach_gt_0p1_mass_fraction, high_mach_mass_fraction);
      maximum_mach_gt_0p3_mass_fraction = std::max(
          maximum_mach_gt_0p3_mass_fraction, very_high_mach_mass_fraction);
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
      maximum_meshblocks = std::max(maximum_meshblocks, meshblock_total);
      // Each history row reports the artificial update from that completed
      // step.  Fresh runs start with a zero-valued time-zero row, whereas
      // restart histories begin with the first newly completed step.
      cumulative_relaxation_inertial_work += relaxation_delta;
      cumulative_relaxation_corot_removed += corot_removed;
      maximum_abs_relaxation_inertial_power = std::max(
          maximum_abs_relaxation_inertial_power, std::abs(relaxation_power));
      maximum_relaxation_corot_power = std::max(
          maximum_relaxation_corot_power, corot_power);
      maximum_material_force_l2 = std::max(
          maximum_material_force_l2, material_force_l2);
      maximum_full_force_l2 = std::max(maximum_full_force_l2, full_force_l2);
      maximum_core_full_force_l2 = std::max(
          maximum_core_full_force_l2, core_full_force_l2);
    }

    const std::vector<double>& final = h.rows.back();
    const double final_mechanical = final[hydro_energy]+final[self_energy]
        +final[core_energy];
    const double final_radiation = has_radiation_energy
        ? final[radiation_energy] : 0.0;
    const double final_gas_internal = has_gas_internal_energy
        ? final[gas_internal_energy] : 0.0;
    const double final_gravity_virial = final[self_virial]+final[core_virial];
    const double final_binding = std::abs(final_gravity_virial);
    const double final_mean_radius = final[r_mass]/final[mass];
    const double final_rms_radius = std::sqrt(final[r2_mass]/final[mass]);
    const double final_available_cells = has_wenoz ? final[wenoz_cells] : 0.0;
    const double final_fallback_cells = has_wenoz ? final[wenoz_fallback] : 0.0;
    const double final_fallback_mass = has_wenoz
        && wenoz_fallback_mass < h.names.size() ? final[wenoz_fallback_mass] : 0.0;
    const double final_high_mach_mass = has_surface_mach
        ? final[mach_gt_0p1_mass] : 0.0;
    const double final_material_force_l2 = has_material_force
        ? NormRatio(final[material_force_error], final[material_force_norm]) : 0.0;
    const double final_full_force_l2 = has_full_force
        ? NormRatio(final[full_force_error], final[full_force_norm]) : 0.0;
    const double final_core_full_force_l2 = has_core_full_force
        ? NormRatio(final[core_full_force_error], final[core_full_force_norm]) : 0.0;
    double initial_meshblock_total = 0.0;
    double final_meshblock_total = 0.0;
    if (has_meshblocks) {
      for (const std::size_t column : meshblocks) {
        initial_meshblock_total += initial[column];
        final_meshblock_total += final[column];
      }
    }
    std::ofstream json(argv[3]);
    if (!json) throw std::runtime_error("cannot create JSON output");
    json << std::scientific << std::setprecision(12)
         << "{\n"
         << "  \"samples\": " << h.rows.size() << ",\n"
         << "  \"final_time_code\": " << h.rows.back()[time] << ",\n"
         << "  \"final_time_over_tdyn\": "
         << h.rows.back()[time]*2.19999657e4/dynamical_time << ",\n"
         << "  \"final_mass_relative_drift\": "
         << Relative(final[mass], initial[mass]) << ",\n"
         << "  \"final_J_relative_drift\": "
         << (initial[angular_momentum] != 0.0
             ? Relative(final[angular_momentum], initial[angular_momentum]) : 0.0)
         << ",\n"
         << "  \"final_angular_momentum_g_cm2_s\": "
         << final[angular_momentum] << ",\n"
         << "  \"final_angular_momentum_breakup_fraction\": "
         << std::abs(final[angular_momentum])
             /(final[mass]*std::sqrt(kGravitationalConstant*final[mass]
                 *final_mean_radius)) << ",\n"
         << "  \"final_mechanical_energy_relative_drift\": "
         << Relative(final_mechanical, initial_mechanical) << ",\n"
         << "  \"final_total_energy_with_radiation_relative_drift\": "
         << Relative(final_mechanical+final_radiation,
                     initial_total_with_radiation) << ",\n"
         << "  \"initial_radiation_energy_erg\": " << initial_radiation << ",\n"
         << "  \"final_radiation_energy_erg\": " << final_radiation << ",\n"
         << "  \"final_gas_internal_energy_erg\": " << final_gas_internal << ",\n"
         << "  \"final_radiation_to_gas_internal_energy\": "
         << (final_gas_internal != 0.0
             ? final_radiation/final_gas_internal : 0.0) << ",\n"
         << "  \"final_radial_mach_rms\": "
         << std::sqrt(std::max(0.0, final[mach2_mass]/final[mass])) << ",\n"
         << "  \"final_radial_mach_max\": " << final[mach_max] << ",\n"
         << "  \"final_radial_mach_max_interior\": "
         << (has_surface_mach ? final[mach_max_interior] : 0.0) << ",\n"
         << "  \"final_radial_mach_max_surface_taper\": "
         << (has_surface_mach ? final[mach_max_surface_taper] : 0.0) << ",\n"
         << "  \"final_radial_mach_max_rho_ge_surface\": "
         << (has_surface_mach ? final[mach_max_dense] : 0.0) << ",\n"
         << "  \"final_radial_mach_max_rho_lt_surface\": "
         << (has_surface_mach ? final[mach_max_thin] : 0.0) << ",\n"
         << "  \"final_radial_mach_gt_0p1_mass_fraction\": "
         << (final[mass] > 0.0 ? final_high_mach_mass/final[mass] : 0.0) << ",\n"
         << "  \"final_radial_mach_gt_0p1_mean_radius_cm\": "
         << (final_high_mach_mass > 0.0
             ? final[mach_gt_0p1_r_mass]/final_high_mach_mass : 0.0) << ",\n"
         << "  \"final_radial_mach_gt_0p1_geometric_mean_density_g_cm3\": "
         << (final_high_mach_mass > 0.0
             ? std::exp(final[mach_gt_0p1_logrho_mass]/final_high_mach_mass)
             : 0.0) << ",\n"
         << "  \"final_radial_mach_gt_0p3_mass_fraction\": "
         << (has_surface_mach && final[mass] > 0.0
             ? final[mach_gt_0p3_mass]/final[mass] : 0.0) << ",\n"
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
         << "  \"final_material_force_l2_relative\": "
         << final_material_force_l2 << ",\n"
         << "  \"final_full_force_l2_relative\": " << final_full_force_l2
         << ",\n"
         << "  \"final_core_region_full_force_l2_relative\": "
         << final_core_full_force_l2 << ",\n"
         << "  \"initial_meshblocks\": " << initial_meshblock_total << ",\n"
         << "  \"final_meshblocks\": " << final_meshblock_total << ",\n"
         << "  \"maximum_meshblocks\": " << maximum_meshblocks << ",\n";
    for (std::size_t level = 0; level < meshblocks.size(); ++level) {
      json << "  \"final_meshblocks_L" << level << "\": "
           << (has_meshblocks ? final[meshblocks[level]] : 0.0) << ",\n";
    }
    json
         << "  \"has_relaxation_diagnostics\": "
         << (has_relaxation ? "true" : "false") << ",\n"
         << "  \"cumulative_relaxation_inertial_work_erg\": "
         << cumulative_relaxation_inertial_work << ",\n"
         << "  \"cumulative_relaxation_corot_ke_removed_erg\": "
         << cumulative_relaxation_corot_removed << ",\n"
         << "  \"maximum_abs_relaxation_inertial_power_erg_s\": "
         << maximum_abs_relaxation_inertial_power << ",\n"
         << "  \"maximum_relaxation_corot_power_erg_s\": "
         << maximum_relaxation_corot_power << ",\n"
         << "  \"maximum_material_force_l2_relative\": "
         << maximum_material_force_l2 << ",\n"
         << "  \"maximum_full_force_l2_relative\": "
         << maximum_full_force_l2 << ",\n"
         << "  \"maximum_core_region_full_force_l2_relative\": "
         << maximum_core_full_force_l2 << ",\n"
         << "  \"maximum_abs_mass_relative_drift\": " << maximum_mass_drift << ",\n"
         << "  \"maximum_abs_J_relative_drift\": " << maximum_j_drift << ",\n"
         << "  \"maximum_angular_momentum_breakup_fraction\": "
         << maximum_angular_momentum_breakup_fraction << ",\n"
         << "  \"maximum_abs_mechanical_energy_relative_drift\": "
         << maximum_energy_drift << ",\n"
         << "  \"maximum_abs_total_energy_with_radiation_relative_drift\": "
         << maximum_total_with_radiation_drift << ",\n"
         << "  \"maximum_radial_mach_rms\": " << maximum_mach_rms << ",\n"
         << "  \"maximum_radial_mach\": " << maximum_mach << ",\n"
         << "  \"maximum_radial_mach_interior\": "
         << maximum_mach_interior << ",\n"
         << "  \"maximum_radial_mach_surface_taper\": "
         << maximum_mach_surface_taper << ",\n"
         << "  \"maximum_radial_mach_rho_ge_surface\": "
         << maximum_mach_dense << ",\n"
         << "  \"maximum_radial_mach_rho_lt_surface\": "
         << maximum_mach_thin << ",\n"
         << "  \"maximum_radial_mach_gt_0p1_mass_fraction\": "
         << maximum_mach_gt_0p1_mass_fraction << ",\n"
         << "  \"maximum_radial_mach_gt_0p3_mass_fraction\": "
         << maximum_mach_gt_0p3_mass_fraction << ",\n"
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
