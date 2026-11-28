#include <glob.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/pgen/scf_relaxation.hpp"

namespace {

constexpr double kGamma = 5.0/3.0;

struct Cell {
  double x;
  double y;
  double density;
  double pressure;
  double vx;
  double vy;
  double vz;
  double stellar_fraction;
  double force_relative;
  double full_force_relative;
  double area;
};

struct Bin {
  std::size_t cells = 0;
  double mass = 0.0;
  double stellar_fraction_mass = 0.0;
  double diluted_mass = 0.0;
  double force2_mass = 0.0;
  double full_force2_mass = 0.0;
  double force_max = 0.0;
  double full_force_max = 0.0;
  double radial_mach2_mass = 0.0;
  double perturbation_mach2_mass = 0.0;
  double radial_mach_max = 0.0;
  double damping_rate_mass = 0.0;
  double radial_ke = 0.0;
};

std::vector<std::string> Glob(const std::string& pattern) {
  glob_t matches;
  const int status = glob(pattern.c_str(), 0, nullptr, &matches);
  if (status != 0) throw std::runtime_error("no files match " + pattern);
  std::vector<std::string> paths;
  for (std::size_t i = 0; i < matches.gl_pathc; ++i) {
    paths.push_back(matches.gl_pathv[i]);
  }
  globfree(&matches);
  return paths;
}

std::vector<std::vector<double> > ReadRows(const std::string& filename) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  std::vector<std::vector<double> > rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream values(line);
    std::vector<double> row;
    double value = 0.0;
    while (values >> value) row.push_back(value);
    if (!row.empty()) rows.push_back(row);
  }
  if (rows.empty()) throw std::runtime_error("no data rows in " + filename);
  return rows;
}

std::vector<Cell> ReadBlock(const std::string& primitive_path) {
  std::string user_path = primitive_path;
  const std::size_t output = user_path.find(".out3.");
  if (output == std::string::npos) {
    throw std::runtime_error("primitive path has no .out3.: " + primitive_path);
  }
  user_path.replace(output, 6, ".out4.");
  const std::vector<std::vector<double> > primitive = ReadRows(primitive_path);
  const std::vector<std::vector<double> > user = ReadRows(user_path);
  if (primitive.size() != user.size()) {
    throw std::runtime_error("paired tab row count differs");
  }
  std::set<double> xs;
  std::set<double> ys;
  std::vector<Cell> cells;
  cells.reserve(primitive.size());
  for (std::size_t i = 0; i < primitive.size(); ++i) {
    const std::vector<double>& p = primitive[i];
    const std::vector<double>& u = user[i];
    if ((p.size() != 11 && p.size() != 12) || u.size() != 25) {
      throw std::runtime_error("unexpected paired tab row width");
    }
    if (p[0] != u[0] || p[1] != u[1] || p[2] != u[2] || p[3] != u[3]) {
      throw std::runtime_error("paired tab coordinates differ");
    }
    Cell cell;
    cell.x = p[1];
    cell.y = p[3];
    cell.density = p[4];
    cell.pressure = p[5];
    cell.vx = p[6];
    cell.vy = p[7];
    cell.vz = p[8];
    cell.stellar_fraction = p[p.size()-2];
    cell.force_relative = u[12] > 0.0 ? std::abs(u[11])/u[12] : 0.0;
    cell.full_force_relative = u[22] > 0.0 ? std::abs(u[21])/u[22] : 0.0;
    cell.area = 0.0;
    cells.push_back(cell);
    xs.insert(cell.x);
    ys.insert(cell.y);
  }
  if (xs.size() < 2 || ys.size() < 2) {
    throw std::runtime_error("incomplete tab block " + primitive_path);
  }
  const double area = (*std::next(xs.begin())-*xs.begin())
      *(*std::next(ys.begin())-*ys.begin());
  for (Cell& cell : cells) cell.area = area;
  return cells;
}

double Fraction(double numerator, double denominator) {
  return denominator > 0.0 ? numerator/denominator : 0.0;
}

void Accumulate(Bin* bin, const Cell& cell, double omega, double tau,
                double core_radius, double core_width,
                double core_strength, double temporal_strength) {
  if (!(cell.density > 0.0) || !(cell.pressure > 0.0)
      || !(cell.stellar_fraction > 0.0)) return;
  const double radius = std::hypot(cell.x, cell.y);
  const double sound_speed2 = kGamma*cell.pressure/cell.density;
  if (!(sound_speed2 > 0.0)) return;
  const double radial_velocity = radius > 0.0
      ? (cell.x*cell.vx+cell.y*cell.vy)/radius : 0.0;
  const double dvx = cell.vx+omega*cell.y;
  const double dvy = cell.vy-omega*cell.x;
  const double perturbation2 = dvx*dvx+dvy*dvy+cell.vz*cell.vz;
  const double weight = cell.density*cell.stellar_fraction*cell.area;
  const double core_weight = scf_relaxation::CoreWeight(
      radius, core_radius, core_width);
  const double strength = scf_relaxation::CombinedStrength(
      temporal_strength, core_weight, core_strength);
  const double damping_rate = strength*cell.stellar_fraction/tau;
  ++bin->cells;
  bin->mass += weight;
  bin->stellar_fraction_mass += weight*cell.stellar_fraction;
  if (cell.stellar_fraction < 0.5) bin->diluted_mass += weight;
  bin->force2_mass += weight*cell.force_relative*cell.force_relative;
  bin->full_force2_mass +=
      weight*cell.full_force_relative*cell.full_force_relative;
  if (cell.stellar_fraction >= 0.5) {
    bin->force_max = std::max(bin->force_max, cell.force_relative);
    bin->full_force_max = std::max(
        bin->full_force_max, cell.full_force_relative);
  }
  bin->radial_mach2_mass +=
      weight*radial_velocity*radial_velocity/sound_speed2;
  bin->perturbation_mach2_mass += weight*perturbation2/sound_speed2;
  bin->radial_mach_max = std::max(
      bin->radial_mach_max, std::abs(radial_velocity)/std::sqrt(sound_speed2));
  bin->damping_rate_mass += weight*damping_rate;
  bin->radial_ke += 0.5*weight*radial_velocity*radial_velocity;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 11 && argc != 12) {
      std::cerr << "usage: " << argv[0]
                << " TAB_DIRECTORY SNAPSHOT OUTPUT.csv SUMMARY.json"
                   " OMEGA_CODE TAU CORE_RADIUS CORE_WIDTH CORE_STRENGTH"
                   " TEMPORAL_STRENGTH [OUTPUT_ID]\n";
      return 2;
    }
    const std::string directory = argv[1];
    const std::string snapshot = argv[2];
    const double omega = std::stod(argv[5]);
    const double tau = std::stod(argv[6]);
    const double core_radius = std::stod(argv[7]);
    const double core_width = std::stod(argv[8]);
    const double core_strength = std::stod(argv[9]);
    const double temporal_strength = std::stod(argv[10]);
    const std::string output_id = argc == 12 ? argv[11] : "out3";
    if (!(tau > 0.0) || core_strength < 0.0 || core_strength > 1.0
        || temporal_strength < 0.0 || temporal_strength > 1.0) {
      throw std::runtime_error("invalid damping parameters");
    }
    const std::vector<double> edges = {0.0, 20.0, 50.0, 65.0, 100.0,
                                       200.0, 400.0, 800.0, 1200.0};
    std::vector<Bin> bins(edges.size()-1);
    const std::string pattern = directory+"/*."+output_id+"."+snapshot+".tab";
    const std::vector<std::string> paths = Glob(pattern);
    for (const std::string& path : paths) {
      for (const Cell& cell : ReadBlock(path)) {
        const double radius = std::hypot(cell.x, cell.y);
        const auto upper = std::upper_bound(edges.begin(), edges.end(), radius);
        if (upper == edges.begin() || upper == edges.end()) continue;
        const std::size_t index = static_cast<std::size_t>(upper-edges.begin()-1);
        Accumulate(&bins[index], cell, omega, tau,
                   core_radius, core_width, core_strength, temporal_strength);
      }
    }
    double total_radial_ke = 0.0;
    for (const Bin& bin : bins) total_radial_ke += bin.radial_ke;

    std::ofstream csv(argv[3]);
    if (!csv) throw std::runtime_error("cannot create CSV output");
    csv << "radius_left_rsun,radius_right_rsun,cells,mass_weight,"
           "mean_stellar_fraction,diluted_mass_fraction,force_relative_rms,"
           "force_relative_max,full_force_relative_rms,full_force_relative_max,"
           "radial_mach_rms,radial_mach_max,perturbation_mach_rms,"
           "effective_damping_rate,effective_damping_tau,radial_ke_fraction\n";
    csv << std::scientific << std::setprecision(12);
    for (std::size_t i = 0; i < bins.size(); ++i) {
      const Bin& bin = bins[i];
      if (!(bin.mass > 0.0)) continue;
      const double effective_rate = Fraction(bin.damping_rate_mass, bin.mass);
      csv << edges[i] << ',' << edges[i+1] << ',' << bin.cells << ','
          << bin.mass << ',' << Fraction(bin.stellar_fraction_mass, bin.mass)
          << ',' << Fraction(bin.diluted_mass, bin.mass) << ','
          << std::sqrt(Fraction(bin.force2_mass, bin.mass)) << ','
          << bin.force_max << ','
          << std::sqrt(Fraction(bin.full_force2_mass, bin.mass)) << ','
          << bin.full_force_max << ','
          << std::sqrt(Fraction(bin.radial_mach2_mass, bin.mass)) << ','
          << bin.radial_mach_max << ','
          << std::sqrt(Fraction(bin.perturbation_mach2_mass, bin.mass)) << ','
          << effective_rate << ','
          << (effective_rate > 0.0 ? 1.0/effective_rate : 0.0) << ','
          << Fraction(bin.radial_ke, total_radial_ke) << '\n';
    }

    Bin inner;
    for (std::size_t i = 0; i < bins.size() && edges[i] < 100.0; ++i) {
      const Bin& bin = bins[i];
      inner.cells += bin.cells;
      inner.mass += bin.mass;
      inner.stellar_fraction_mass += bin.stellar_fraction_mass;
      inner.diluted_mass += bin.diluted_mass;
      inner.force2_mass += bin.force2_mass;
      inner.full_force2_mass += bin.full_force2_mass;
      inner.force_max = std::max(inner.force_max, bin.force_max);
      inner.full_force_max = std::max(inner.full_force_max, bin.full_force_max);
      inner.radial_mach2_mass += bin.radial_mach2_mass;
      inner.perturbation_mach2_mass += bin.perturbation_mach2_mass;
      inner.radial_mach_max = std::max(inner.radial_mach_max, bin.radial_mach_max);
      inner.damping_rate_mass += bin.damping_rate_mass;
      inner.radial_ke += bin.radial_ke;
    }
    const double inner_effective_rate =
        Fraction(inner.damping_rate_mass, inner.mass);
    std::ofstream json(argv[4]);
    if (!json) throw std::runtime_error("cannot create JSON output");
    json << std::scientific << std::setprecision(12)
         << "{\n"
         << "  \"snapshot\": \"" << snapshot << "\",\n"
         << "  \"tab_files\": " << paths.size() << ",\n"
         << "  \"omega_code\": " << omega << ",\n"
         << "  \"relaxation_tau_code\": " << tau << ",\n"
         << "  \"core_radius_rsun\": " << core_radius << ",\n"
         << "  \"core_taper_width_rsun\": " << core_width << ",\n"
         << "  \"temporal_strength\": " << temporal_strength << ",\n"
         << "  \"inner_r_lt_100_rsun_cells\": " << inner.cells << ",\n"
         << "  \"inner_mean_stellar_fraction\": "
         << Fraction(inner.stellar_fraction_mass, inner.mass) << ",\n"
         << "  \"inner_diluted_mass_fraction\": "
         << Fraction(inner.diluted_mass, inner.mass) << ",\n"
         << "  \"inner_force_relative_rms\": "
         << std::sqrt(Fraction(inner.force2_mass, inner.mass)) << ",\n"
         << "  \"inner_force_relative_max\": " << inner.force_max << ",\n"
         << "  \"inner_full_force_relative_rms\": "
         << std::sqrt(Fraction(inner.full_force2_mass, inner.mass)) << ",\n"
         << "  \"inner_full_force_relative_max\": "
         << inner.full_force_max << ",\n"
         << "  \"inner_radial_mach_rms\": "
         << std::sqrt(Fraction(inner.radial_mach2_mass, inner.mass)) << ",\n"
         << "  \"inner_radial_mach_max\": " << inner.radial_mach_max << ",\n"
         << "  \"inner_perturbation_mach_rms\": "
         << std::sqrt(Fraction(inner.perturbation_mach2_mass, inner.mass)) << ",\n"
         << "  \"inner_effective_damping_rate\": " << inner_effective_rate << ",\n"
         << "  \"inner_effective_damping_tau\": "
         << (inner_effective_rate > 0.0 ? 1.0/inner_effective_rate : 0.0) << ",\n"
         << "  \"inner_radial_ke_fraction\": "
         << Fraction(inner.radial_ke, total_radial_ke) << "\n"
         << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "audit_core_relaxation_slice: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
