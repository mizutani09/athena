#include <glob.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kOmegaCode = 4.0e-8*2.19999657e4;
constexpr int kMaximumMode = 4;

struct Cell {
  double x;
  double y;
  double density;
  double vx;
  double vy;
  double vz;
  double stellar_fraction;
  double area;
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

std::vector<Cell> ReadBlock(const std::string& filename) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  std::vector<Cell> cells;
  std::set<double> xs;
  std::set<double> ys;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream values(line);
    std::vector<double> fields;
    double value = 0.0;
    while (values >> value) fields.push_back(value);
    if (fields.size() != 11 && fields.size() != 12) {
      throw std::runtime_error("unexpected primitive tab row in " + filename);
    }
    Cell cell;
    cell.x = fields[1];
    cell.y = fields[3];
    cell.density = fields[4];
    cell.vx = fields[6];
    cell.vy = fields[7];
    cell.vz = fields[8];
    cell.stellar_fraction = fields[fields.size()-2];
    cell.area = 0.0;
    cells.push_back(cell);
    xs.insert(cell.x);
    ys.insert(cell.y);
  }
  if (cells.empty() || xs.size() < 2 || ys.size() < 2) {
    throw std::runtime_error("incomplete tab block " + filename);
  }
  const double dx = *std::next(xs.begin())-*xs.begin();
  const double dy = *std::next(ys.begin())-*ys.begin();
  for (Cell& cell : cells) cell.area = dx*dy;
  return cells;
}

struct Region {
  explicit Region(double input_radius) : radius(input_radius) {}

  double radius;
  double mass = 0.0;
  double x_mass = 0.0;
  double y_mass = 0.0;
  double dvx_mass = 0.0;
  double dvy_mass = 0.0;
  double vz_mass = 0.0;
  double moment = 0.0;
  double angular_momentum = 0.0;
  double vphi2_mass = 0.0;
  double radial_speed2_mass = 0.0;
  double tangential_perturbation2_mass = 0.0;
  double vertical_speed2_mass = 0.0;
  std::array<double, kMaximumMode+1> density_cos{};
  std::array<double, kMaximumMode+1> density_sin{};
  std::array<double, kMaximumMode+1> radial_cos{};
  std::array<double, kMaximumMode+1> radial_sin{};
  std::array<double, kMaximumMode+1> tangential_cos{};
  std::array<double, kMaximumMode+1> tangential_sin{};
  std::size_t cells = 0;
};

double Coherence(double real, double imaginary, double mass,
                 double speed2_mass) {
  if (mass <= 0.0 || speed2_mass <= 0.0) return 0.0;
  return std::hypot(real, imaginary)/std::sqrt(mass*speed2_mass);
}

void Accumulate(Region* region, const Cell& cell) {
  const double r = std::hypot(cell.x, cell.y);
  if (r <= 0.0 || r >= region->radius || cell.stellar_fraction <= 0.5
      || cell.density <= 0.0) return;
  const double weight = cell.density*cell.stellar_fraction*cell.area;
  const double cosine = cell.x/r;
  const double sine = cell.y/r;
  const double radial_velocity = cosine*cell.vx+sine*cell.vy;
  const double tangential_velocity = -sine*cell.vx+cosine*cell.vy;
  const double tangential_perturbation = tangential_velocity-kOmegaCode*r;
  const double dvx = cell.vx+kOmegaCode*cell.y;
  const double dvy = cell.vy-kOmegaCode*cell.x;

  region->mass += weight;
  region->x_mass += weight*cell.x;
  region->y_mass += weight*cell.y;
  region->dvx_mass += weight*dvx;
  region->dvy_mass += weight*dvy;
  region->vz_mass += weight*cell.vz;
  region->moment += weight*r*r;
  region->angular_momentum += weight*r*tangential_velocity;
  region->vphi2_mass += weight*tangential_velocity*tangential_velocity;
  region->radial_speed2_mass += weight*radial_velocity*radial_velocity;
  region->tangential_perturbation2_mass +=
      weight*tangential_perturbation*tangential_perturbation;
  region->vertical_speed2_mass += weight*cell.vz*cell.vz;
  const double theta = std::atan2(cell.y, cell.x);
  for (int mode = 1; mode <= kMaximumMode; ++mode) {
    const double c = std::cos(mode*theta);
    const double s = std::sin(mode*theta);
    region->density_cos[mode] += weight*c;
    region->density_sin[mode] += weight*s;
    region->radial_cos[mode] += weight*radial_velocity*c;
    region->radial_sin[mode] += weight*radial_velocity*s;
    region->tangential_cos[mode] += weight*tangential_perturbation*c;
    region->tangential_sin[mode] += weight*tangential_perturbation*s;
  }
  ++region->cells;
}

void WriteHeader(std::ofstream* output) {
  *output << "snapshot,time_over_tdyn,radius_rsun,cells,slice_mass,"
      "com_x_rsun,com_y_rsun,mean_dvx,mean_dvy,mean_vz,"
      "omega_fit_code,omega_fit_over_target_minus_one,"
      "translation_ke_fraction,radial_ke_fraction,"
      "target_tangential_ke_fraction,best_fit_rotation_residual_ke_fraction,"
      "vertical_ke_fraction";
  for (int mode = 1; mode <= kMaximumMode; ++mode) {
    *output << ",density_m" << mode << ",radial_velocity_m" << mode
            << ",tangential_velocity_m" << mode;
  }
  *output << '\n';
}

void WriteRegion(std::ofstream* output, const std::string& snapshot,
                 const Region& region) {
  if (region.mass <= 0.0 || region.moment <= 0.0) {
    throw std::runtime_error("empty core region");
  }
  const double omega_fit = region.angular_momentum/region.moment;
  const double translation_speed2 =
      std::pow(region.dvx_mass/region.mass, 2)
      +std::pow(region.dvy_mass/region.mass, 2)
      +std::pow(region.vz_mass/region.mass, 2);
  const double best_fit_rotation_residual = std::max(
      0.0, region.vphi2_mass
      -region.angular_momentum*region.angular_momentum/region.moment);
  const double total_perturbation2 = region.radial_speed2_mass
      +region.tangential_perturbation2_mass+region.vertical_speed2_mass;
  const int snapshot_number = std::stoi(snapshot);
  *output << snapshot << ',' << snapshot_number/100.0 << ',' << region.radius
          << ',' << region.cells << ',' << region.mass << ','
          << region.x_mass/region.mass << ',' << region.y_mass/region.mass << ','
          << region.dvx_mass/region.mass << ',' << region.dvy_mass/region.mass << ','
          << region.vz_mass/region.mass << ',' << omega_fit << ','
          << omega_fit/kOmegaCode-1.0 << ','
          << region.mass*translation_speed2/total_perturbation2 << ','
          << region.radial_speed2_mass/total_perturbation2 << ','
          << region.tangential_perturbation2_mass/total_perturbation2 << ','
          << best_fit_rotation_residual/total_perturbation2 << ','
          << region.vertical_speed2_mass/total_perturbation2;
  for (int mode = 1; mode <= kMaximumMode; ++mode) {
    *output << ',' << std::hypot(region.density_cos[mode],
                                 region.density_sin[mode])/region.mass
            << ',' << Coherence(region.radial_cos[mode],
                                 region.radial_sin[mode], region.mass,
                                 region.radial_speed2_mass)
            << ',' << Coherence(region.tangential_cos[mode],
                                 region.tangential_sin[mode], region.mass,
                                 region.tangential_perturbation2_mass);
  }
  *output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: " << argv[0]
                << " TAB_DIRECTORY OUTPUT.csv SNAPSHOT [SNAPSHOT ...]\n";
      return 2;
    }
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot create output CSV");
    output << std::scientific << std::setprecision(12);
    WriteHeader(&output);
    for (int argument = 3; argument < argc; ++argument) {
      const std::string snapshot = argv[argument];
      const std::string pattern = std::string(argv[1])+"/*.out2."
          +snapshot+".tab";
      const std::vector<std::string> paths = Glob(pattern);
      std::array<Region, 3> regions{{Region(5.0), Region(10.0), Region(20.0)}};
      for (const std::string& path : paths) {
        for (const Cell& cell : ReadBlock(path)) {
          for (Region& region : regions) Accumulate(&region, cell);
        }
      }
      for (const Region& region : regions) WriteRegion(&output, snapshot, region);
    }
  } catch (const std::exception& error) {
    std::cerr << "analyze_core_mode: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
