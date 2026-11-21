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

namespace {

constexpr double kGamma = 5.0/3.0;
constexpr double kDefaultOmegaCode = 4.0e-8*2.19999657e4;

struct Cell {
  double x;
  double y;
  double density;
  double pressure;
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
    cell.pressure = fields[5];
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

struct Bin {
  double mass = 0.0;
  double mach2_mass = 0.0;
  double radial_ke = 0.0;
  double nonradial_ke = 0.0;
  double maximum_mach = 0.0;
};

double SafeFraction(double numerator, double denominator) {
  return denominator > 0.0 ? numerator/denominator : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 5 || argc > 7) {
      std::cerr << "usage: " << argv[0]
                << " TAB_DIRECTORY SNAPSHOT OUTPUT.csv SUMMARY.json"
                   " [OMEGA_CODE [OUTPUT_ID]]\n";
      return 2;
    }
    const double omega_code = argc >= 6 ? std::stod(argv[5]) : kDefaultOmegaCode;
    const std::string output_id = argc >= 7 ? argv[6] : "out2";
    const std::string pattern = std::string(argv[1])+"/*."+output_id+"."
        +argv[2]+".tab";
    const std::vector<std::string> paths = Glob(pattern);
    const double bin_width = 25.0;
    const int number_bins = 48;
    std::vector<Bin> bins(number_bins);
    double total_radial_ke = 0.0;
    double core_radial_ke = 0.0;
    double inner_radial_ke = 0.0;
    double envelope_radial_ke = 0.0;
    double surface_radial_ke = 0.0;
    double total_nonradial_ke = 0.0;
    double core_nonradial_ke = 0.0;
    double inner_nonradial_ke = 0.0;
    double envelope_nonradial_ke = 0.0;
    double surface_nonradial_ke = 0.0;
    std::size_t stellar_cells = 0;

    for (const std::string& path : paths) {
      for (const Cell& cell : ReadBlock(path)) {
        if (cell.stellar_fraction <= 0.5 || cell.density <= 0.0
            || cell.pressure <= 0.0) continue;
        const double radius = std::hypot(cell.x, cell.y);
        const int index = static_cast<int>(radius/bin_width);
        if (index < 0 || index >= number_bins) continue;
        const double radial_velocity = radius > 0.0
            ? (cell.x*cell.vx+cell.y*cell.vy)/radius : 0.0;
        const double target_vx = -omega_code*cell.y;
        const double target_vy = omega_code*cell.x;
        const double dvx = cell.vx-target_vx;
        const double dvy = cell.vy-target_vy;
        const double perturbation_speed2 = dvx*dvx+dvy*dvy+cell.vz*cell.vz;
        const double sound_speed2 = kGamma*cell.pressure/cell.density;
        const double weight = cell.density*cell.stellar_fraction*cell.area;
        const double radial_ke = 0.5*weight*radial_velocity*radial_velocity;
        const double nonradial_ke = 0.5*weight*std::max(
            0.0, perturbation_speed2-radial_velocity*radial_velocity);
        Bin& bin = bins[index];
        bin.mass += weight;
        bin.mach2_mass += weight*radial_velocity*radial_velocity/sound_speed2;
        bin.radial_ke += radial_ke;
        bin.nonradial_ke += nonradial_ke;
        bin.maximum_mach = std::max(
            bin.maximum_mach, std::abs(radial_velocity)/std::sqrt(sound_speed2));
        total_radial_ke += radial_ke;
        total_nonradial_ke += nonradial_ke;
        if (radius < 20.0) {
          core_radial_ke += radial_ke;
          core_nonradial_ke += nonradial_ke;
        } else if (radius < 100.0) {
          inner_radial_ke += radial_ke;
          inner_nonradial_ke += nonradial_ke;
        } else if (radius < 800.0) {
          envelope_radial_ke += radial_ke;
          envelope_nonradial_ke += nonradial_ke;
        } else {
          surface_radial_ke += radial_ke;
          surface_nonradial_ke += nonradial_ke;
        }
        ++stellar_cells;
      }
    }
    std::ofstream csv(argv[3]);
    if (!csv) throw std::runtime_error("cannot create CSV output");
    csv << "radius_left_rsun,radius_right_rsun,slice_mass_weight,"
           "radial_mach_rms,radial_mach_max,radial_ke_fraction,"
           "nonradial_to_radial_ke\n";
    csv << std::scientific << std::setprecision(12);
    for (int i = 0; i < number_bins; ++i) {
      const Bin& bin = bins[i];
      if (bin.mass <= 0.0) continue;
      csv << i*bin_width << ',' << (i+1)*bin_width << ',' << bin.mass << ','
          << std::sqrt(bin.mach2_mass/bin.mass) << ',' << bin.maximum_mach << ','
          << SafeFraction(bin.radial_ke, total_radial_ke) << ','
          << SafeFraction(bin.nonradial_ke, bin.radial_ke) << '\n';
    }

    std::ofstream json(argv[4]);
    if (!json) throw std::runtime_error("cannot create JSON output");
    json << std::scientific << std::setprecision(12)
         << "{\n"
         << "  \"snapshot\": \"" << argv[2] << "\",\n"
         << "  \"omega_code\": " << omega_code << ",\n"
         << "  \"output_id\": \"" << output_id << "\",\n"
         << "  \"tab_files\": " << paths.size() << ",\n"
         << "  \"stellar_slice_cells\": " << stellar_cells << ",\n"
         << "  \"radial_ke_fraction_r_lt_20_rsun\": "
         << SafeFraction(core_radial_ke, total_radial_ke) << ",\n"
         << "  \"radial_ke_fraction_20_to_100_rsun\": "
         << SafeFraction(inner_radial_ke, total_radial_ke) << ",\n"
         << "  \"radial_ke_fraction_100_to_800_rsun\": "
         << SafeFraction(envelope_radial_ke, total_radial_ke) << ",\n"
         << "  \"radial_ke_fraction_r_gt_800_rsun\": "
         << SafeFraction(surface_radial_ke, total_radial_ke) << ",\n"
         << "  \"nonradial_ke_fraction_r_lt_20_rsun\": "
         << SafeFraction(core_nonradial_ke, total_nonradial_ke) << ",\n"
         << "  \"nonradial_ke_fraction_20_to_100_rsun\": "
         << SafeFraction(inner_nonradial_ke, total_nonradial_ke) << ",\n"
         << "  \"nonradial_ke_fraction_100_to_800_rsun\": "
         << SafeFraction(envelope_nonradial_ke, total_nonradial_ke) << ",\n"
         << "  \"nonradial_ke_fraction_r_gt_800_rsun\": "
         << SafeFraction(surface_nonradial_ke, total_nonradial_ke) << "\n"
         << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "analyze_relaxation_slice: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
