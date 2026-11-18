#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kG = 6.67430e-8;
constexpr double kRsun = 6.957e10;
constexpr double kDay = 86400.0;
constexpr double kPi = 3.14159265358979323846;

struct Point {
  double radius;
  double mu;
  double cylindrical_radius;
  double density;
  double pressure;
  double enthalpy;
  int layer;
};

std::vector<std::string> Split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

double Parse(const std::string& value) {
  char* end = nullptr;
  const double result = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    throw std::runtime_error("invalid number in SCF CSV: " + value);
  }
  return result;
}

std::vector<Point> ReadEquator(const std::string& filename) {
  std::ifstream input(filename.c_str());
  if (!input) throw std::runtime_error("cannot open " + filename);
  std::string line;
  std::getline(input, line);
  std::vector<Point> all;
  double minimum_abs_mu = std::numeric_limits<double>::infinity();
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> f = Split(line);
    if (f.size() != 10) throw std::runtime_error("unexpected SCF CSV schema");
    Point p;
    p.radius = Parse(f[0]);
    p.mu = Parse(f[1]);
    p.cylindrical_radius = Parse(f[2]);
    p.density = Parse(f[3]);
    p.pressure = Parse(f[4]);
    p.enthalpy = Parse(f[5]);
    p.layer = static_cast<int>(Parse(f[9]));
    all.push_back(p);
    minimum_abs_mu = std::min(minimum_abs_mu, std::abs(p.mu));
  }
  std::vector<Point> equator;
  for (const Point& p : all) {
    if (p.mu > 0.0 && std::abs(std::abs(p.mu)-minimum_abs_mu) < 1.0e-12) {
      equator.push_back(p);
    }
  }
  std::sort(equator.begin(), equator.end(),
            [](const Point& a, const Point& b) { return a.radius < b.radius; });
  if (equator.size() < 4) throw std::runtime_error("equatorial ray is incomplete");
  return equator;
}

double LogInterpolate(double x0, double y0, double x1, double y1, double y) {
  const double fraction = (std::log(y)-std::log(y0))/(std::log(y1)-std::log(y0));
  return x0+fraction*(x1-x0);
}

struct SurfaceSample {
  double density;
  double radius;
  double pressure;
  double pressure_scale_height;
  double sound_speed;
};

SurfaceSample SampleAtDensity(const std::vector<Point>& ray, double target,
                              double gamma) {
  for (std::size_t i = 1; i+1 < ray.size(); ++i) {
    if (ray[i-1].density >= target && ray[i].density <= target
        && ray[i].density > 0.0) {
      const double r = LogInterpolate(ray[i-1].radius, ray[i-1].density,
                                      ray[i].radius, ray[i].density, target);
      const double log_pressure = std::log(ray[i-1].pressure)
          +(r-ray[i-1].radius)/(ray[i].radius-ray[i-1].radius)
          *(std::log(ray[i].pressure)-std::log(ray[i-1].pressure));
      const std::size_t lo = i >= 2 ? i-2 : i-1;
      const std::size_t hi = std::min(i+1, ray.size()-1);
      if (ray[lo].pressure <= 0.0 || ray[hi].pressure <= 0.0) {
        throw std::runtime_error("cannot estimate pressure derivative");
      }
      const double dlogp_dr = (std::log(ray[hi].pressure)-std::log(ray[lo].pressure))
          /(ray[hi].radius-ray[lo].radius);
      SurfaceSample sample;
      sample.density = target;
      sample.radius = r;
      sample.pressure = std::exp(log_pressure);
      sample.pressure_scale_height = 1.0/std::abs(dlogp_dr);
      sample.sound_speed = std::sqrt(gamma*sample.pressure/target);
      return sample;
    }
  }
  throw std::runtime_error("density threshold not bracketed");
}

double AcousticCrossingTime(const std::vector<Point>& ray, double gamma,
                            double surface_radius) {
  std::vector<Point> positive;
  for (const Point& p : ray) {
    if (p.density > 0.0 && p.pressure > 0.0) positive.push_back(p);
  }
  double time = 0.0;
  for (std::size_t i = 1; i < positive.size(); ++i) {
    const double c0 = std::sqrt(gamma*positive[i-1].pressure/positive[i-1].density);
    const double c1 = std::sqrt(gamma*positive[i].pressure/positive[i].density);
    time += 0.5*(1.0/c0+1.0/c1)*(positive[i].radius-positive[i-1].radius);
  }
  const Point& last = positive.back();
  const double clast = std::sqrt(gamma*last.pressure/last.density);
  // For a polytropic surface c_s is proportional to sqrt(R-r), so the final
  // unresolved integral is twice the constant-c_s estimate at the last point.
  time += 2.0*std::max(0.0, surface_radius-last.radius)/clast;
  return time;
}

double OblateSurfaceArea(double a, double c) {
  const double eccentricity = std::sqrt(std::max(0.0, 1.0-c*c/(a*a)));
  if (eccentricity < 1.0e-12) return 4.0*kPi*a*a;
  return 2.0*kPi*a*a*(1.0+(1.0-eccentricity*eccentricity)
      *std::atanh(eccentricity)/eccentricity);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: " << argv[0] << " SCF_PROFILE.csv\n";
      return 2;
    }
    const std::vector<Point> ray = ReadEquator(argv[1]);

    // WOR-159/160 production model and WOR-162 local benchmark constants.
    const double gas_mass = 3.041623627857687e34;
    const double core_mass = 1.134037100608219e34;
    const double total_mass = gas_mass+core_mass;
    const double equatorial_radius = 6.679757551079786e13;
    const double polar_radius = 6.102703576377927e13;
    const double core_support_radius = 3.541497158802513e11;
    const double omega = 4.0e-8;
    const double gamma = 5.0/3.0;
    const double domain_width_rsun = 2200.0;
    const int root_cells = 64;
    const int existing_levels = 5;
    const int block_cells = 16;
    const double cfl = 0.3;
    const double rho_unit = 1.0e-8;
    const double pressure_unit = 1.0e5;
    const double length_unit = kRsun;
    const double time_unit = length_unit/std::sqrt(pressure_unit/rho_unit);
    const double benchmark_cpu_seconds = 103.245726;
    const double benchmark_zone_cycles = 23592960.0;
    const double benchmark_memory_bytes = 1831748.0*1024.0;
    const double benchmark_cells = 1179648.0;
    const double measured_root32_dt_code = 3.3361551227632839e-2;

    const double dynamical_time = std::sqrt(std::pow(equatorial_radius,3)
                                            /(kG*total_mass));
    const double free_fall_time = kPi/std::sqrt(8.0)*dynamical_time;
    const double rotation_period = 2.0*kPi/omega;
    const double kepler_omega = 1.0/dynamical_time;
    const double equatorial_speed = omega*equatorial_radius;
    const double escape_speed = std::sqrt(2.0*kG*total_mass/equatorial_radius);
    const double mean_density = total_mass/(4.0*kPi/3.0
        *equatorial_radius*equatorial_radius*polar_radius);
    const double acoustic_time = AcousticCrossingTime(ray, gamma, equatorial_radius);
    double maximum_sound_speed = 0.0;
    double interface_radius = 0.0;
    for (const Point& p : ray) {
      if (p.density > 0.0 && p.pressure > 0.0) {
        maximum_sound_speed = std::max(maximum_sound_speed,
                                      std::sqrt(gamma*p.pressure/p.density));
      }
      if (p.layer == 2) interface_radius = std::max(interface_radius, p.radius);
    }

    const double thresholds[] = {1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11};
    std::vector<SurfaceSample> samples;
    for (double threshold : thresholds) {
      samples.push_back(SampleAtDensity(ray, threshold, gamma));
    }

    const double root_dx_rsun = domain_width_rsun/root_cells;
    const double finest_dx_cm = root_dx_rsun*kRsun/std::pow(2.0, existing_levels);
    const double cfl_dt = cfl*finest_dx_cm/(maximum_sound_speed+equatorial_speed);
    const double projected_root64_dt = 0.5*measured_root32_dt_code*time_unit;
    const double cpu_seconds_per_zone_cycle = benchmark_cpu_seconds/benchmark_zone_cycles;
    const double bytes_per_cell = benchmark_memory_bytes/benchmark_cells;
    const double surface_area_rsun2 = OblateSurfaceArea(equatorial_radius/kRsun,
                                                        polar_radius/kRsun);
    const double base_cells = (std::pow(root_cells/block_cells,3)
        +existing_levels*56.0)*std::pow(block_cells,3);

    std::cout << std::scientific << std::setprecision(8);
    std::cout << "{\n";
    std::cout << "  \"equatorial_radius_cm\": " << equatorial_radius << ",\n";
    std::cout << "  \"polar_radius_cm\": " << polar_radius << ",\n";
    std::cout << "  \"interface_radius_cm\": " << interface_radius << ",\n";
    std::cout << "  \"core_support_radius_cm\": " << core_support_radius << ",\n";
    std::cout << "  \"mean_density_g_cm3\": " << mean_density << ",\n";
    std::cout << "  \"dynamical_time_s\": " << dynamical_time << ",\n";
    std::cout << "  \"free_fall_time_s\": " << free_fall_time << ",\n";
    std::cout << "  \"equatorial_acoustic_crossing_time_s\": " << acoustic_time << ",\n";
    std::cout << "  \"rotation_period_s\": " << rotation_period << ",\n";
    std::cout << "  \"omega_over_omega_kepler\": " << omega/kepler_omega << ",\n";
    std::cout << "  \"equatorial_rotation_speed_cm_s\": " << equatorial_speed << ",\n";
    std::cout << "  \"surface_escape_speed_cm_s\": " << escape_speed << ",\n";
    std::cout << "  \"maximum_sound_speed_cm_s\": " << maximum_sound_speed << ",\n";
    std::cout << "  \"code_time_s\": " << time_unit << ",\n";
    std::cout << "  \"estimated_root64_cfl_dt_s\": " << cfl_dt << ",\n";
    std::cout << "  \"projected_root64_dt_from_benchmark_s\": "
              << projected_root64_dt << ",\n";
    std::cout << "  \"estimated_steps_per_dynamical_time\": "
              << dynamical_time/projected_root64_dt << ",\n";
    std::cout << "  \"surface_samples\": [\n";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const SurfaceSample& s = samples[i];
      std::cout << "    {\"density_g_cm3\": " << s.density
                << ", \"radius_cm\": " << s.radius
                << ", \"pressure_scale_height_cm\": " << s.pressure_scale_height
                << ", \"sound_speed_cm_s\": " << s.sound_speed << "}"
                << (i+1 == samples.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n";
    std::cout << "  \"resolution_levels\": [\n";
    double cumulative_shell_cells = 0.0;
    for (int level = 0; level <= 5; ++level) {
      const double dx = root_dx_rsun/std::pow(2.0, level);
      if (level > 0) {
        cumulative_shell_cells += surface_area_rsun2*block_cells/(dx*dx);
      }
      const double cells = base_cells+cumulative_shell_cells;
      const double steps = dynamical_time/projected_root64_dt;
      const double cpu_hours_per_tdyn = cells*steps*cpu_seconds_per_zone_cycle/3600.0;
      const double memory_gib = cells*bytes_per_cell/std::pow(1024.0,3);
      std::cout << "    {\"surface_level\": " << level
                << ", \"dx_rsun\": " << dx
                << ", \"estimated_cells\": " << cells
                << ", \"estimated_memory_gib\": " << memory_gib
                << ", \"estimated_cpu_hours_per_tdyn\": " << cpu_hours_per_tdyn
                << ", \"cells_per_H_rho1e-9\": "
                << samples[1].pressure_scale_height/(dx*kRsun) << "}"
                << (level == 5 ? "\n" : ",\n");
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
  } catch (const std::exception& error) {
    std::cerr << "estimate_relaxation_scales: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
