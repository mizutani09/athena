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

constexpr double kPi = 3.14159265358979323846;
constexpr double kG = 6.67430e-8;
constexpr double kSolarRadiusCm = 6.957e10;
constexpr double kSolarMassG = 1.98847e33;
constexpr double kGasConstant = 8.31446261815324e7;
constexpr double kMeanMolecularWeight = 0.6353437771420501;
constexpr double kTotalMassG = 4.1756607284659057e34;
constexpr double kGasMassG = 3.2091112192814024e34;
constexpr double kSurfaceL2DxRsun = 8.59375;
constexpr double kSurfaceL3DxRsun = 4.296875;

struct Point {
  double radius;
  double mu;
  double density;
  double pressure;
};

std::vector<std::string> Split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

double Parse(const std::string& text) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error("invalid SCF numeric field: "+text);
  }
  return value;
}

std::vector<Point> ReadEquator(const std::string& path) {
  std::ifstream input(path.c_str());
  if (!input) throw std::runtime_error("cannot open "+path);
  std::string line;
  std::getline(input, line);
  std::vector<Point> all;
  double minimum_abs_mu = std::numeric_limits<double>::infinity();
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> field = Split(line);
    if (field.size() != 10) throw std::runtime_error("unexpected SCF schema");
    Point point = {Parse(field[0]), Parse(field[1]),
                   Parse(field[3]), Parse(field[4])};
    all.push_back(point);
    minimum_abs_mu = std::min(minimum_abs_mu, std::abs(point.mu));
  }
  std::vector<Point> ray;
  for (const Point& point : all) {
    if (point.mu > 0.0
        && std::abs(std::abs(point.mu)-minimum_abs_mu) < 1.0e-12) {
      ray.push_back(point);
    }
  }
  std::sort(ray.begin(), ray.end(), [](const Point& a, const Point& b) {
    return a.radius < b.radius;
  });
  if (ray.size() < 4) throw std::runtime_error("equatorial ray is incomplete");
  return ray;
}

double LogInterpolate(double x0, double y0, double x1, double y1, double y) {
  return x0+(x1-x0)*(std::log(y)-std::log(y0))
      /(std::log(y1)-std::log(y0));
}

double LinearAt(double x0, double y0, double x1, double y1, double x) {
  return y0+(y1-y0)*(x-x0)/(x1-x0);
}

// Exact integral of 4*pi*r^2 times a linearly interpolated density.
double SegmentMass(double r0, double rho0, double r1, double rho1,
                   double lower) {
  const double a = (rho1-rho0)/(r1-r0);
  const double b = rho0-a*r0;
  const double lo = std::max(r0, lower);
  if (lo >= r1) return 0.0;
  return 4.0*kPi*(a/4.0*(std::pow(r1, 4)-std::pow(lo, 4))
                   +b/3.0*(std::pow(r1, 3)-std::pow(lo, 3)));
}

struct Sample {
  double radius;
  double pressure;
  double pressure_scale_height;
  double density_scale_height;
};

Sample AtDensity(const std::vector<Point>& ray, double target) {
  for (std::size_t i = 1; i+1 < ray.size(); ++i) {
    if (ray[i-1].density >= target && ray[i].density <= target
        && ray[i].density > 0.0) {
      const double radius = LogInterpolate(
          ray[i-1].radius, ray[i-1].density,
          ray[i].radius, ray[i].density, target);
      const double pressure = std::exp(LinearAt(
          ray[i-1].radius, std::log(ray[i-1].pressure),
          ray[i].radius, std::log(ray[i].pressure), radius));
      const std::size_t lower = i >= 2 ? i-2 : i-1;
      const std::size_t upper = std::min(i+1, ray.size()-1);
      const double dlogp = (std::log(ray[upper].pressure)
          -std::log(ray[lower].pressure))/(ray[upper].radius-ray[lower].radius);
      const double dlogrho = (std::log(ray[upper].density)
          -std::log(ray[lower].density))/(ray[upper].radius-ray[lower].radius);
      return {radius, pressure, 1.0/std::abs(dlogp),
              1.0/std::abs(dlogrho)};
    }
  }
  throw std::runtime_error("density threshold not bracketed");
}

double ExteriorMass(const std::vector<Point>& ray, double lower_radius) {
  double mass = 0.0;
  for (std::size_t i = 1; i < ray.size(); ++i) {
    mass += SegmentMass(ray[i-1].radius, ray[i-1].density,
                        ray[i].radius, ray[i].density, lower_radius);
  }
  return mass;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: audit_surface_match PROFILE.csv OUTPUT.csv\n";
      return 2;
    }
    const std::vector<Point> ray = ReadEquator(argv[1]);
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error(std::string("cannot open ")+argv[2]);
    output << "density_g_cm3,radius_Rsun,pressure_dyn_cm2,temperature_gas_K,"
              "pressure_scale_height_Rsun,density_scale_height_Rsun,"
              "hse_pressure_scale_height_Rsun,cells_per_Hp_surface_L2,"
              "cells_per_Hp_surface_L3,exterior_mass_Msun,"
              "exterior_fraction_of_gas\n";
    output << std::scientific << std::setprecision(16);
    const double thresholds[] = {
      1.0e-8, 8.0e-9, 6.0e-9, 5.0e-9, 4.0e-9, 3.0e-9,
      2.0e-9, 1.5e-9, 1.0e-9
    };
    for (double density : thresholds) {
      const Sample sample = AtDensity(ray, density);
      const double hp_rsun = sample.pressure_scale_height/kSolarRadiusCm;
      const double gravity = kG*kTotalMassG/(sample.radius*sample.radius);
      const double hse_hp = sample.pressure/(density*gravity);
      const double exterior_mass = ExteriorMass(ray, sample.radius);
      output << density << ',' << sample.radius/kSolarRadiusCm << ','
             << sample.pressure << ','
             << sample.pressure*kMeanMolecularWeight/(density*kGasConstant)
             << ',' << hp_rsun << ','
             << sample.density_scale_height/kSolarRadiusCm << ','
             << hse_hp/kSolarRadiusCm << ','
             << hp_rsun/kSurfaceL2DxRsun << ','
             << hp_rsun/kSurfaceL3DxRsun << ','
             << exterior_mass/kSolarMassG << ',' << exterior_mass/kGasMassG
             << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "audit_surface_match: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
