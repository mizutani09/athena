#include <array>
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

enum Column {
  kRadius = 0,
  kMu = 1,
  kCylindricalRadius = 2,
  kDensity = 3,
  kPressure = 4,
  kEnthalpy = 5,
  kGasPotential = 6,
  kCorePotential = 7,
  kRotationalPotential = 8,
  kLayer = 9,
  kColumns = 10
};

using Row = std::array<double, kColumns>;

std::vector<std::string> Split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

double Parse(const std::string& text, const std::string& description) {
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
    throw std::runtime_error("invalid "+description+": "+text);
  }
  return value;
}

struct Profile {
  std::string header;
  std::vector<Row> rows;
  std::size_t radial_points;
  std::size_t angular_points;
};

Profile ReadProfile(const std::string& path) {
  std::ifstream input(path.c_str());
  if (!input) throw std::runtime_error("cannot open input profile "+path);
  Profile profile;
  if (!std::getline(input, profile.header)) {
    throw std::runtime_error("empty input profile "+path);
  }
  const std::string expected =
      "radius_cm,mu,cylindrical_radius_cm,density_g_cm3,pressure_dyn_cm2,"
      "enthalpy_cm2_s2,gas_potential_cm2_s2,core_potential_cm2_s2,"
      "rotational_potential_cm2_s2,layer";
  if (profile.header != expected) {
    throw std::runtime_error("unexpected SCF profile schema");
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> fields = Split(line);
    if (fields.size() != kColumns) {
      throw std::runtime_error("malformed SCF profile row");
    }
    Row row;
    for (std::size_t column = 0; column < row.size(); ++column) {
      row[column] = Parse(fields[column], "SCF numeric field");
    }
    if (!(row[kRadius] >= 0.0) || row[kMu] < -1.0 || row[kMu] > 1.0
        || row[kDensity] < 0.0 || row[kPressure] < 0.0) {
      throw std::runtime_error("out-of-domain SCF value");
    }
    profile.rows.push_back(row);
  }
  if (profile.rows.size() < 4) {
    throw std::runtime_error("input profile is too short");
  }
  profile.angular_points = 1;
  while (profile.angular_points < profile.rows.size()
         && profile.rows[profile.angular_points][kRadius]
             == profile.rows[0][kRadius]) {
    ++profile.angular_points;
  }
  if (profile.angular_points < 2
      || profile.rows.size()%profile.angular_points != 0) {
    throw std::runtime_error("profile is not a rectangular (radius,mu) grid");
  }
  profile.radial_points = profile.rows.size()/profile.angular_points;
  for (std::size_t radial = 0; radial < profile.radial_points; ++radial) {
    const double radius = profile.rows[radial*profile.angular_points][kRadius];
    if (radial > 0
        && !(radius > profile.rows[(radial-1)*profile.angular_points][kRadius])) {
      throw std::runtime_error("SCF radii are not strictly increasing");
    }
    for (std::size_t angular = 0; angular < profile.angular_points; ++angular) {
      const Row& row = profile.rows[radial*profile.angular_points+angular];
      if (row[kRadius] != radius
          || row[kMu] != profile.rows[angular][kMu]) {
        throw std::runtime_error("inconsistent rectangular SCF grid");
      }
      if (angular > 0
          && !(row[kMu] > profile.rows[angular-1][kMu])) {
        throw std::runtime_error("SCF mu nodes are not strictly increasing");
      }
    }
  }
  return profile;
}

double EffectivePotential(const Row& row) {
  return row[kGasPotential]+row[kCorePotential]+row[kRotationalPotential];
}

std::vector<double> GaussLegendreWeights(const Profile& profile) {
  const int order = static_cast<int>(profile.angular_points);
  std::vector<double> weights(profile.angular_points);
  for (std::size_t angular = 0; angular < profile.angular_points; ++angular) {
    const double x = profile.rows[angular][kMu];
    double pnm2 = 1.0;
    double pnm1 = x;
    double pn = order == 1 ? pnm2 : pnm1;
    for (int degree = 2; degree <= order; ++degree) {
      pn = ((2.0*degree-1.0)*x*pnm1-(degree-1.0)*pnm2)/degree;
      pnm2 = pnm1;
      pnm1 = pn;
    }
    const double pprevious = order == 1 ? 1.0 : pnm2;
    const double derivative = order*(x*pn-pprevious)/(x*x-1.0);
    weights[angular] = 2.0/((1.0-x*x)*derivative*derivative);
    if (!std::isfinite(weights[angular]) || !(weights[angular] > 0.0)
        || std::abs(pn) > 1.0e-10) {
      throw std::runtime_error("mu coordinates are not Gauss-Legendre nodes");
    }
  }
  return weights;
}

double SegmentIntegral(double r0, double f0, double r1, double f1) {
  const double slope = (f1-f0)/(r1-r0);
  const double intercept = f0-slope*r0;
  return slope*(std::pow(r1, 4)-std::pow(r0, 4))/4.0
      +intercept*(std::pow(r1, 3)-std::pow(r0, 3))/3.0;
}

double VolumeIntegral(const Profile& profile, int column,
                      const std::vector<double>& weights) {
  double integral = 0.0;
  for (std::size_t angular = 0; angular < profile.angular_points; ++angular) {
    double radial_integral = 0.0;
    for (std::size_t radial = 1; radial < profile.radial_points; ++radial) {
      const Row& lower = profile.rows[(radial-1)*profile.angular_points+angular];
      const Row& upper = profile.rows[radial*profile.angular_points+angular];
      radial_integral += SegmentIntegral(
          lower[kRadius], lower[column], upper[kRadius], upper[column]);
    }
    integral += 2.0*kPi*weights[angular]*radial_integral;
  }
  return integral;
}

struct TaperAudit {
  double minimum_match_radius = std::numeric_limits<double>::infinity();
  double maximum_match_radius = 0.0;
  double minimum_zero_radius = std::numeric_limits<double>::infinity();
  double maximum_zero_radius = 0.0;
  double maximum_anchor_density_relative_error = 0.0;
  double maximum_anchor_enthalpy_relative_change = 0.0;
};

TaperAudit ApplyTaper(Profile* profile, double match_density, double gamma) {
  TaperAudit audit;
  for (std::size_t angular = 0; angular < profile->angular_points; ++angular) {
    std::size_t anchor = profile->radial_points;
    for (std::size_t radial = 1; radial < profile->radial_points; ++radial) {
      const Row& lower = profile->rows[(radial-1)*profile->angular_points+angular];
      const Row& upper = profile->rows[radial*profile->angular_points+angular];
      if (lower[kDensity] > match_density
          && upper[kDensity] <= match_density && upper[kDensity] > 0.0) {
        anchor = radial;
        break;
      }
    }
    if (anchor == profile->radial_points) {
      throw std::runtime_error("match density is not bracketed on every ray");
    }
    Row& match = profile->rows[anchor*profile->angular_points+angular];
    const double density = match[kDensity];
    const double pressure = match[kPressure];
    if (!(density > 0.0) || !(pressure > 0.0)) {
      throw std::runtime_error("nonpositive surface-taper anchor state");
    }
    const double polytropic_constant = pressure/std::pow(density, gamma);
    const double match_enthalpy = gamma/(gamma-1.0)*pressure/density;
    const double original_enthalpy = match[kEnthalpy];
    const double match_potential = EffectivePotential(match);
    audit.minimum_match_radius = std::min(audit.minimum_match_radius,
                                          match[kRadius]);
    audit.maximum_match_radius = std::max(audit.maximum_match_radius,
                                          match[kRadius]);
    audit.maximum_anchor_density_relative_error = std::max(
        audit.maximum_anchor_density_relative_error,
        std::abs(density/match_density-1.0));
    if (original_enthalpy != 0.0) {
      audit.maximum_anchor_enthalpy_relative_change = std::max(
          audit.maximum_anchor_enthalpy_relative_change,
          std::abs(match_enthalpy/original_enthalpy-1.0));
    }
    match[kEnthalpy] = match_enthalpy;
    bool found_zero = false;
    for (std::size_t radial = anchor+1;
         radial < profile->radial_points; ++radial) {
      Row& row = profile->rows[radial*profile->angular_points+angular];
      const double enthalpy = match_enthalpy
          -(EffectivePotential(row)-match_potential);
      if (enthalpy > 0.0) {
        const double tapered_density = std::pow(
            enthalpy*(gamma-1.0)/(gamma*polytropic_constant),
            1.0/(gamma-1.0));
        row[kDensity] = tapered_density;
        row[kPressure] = polytropic_constant*std::pow(tapered_density, gamma);
        row[kEnthalpy] = enthalpy;
        row[kLayer] = 1.0;
      } else {
        row[kDensity] = 0.0;
        row[kPressure] = 0.0;
        row[kEnthalpy] = 0.0;
        row[kLayer] = 0.0;
        if (!found_zero) {
          audit.minimum_zero_radius = std::min(audit.minimum_zero_radius,
                                                row[kRadius]);
          audit.maximum_zero_radius = std::max(audit.maximum_zero_radius,
                                                row[kRadius]);
          found_zero = true;
        }
      }
    }
    if (!found_zero) {
      throw std::runtime_error("surface taper does not reach zero in the domain");
    }
  }
  return audit;
}

void WriteProfile(const Profile& profile, const std::string& path) {
  std::ofstream output(path.c_str());
  if (!output) throw std::runtime_error("cannot open output profile "+path);
  output << profile.header << '\n' << std::scientific << std::setprecision(16);
  for (const Row& row : profile.rows) {
    for (std::size_t column = 0; column < row.size(); ++column) {
      if (column != 0) output << ',';
      if (column == kLayer) output << static_cast<int>(std::lround(row[column]));
      else output << row[column];
    }
    output << '\n';
  }
}

void WriteAudit(const std::string& path, double match_density, double gamma,
                const Profile& profile, const TaperAudit& taper,
                double original_mass, double tapered_mass,
                double original_pressure, double tapered_pressure) {
  std::ofstream output(path.c_str());
  if (!output) throw std::runtime_error("cannot open audit output "+path);
  output << "match_density_g_cm3,gamma,radial_points,angular_points,"
            "minimum_match_radius_cm,maximum_match_radius_cm,"
            "minimum_zero_radius_cm,maximum_zero_radius_cm,"
            "maximum_anchor_density_relative_error,"
            "maximum_anchor_enthalpy_relative_change,original_mass_g,"
            "tapered_mass_g,mass_relative_change,"
            "original_pressure_volume_integral_erg,"
            "tapered_pressure_volume_integral_erg,"
            "pressure_volume_relative_change\n"
         << std::scientific << std::setprecision(16)
         << match_density << ',' << gamma << ',' << profile.radial_points << ','
         << profile.angular_points << ',' << taper.minimum_match_radius << ','
         << taper.maximum_match_radius << ',' << taper.minimum_zero_radius << ','
         << taper.maximum_zero_radius << ','
         << taper.maximum_anchor_density_relative_error << ','
         << taper.maximum_anchor_enthalpy_relative_change << ','
         << original_mass << ',' << tapered_mass << ','
         << tapered_mass/original_mass-1.0 << ',' << original_pressure << ','
         << tapered_pressure << ',' << tapered_pressure/original_pressure-1.0
         << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: scf_surface_taper INPUT.csv OUTPUT.csv AUDIT.csv "
                   "MATCH_DENSITY_G_CM3 GAMMA\n";
      return 2;
    }
    const double match_density = Parse(argv[4], "match density");
    const double gamma = Parse(argv[5], "polytropic gamma");
    if (!(match_density > 0.0) || !(gamma > 1.0)) {
      throw std::invalid_argument("match density and gamma must be valid");
    }
    Profile profile = ReadProfile(argv[1]);
    const std::vector<double> weights = GaussLegendreWeights(profile);
    const double original_mass = VolumeIntegral(profile, kDensity, weights);
    const double original_pressure = VolumeIntegral(profile, kPressure, weights);
    const TaperAudit taper = ApplyTaper(&profile, match_density, gamma);
    const double tapered_mass = VolumeIntegral(profile, kDensity, weights);
    const double tapered_pressure = VolumeIntegral(profile, kPressure, weights);
    WriteProfile(profile, argv[2]);
    WriteAudit(argv[3], match_density, gamma, profile, taper,
               original_mass, tapered_mass, original_pressure,
               tapered_pressure);
    std::cout << std::scientific << std::setprecision(8)
              << "mass_relative_change=" << tapered_mass/original_mass-1.0
              << " pressure_volume_relative_change="
              << tapered_pressure/original_pressure-1.0
              << " match_radius_cm=[" << taper.minimum_match_radius << ','
              << taper.maximum_match_radius << "] zero_radius_cm=["
              << taper.minimum_zero_radius << ',' << taper.maximum_zero_radius
              << "]\n";
  } catch (const std::exception& error) {
    std::cerr << "scf_surface_taper: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
