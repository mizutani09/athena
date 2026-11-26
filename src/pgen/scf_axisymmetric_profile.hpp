#ifndef ATHENA_PGEN_SCF_AXISYMMETRIC_PROFILE_HPP_
#define ATHENA_PGEN_SCF_AXISYMMETRIC_PROFILE_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scf_map {

constexpr double kGasConstantCgs = 8.31446261815324e7;
constexpr double kRadiationConstantCgs = 7.5657e-15;

struct State {
  double density;
  double pressure;
  double enthalpy;
  double gas_potential;
  double core_potential;
  double rotational_potential;
  double temperature;
  double radiative_fraction;
};

struct Conserved {
  double density;
  double momentum_x;
  double momentum_y;
  double momentum_z;
  double total_energy;
  double pressure;
  double stellar_density;
  double atmosphere_density;
  double radiative_density;
  double density_temperature;
  double angular_momentum_z;
  double internal_energy;
  double gas_internal_energy;
  double radiation_energy;
  double kinetic_energy;
};

struct AtmosphereModel {
  double density_reference;
  double pressure_reference;
  double density_floor;
  double pressure_floor;
  double sound_speed_squared;
  double reference_radius;
  double monopole_gm;
  bool isothermal_monopole;
  bool rotate;
};

inline AtmosphereModel UniformAtmosphere(double density, double pressure,
                                         bool rotate) {
  AtmosphereModel model = {};
  model.density_reference = density;
  model.pressure_reference = pressure;
  model.density_floor = density;
  model.pressure_floor = pressure;
  model.rotate = rotate;
  return model;
}

inline void AtmosphereState(const AtmosphereModel& model,
                            double x, double y, double z,
                            double* density, double* pressure) {
  if (!(model.density_reference > 0.0)
      || !(model.pressure_reference > 0.0)) {
    throw std::invalid_argument("atmosphere reference state must be positive");
  }
  *density = model.density_reference;
  *pressure = model.pressure_reference;
  if (model.isothermal_monopole) {
    if (!(model.sound_speed_squared > 0.0)
        || !(model.reference_radius > 0.0) || !(model.monopole_gm > 0.0)) {
      throw std::invalid_argument("invalid isothermal monopole atmosphere");
    }
    const double radius = std::sqrt(x*x+y*y+z*z);
    if (radius > 0.0) {
      const double delta_potential = model.monopole_gm
          *(1.0/model.reference_radius-1.0/radius);
      const double exponent = std::max(-80.0, std::min(
          0.0, -delta_potential/model.sound_speed_squared));
      *density = model.density_reference*std::exp(exponent);
      *pressure = *density*model.sound_speed_squared;
    }
    *density = std::max(*density, model.density_floor);
    *pressure = std::max(*pressure, model.pressure_floor);
  }
}

inline Conserved ZeroConserved() {
  Conserved value = {};
  return value;
}

inline std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream input(line);
  std::string field;
  while (std::getline(input, field, ',')) fields.push_back(field);
  return fields;
}

inline double ParseFinite(const std::string& text, const std::string& field) {
  std::size_t used = 0;
  const double value = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid finite value in SCF column " + field);
  }
  return value;
}

inline double GasRadiationTemperature(double density, double pressure,
                                      double mean_molecular_weight) {
  if (!(density > 0.0) || !(pressure > 0.0)) return 0.0;
  const double gas_coefficient = density*kGasConstantCgs/mean_molecular_weight;
  double low = 0.0;
  double high = std::max(pressure/gas_coefficient,
      std::pow(3.0*pressure/kRadiationConstantCgs, 0.25));
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double middle = 0.5*(low+high);
    const double reconstructed = gas_coefficient*middle
        +kRadiationConstantCgs*std::pow(middle, 4)/3.0;
    if (reconstructed > pressure) high = middle;
    else low = middle;
  }
  return 0.5*(low+high);
}

class AxisymmetricProfile {
 public:
  void Load(const std::string& filename, double mean_molecular_weight) {
    if (!(mean_molecular_weight > 0.0)) {
      throw std::invalid_argument("mean molecular weight must be positive");
    }
    mean_molecular_weight_ = mean_molecular_weight;
    std::ifstream input(filename.c_str());
    if (!input) throw std::runtime_error("cannot open SCF profile " + filename);
    std::string line;
    if (!std::getline(input, line)) {
      throw std::runtime_error("empty SCF profile " + filename);
    }
    const std::vector<std::string> header = SplitCsv(line);
    std::map<std::string, std::size_t> column;
    for (std::size_t index = 0; index < header.size(); ++index) {
      column[header[index]] = index;
    }
    const char* required[] = {
      "radius_cm", "mu", "density_g_cm3", "pressure_dyn_cm2",
      "enthalpy_cm2_s2", "gas_potential_cm2_s2",
      "core_potential_cm2_s2", "rotational_potential_cm2_s2", "layer"
    };
    for (const char* name : required) {
      if (column.count(name) == 0) {
        throw std::runtime_error("missing SCF profile column "
                                 +std::string(name));
      }
    }

    struct Row {
      double radius;
      double mu;
      State state;
    };
    std::vector<Row> rows;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      const std::vector<std::string> fields = SplitCsv(line);
      if (fields.size() != header.size()) {
        throw std::runtime_error("malformed row in SCF profile " + filename);
      }
      auto value = [&](const char* name) {
        return ParseFinite(fields[column[name]], name);
      };
      Row row;
      row.radius = value("radius_cm");
      row.mu = value("mu");
      row.state.density = value("density_g_cm3");
      row.state.pressure = value("pressure_dyn_cm2");
      row.state.enthalpy = value("enthalpy_cm2_s2");
      row.state.gas_potential = value("gas_potential_cm2_s2");
      row.state.core_potential = value("core_potential_cm2_s2");
      row.state.rotational_potential = value("rotational_potential_cm2_s2");
      row.state.temperature = GasRadiationTemperature(
          row.state.density, row.state.pressure, mean_molecular_weight);
      const double layer = value("layer");
      row.state.radiative_fraction = layer > 1.5 ? 1.0 : 0.0;
      if (!(row.radius >= 0.0) || row.mu < -1.0 || row.mu > 1.0
          || row.state.density < 0.0 || row.state.pressure < 0.0) {
        throw std::runtime_error("out-of-domain value in SCF profile");
      }
      rows.push_back(row);
    }
    if (rows.size() < 4) throw std::runtime_error("SCF profile is too short");

    radii_.clear();
    mu_.clear();
    for (const Row& row : rows) {
      if (radii_.empty() || row.radius != radii_.back()) {
        radii_.push_back(row.radius);
      }
    }
    for (const Row& row : rows) {
      if (row.radius == radii_.front()) mu_.push_back(row.mu);
      else break;
    }
    if (radii_.size() < 2 || mu_.size() < 2
        || rows.size() != radii_.size()*mu_.size()) {
      throw std::runtime_error("SCF profile is not a rectangular (radius,mu) grid");
    }
    for (std::size_t radial = 1; radial < radii_.size(); ++radial) {
      if (!(radii_[radial] > radii_[radial-1])) {
        throw std::runtime_error("SCF radii are not strictly increasing");
      }
    }
    for (std::size_t angular = 1; angular < mu_.size(); ++angular) {
      if (!(mu_[angular] > mu_[angular-1])) {
        throw std::runtime_error("SCF mu nodes are not strictly increasing");
      }
    }
    states_.resize(rows.size());
    for (std::size_t radial = 0; radial < radii_.size(); ++radial) {
      for (std::size_t angular = 0; angular < mu_.size(); ++angular) {
        const Row& row = rows[radial*mu_.size()+angular];
        if (row.radius != radii_[radial] || row.mu != mu_[angular]) {
          throw std::runtime_error("SCF profile row ordering is inconsistent");
        }
        states_[radial*mu_.size()+angular] = row.state;
      }
    }
  }

  State Interpolate(double cylindrical_radius, double z) const {
    if (states_.empty()) throw std::runtime_error("SCF profile was not loaded");
    const double radius = std::hypot(cylindrical_radius, z);
    if (radius > radii_.back()) {
      State vacuum = {};
      return vacuum;
    }
    const double query_mu = radius > 0.0 ? z/radius : 0.0;
    const Bracket radial = FindBracket(radii_, radius);
    const Bracket angular = FindBracket(mu_, query_mu);
    const State& s00 = At(radial.lower, angular.lower);
    const State& s01 = At(radial.lower, angular.upper);
    const State& s10 = At(radial.upper, angular.lower);
    const State& s11 = At(radial.upper, angular.upper);
    State result;
    result.density = Bilinear(s00.density, s01.density, s10.density,
                              s11.density, radial.fraction, angular.fraction);
    result.pressure = Bilinear(s00.pressure, s01.pressure, s10.pressure,
                               s11.pressure, radial.fraction, angular.fraction);
    result.enthalpy = Bilinear(s00.enthalpy, s01.enthalpy, s10.enthalpy,
                               s11.enthalpy, radial.fraction, angular.fraction);
    result.gas_potential = Bilinear(
        s00.gas_potential, s01.gas_potential,
        s10.gas_potential, s11.gas_potential,
        radial.fraction, angular.fraction);
    result.core_potential = Bilinear(
        s00.core_potential, s01.core_potential,
        s10.core_potential, s11.core_potential,
        radial.fraction, angular.fraction);
    result.rotational_potential = Bilinear(
        s00.rotational_potential, s01.rotational_potential,
        s10.rotational_potential, s11.rotational_potential,
        radial.fraction, angular.fraction);
    result.temperature = Bilinear(
        s00.temperature, s01.temperature, s10.temperature, s11.temperature,
        radial.fraction, angular.fraction);
    result.radiative_fraction = std::max(0.0, std::min(1.0, Bilinear(
        s00.radiative_fraction, s01.radiative_fraction,
        s10.radiative_fraction, s11.radiative_fraction,
        radial.fraction, angular.fraction)));
    result.density = std::max(result.density, 0.0);
    result.pressure = std::max(result.pressure, 0.0);
    return result;
  }

  std::size_t radial_points() const { return radii_.size(); }
  std::size_t angular_points() const { return mu_.size(); }
  double maximum_radius() const { return radii_.back(); }
  double mean_molecular_weight() const { return mean_molecular_weight_; }
  double radius(std::size_t radial) const { return radii_.at(radial); }
  double mu(std::size_t angular) const { return mu_.at(angular); }
  const State& state(std::size_t radial, std::size_t angular) const {
    return At(radial, angular);
  }
  State& mutable_state(std::size_t radial, std::size_t angular) {
    return states_.at(radial*mu_.size()+angular);
  }

 private:
  struct Bracket {
    std::size_t lower;
    std::size_t upper;
    double fraction;
  };

  static Bracket FindBracket(const std::vector<double>& coordinates,
                             double value) {
    if (value <= coordinates.front()) return {0, 1, 0.0};
    if (value >= coordinates.back()) {
      return {coordinates.size()-2, coordinates.size()-1, 1.0};
    }
    const auto upper = std::upper_bound(coordinates.begin(), coordinates.end(),
                                        value);
    const std::size_t hi = static_cast<std::size_t>(
        upper-coordinates.begin());
    const std::size_t lo = hi-1;
    return {lo, hi, (value-coordinates[lo])
                    /(coordinates[hi]-coordinates[lo])};
  }

  static double Bilinear(double f00, double f01, double f10, double f11,
                         double radial_fraction, double angular_fraction) {
    const double inner = f00+angular_fraction*(f01-f00);
    const double outer = f10+angular_fraction*(f11-f10);
    return inner+radial_fraction*(outer-inner);
  }

  const State& At(std::size_t radial, std::size_t angular) const {
    return states_[radial*mu_.size()+angular];
  }

  std::vector<double> radii_;
  std::vector<double> mu_;
  std::vector<State> states_;
  double mean_molecular_weight_ = 0.0;
};

inline Conserved Sample(const AxisymmetricProfile& profile,
                        double x, double y, double z,
                        double angular_velocity, double gamma,
                        const AtmosphereModel& atmosphere) {
  if (!(gamma > 1.0)) throw std::invalid_argument("invalid gamma");
  double atmosphere_density = 0.0;
  double atmosphere_pressure = 0.0;
  AtmosphereState(atmosphere, x, y, z,
                  &atmosphere_density, &atmosphere_pressure);
  const State source = profile.Interpolate(std::hypot(x, y), z);
  const bool stellar = source.density > atmosphere_density
      && source.pressure > atmosphere_pressure;
  const double density = stellar ? source.density : atmosphere_density;
  const double pressure = stellar ? source.pressure : atmosphere_pressure;
  const double temperature = stellar ? source.temperature
      : pressure*profile.mean_molecular_weight()
          /(density*kGasConstantCgs);
  // The numerical atmosphere is not part of the LTE stellar model.  Its
  // radiation bath is supplied separately by the FLD problem generator.
  const double radiation_energy = stellar ? kRadiationConstantCgs
      *std::pow(temperature, 4) : 0.0;
  const double gas_pressure = stellar
      ? std::max(pressure-radiation_energy/3.0, 0.0) : pressure;
  const double omega = stellar || atmosphere.rotate ? angular_velocity : 0.0;
  const double vx = -omega*y;
  const double vy = omega*x;
  Conserved result;
  result.density = density;
  result.momentum_x = density*vx;
  result.momentum_y = density*vy;
  result.momentum_z = 0.0;
  result.total_energy = pressure/(gamma-1.0)
      +0.5*density*(vx*vx+vy*vy);
  result.internal_energy = pressure/(gamma-1.0);
  result.gas_internal_energy = gas_pressure/(gamma-1.0);
  result.radiation_energy = radiation_energy;
  result.kinetic_energy = 0.5*density*(vx*vx+vy*vy);
  result.angular_momentum_z = x*result.momentum_y-y*result.momentum_x;
  result.pressure = pressure;
  result.stellar_density = stellar ? density : 0.0;
  result.atmosphere_density = stellar ? 0.0 : density;
  result.radiative_density = stellar
      ? density*source.radiative_fraction : 0.0;
  result.density_temperature = density*temperature;
  return result;
}

inline Conserved Sample(const AxisymmetricProfile& profile,
                        double x, double y, double z,
                        double angular_velocity, double gamma,
                        double atmosphere_density, double atmosphere_pressure,
                        bool rotate_atmosphere) {
  return Sample(profile, x, y, z, angular_velocity, gamma,
                UniformAtmosphere(atmosphere_density, atmosphere_pressure,
                                  rotate_atmosphere));
}

inline void GaussLegendreRule(int order, std::vector<double>* nodes,
                              std::vector<double>* weights) {
  nodes->clear();
  weights->clear();
  if (order == 1) {
    *nodes = {0.0};
    *weights = {2.0};
  } else if (order == 2) {
    const double node = 1.0/std::sqrt(3.0);
    *nodes = {-node, node};
    *weights = {1.0, 1.0};
  } else if (order == 3) {
    const double node = std::sqrt(3.0/5.0);
    *nodes = {-node, 0.0, node};
    *weights = {5.0/9.0, 8.0/9.0, 5.0/9.0};
  } else {
    throw std::invalid_argument("cell quadrature order must be 1, 2, or 3");
  }
}

inline Conserved CellAverage(const AxisymmetricProfile& profile,
                             double x_lower, double x_upper,
                             double y_lower, double y_upper,
                             double z_lower, double z_upper,
                             int quadrature_order,
                             double angular_velocity, double gamma,
                             const AtmosphereModel& atmosphere) {
  if (!(x_upper > x_lower) || !(y_upper > y_lower)
      || !(z_upper > z_lower)) {
    throw std::invalid_argument("cell bounds must have positive volume");
  }
  std::vector<double> nodes, weights;
  GaussLegendreRule(quadrature_order, &nodes, &weights);
  Conserved average = ZeroConserved();
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const double z = 0.5*((z_upper+z_lower)
                         +nodes[k]*(z_upper-z_lower));
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      const double y = 0.5*((y_upper+y_lower)
                           +nodes[j]*(y_upper-y_lower));
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double x = 0.5*((x_upper+x_lower)
                             +nodes[i]*(x_upper-x_lower));
        const double weight = weights[i]*weights[j]*weights[k]/8.0;
        const Conserved point = Sample(
            profile, x, y, z, angular_velocity, gamma,
            atmosphere);
        average.density += weight*point.density;
        average.momentum_x += weight*point.momentum_x;
        average.momentum_y += weight*point.momentum_y;
        average.momentum_z += weight*point.momentum_z;
        average.total_energy += weight*point.total_energy;
        average.pressure += weight*point.pressure;
        average.stellar_density += weight*point.stellar_density;
        average.atmosphere_density += weight*point.atmosphere_density;
        average.radiative_density += weight*point.radiative_density;
        average.density_temperature += weight*point.density_temperature;
        average.angular_momentum_z += weight*point.angular_momentum_z;
        average.internal_energy += weight*point.internal_energy;
        average.gas_internal_energy += weight*point.gas_internal_energy;
        average.radiation_energy += weight*point.radiation_energy;
        average.kinetic_energy += weight*point.kinetic_energy;
      }
    }
  }
  return average;
}


inline Conserved CellAverage(const AxisymmetricProfile& profile,
                             double x_lower, double x_upper,
                             double y_lower, double y_upper,
                             double z_lower, double z_upper,
                             int quadrature_order,
                             double angular_velocity, double gamma,
                             double atmosphere_density,
                             double atmosphere_pressure,
                             bool rotate_atmosphere) {
  return CellAverage(profile, x_lower, x_upper, y_lower, y_upper,
                     z_lower, z_upper, quadrature_order,
                     angular_velocity, gamma,
                     UniformAtmosphere(atmosphere_density, atmosphere_pressure,
                                       rotate_atmosphere));
}

}  // namespace scf_map

#endif  // ATHENA_PGEN_SCF_AXISYMMETRIC_PROFILE_HPP_
