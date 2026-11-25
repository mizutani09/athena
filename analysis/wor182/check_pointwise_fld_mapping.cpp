#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/pgen/scf_axisymmetric_profile.hpp"

namespace {

constexpr double kLengthUnit = 6.957e10;
constexpr double kMeanMolecularWeight = 0.6353437771420501;
constexpr double kGamma = 5.0/3.0;
constexpr double kAngularVelocity = 4.0100850320255668e-8;
constexpr double kAtmosphereTemperature = 1.0e4;
constexpr double kAtmosphereReferenceRadius = 7.0479108661544016e13;
constexpr double kGasMass = 2.8801452052462170e34;
constexpr double kCoreMass = 1.2962446564904497e34;
constexpr double kGravity = 6.67430e-8;
constexpr double kPressureFloor = 1.0e-4;
constexpr double kExternalRadiationTemperature = 1.0e2;
constexpr double kSurfaceL2CellWidth = 9.375;

struct Row {
  double radius;
  double density;
  double stellar_mass_fraction;
  double stellar_volume_fraction;
  double old_gas_pressure;
  double new_gas_pressure;
  double old_radiation_energy;
  double new_radiation_energy;
  double source_density;
  double source_pressure;
  double atmosphere_density;
  double atmosphere_pressure;
};

double StellarVolumeFraction(const scf_map::AxisymmetricProfile& profile,
                             const scf_map::AtmosphereModel& atmosphere,
                             double ray_lower, double width, bool polar) {
  std::vector<double> nodes, weights;
  scf_map::GaussLegendreRule(3, &nodes, &weights);
  double fraction = 0.0;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const double z = polar
        ? ray_lower+0.5*width*(1.0+nodes[k])
        : 0.5*width*(1.0+nodes[k]);
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      const double y = 0.5*width*(1.0+nodes[j]);
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double x = polar
            ? 0.5*width*(1.0+nodes[i])
            : ray_lower+0.5*width*(1.0+nodes[i]);
        const scf_map::Conserved point = scf_map::Sample(
            profile, x, y, z, kAngularVelocity, kGamma, atmosphere);
        const double weight = weights[i]*weights[j]*weights[k]/8.0;
        if (point.stellar_density > 0.0) fraction += weight;
      }
    }
  }
  return fraction;
}

Row MapCell(const scf_map::AxisymmetricProfile& profile,
            const scf_map::AtmosphereModel& atmosphere,
            double ray_lower_code, bool polar) {
  const double dx = kSurfaceL2CellWidth*kLengthUnit;
  const double ray_lower = ray_lower_code*kLengthUnit;
  const double x_lower = polar ? 0.0 : ray_lower;
  const double z_lower = polar ? ray_lower : 0.0;
  const scf_map::Conserved mapped = scf_map::CellAverage(
      profile, x_lower, x_lower+dx, 0.0, dx, z_lower, z_lower+dx, 3,
      kAngularVelocity, kGamma, atmosphere);
  const double stellar_volume_fraction = StellarVolumeFraction(
      profile, atmosphere, ray_lower, dx, polar);

  const double old_temperature = mapped.stellar_density > 0.0
      ? scf_map::GasRadiationTemperature(
          mapped.density, mapped.pressure, kMeanMolecularWeight)
      : kAtmosphereTemperature;
  const double old_radiation_energy = mapped.stellar_density > 0.0
      ? scf_map::kRadiationConstantCgs*std::pow(old_temperature, 4)
      : scf_map::kRadiationConstantCgs
          *std::pow(kExternalRadiationTemperature, 4);
  const double old_gas_pressure = mapped.stellar_density > 0.0
      ? std::max(mapped.pressure-old_radiation_energy/3.0, kPressureFloor)
      : mapped.pressure;
  const double atmosphere_volume_fraction = std::max(
      0.0, std::min(1.0, 1.0-stellar_volume_fraction));
  const double new_radiation_energy = mapped.radiation_energy
      +atmosphere_volume_fraction*scf_map::kRadiationConstantCgs
          *std::pow(kExternalRadiationTemperature, 4);
  const double new_gas_pressure = (kGamma-1.0)*std::max(
      mapped.gas_internal_energy, kPressureFloor/(kGamma-1.0));

  Row row = {};
  row.radius = ray_lower_code+0.5*kSurfaceL2CellWidth;
  row.density = mapped.density;
  row.stellar_mass_fraction = mapped.stellar_density/mapped.density;
  row.stellar_volume_fraction = stellar_volume_fraction;
  row.old_gas_pressure = old_gas_pressure;
  row.new_gas_pressure = new_gas_pressure;
  row.old_radiation_energy = old_radiation_energy;
  row.new_radiation_energy = new_radiation_energy;
  const double center = (ray_lower_code+0.5*kSurfaceL2CellWidth)*kLengthUnit;
  const double transverse_center = 0.5*dx;
  const scf_map::State source = polar
      ? profile.Interpolate(std::sqrt(2.0)*transverse_center, center)
      : profile.Interpolate(std::hypot(center, transverse_center),
                            transverse_center);
  row.source_density = source.density;
  row.source_pressure = source.pressure;
  scf_map::AtmosphereState(atmosphere,
                           polar ? transverse_center : center,
                           transverse_center,
                           polar ? center : transverse_center,
                           &row.atmosphere_density,
                           &row.atmosphere_pressure);
  return row;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " SCF_PROFILE.csv ATMOSPHERE_DENSITY_G_CM3 "
              << "equator|polar\n";
    return EXIT_FAILURE;
  }
  const double atmosphere_density = std::strtod(argv[2], nullptr);
  if (!(atmosphere_density > 0.0)) {
    std::cerr << "atmosphere density must be positive\n";
    return EXIT_FAILURE;
  }
  const std::string direction = argv[3];
  if (direction != "equator" && direction != "polar") {
    std::cerr << "direction must be equator or polar\n";
    return EXIT_FAILURE;
  }
  const bool polar = direction == "polar";

  scf_map::AxisymmetricProfile profile;
  profile.Load(argv[1], kMeanMolecularWeight);
  scf_map::AtmosphereModel atmosphere = {};
  atmosphere.density_reference = atmosphere_density;
  atmosphere.pressure_reference = atmosphere_density
      *scf_map::kGasConstantCgs*kAtmosphereTemperature
      /kMeanMolecularWeight;
  atmosphere.density_floor = 1.0e-16;
  atmosphere.pressure_floor = kPressureFloor;
  atmosphere.sound_speed_squared =
      atmosphere.pressure_reference/atmosphere.density_reference;
  atmosphere.reference_radius = kAtmosphereReferenceRadius;
  atmosphere.monopole_gm = kGravity*(kGasMass+kCoreMass);
  atmosphere.isothermal_monopole = true;
  atmosphere.rotate = false;

  std::vector<Row> rows;
  for (double x = 825.0; x < 1100.0; x += kSurfaceL2CellWidth) {
    rows.push_back(MapCell(profile, atmosphere, x, polar));
  }

  std::cout << std::scientific << std::setprecision(9)
            << "# r_code rho_cgs stellar_mass_frac stellar_volume_frac "
            << "Pgas_old Pgas_pointwise Erad_old Erad_pointwise "
            << "pressure_accel_old pressure_accel_pointwise "
            << "source_rho source_pressure atmosphere_rho atmosphere_pressure\n";
  for (std::size_t i = 1; i+1 < rows.size(); ++i) {
    const Row& row = rows[i];
    const double dr = (rows[i+1].radius-rows[i-1].radius)*kLengthUnit;
    const double old_acceleration = -(rows[i+1].old_gas_pressure
        -rows[i-1].old_gas_pressure)/(dr*row.density);
    const double new_acceleration = -(rows[i+1].new_gas_pressure
        -rows[i-1].new_gas_pressure)/(dr*row.density);
    std::cout << row.radius << ' ' << row.density << ' '
              << row.stellar_mass_fraction << ' '
              << row.stellar_volume_fraction << ' '
              << row.old_gas_pressure << ' ' << row.new_gas_pressure << ' '
              << row.old_radiation_energy << ' '
              << row.new_radiation_energy << ' '
              << old_acceleration << ' ' << new_acceleration << ' '
              << row.source_density << ' ' << row.source_pressure << ' '
              << row.atmosphere_density << ' '
              << row.atmosphere_pressure << '\n';
  }
  return EXIT_SUCCESS;
}
