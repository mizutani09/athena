#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/pgen/scf_axisymmetric_profile.hpp"

namespace {

double D(const char* text) { return std::stod(text); }
int I(const char* text) { return std::stoi(text); }

std::size_t Index(int i, int j, int k, int points) {
  return static_cast<std::size_t>((k*points+j)*points+i);
}

struct Cell {
  scf_map::Conserved conserved;
  double potential;
};

struct Totals {
  double mass = 0.0;
  double stellar_mass = 0.0;
  double atmosphere_mass = 0.0;
  double exact_j = 0.0;
  double discrete_j = 0.0;
  double total_energy = 0.0;
  double internal_energy = 0.0;
  double kinetic_energy = 0.0;
  double radiative_mass = 0.0;
  double pressure_error = 0.0;
  double pressure_norm = 0.0;
  double density_error = 0.0;
  double density_norm = 0.0;
  double temperature_error = 0.0;
  double temperature_norm = 0.0;
  double force_error_squared = 0.0;
  double force_norm_squared = 0.0;
  double force_max = 0.0;
};

bool IsRefined(int i, int j, int k, int points, int level, int levels) {
  if (level >= levels) return false;
  const int lower = points/4;
  const int upper = 3*points/4;
  return i >= lower && i < upper
      && j >= lower && j < upper
      && k >= lower && k < upper;
}

void AccumulateLevel(const scf_map::AxisymmetricProfile& profile,
                     int points, double half_width, int level, int levels,
                     int quadrature, double omega, double gamma,
                     double rho_atm, double pressure_atm,
                     Totals* totals, std::ofstream* slice) {
  const double dx = 2.0*half_width/points;
  const double volume = dx*dx*dx;
  std::vector<Cell> cells(static_cast<std::size_t>(points)*points*points);
  for (int k = 0; k < points; ++k) {
    const double zl = -half_width+k*dx;
    const double zu = zl+dx;
    const double zc = 0.5*(zl+zu);
    for (int j = 0; j < points; ++j) {
      const double yl = -half_width+j*dx;
      const double yu = yl+dx;
      const double yc = 0.5*(yl+yu);
      for (int i = 0; i < points; ++i) {
        const double xl = -half_width+i*dx;
        const double xu = xl+dx;
        const double xc = 0.5*(xl+xu);
        Cell& cell = cells[Index(i,j,k,points)];
        cell.conserved = scf_map::CellAverage(
            profile, xl, xu, yl, yu, zl, zu, quadrature,
            omega, gamma, rho_atm, pressure_atm, false);
        const scf_map::State center_state = profile.Interpolate(
            std::hypot(xc,yc), zc);
        cell.potential = center_state.gas_potential+center_state.core_potential;
        if (IsRefined(i,j,k,points,level,levels)) continue;

        const scf_map::Conserved center = scf_map::Sample(
            profile, xc, yc, zc, omega, gamma, rho_atm, pressure_atm, false);
        const scf_map::Conserved& mapped = cell.conserved;
        totals->mass += mapped.density*volume;
        totals->stellar_mass += mapped.stellar_density*volume;
        totals->atmosphere_mass += mapped.atmosphere_density*volume;
        totals->exact_j += mapped.angular_momentum_z*volume;
        totals->discrete_j += (xc*mapped.momentum_y-yc*mapped.momentum_x)*volume;
        totals->total_energy += mapped.total_energy*volume;
        totals->internal_energy += mapped.internal_energy*volume;
        totals->kinetic_energy += mapped.kinetic_energy*volume;
        totals->radiative_mass += mapped.radiative_density*volume;
        const double resolved_kinetic = 0.5*(
            mapped.momentum_x*mapped.momentum_x
            +mapped.momentum_y*mapped.momentum_y
            +mapped.momentum_z*mapped.momentum_z)/mapped.density;
        const double recovered_pressure = (gamma-1.0)
            *(mapped.total_energy-resolved_kinetic);
        totals->pressure_error += std::abs(
            recovered_pressure-mapped.pressure)*volume;
        totals->pressure_norm += mapped.pressure*volume;
        totals->density_error += std::abs(center.density-mapped.density)*volume;
        totals->density_norm += mapped.density*volume;
        const double mapped_temperature = mapped.density_temperature/mapped.density;
        const double center_temperature = center.density_temperature/center.density;
        totals->temperature_error += std::abs(
            mapped_temperature-center_temperature)*mapped.density*volume;
        totals->temperature_norm += mapped_temperature*mapped.density*volume;
        if (slice != nullptr && k == points/2) {
          *slice << level << ',' << dx << ',' << xc << ',' << yc << ','
                 << mapped.density << ',' << mapped.pressure << ','
                 << mapped.stellar_density/mapped.density << ','
                 << mapped.radiative_density/mapped.density << '\n';
        }
      }
    }
  }

  for (int k = 1; k < points-1; ++k) {
    for (int j = 1; j < points-1; ++j) {
      const double y = -half_width+(j+0.5)*dx;
      for (int i = 1; i < points-1; ++i) {
        const double x = -half_width+(i+0.5)*dx;
        if (IsRefined(i,j,k,points,level,levels)
            || IsRefined(i-1,j,k,points,level,levels)
            || IsRefined(i+1,j,k,points,level,levels)
            || IsRefined(i,j-1,k,points,level,levels)
            || IsRefined(i,j+1,k,points,level,levels)
            || IsRefined(i,j,k-1,points,level,levels)
            || IsRefined(i,j,k+1,points,level,levels)) continue;
        const Cell& center = cells[Index(i,j,k,points)];
        const double stellar_fraction = center.conserved.stellar_density
            /center.conserved.density;
        if (stellar_fraction < 0.999
            || center.conserved.density < rho_atm*100.0) continue;
        const double dpx = (cells[Index(i+1,j,k,points)].conserved.pressure
            -cells[Index(i-1,j,k,points)].conserved.pressure)/(2.0*dx);
        const double dpy = (cells[Index(i,j+1,k,points)].conserved.pressure
            -cells[Index(i,j-1,k,points)].conserved.pressure)/(2.0*dx);
        const double dpz = (cells[Index(i,j,k+1,points)].conserved.pressure
            -cells[Index(i,j,k-1,points)].conserved.pressure)/(2.0*dx);
        const double dphix = (cells[Index(i+1,j,k,points)].potential
            -cells[Index(i-1,j,k,points)].potential)/(2.0*dx);
        const double dphiy = (cells[Index(i,j+1,k,points)].potential
            -cells[Index(i,j-1,k,points)].potential)/(2.0*dx);
        const double dphiz = (cells[Index(i,j,k+1,points)].potential
            -cells[Index(i,j,k-1,points)].potential)/(2.0*dx);
        const double rho = center.conserved.density;
        const double gx = rho*(dphix-omega*omega*x);
        const double gy = rho*(dphiy-omega*omega*y);
        const double gz = rho*dphiz;
        const double residual = std::sqrt(
            (dpx+gx)*(dpx+gx)+(dpy+gy)*(dpy+gy)+(dpz+gz)*(dpz+gz));
        const double scale = std::sqrt(dpx*dpx+dpy*dpy+dpz*dpz)
            +std::sqrt(gx*gx+gy*gy+gz*gz);
        if (scale > 0.0) {
          totals->force_error_squared += residual*residual*volume;
          totals->force_norm_squared += scale*scale*volume;
          totals->force_max = std::max(totals->force_max, residual/scale);
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 18) {
    std::cerr << "usage: scf_smr_map_audit PROFILE AUDIT_CSV SLICE_CSV N "
                 "HALF_WIDTH_CM LEVELS QUADRATURE OMEGA GAMMA RHO_ATM T_ATM MU "
                 "EXPECTED_MASS EXPECTED_J EXPECTED_PI EXPECTED_T LABEL\n";
    return 1;
  }
  try {
    const std::string profile_file = argv[1];
    const std::string audit_file = argv[2];
    const std::string slice_file = argv[3];
    const int points = I(argv[4]);
    const double root_half_width = D(argv[5]);
    const int levels = I(argv[6]);
    const int quadrature = I(argv[7]);
    const double omega = D(argv[8]);
    const double gamma = D(argv[9]);
    const double rho_atm = D(argv[10]);
    const double temperature_atm = D(argv[11]);
    const double mean_molecular_weight = D(argv[12]);
    const double expected_mass = D(argv[13]);
    const double expected_j = D(argv[14]);
    const double expected_pressure_integral = D(argv[15]);
    const double expected_kinetic = D(argv[16]);
    const std::string label = argv[17];
    if (points < 8 || points % 4 != 0 || levels < 0
        || !(root_half_width > 0.0) || !(gamma > 1.0)) {
      throw std::invalid_argument("invalid SMR audit controls");
    }
    const double pressure_atm = rho_atm*scf_map::kGasConstantCgs
        *temperature_atm/mean_molecular_weight;
    scf_map::AxisymmetricProfile profile;
    profile.Load(profile_file, mean_molecular_weight);
    Totals totals;
    std::ofstream slice(slice_file.c_str());
    if (!slice) throw std::runtime_error("cannot open slice output");
    slice << std::scientific << std::setprecision(16)
          << "level,dx_cm,x_cm,y_cm,density_g_cm3,pressure_dyn_cm2,"
             "stellar_fraction,radiative_fraction\n";
    for (int level = 0; level <= levels; ++level) {
      AccumulateLevel(profile, points, root_half_width/std::pow(2.0,level),
                      level, levels, quadrature, omega, gamma,
                      rho_atm, pressure_atm, &totals, &slice);
    }
    const double expected_energy = expected_pressure_integral/(gamma-1.0)
        +expected_kinetic;
    const double minimum_dx = 2.0*root_half_width/points/std::pow(2.0,levels);
    const double force_l2 = std::sqrt(
        totals.force_error_squared/totals.force_norm_squared);
    std::ofstream audit(audit_file.c_str());
    if (!audit) throw std::runtime_error("cannot open audit output");
    audit << std::scientific << std::setprecision(16)
          << "label,root_points,refinement_levels,quadrature,min_dx_cm,"
             "total_mass_g,stellar_mass_g,atmosphere_mass_g,mass_relative_error,"
             "exact_Jz_g_cm2_s,discrete_Jz_g_cm2_s,exact_J_relative_error,"
             "discrete_J_relative_error,total_energy_erg,internal_energy_erg,"
             "kinetic_energy_erg,energy_relative_error,radiative_mass_g,"
             "pressure_recovery_l1_relative,density_center_l1_relative,"
             "temperature_center_l1_relative,force_l2_relative,"
             "force_max_relative\n"
          << label << ',' << points << ',' << levels << ',' << quadrature << ','
          << minimum_dx << ',' << totals.mass << ',' << totals.stellar_mass << ','
          << totals.atmosphere_mass << ','
          << totals.stellar_mass/expected_mass-1.0 << ',' << totals.exact_j << ','
          << totals.discrete_j << ',' << totals.exact_j/expected_j-1.0 << ','
          << totals.discrete_j/expected_j-1.0 << ',' << totals.total_energy << ','
          << totals.internal_energy << ',' << totals.kinetic_energy << ','
          << (totals.internal_energy+totals.kinetic_energy)/expected_energy-1.0
          << ',' << totals.radiative_mass << ','
          << totals.pressure_error/totals.pressure_norm << ','
          << totals.density_error/totals.density_norm << ','
          << totals.temperature_error/totals.temperature_norm << ','
          << force_l2 << ',' << totals.force_max << '\n';
    std::cout << "root_points=" << points << " levels=" << levels
              << " min_dx_cm=" << minimum_dx
              << " mass_error=" << totals.stellar_mass/expected_mass-1.0
              << " J_error=" << totals.discrete_j/expected_j-1.0
              << " energy_error="
              << (totals.internal_energy+totals.kinetic_energy)/expected_energy-1.0
              << " force_l2=" << force_l2 << '\n';
  } catch (const std::exception& error) {
    std::cerr << "scf_smr_map_audit: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
