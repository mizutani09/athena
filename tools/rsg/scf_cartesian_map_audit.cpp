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

struct CellFields {
  double density;
  double pressure;
  double potential;
  double stellar_fraction;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 17) {
    std::cerr << "usage: scf_cartesian_map_audit PROFILE AUDIT_CSV SLICE_CSV "
                 "N HALF_WIDTH_CM QUADRATURE OMEGA GAMMA RHO_ATM T_ATM MU "
                 "EXPECTED_MASS EXPECTED_J EXPECTED_PI EXPECTED_T LABEL\n";
    return 1;
  }
  try {
    const std::string profile_file = argv[1];
    const std::string audit_file = argv[2];
    const std::string slice_file = argv[3];
    const int points = I(argv[4]);
    const double half_width = D(argv[5]);
    const int quadrature = I(argv[6]);
    const double omega = D(argv[7]);
    const double gamma = D(argv[8]);
    const double rho_atm = D(argv[9]);
    const double temperature_atm = D(argv[10]);
    const double mean_molecular_weight = D(argv[11]);
    const double expected_mass = D(argv[12]);
    const double expected_j = D(argv[13]);
    const double expected_pressure_integral = D(argv[14]);
    const double expected_kinetic = D(argv[15]);
    const std::string label = argv[16];
    if (points < 8 || !(half_width > 0.0) || !(gamma > 1.0)) {
      throw std::invalid_argument("invalid Cartesian audit controls");
    }
    const double pressure_atm = rho_atm*scf_map::kGasConstantCgs
        *temperature_atm/mean_molecular_weight;
    scf_map::AxisymmetricProfile profile;
    profile.Load(profile_file, mean_molecular_weight);
    const double dx = 2.0*half_width/points;
    const double volume = dx*dx*dx;
    std::vector<CellFields> fields(static_cast<std::size_t>(points)*points*points);
    double total_mass = 0.0;
    double stellar_mass = 0.0;
    double atmosphere_mass = 0.0;
    double exact_j = 0.0;
    double discrete_j = 0.0;
    double total_energy = 0.0;
    double internal_energy = 0.0;
    double kinetic_energy = 0.0;
    double radiative_mass = 0.0;
    double pressure_recovery_l1 = 0.0;
    double pressure_norm = 0.0;
    double density_center_l1 = 0.0;
    double density_norm = 0.0;
    double temperature_center_l1 = 0.0;
    double temperature_norm = 0.0;
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
          const scf_map::Conserved cell = scf_map::CellAverage(
              profile, xl, xu, yl, yu, zl, zu, quadrature,
              omega, gamma, rho_atm, pressure_atm, false);
          const scf_map::Conserved center = scf_map::Sample(
              profile, xc, yc, zc, omega, gamma,
              rho_atm, pressure_atm, false);
          total_mass += cell.density*volume;
          stellar_mass += cell.stellar_density*volume;
          atmosphere_mass += cell.atmosphere_density*volume;
          exact_j += cell.angular_momentum_z*volume;
          discrete_j += (xc*cell.momentum_y-yc*cell.momentum_x)*volume;
          total_energy += cell.total_energy*volume;
          internal_energy += cell.internal_energy*volume;
          kinetic_energy += cell.kinetic_energy*volume;
          radiative_mass += cell.radiative_density*volume;
          const double resolved_kinetic = 0.5*(
              cell.momentum_x*cell.momentum_x
              +cell.momentum_y*cell.momentum_y
              +cell.momentum_z*cell.momentum_z)/cell.density;
          const double recovered_pressure = (gamma-1.0)
              *(cell.total_energy-resolved_kinetic);
          pressure_recovery_l1 += std::abs(recovered_pressure-cell.pressure)*volume;
          pressure_norm += cell.pressure*volume;
          density_center_l1 += std::abs(center.density-cell.density)*volume;
          density_norm += cell.density*volume;
          const double mapped_temperature = cell.density_temperature/cell.density;
          const double center_temperature = center.density_temperature/center.density;
          temperature_center_l1 += std::abs(
              mapped_temperature-center_temperature)*cell.density*volume;
          temperature_norm += mapped_temperature*cell.density*volume;
          const scf_map::State source_center = profile.Interpolate(
              std::hypot(xc,yc), zc);
          CellFields stored;
          stored.density = cell.density;
          stored.pressure = cell.pressure;
          stored.potential = source_center.gas_potential
              +source_center.core_potential;
          stored.stellar_fraction = cell.stellar_density/cell.density;
          fields[Index(i,j,k,points)] = stored;
        }
      }
    }

    double force_residual_squared = 0.0;
    double force_scale_squared = 0.0;
    double force_max_relative = 0.0;
    for (int k = 1; k < points-1; ++k) {
      for (int j = 1; j < points-1; ++j) {
        const double y = -half_width+(j+0.5)*dx;
        for (int i = 1; i < points-1; ++i) {
          const double x = -half_width+(i+0.5)*dx;
          const CellFields& center = fields[Index(i,j,k,points)];
          if (center.stellar_fraction < 0.999 || center.density < rho_atm*100.0) {
            continue;
          }
          const double dpx = (fields[Index(i+1,j,k,points)].pressure
              -fields[Index(i-1,j,k,points)].pressure)/(2.0*dx);
          const double dpy = (fields[Index(i,j+1,k,points)].pressure
              -fields[Index(i,j-1,k,points)].pressure)/(2.0*dx);
          const double dpz = (fields[Index(i,j,k+1,points)].pressure
              -fields[Index(i,j,k-1,points)].pressure)/(2.0*dx);
          const double dphix = (fields[Index(i+1,j,k,points)].potential
              -fields[Index(i-1,j,k,points)].potential)/(2.0*dx);
          const double dphiy = (fields[Index(i,j+1,k,points)].potential
              -fields[Index(i,j-1,k,points)].potential)/(2.0*dx);
          const double dphiz = (fields[Index(i,j,k+1,points)].potential
              -fields[Index(i,j,k-1,points)].potential)/(2.0*dx);
          const double gx = center.density*(dphix-omega*omega*x);
          const double gy = center.density*(dphiy-omega*omega*y);
          const double gz = center.density*dphiz;
          const double residual = std::sqrt(
              (dpx+gx)*(dpx+gx)+(dpy+gy)*(dpy+gy)+(dpz+gz)*(dpz+gz));
          const double scale = std::sqrt(dpx*dpx+dpy*dpy+dpz*dpz)
              +std::sqrt(gx*gx+gy*gy+gz*gz);
          if (scale > 0.0) {
            force_residual_squared += residual*residual*volume;
            force_scale_squared += scale*scale*volume;
            force_max_relative = std::max(force_max_relative, residual/scale);
          }
        }
      }
    }
    const double force_l2_relative = std::sqrt(
        force_residual_squared/force_scale_squared);

    std::ofstream audit(audit_file.c_str());
    if (!audit) throw std::runtime_error("cannot open audit output");
    audit << std::scientific << std::setprecision(16)
          << "label,points,quadrature,dx_cm,total_mass_g,stellar_mass_g,"
             "atmosphere_mass_g,mass_relative_error,exact_Jz_g_cm2_s,"
             "discrete_Jz_g_cm2_s,exact_J_relative_error,"
             "discrete_J_relative_error,total_energy_erg,internal_energy_erg,"
             "kinetic_energy_erg,energy_relative_error,radiative_mass_g,"
             "pressure_recovery_l1_relative,density_center_l1_relative,"
             "temperature_center_l1_relative,force_l2_relative,"
             "force_max_relative\n"
          << label << ',' << points << ',' << quadrature << ',' << dx << ','
          << total_mass << ',' << stellar_mass << ',' << atmosphere_mass << ','
          << stellar_mass/expected_mass-1.0 << ',' << exact_j << ','
          << discrete_j << ',' << exact_j/expected_j-1.0 << ','
          << discrete_j/expected_j-1.0 << ',' << total_energy << ','
          << internal_energy << ',' << kinetic_energy << ','
          << (internal_energy+kinetic_energy)
              /(expected_pressure_integral/(gamma-1.0)+expected_kinetic)-1.0
          << ',' << radiative_mass << ','
          << pressure_recovery_l1/pressure_norm << ','
          << density_center_l1/density_norm << ','
          << temperature_center_l1/temperature_norm << ','
          << force_l2_relative << ',' << force_max_relative << '\n';

    std::ofstream slice(slice_file.c_str());
    if (!slice) throw std::runtime_error("cannot open slice output");
    slice << std::scientific << std::setprecision(16)
          << "x_cm,y_cm,density_g_cm3,pressure_dyn_cm2,stellar_fraction,"
             "radiative_fraction\n";
    const int k = points/2;
    for (int j = 0; j < points; ++j) {
      const double y = -half_width+(j+0.5)*dx;
      for (int i = 0; i < points; ++i) {
        const double x = -half_width+(i+0.5)*dx;
        const CellFields& cell = fields[Index(i,j,k,points)];
        const scf_map::State source = profile.Interpolate(std::hypot(x,y),0.0);
        slice << x << ',' << y << ',' << cell.density << ',' << cell.pressure
              << ',' << cell.stellar_fraction << ','
              << source.radiative_fraction*cell.stellar_fraction << '\n';
      }
    }
    std::cout << "points=" << points
              << " mass_error=" << stellar_mass/expected_mass-1.0
              << " J_error=" << discrete_j/expected_j-1.0
              << " energy_error=" << (internal_energy+kinetic_energy)
                  /(expected_pressure_integral/(gamma-1.0)+expected_kinetic)-1.0
              << " force_l2=" << force_l2_relative << '\n';
  } catch (const std::exception& error) {
    std::cerr << "scf_cartesian_map_audit: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
