//========================================================================================
//! \file scf_rsg.cpp
//! \brief Conservative Cartesian mapping of an axisymmetric rotating SCF RSG.
//!
//! The input table is sampled in physical cgs coordinates.  Each Athena++ cell stores
//! tensor-Gauss volume averages of rho, rho*v, and total HD energy.  The inertial-frame
//! velocity is v=(-Omega*y, Omega*x, 0).  This is deliberately independent of the
//! existing preCE and NR-FLD problem generators.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "scf_axisymmetric_profile.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "scf_rsg is a hydrodynamic problem and must be built without -b"
#endif
#if !NON_BAROTROPIC_EOS
#error "scf_rsg requires an energy equation (--eos=adiabatic)"
#endif
#if NSCALARS < 2
#error "scf_rsg requires --nscalars=2 (stellar and radiative-layer tracers)"
#endif

namespace {

scf_map::AxisymmetricProfile profile;
Real rho_unit, pressure_unit, length_unit, velocity_unit;
Real angular_velocity, gamma_gas, atmosphere_density, atmosphere_pressure;
Real mean_molecular_weight;
int quadrature_order;
bool rotate_atmosphere;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::stringstream error;
    error << "scf_rsg: " << message;
    ATHENA_ERROR(error);
  }
}

Real HistoryMass(MeshBlock* block, int) {
  Real total = 0.0;
  const Real scale = rho_unit*std::pow(length_unit, 3);
  for (int k = block->ks; k <= block->ke; ++k)
    for (int j = block->js; j <= block->je; ++j)
      for (int i = block->is; i <= block->ie; ++i)
        total += block->phydro->u(IDN,k,j,i)
            *block->pcoord->GetCellVolume(k,j,i)*scale;
  return total;
}

Real HistoryStellarMass(MeshBlock* block, int) {
  Real total = 0.0;
#if NSCALARS > 0
  const Real scale = rho_unit*std::pow(length_unit, 3);
  for (int k = block->ks; k <= block->ke; ++k)
    for (int j = block->js; j <= block->je; ++j)
      for (int i = block->is; i <= block->ie; ++i)
        total += block->pscalars->s(0,k,j,i)
            *block->pcoord->GetCellVolume(k,j,i)*scale;
#endif
  return total;
}

Real HistoryAngularMomentum(MeshBlock* block, int) {
  Real total = 0.0;
  const Real scale = rho_unit*velocity_unit*std::pow(length_unit, 4);
  for (int k = block->ks; k <= block->ke; ++k)
    for (int j = block->js; j <= block->je; ++j)
      for (int i = block->is; i <= block->ie; ++i) {
        const Real x = block->pcoord->x1v(i);
        const Real y = block->pcoord->x2v(j);
        total += (x*block->phydro->u(IM2,k,j,i)
                 -y*block->phydro->u(IM1,k,j,i))
            *block->pcoord->GetCellVolume(k,j,i)*scale;
      }
  return total;
}

Real HistoryTotalEnergy(MeshBlock* block, int) {
  Real total = 0.0;
  const Real scale = pressure_unit*std::pow(length_unit, 3);
  for (int k = block->ks; k <= block->ke; ++k)
    for (int j = block->js; j <= block->je; ++j)
      for (int i = block->is; i <= block->ie; ++i)
        total += block->phydro->u(IEN,k,j,i)
            *block->pcoord->GetCellVolume(k,j,i)*scale;
  return total;
}

}  // namespace

void Mesh::InitUserMeshData(ParameterInput* pin) {
  Require(mesh_size.nx2 > 1 && mesh_size.nx3 > 1,
          "a three-dimensional Cartesian mesh is required");
  const std::string frame = pin->GetOrAddString("problem", "frame", "inertial");
  Require(frame == "inertial",
          "only problem/frame=inertial is implemented");
  const std::string filename = pin->GetString("problem", "scf_profile_file");
  rho_unit = pin->GetReal("hydro", "rho_unit");
  pressure_unit = pin->GetReal("hydro", "egas_unit");
  length_unit = pin->GetReal("hydro", "leng_unit");
  gamma_gas = pin->GetReal("hydro", "gamma");
  mean_molecular_weight = pin->GetReal("hydro", "mu");
  angular_velocity = pin->GetReal("problem", "angular_velocity_rad_s");
  atmosphere_density = pin->GetReal("problem", "atmosphere_density_g_cm3");
  const Real atmosphere_temperature =
      pin->GetReal("problem", "atmosphere_temperature_k");
  quadrature_order = pin->GetOrAddInteger("problem", "cell_quadrature_order", 2);
  rotate_atmosphere = pin->GetOrAddBoolean(
      "problem", "rotate_atmosphere", false);
  Require(rho_unit > 0.0 && pressure_unit > 0.0 && length_unit > 0.0,
          "hydro units must be positive");
  Require(gamma_gas > 1.0 && mean_molecular_weight > 0.0,
          "gamma and mean molecular weight must be valid");
  Require(angular_velocity >= 0.0 && atmosphere_density > 0.0
          && atmosphere_temperature > 0.0,
          "rotation and atmosphere parameters must be nonnegative/positive");
  Require(quadrature_order >= 1 && quadrature_order <= 3,
          "cell_quadrature_order must be 1, 2, or 3");
  atmosphere_pressure = atmosphere_density*scf_map::kGasConstantCgs
      *atmosphere_temperature/mean_molecular_weight;
  velocity_unit = std::sqrt(pressure_unit/rho_unit);
  profile.Load(filename, mean_molecular_weight);

  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, HistoryMass, "mapped_mass_g");
  EnrollUserHistoryOutput(1, HistoryStellarMass, "mapped_stellar_mass_g");
  EnrollUserHistoryOutput(2, HistoryAngularMomentum, "mapped_Jz_g_cm2_s");
  EnrollUserHistoryOutput(3, HistoryTotalEnergy, "mapped_E_erg");
  if (Globals::my_rank == 0) {
    std::cout << std::scientific << std::setprecision(8)
              << "scf_rsg profile=" << filename
              << " nr=" << profile.radial_points()
              << " nmu=" << profile.angular_points()
              << " omega_rad_s=" << angular_velocity
              << " quadrature_order=" << quadrature_order
              << " atmosphere_rho=" << atmosphere_density
              << " atmosphere_pressure=" << atmosphere_pressure
              << " frame=inertial\n";
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput*) {
  AllocateUserOutputVariables(5);
  SetUserOutputVariableName(0, "T_source");
  SetUserOutputVariableName(1, "stellar_frac");
  SetUserOutputVariableName(2, "radiative_frac");
  SetUserOutputVariableName(3, "Phi_gas");
  SetUserOutputVariableName(4, "Phi_core");
}

void MeshBlock::ProblemGenerator(ParameterInput*) {
  const Real density_scale = rho_unit;
  const Real momentum_scale = rho_unit*velocity_unit;
  const Real energy_scale = pressure_unit;
  for (int k = ks; k <= ke; ++k) {
    const Real z_lower = pcoord->x3f(k)*length_unit;
    const Real z_upper = pcoord->x3f(k+1)*length_unit;
    for (int j = js; j <= je; ++j) {
      const Real y_lower = pcoord->x2f(j)*length_unit;
      const Real y_upper = pcoord->x2f(j+1)*length_unit;
      for (int i = is; i <= ie; ++i) {
        const Real x_lower = pcoord->x1f(i)*length_unit;
        const Real x_upper = pcoord->x1f(i+1)*length_unit;
        const scf_map::Conserved mapped = scf_map::CellAverage(
            profile, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper,
            quadrature_order, angular_velocity, gamma_gas,
            atmosphere_density, atmosphere_pressure, rotate_atmosphere);
        phydro->u(IDN,k,j,i) = mapped.density/density_scale;
        phydro->u(IM1,k,j,i) = mapped.momentum_x/momentum_scale;
        phydro->u(IM2,k,j,i) = mapped.momentum_y/momentum_scale;
        phydro->u(IM3,k,j,i) = mapped.momentum_z/momentum_scale;
        phydro->u(IEN,k,j,i) = mapped.total_energy/energy_scale;
#if NSCALARS > 0
        pscalars->s(0,k,j,i) = mapped.stellar_density/density_scale;
#endif
#if NSCALARS > 1
        pscalars->s(1,k,j,i) = mapped.radiative_density/density_scale;
#endif
        const Real rho = std::max(mapped.density, TINY_NUMBER);
        user_out_var(0,k,j,i) = mapped.density_temperature/rho;
        user_out_var(1,k,j,i) = mapped.stellar_density/rho;
        user_out_var(2,k,j,i) = mapped.radiative_density/rho;
        const Real x = pcoord->x1v(i)*length_unit;
        const Real y = pcoord->x2v(j)*length_unit;
        const Real z = pcoord->x3v(k)*length_unit;
        const scf_map::State center = profile.Interpolate(std::hypot(x,y), z);
        user_out_var(3,k,j,i) = center.gas_potential
            /(velocity_unit*velocity_unit);
        user_out_var(4,k,j,i) = center.core_potential
            /(velocity_unit*velocity_unit);
      }
    }
  }
}
