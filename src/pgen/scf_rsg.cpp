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
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "scf_axisymmetric_profile.hpp"
#include "scf_core_softening.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "scf_rsg is a hydrodynamic problem and must be built without -b"
#endif
#if !NON_BAROTROPIC_EOS
#error "scf_rsg requires an energy equation (--eos=adiabatic)"
#endif
#if NSCALARS < 2
#error "scf_rsg requires --nscalars=2 (stellar and radiative-layer tracers)"
#endif
#if SELF_GRAVITY_ENABLED != 2
#error "scf_rsg requires multigrid self-gravity (--grav=mg)"
#endif

namespace {

scf_map::AxisymmetricProfile profile;
Real rho_unit, pressure_unit, length_unit, velocity_unit;
Real time_unit, mass_unit, gravity_code, four_pi_gravity_code;
Real angular_velocity, angular_velocity_code, gamma_gas;
Real atmosphere_density, atmosphere_pressure, atmosphere_sound_speed_squared;
Real atmosphere_density_floor, atmosphere_pressure_floor;
Real atmosphere_reference_radius, atmosphere_monopole_gm;
Real mean_molecular_weight, core_mass, core_support_radius;
Real core_mass_code, core_support_radius_code, core_gm_code;
int quadrature_order;
bool rotate_atmosphere, isothermal_atmosphere;
scf_map::AtmosphereModel atmosphere_model;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::stringstream error;
    error << "scf_rsg: " << message;
    ATHENA_ERROR(error);
  }
}

Real MonopolePotentialCode(Real x, Real y, Real z) {
  const Real radius = std::sqrt(x*x+y*y+z*z);
  return radius > 0.0 ? -atmosphere_monopole_gm
      /(velocity_unit*velocity_unit*radius*length_unit) : 0.0;
}

Real CorePotentialCode(Real x, Real y, Real z) {
  const Real radius = std::sqrt(x*x+y*y+z*z);
  return scf_core::Potential(radius, core_gm_code, core_support_radius_code);
}

void CoreAccelerationCode(Coordinates* coordinates, int k, int j, int i,
                          Real* ax, Real* ay, Real* az) {
  const Real x = coordinates->x1v(i);
  const Real y = coordinates->x2v(j);
  const Real z = coordinates->x3v(k);
  *ax = -(CorePotentialCode(coordinates->x1v(i+1), y, z)
          -CorePotentialCode(coordinates->x1v(i-1), y, z))
      /(coordinates->x1v(i+1)-coordinates->x1v(i-1));
  *ay = -(CorePotentialCode(x, coordinates->x2v(j+1), z)
          -CorePotentialCode(x, coordinates->x2v(j-1), z))
      /(coordinates->x2v(j+1)-coordinates->x2v(j-1));
  *az = -(CorePotentialCode(x, y, coordinates->x3v(k+1))
          -CorePotentialCode(x, y, coordinates->x3v(k-1)))
      /(coordinates->x3v(k+1)-coordinates->x3v(k-1));
}

void AtmosphereGravityMask(AthenaArray<Real>& source,
                           int is, int ie, int js, int je,
                           int ks, int ke, const MGCoordinates& coordinates) {
  for (int k = ks; k <= ke; ++k) {
    const Real z = coordinates.x3v(k)*length_unit;
    for (int j = js; j <= je; ++j) {
      const Real y = coordinates.x2v(j)*length_unit;
      for (int i = is; i <= ie; ++i) {
        const Real x = coordinates.x1v(i)*length_unit;
        double density = 0.0, pressure = 0.0;
        scf_map::AtmosphereState(atmosphere_model, x, y, z,
                                 &density, &pressure);
        source(k,j,i) = std::max(
            static_cast<Real>(0.0),
            source(k,j,i)-four_pi_gravity_code*density/rho_unit);
      }
    }
  }
}

void CoreGravitySource(MeshBlock* block, const Real, const Real dt,
                       const AthenaArray<Real>& primitive,
                       const AthenaArray<Real>&, const AthenaArray<Real>&,
                       AthenaArray<Real>& conserved, AthenaArray<Real>&) {
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        Real ax, ay, az;
        CoreAccelerationCode(block->pcoord, k, j, i, &ax, &ay, &az);
        const Real density = primitive(IDN,k,j,i);
        const Real dpx = dt*density*ax;
        const Real dpy = dt*density*ay;
        const Real dpz = dt*density*az;
        conserved(IM1,k,j,i) += dpx;
        conserved(IM2,k,j,i) += dpy;
        conserved(IM3,k,j,i) += dpz;
        // Midpoint kick: the external force changes kinetic, not internal,
        // energy when hydrodynamic fluxes are absent.
        conserved(IEN,k,j,i) += primitive(IVX,k,j,i)*dpx
            +primitive(IVY,k,j,i)*dpy+primitive(IVZ,k,j,i)*dpz
            +0.5*(dpx*dpx+dpy*dpy+dpz*dpz)/density;
      }
    }
  }
}

void HydrostaticOpenBoundary(MeshBlock* block, Coordinates* coordinates,
                             AthenaArray<Real>& primitive,
                             int axis, bool inner,
                             int is, int ie, int js, int je,
                             int ks, int ke, int ngh) {
  int lower[3] = {is, js, ks};
  int upper[3] = {ie, je, ke};
  const int reference = inner ? lower[axis] : upper[axis];
  lower[axis] = inner ? reference-ngh : reference+1;
  upper[axis] = inner ? reference-1 : reference+ngh;
  for (int k = lower[2]; k <= upper[2]; ++k) {
    for (int j = lower[1]; j <= upper[1]; ++j) {
      for (int i = lower[0]; i <= upper[0]; ++i) {
        int q[3] = {i, j, k};
        q[axis] = reference;
        const Real x = coordinates->x1v(i);
        const Real y = coordinates->x2v(j);
        const Real z = coordinates->x3v(k);
        const Real xr = coordinates->x1v(q[0]);
        const Real yr = coordinates->x2v(q[1]);
        const Real zr = coordinates->x3v(q[2]);
        const Real delta_phi = MonopolePotentialCode(x,y,z)
            -MonopolePotentialCode(xr,yr,zr);
        Real factor = 1.0;
        if (isothermal_atmosphere) {
          factor = std::exp(std::max(
              static_cast<Real>(-80.0), std::min(static_cast<Real>(0.0),
              -delta_phi/(atmosphere_sound_speed_squared
                          /(velocity_unit*velocity_unit)))));
        }
        for (int variable = 0; variable < NHYDRO; ++variable) {
          primitive(variable,k,j,i) = primitive(variable,q[2],q[1],q[0]);
        }
        const Real density_reference = primitive(IDN,q[2],q[1],q[0]);
        primitive(IDN,k,j,i) = std::max(
            atmosphere_density_floor/rho_unit, density_reference*factor);
        primitive(IPR,k,j,i) = std::max(
            atmosphere_pressure_floor/pressure_unit,
            primitive(IPR,q[2],q[1],q[0])
                *primitive(IDN,k,j,i)/density_reference);
        const int normal_velocity = IVX+axis;
        if ((inner && primitive(normal_velocity,k,j,i) > 0.0)
            || (!inner && primitive(normal_velocity,k,j,i) < 0.0)) {
          primitive(normal_velocity,k,j,i) = 0.0;
        }
        for (int scalar = 0; scalar < NSCALARS; ++scalar) {
          block->pscalars->r(scalar,k,j,i) = 0.0;
        }
      }
    }
  }
}

#define SCF_HYDRO_BOUNDARY(NAME, AXIS, INNER) \
void NAME(MeshBlock* block, Coordinates* coordinates, AthenaArray<Real>& primitive, \
          FaceField&, Real, Real, int is, int ie, int js, int je, \
          int ks, int ke, int ngh) { \
  HydrostaticOpenBoundary(block, coordinates, primitive, AXIS, INNER, \
                          is, ie, js, je, ks, ke, ngh); \
}
SCF_HYDRO_BOUNDARY(InnerX1, 0, true)
SCF_HYDRO_BOUNDARY(OuterX1, 0, false)
SCF_HYDRO_BOUNDARY(InnerX2, 1, true)
SCF_HYDRO_BOUNDARY(OuterX2, 1, false)
SCF_HYDRO_BOUNDARY(InnerX3, 2, true)
SCF_HYDRO_BOUNDARY(OuterX3, 2, false)
#undef SCF_HYDRO_BOUNDARY

void UpdateGravityDiagnostics(MeshBlock* block) {
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const Real density = block->phydro->w(IDN,k,j,i);
        const Real dpx = (block->phydro->w(IPR,k,j,i+1)
                         -block->phydro->w(IPR,k,j,i-1))
            /(block->pcoord->x1v(i+1)-block->pcoord->x1v(i-1));
        const Real dpy = (block->phydro->w(IPR,k,j+1,i)
                         -block->phydro->w(IPR,k,j-1,i))
            /(block->pcoord->x2v(j+1)-block->pcoord->x2v(j-1));
        const Real dpz = (block->phydro->w(IPR,k+1,j,i)
                         -block->phydro->w(IPR,k-1,j,i))
            /(block->pcoord->x3v(k+1)-block->pcoord->x3v(k-1));
        const Real apx = -dpx/density;
        const Real apy = -dpy/density;
        const Real apz = -dpz/density;
        const Real asx = -(block->pgrav->phi(k,j,i+1)
                          -block->pgrav->phi(k,j,i-1))
            /(block->pcoord->x1v(i+1)-block->pcoord->x1v(i-1));
        const Real asy = -(block->pgrav->phi(k,j+1,i)
                          -block->pgrav->phi(k,j-1,i))
            /(block->pcoord->x2v(j+1)-block->pcoord->x2v(j-1));
        const Real asz = -(block->pgrav->phi(k+1,j,i)
                          -block->pgrav->phi(k-1,j,i))
            /(block->pcoord->x3v(k+1)-block->pcoord->x3v(k-1));
        Real acx, acy, acz;
        CoreAccelerationCode(block->pcoord, k, j, i, &acx, &acy, &acz);
        const Real x = block->pcoord->x1v(i);
        const Real y = block->pcoord->x2v(j);
        const Real z = block->pcoord->x3v(k);
        const Real centrifugal_x = angular_velocity_code*angular_velocity_code*x;
        const Real centrifugal_y = angular_velocity_code*angular_velocity_code*y;
        const Real residual_x = apx+asx+acx+centrifugal_x;
        const Real residual_y = apy+asy+acy+centrifugal_y;
        const Real residual_z = apz+asz+acz;
        const Real pressure_acceleration = std::sqrt(apx*apx+apy*apy+apz*apz);
        const Real gravity_acceleration = std::sqrt(
            SQR(asx+acx)+SQR(asy+acy)+SQR(asz+acz));
        const Real centrifugal_acceleration = std::sqrt(
            centrifugal_x*centrifugal_x+centrifugal_y*centrifugal_y);
        const scf_map::State source = profile.Interpolate(
            std::hypot(x,y)*length_unit, z*length_unit);
        block->user_out_var(3,k,j,i) = block->pgrav->phi(k,j,i);
        block->user_out_var(4,k,j,i) = CorePotentialCode(x,y,z);
        block->user_out_var(5,k,j,i) = source.gas_potential
            /(velocity_unit*velocity_unit);
        block->user_out_var(6,k,j,i) = block->pgrav->phi(k,j,i)
            -block->user_out_var(5,k,j,i);
        block->user_out_var(7,k,j,i) = std::sqrt(
            residual_x*residual_x+residual_y*residual_y+residual_z*residual_z);
        block->user_out_var(8,k,j,i) = pressure_acceleration+gravity_acceleration
            +centrifugal_acceleration;
        block->user_out_var(9,k,j,i) = pressure_acceleration;
        block->user_out_var(10,k,j,i) = gravity_acceleration;
        block->user_out_var(11,k,j,i) = density*(block->phydro->w(IVX,k,j,i)*acx
            +block->phydro->w(IVY,k,j,i)*acy+block->phydro->w(IVZ,k,j,i)*acz);
        block->user_out_var(12,k,j,i) = density*(x*(asy+acy)-y*(asx+acx));
      }
    }
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

Real HistoryGravityDiagnostic(MeshBlock* block, int output) {
  if (output == 4) UpdateGravityDiagnostics(block);
  Real total = 0.0;
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const Real volume = block->pcoord->GetCellVolume(k,j,i);
        const Real stellar_density = block->pscalars->s(0,k,j,i);
        const Real density = block->phydro->u(IDN,k,j,i);
        if (output == 4) total += stellar_density
            *SQR(block->user_out_var(6,k,j,i))*volume;
        if (output == 5) total += stellar_density
            *SQR(block->user_out_var(5,k,j,i))*volume;
        if (output == 6) total += stellar_density
            *SQR(block->user_out_var(7,k,j,i))*volume;
        if (output == 7) total += stellar_density
            *SQR(block->user_out_var(8,k,j,i))*volume;
        if (output == 8) total += block->user_out_var(11,k,j,i)*volume;
        if (output == 9) total += block->user_out_var(12,k,j,i)*volume;
        if (output == 10) total += (density-stellar_density)*volume
            *rho_unit*std::pow(length_unit,3);
        if (output == 11 && stellar_density/density > 0.999
            && block->user_out_var(8,k,j,i) > 0.0) {
          total = std::max(total, block->user_out_var(7,k,j,i)
                                    /block->user_out_var(8,k,j,i));
        }
        if (output == 12) total += stellar_density
            *block->user_out_var(4,k,j,i)*volume;
        if (output == 13) total += 0.5*stellar_density
            *block->user_out_var(3,k,j,i)*volume;
      }
    }
  }
  if (output == 8) {
    return total*pressure_unit*std::pow(length_unit,3)/time_unit;
  }
  if (output == 9) {
    return total*pressure_unit*std::pow(length_unit,3);
  }
  if (output == 12 || output == 13) {
    return total*pressure_unit*std::pow(length_unit,3);
  }
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
  const std::string atmosphere_mode = pin->GetOrAddString(
      "problem", "atmosphere_mode", "isothermal_monopole");
  isothermal_atmosphere = atmosphere_mode == "isothermal_monopole";
  Require(isothermal_atmosphere || atmosphere_mode == "uniform",
          "atmosphere_mode must be isothermal_monopole or uniform");
  core_mass = pin->GetReal("problem", "core_mass_g");
  core_support_radius = pin->GetReal("problem", "core_support_radius_cm");
  const Real gas_mass = pin->GetReal("problem", "gas_mass_g");
  atmosphere_reference_radius = pin->GetReal(
      "problem", "atmosphere_reference_radius_cm");
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
  Require(core_mass > 0.0 && core_support_radius > 0.0 && gas_mass > 0.0
          && atmosphere_reference_radius > 0.0,
          "core, gas, and atmosphere reference parameters must be positive");
  Require(quadrature_order >= 1 && quadrature_order <= 3,
          "cell_quadrature_order must be 1, 2, or 3");
  atmosphere_pressure = atmosphere_density*scf_map::kGasConstantCgs
      *atmosphere_temperature/mean_molecular_weight;
  atmosphere_sound_speed_squared = atmosphere_pressure/atmosphere_density;
  velocity_unit = std::sqrt(pressure_unit/rho_unit);
  time_unit = length_unit/velocity_unit;
  mass_unit = rho_unit*std::pow(length_unit,3);
  gravity_code = 6.67430e-8*rho_unit*time_unit*time_unit;
  four_pi_gravity_code = 4.0*PI*gravity_code;
  angular_velocity_code = angular_velocity*time_unit;
  core_mass_code = core_mass/mass_unit;
  core_support_radius_code = core_support_radius/length_unit;
  core_gm_code = gravity_code*core_mass_code;
  atmosphere_monopole_gm = 6.67430e-8*(gas_mass+core_mass);
  atmosphere_density_floor = pin->GetReal("hydro", "dfloor")*rho_unit;
  atmosphere_pressure_floor = pin->GetReal("hydro", "pfloor")*pressure_unit;
  Require(atmosphere_density_floor > 0.0
          && atmosphere_density_floor <= atmosphere_density,
          "hydro density floor must be positive and no larger than atmosphere rho");
  Require(atmosphere_pressure_floor > 0.0
          && atmosphere_pressure_floor <= atmosphere_pressure,
          "hydro pressure floor must be positive and no larger than atmosphere pressure");
  atmosphere_model = {};
  atmosphere_model.density_reference = atmosphere_density;
  atmosphere_model.pressure_reference = atmosphere_pressure;
  atmosphere_model.density_floor = atmosphere_density_floor;
  atmosphere_model.pressure_floor = atmosphere_pressure_floor;
  atmosphere_model.sound_speed_squared = atmosphere_sound_speed_squared;
  atmosphere_model.reference_radius = atmosphere_reference_radius;
  atmosphere_model.monopole_gm = atmosphere_monopole_gm;
  atmosphere_model.isothermal_monopole = isothermal_atmosphere;
  atmosphere_model.rotate = rotate_atmosphere;
  profile.Load(filename, mean_molecular_weight);

  SetGravitationalConstant(gravity_code);
  EnrollUserMGGravitySourceMaskFunction(AtmosphereGravityMask);
  EnrollUserExplicitSourceFunction(CoreGravitySource);
  const char* boundary_names[6] = {
      "ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"};
  BValFunc boundary_functions[6] = {
      InnerX1, OuterX1, InnerX2, OuterX2, InnerX3, OuterX3};
  for (int face = 0; face < 6; ++face) {
    Require(pin->GetString("mesh", boundary_names[face]) == "user",
            "all hydrodynamic boundaries must be user");
    Require(pin->GetString("gravity", boundary_names[face]) == "multipole",
            "all gravity boundaries must be isolated multipole");
    EnrollUserBoundaryFunction(static_cast<BoundaryFace>(face),
                               boundary_functions[face]);
  }
  Require(pin->GetOrAddBoolean("gravity", "fill_ghost", true),
          "gravity/fill_ghost must be true for force diagnostics");

  AllocateUserHistoryOutput(14);
  EnrollUserHistoryOutput(0, HistoryMass, "mapped_mass_g");
  EnrollUserHistoryOutput(1, HistoryStellarMass, "mapped_stellar_mass_g");
  EnrollUserHistoryOutput(2, HistoryAngularMomentum, "mapped_Jz_g_cm2_s");
  EnrollUserHistoryOutput(3, HistoryTotalEnergy, "mapped_E_erg");
  EnrollUserHistoryOutput(4, HistoryGravityDiagnostic, "phi_error2");
  EnrollUserHistoryOutput(5, HistoryGravityDiagnostic, "phi_norm2");
  EnrollUserHistoryOutput(6, HistoryGravityDiagnostic, "force_error2");
  EnrollUserHistoryOutput(7, HistoryGravityDiagnostic, "force_norm2");
  EnrollUserHistoryOutput(8, HistoryGravityDiagnostic, "core_power_erg_s");
  EnrollUserHistoryOutput(9, HistoryGravityDiagnostic, "gravity_torque_erg");
  EnrollUserHistoryOutput(10, HistoryGravityDiagnostic, "atmosphere_mass_g");
  EnrollUserHistoryOutput(11, HistoryGravityDiagnostic, "force_relative_max",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(12, HistoryGravityDiagnostic,
                          "core_potential_energy_erg");
  EnrollUserHistoryOutput(13, HistoryGravityDiagnostic,
                          "self_potential_energy_erg");
  if (Globals::my_rank == 0) {
    std::cout << std::scientific << std::setprecision(8)
              << "scf_rsg profile=" << filename
              << " nr=" << profile.radial_points()
              << " nmu=" << profile.angular_points()
              << " omega_rad_s=" << angular_velocity
              << " quadrature_order=" << quadrature_order
              << " atmosphere_rho=" << atmosphere_density
              << " atmosphere_pressure=" << atmosphere_pressure
              << " atmosphere_mode=" << atmosphere_mode
              << " core_mass_g=" << core_mass
              << " core_support_radius_cm=" << core_support_radius
              << " G_code=" << gravity_code
              << " frame=inertial\n";
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput*) {
  AllocateUserOutputVariables(13);
  SetUserOutputVariableName(0, "T_source");
  SetUserOutputVariableName(1, "stellar_frac");
  SetUserOutputVariableName(2, "radiative_frac");
  SetUserOutputVariableName(3, "Phi_self_mg");
  SetUserOutputVariableName(4, "Phi_core");
  SetUserOutputVariableName(5, "Phi_self_source");
  SetUserOutputVariableName(6, "Phi_self_error");
  SetUserOutputVariableName(7, "force_residual");
  SetUserOutputVariableName(8, "force_scale");
  SetUserOutputVariableName(9, "pressure_accel");
  SetUserOutputVariableName(10, "gravity_accel");
  SetUserOutputVariableName(11, "core_work_density");
  SetUserOutputVariableName(12, "gravity_torque_density");
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
            atmosphere_model);
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
        user_out_var(4,k,j,i) = center.core_potential
            /(velocity_unit*velocity_unit);
        user_out_var(5,k,j,i) = center.gas_potential
            /(velocity_unit*velocity_unit);
      }
    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput*) {
  UpdateGravityDiagnostics(this);
}
