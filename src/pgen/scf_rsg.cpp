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
#include <limits>
#include <sstream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#if NRMGFLD_ENABLED
#include "../fld/fld.hpp"
#include "../fld/opacity_table.hpp"
#endif
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#if EOS_TABLE_ENABLED
#include "../utils/interp_table.hpp"
#endif
#include "scf_axisymmetric_profile.hpp"
#include "scf_core_softening.hpp"
#include "scf_fld_diagnostic.hpp"
#include "scf_luminosity.hpp"
#include "scf_relaxation.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "scf_rsg is a hydrodynamic problem and must be built without -b"
#endif
#if !NON_BAROTROPIC_EOS
#error "scf_rsg requires an energy equation (--eos=adiabatic)"
#endif
#if GENERAL_EOS && !EOS_TABLE_ENABLED
#error "scf_rsg general-EOS support currently requires --eos=general/eos_table"
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
Real temperature_unit, lte_temperature_min, lte_temperature_max;
Real atmosphere_density, atmosphere_pressure, atmosphere_sound_speed_squared;
Real atmosphere_density_floor, atmosphere_pressure_floor;
Real atmosphere_reference_radius, surface_polar_radius, atmosphere_monopole_gm;
Real surface_match_equatorial_radius, surface_match_polar_radius;
Real surface_mach_diagnostic_threshold;
Real surface_pressure_ratio_diagnostic_threshold;
Real relaxation_velocity_tau, relaxation_taper_start, relaxation_taper_end;
#if EOS_TABLE_ENABLED
Real eos_dens_pow;
#endif
Real mean_molecular_weight, core_mass, core_support_radius;
Real core_mass_code, core_support_radius_code, core_gm_code;
Real core_luminosity_lsun, core_luminosity_code;
Real luminosity_heating_radius, luminosity_heating_radius_code;
Real luminosity_kernel_normalization_code;
int quadrature_order;
bool rotate_atmosphere, isothermal_atmosphere, surface_refinement_use_tracer;
scf_map::AtmosphereModel atmosphere_model;
int root_logical_level, surface_refinement_level;
Real surface_tracer_lower, surface_tracer_upper, surface_density_code;
#if NRMGFLD_ENABLED
UserOpacityTable* opacity_table = nullptr;
Real external_radiation_energy_code, planck_taper_radius;
Real planck_taper_polar_radius, planck_taper_width;
Real marshak_alpha;
Real radiation_startup_dt_cap, radiation_startup_dt_end;
Real taper_exchange_diagnostic_time_threshold;
#endif
constexpr int kWenozHistoryStart = 29;
// WOR-177's compact softened core needs an L8/L9 resolution comparison on the
// root64 mesh.  Keep one history slot for every level from L0 through L9.
constexpr int kTrackedMeshLevels = 10;
constexpr int kWenozLevelHistoryStart = 33;
constexpr int kWenozFallbackMassHistory =
    kWenozLevelHistoryStart+kTrackedMeshLevels;
constexpr int kMeshBlockHistoryStart = kWenozFallbackMassHistory+1;
constexpr int kSurfaceMachHistoryStart =
    kMeshBlockHistoryStart+kTrackedMeshLevels;
constexpr int kSurfaceMachHistoryCount = 8;
constexpr int kBaseHistoryCount =
    kSurfaceMachHistoryStart+kSurfaceMachHistoryCount;
#if NRMGFLD_ENABLED
constexpr int kRadiationHistoryCount = 5;
#else
constexpr int kRadiationHistoryCount = 0;
#endif
#if EOS_TABLE_ENABLED
constexpr int kEosTableHistoryStart =
    kBaseHistoryCount+kRadiationHistoryCount;
constexpr int kEosTableHistoryCount = 4;
#else
constexpr int kEosTableHistoryCount = 0;
#endif
constexpr int kTotalHistoryCount =
    kBaseHistoryCount+kRadiationHistoryCount+kEosTableHistoryCount;
constexpr int kTaperRadiationHistoryStart = kTotalHistoryCount;
#if NRMGFLD_ENABLED
constexpr int kTaperRadiationHistoryCount = 9;
#else
constexpr int kTaperRadiationHistoryCount = 0;
#endif
constexpr int kFinalHistoryCount =
    kTotalHistoryCount+kTaperRadiationHistoryCount;
constexpr int kRelaxationHistoryStart = kFinalHistoryCount;
constexpr int kRelaxationHistoryCount = 4;
constexpr int kAllHistoryCount = kFinalHistoryCount+kRelaxationHistoryCount;
constexpr int kLuminosityHistory = kAllHistoryCount;
constexpr int kHistoryCount = kAllHistoryCount+1;

int SurfaceRefinementCondition(MeshBlock* block);

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::stringstream error;
    error << "scf_rsg: " << message;
    ATHENA_ERROR(error);
  }
}

Real AdiabaticSoundSpeedSquared(MeshBlock* block, Real density,
                                Real pressure) {
#if GENERAL_EOS
  return block->peos->AsqFromRhoP(density, pressure);
#else
  (void) block;
  return gamma_gas*pressure/density;
#endif
}

#if NRMGFLD_ENABLED
Real GasTemperatureCgs(MeshBlock* block, Real density_code,
                       Real pressure_code) {
#if GENERAL_EOS
  const Real gas_energy = block->peos->EgasFromRhoP(
      density_code, pressure_code);
  const Real temperature = block->peos->TempFromRhoEg(
      density_code, gas_energy)*temperature_unit;
  Require(std::isfinite(gas_energy) && gas_energy > 0.0
              && std::isfinite(temperature) && temperature > 0.0,
          "tabulated EOS returned an invalid gas temperature");
  return temperature;
#else
  (void) block;
  return pressure_code*pressure_unit*mean_molecular_weight
      /(density_code*rho_unit*scf_map::kGasConstantCgs);
#endif
}

struct LteSplit {
  Real temperature;
  Real gas_energy;
  Real gas_pressure;
  Real radiation_energy;
};

LteSplit SplitTotalPressureLte(MeshBlock* block, Real density_cgs,
                               Real total_pressure_cgs) {
  const Real density = density_cgs/rho_unit;
  const Real total_pressure = total_pressure_cgs/pressure_unit;
#if GENERAL_EOS
  auto evaluate = [block, density](Real gas_energy, LteSplit* state) {
    if (!std::isfinite(gas_energy) || gas_energy <= 0.0) return false;
    const Real gas_pressure = block->peos->PresFromRhoEg(density, gas_energy);
    const Real temperature_cgs = block->peos->TempFromRhoEg(
        density, gas_energy)*temperature_unit;
    const Real radiation_energy = scf_map::kRadiationConstantCgs
        *std::pow(temperature_cgs, 4)/pressure_unit;
    if (!std::isfinite(gas_pressure) || gas_pressure <= 0.0
        || !std::isfinite(temperature_cgs) || temperature_cgs <= 0.0
        || !std::isfinite(radiation_energy)) return false;
    state->temperature = temperature_cgs;
    state->gas_energy = gas_energy;
    state->gas_pressure = gas_pressure;
    state->radiation_energy = radiation_energy;
    return true;
  };

  Real lower_energy = block->peos->EgasFromRhoT(
      density, lte_temperature_min/temperature_unit);
  Real upper_energy = block->peos->EgasFromRhoP(density, total_pressure);
  LteSplit lower_state{}, upper_state{};
  bool bracketed = evaluate(lower_energy, &lower_state)
      && evaluate(upper_energy, &upper_state);
  Real lower_residual = bracketed ? lower_state.gas_pressure
      +lower_state.radiation_energy/3.0-total_pressure : 0.0;
  Real upper_residual = bracketed ? upper_state.gas_pressure
      +upper_state.radiation_energy/3.0-total_pressure : 0.0;
  bracketed = bracketed && lower_residual <= 0.0 && upper_residual >= 0.0;
  if (!bracketed) {
    std::stringstream message;
    message << std::scientific << std::setprecision(16)
            << "unable to split tabulated-EOS LTE pressure: rho_cgs="
            << density_cgs << " Ptot_cgs=" << total_pressure_cgs
            << " Tmin=" << lte_temperature_min
            << " Tmax=" << lte_temperature_max
            << " lower_energy=" << lower_energy
            << " upper_energy=" << upper_energy
            << " lower_residual=" << lower_residual
            << " upper_residual=" << upper_residual;
    Require(false, message.str());
  }
  for (int iteration = 0; iteration < 64; ++iteration) {
    const Real middle_energy = std::sqrt(lower_energy*upper_energy);
    LteSplit middle_state{};
    Require(evaluate(middle_energy, &middle_state),
            "tabulated EOS became invalid inside an LTE pressure bracket");
    const Real residual = middle_state.gas_pressure
        +middle_state.radiation_energy/3.0-total_pressure;
    if (residual < 0.0) {
      lower_energy = middle_energy;
    } else {
      upper_energy = middle_energy;
      upper_state = middle_state;
    }
  }
  Require(upper_state.temperature <= lte_temperature_max,
          "tabulated-EOS LTE solution exceeds lte_temperature_max_k");
  return upper_state;
#else
  const Real temperature = scf_map::GasRadiationTemperature(
      density_cgs, total_pressure_cgs, mean_molecular_weight);
  const Real radiation_energy_cgs = scf_map::kRadiationConstantCgs
      *std::pow(temperature, 4);
  const Real gas_pressure_cgs = std::max(
      total_pressure_cgs-radiation_energy_cgs/3.0,
      atmosphere_pressure_floor);
  return {temperature, gas_pressure_cgs/(gamma_gas-1.0)/pressure_unit,
          gas_pressure_cgs/pressure_unit,
          radiation_energy_cgs/pressure_unit};
#endif
}
#endif

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
        const Real x = block->pcoord->x1v(i);
        const Real y = block->pcoord->x2v(j);
        const Real z = block->pcoord->x3v(k);
        const Real phi = CorePotentialCode(x,y,z);
        const Real phi1_l = CorePotentialCode(block->pcoord->x1v(i-1),y,z);
        const Real phi1_r = CorePotentialCode(block->pcoord->x1v(i+1),y,z);
        const Real phi2_l = CorePotentialCode(x,block->pcoord->x2v(j-1),z);
        const Real phi2_r = CorePotentialCode(x,block->pcoord->x2v(j+1),z);
        const Real phi3_l = CorePotentialCode(x,y,block->pcoord->x3v(k-1));
        const Real phi3_r = CorePotentialCode(x,y,block->pcoord->x3v(k+1));
        const Real ax = -(phi1_r-phi1_l)
            /(block->pcoord->x1v(i+1)-block->pcoord->x1v(i-1));
        const Real ay = -(phi2_r-phi2_l)
            /(block->pcoord->x2v(j+1)-block->pcoord->x2v(j-1));
        const Real az = -(phi3_r-phi3_l)
            /(block->pcoord->x3v(k+1)-block->pcoord->x3v(k-1));
        const Real density = primitive(IDN,k,j,i);
        const Real dpx = dt*density*ax;
        const Real dpy = dt*density*ay;
        const Real dpz = dt*density*az;
        conserved(IM1,k,j,i) += dpx;
        conserved(IM2,k,j,i) += dpy;
        conserved(IM3,k,j,i) += dpz;
        // Use the same face-mass-flux work discretization as Athena++
        // self-gravity.  For this static potential it makes the exchange
        // between gas energy and integral(rho*Phi_core) conservative apart
        // from boundary and AMR reflux errors.
        const Real dphi1_l = -(phi-phi1_l);
        const Real dphi1_r = -(phi1_r-phi);
        conserved(IEN,k,j,i) += 0.5*dt/block->pcoord->dx1v(i)
            *(block->phydro->flux[X1DIR](IDN,k,j,i)*dphi1_l
              +block->phydro->flux[X1DIR](IDN,k,j,i+1)*dphi1_r);
        if (block->block_size.nx2 > 1) {
          const Real dphi2_l = -(phi-phi2_l);
          const Real dphi2_r = -(phi2_r-phi);
          conserved(IEN,k,j,i) += 0.5*dt/block->pcoord->dx2v(j)
              *(block->phydro->flux[X2DIR](IDN,k,j,i)*dphi2_l
                +block->phydro->flux[X2DIR](IDN,k,j+1,i)*dphi2_r);
        }
        if (block->block_size.nx3 > 1) {
          const Real dphi3_l = -(phi-phi3_l);
          const Real dphi3_r = -(phi3_r-phi);
          conserved(IEN,k,j,i) += 0.5*dt/block->pcoord->dx3v(k)
              *(block->phydro->flux[X3DIR](IDN,k,j,i)*dphi3_l
                +block->phydro->flux[X3DIR](IDN,k+1,j,i)*dphi3_r);
        }
        if (core_luminosity_code > 0.0) {
          const Real radius = std::sqrt(x*x+y*y+z*z);
          const Real weight = scf_luminosity::Weight(
              radius, luminosity_heating_radius_code);
          conserved(IEN,k,j,i) += dt*core_luminosity_code*weight
              /luminosity_kernel_normalization_code;
        }
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

#if NRMGFLD_ENABLED
void RadiationBoundary(AthenaArray<Real>& radiation,
                       AthenaArray<Real>* gas_energy, int axis, bool inner,
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
        radiation(k,j,i) = radiation(q[2],q[1],q[0]);
        if (gas_energy != nullptr) {
          (*gas_energy)(k,j,i) = (*gas_energy)(q[2],q[1],q[0]);
        }
      }
    }
  }
}

void CorrectionBoundary(AthenaArray<Real>& correction, int variables,
                        int axis, bool inner, int is, int ie, int js, int je,
                        int ks, int ke, int ngh) {
  int lower[3] = {is, js, ks};
  int upper[3] = {ie, je, ke};
  const int reference = inner ? lower[axis] : upper[axis];
  lower[axis] = inner ? reference-ngh : reference+1;
  upper[axis] = inner ? reference-1 : reference+ngh;
  for (int n = 0; n < variables; ++n) {
    for (int k = lower[2]; k <= upper[2]; ++k) {
      for (int j = lower[1]; j <= upper[1]; ++j) {
        for (int i = lower[0]; i <= upper[0]; ++i) {
          correction(n,k,j,i) = 0.0;
        }
      }
    }
  }
}

#define SCF_RADIATION_BOUNDARIES(TAG, AXIS, INNER) \
void FLD##TAG(MeshBlock*, Coordinates*, FLD*, const AthenaArray<Real>&, \
              AthenaArray<Real>& radiation, Real, Real, \
              int is, int ie, int js, int je, int ks, int ke, int ngh) { \
  RadiationBoundary(radiation, nullptr, AXIS, INNER, \
                    is, ie, js, je, ks, ke, ngh); \
} \
void NR##TAG(MeshBlock*, AthenaArray<Real>& radiation, \
             AthenaArray<Real>& gas_energy, Coordinates*, \
             const AthenaArray<Real>&, Real, Real, \
             int is, int ie, int js, int je, int ks, int ke, int ngh) { \
  RadiationBoundary(radiation, &gas_energy, AXIS, INNER, \
                    is, ie, js, je, ks, ke, ngh); \
} \
void MG##TAG(AthenaArray<Real>& correction, Real, int variables, \
             int is, int ie, int js, int je, int ks, int ke, int ngh, \
             const MGCoordinates&) { \
  CorrectionBoundary(correction, variables, AXIS, INNER, \
                     is, ie, js, je, ks, ke, ngh); \
}
SCF_RADIATION_BOUNDARIES(IX1, 0, true)
SCF_RADIATION_BOUNDARIES(OX1, 0, false)
SCF_RADIATION_BOUNDARIES(IX2, 1, true)
SCF_RADIATION_BOUNDARIES(OX2, 1, false)
SCF_RADIATION_BOUNDARIES(IX3, 2, true)
SCF_RADIATION_BOUNDARIES(OX3, 2, false)
#undef SCF_RADIATION_BOUNDARIES

void Opacity(MeshBlock* block, AthenaArray<Real>& radiation,
             AthenaArray<Real>& primitive) {
  (void) radiation;
  for (int k = block->ks-NGHOST; k <= block->ke+NGHOST; ++k) {
    for (int j = block->js-NGHOST; j <= block->je+NGHOST; ++j) {
      for (int i = block->is-NGHOST; i <= block->ie+NGHOST; ++i) {
        const Real density = std::max(
            primitive(IDN,k,j,i)*rho_unit, atmosphere_density_floor);
        const Real gas_pressure = std::max(
            primitive(IPR,k,j,i)*pressure_unit, atmosphere_pressure_floor);
        const Real temperature = GasTemperatureCgs(
            block, density/rho_unit, gas_pressure/pressure_unit);
        const Real planck = opacity_table->GetOpacity(
            RadFLD::SIGMA_P, density, temperature);
        const Real rosseland = opacity_table->GetOpacity(
            RadFLD::SIGMA_R, density, temperature);
        const Real x = block->pcoord->x1v(i)*length_unit;
        const Real y = block->pcoord->x2v(j)*length_unit;
        const Real z = block->pcoord->x3v(k)*length_unit;
        const Real cylindrical_radius = std::hypot(x,y);
        const Real radius = std::hypot(cylindrical_radius,z);
        const Real ellipsoidal_radius = std::sqrt(
            SQR(cylindrical_radius/planck_taper_radius)
            +SQR(z/planck_taper_polar_radius));
        const Real surface_radius_along_ray = ellipsoidal_radius > 0.0
            ? radius/ellipsoidal_radius : planck_taper_radius;
        const Real taper_distance =
            (ellipsoidal_radius-1.0)*surface_radius_along_ray;
        Real planck_weight = 1.0;
        if (taper_distance > 0.0) {
          planck_weight = 0.0;
          if (planck_taper_width > 0.0
              && taper_distance < planck_taper_width) {
            const Real fraction = taper_distance/planck_taper_width;
            planck_weight = 0.5*(1.0+std::cos(PI*fraction));
          }
        }
        block->prfld->sigma_p(k,j,i) = std::max(
            planck_weight*planck*density*length_unit, TINY_NUMBER);
        block->prfld->sigma_r(k,j,i) = std::max(
            rosseland*density*length_unit, TINY_NUMBER);
      }
    }
  }
}

Real RadiationStartupTimeStep(MeshBlock* block) {
  if (radiation_startup_dt_cap > 0.0
      && block->pmy_mesh->time < radiation_startup_dt_end) {
    return radiation_startup_dt_cap;
  }
  return std::numeric_limits<Real>::max();
}
#endif

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
#if NRMGFLD_ENABLED
        Real arx = 0.0, ary = 0.0, arz = 0.0;
        block->prfld->RadiationAcceleration(
            k, j, i, density, &arx, &ary, &arz);
        const Real radiation_acceleration =
            std::sqrt(arx*arx+ary*ary+arz*arz);
        const Real full_residual_x = residual_x+arx;
        const Real full_residual_y = residual_y+ary;
        const Real full_residual_z = residual_z+arz;
        const Real full_force_residual = std::sqrt(
            full_residual_x*full_residual_x
            +full_residual_y*full_residual_y
            +full_residual_z*full_residual_z);
#endif
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
#if NRMGFLD_ENABLED
        block->user_out_var(16,k,j,i) = radiation_acceleration;
        block->user_out_var(17,k,j,i) = full_force_residual;
        block->user_out_var(18,k,j,i) = pressure_acceleration
            +radiation_acceleration+gravity_acceleration
            +centrifugal_acceleration;
#endif
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

Real HistoryCoreLuminosity(MeshBlock* block, int) {
  Real total_code = 0.0;
  if (core_luminosity_code <= 0.0) return total_code;
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const Real x = block->pcoord->x1v(i);
        const Real y = block->pcoord->x2v(j);
        const Real z = block->pcoord->x3v(k);
        const Real radius = std::sqrt(x*x+y*y+z*z);
        total_code += core_luminosity_code
            *scf_luminosity::Weight(radius, luminosity_heating_radius_code)
            /luminosity_kernel_normalization_code
            *block->pcoord->GetCellVolume(k,j,i);
      }
    }
  }
  return total_code*pressure_unit*std::pow(length_unit,3)/time_unit;
}

Real HistoryMeshLevel(MeshBlock* block, int output) {
  const int relative_level = block->loc.level-root_logical_level;
  return relative_level == output-kMeshBlockHistoryStart ? 1.0 : 0.0;
}

Real HistoryWenozCells(MeshBlock* block, int output) {
  const AthenaArray<int>& mask = block->phydro->GetTroubledCellMask();
  const Hydro::TroubledCellMaskBounds& bounds =
      block->phydro->GetTroubledCellMaskBounds();
  if (mask.GetDim1() == 0 || bounds.il > bounds.iu || bounds.jl > bounds.ju
      || bounds.kl > bounds.ku) return 0.0;
  if (output == kWenozHistoryStart+1) {
    return static_cast<Real>(block->phydro->GetWenozSensorFallbackCells());
  }
  if (output == kWenozHistoryStart+2) {
    return static_cast<Real>(block->phydro->GetWenozAmrFallbackCells());
  }
  const int relative_level = block->loc.level-root_logical_level;
  if (output >= kWenozLevelHistoryStart
      && output < kWenozLevelHistoryStart+kTrackedMeshLevels
      && output != kWenozLevelHistoryStart+relative_level) return 0.0;
  Real total = 0.0;
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const bool available = i >= bounds.il && i <= bounds.iu
            && j >= bounds.jl && j <= bounds.ju
            && k >= bounds.kl && k <= bounds.ku;
        if (output == kWenozHistoryStart && available) total += 1.0;
        if (output == kWenozHistoryStart+3 && available && mask(k,j,i) != 0) {
          total += 1.0;
        }
        if (output == kWenozLevelHistoryStart+relative_level
            && available && mask(k,j,i) != 0) total += 1.0;
      }
    }
  }
  return total;
}

Real HistoryWenozFallbackMass(MeshBlock* block, int) {
  const AthenaArray<int>& mask = block->phydro->GetTroubledCellMask();
  const Hydro::TroubledCellMaskBounds& bounds =
      block->phydro->GetTroubledCellMaskBounds();
  if (mask.GetDim1() == 0 || bounds.il > bounds.iu || bounds.jl > bounds.ju
      || bounds.kl > bounds.ku) return 0.0;
  Real total = 0.0;
  const Real scale = rho_unit*std::pow(length_unit, 3);
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const bool available = i >= bounds.il && i <= bounds.iu
            && j >= bounds.jl && j <= bounds.ju
            && k >= bounds.kl && k <= bounds.ku;
        if (available && mask(k,j,i) != 0) {
          total += block->pscalars->s(0,k,j,i)
              *block->pcoord->GetCellVolume(k,j,i)*scale;
        }
      }
    }
  }
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

#if NRMGFLD_ENABLED
void UpdateTaperRadiationDiagnostics(MeshBlock* block) {
  constexpr int kDiffusionHeating = 19;
  constexpr int kThermalExchange = 20;
  const AthenaArray<Real>& radiation = block->prfld->u_rad;
  const AthenaArray<Real>& diffusion = block->prfld->marshak_dface_all;
  for (int k = block->ks; k <= block->ke; ++k) {
    const Real dz = block->pcoord->dx3f(k);
    for (int j = block->js; j <= block->je; ++j) {
      const Real dy = block->pcoord->dx2f(j);
      for (int i = block->is; i <= block->ie; ++i) {
        const Real dx = block->pcoord->dx1f(i);
        const Real center = radiation(k,j,i);
        Require(std::isfinite(center) && center >= 0.0,
                "taper diagnostic requires nonnegative active radiation energy");
        Real dface[6];
        for (int face = 0; face < 6; ++face) {
          dface[face] = diffusion(face,k,j,i);
          Require(std::isfinite(dface[face]) && dface[face] >= 0.0,
                  "taper diagnostic requires nonnegative FLD diffusion coefficients");
        }
        // The implicit solve bounds finite AMR interpolation undershoots in
        // its private ghost stencil.  Reproduce that exact admissibility rule
        // here while retaining the active-cell hard check above.
        const auto bounded_ghost = [](Real value) {
          Require(std::isfinite(value),
                  "taper diagnostic requires finite radiation ghosts");
          return std::max(value, TINY_NUMBER);
        };
        const Real xm = bounded_ghost(radiation(k,j,i-1));
        const Real xp = bounded_ghost(radiation(k,j,i+1));
        const Real ym = bounded_ghost(radiation(k,j-1,i));
        const Real yp = bounded_ghost(radiation(k,j+1,i));
        const Real zm = bounded_ghost(radiation(k-1,j,i));
        const Real zp = bounded_ghost(radiation(k+1,j,i));
        block->user_out_var(kDiffusionHeating,k,j,i) =
            scf_fld_diagnostic::DiffusionHeating(
                center, xm, xp, ym, yp, zm, zp,
                dface[0], dface[1], dface[2], dface[3], dface[4], dface[5],
                dx, dy, dz);

        const Real density = block->phydro->u(IDN,k,j,i);
        const Real mx = block->phydro->u(IM1,k,j,i);
        const Real my = block->phydro->u(IM2,k,j,i);
        const Real mz = block->phydro->u(IM3,k,j,i);
        const Real gas_energy = block->phydro->u(IEN,k,j,i)
            -0.5*(mx*mx+my*my+mz*mz)/density;
#if GENERAL_EOS
        const Real temperature = block->peos->TempFromRhoEg(density, gas_energy);
#else
        const Real temperature = (gamma_gas-1.0)*gas_energy/density;
#endif
        const Real equilibrium = block->prfld->a_r*std::pow(temperature, 4);
        block->user_out_var(kThermalExchange,k,j,i) =
            block->prfld->c_ph*block->prfld->sigma_p(k,j,i)
            *scf_fld_diagnostic::ResolvableThermalDifference(
                equilibrium, center);
      }
    }
  }
}

Real HistoryRadiationEnergy(MeshBlock* block, int) {
  Real total = 0.0;
  const Real scale = pressure_unit*std::pow(length_unit, 3);
  for (int k = block->ks; k <= block->ke; ++k)
    for (int j = block->js; j <= block->je; ++j)
      for (int i = block->is; i <= block->ie; ++i)
        total += block->prfld->u_rad(k,j,i)
            *block->pcoord->GetCellVolume(k,j,i)*scale;
  return total;
}

Real HistoryGasInternalEnergy(MeshBlock* block, int) {
  Real total = 0.0;
  const Real scale = pressure_unit*std::pow(length_unit, 3);
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const Real density = block->phydro->u(IDN,k,j,i);
        const Real mx = block->phydro->u(IM1,k,j,i);
        const Real my = block->phydro->u(IM2,k,j,i);
        const Real mz = block->phydro->u(IM3,k,j,i);
        const Real kinetic = 0.5*(mx*mx+my*my+mz*mz)
            /std::max(density, TINY_NUMBER);
        total += (block->phydro->u(IEN,k,j,i)-kinetic)
            *block->pcoord->GetCellVolume(k,j,i)*scale;
      }
    }
  }
  return total;
}

Real HistoryTaperRadiationDiagnostic(MeshBlock* block, int output) {
  constexpr int kDiffusionHeating = 19;
  constexpr int kThermalExchange = 20;
  const int diagnostic = output-kTaperRadiationHistoryStart;
  // HistoryOutput visits user fields in ascending order for each MeshBlock.
  // Unlike regular data outputs it does not call UserWorkBeforeOutput, so the
  // first taper field refreshes the two cell diagnostics exactly once and the
  // remaining seven fields consume that same snapshot.
  if (diagnostic == 0) UpdateTaperRadiationDiagnostics(block);
  const Real energy_scale = pressure_unit*std::pow(length_unit, 3);
  const Real power_scale = energy_scale/time_unit;
  Real total = (diagnostic == 7 || diagnostic == 8)
      ? std::numeric_limits<Real>::infinity() : 0.0;
  int minimum_i = -1, minimum_j = -1, minimum_k = -1;
  Real minimum_ellipsoidal_radius = 0.0;
  for (int k = block->ks; k <= block->ke; ++k) {
    const Real z = block->pcoord->x3v(k);
    for (int j = block->js; j <= block->je; ++j) {
      const Real y = block->pcoord->x2v(j);
      for (int i = block->is; i <= block->ie; ++i) {
        const Real x = block->pcoord->x1v(i);
        const Real cylindrical_radius = std::hypot(x,y);
        const Real ellipsoidal_radius = std::sqrt(
            SQR(cylindrical_radius*length_unit/surface_match_equatorial_radius)
            +SQR(z*length_unit/surface_match_polar_radius));
        const Real density = block->phydro->u(IDN,k,j,i);
        const Real stellar_density = block->pscalars->s(0,k,j,i);
        if (!(density > 0.0) || stellar_density/density <= 0.5
            || ellipsoidal_radius <= 1.0) continue;
        const Real volume = block->pcoord->GetCellVolume(k,j,i);
        const Real mx = block->phydro->u(IM1,k,j,i);
        const Real my = block->phydro->u(IM2,k,j,i);
        const Real mz = block->phydro->u(IM3,k,j,i);
        const Real kinetic = 0.5*(mx*mx+my*my+mz*mz)/density;
        const Real gas_energy = std::max(
            block->phydro->u(IEN,k,j,i)-kinetic, TINY_NUMBER);
        const Real radiation_energy = std::max(
            block->prfld->u_rad(k,j,i), TINY_NUMBER);
        const Real diffusion_heating =
            block->user_out_var(kDiffusionHeating,k,j,i);
        const Real thermal_exchange =
            block->user_out_var(kThermalExchange,k,j,i);
        if (diagnostic == 0) total += stellar_density*volume*mass_unit;
        if (diagnostic == 1) total += radiation_energy*volume*energy_scale;
        if (diagnostic == 2) total += gas_energy*volume*energy_scale;
        if (diagnostic == 3) {
          total += diffusion_heating*volume*power_scale;
        }
        if (diagnostic == 4) {
          total += std::abs(diffusion_heating)*volume*power_scale;
        }
        if (diagnostic == 5) {
          total += thermal_exchange*volume*power_scale;
        }
        if (diagnostic == 6) {
          total += std::abs(thermal_exchange)*volume*power_scale;
        }
        const Real total_energy = gas_energy+radiation_energy;
        if (diagnostic == 7 && std::abs(diffusion_heating) > 0.0) {
          total = std::min(total,
              total_energy/std::abs(diffusion_heating)*time_unit);
        }
        if (diagnostic == 8 && std::abs(thermal_exchange) > 0.0) {
          const Real exchange_time =
              total_energy/std::abs(thermal_exchange)*time_unit;
          if (exchange_time < total) {
            total = exchange_time;
            minimum_i = i;
            minimum_j = j;
            minimum_k = k;
            minimum_ellipsoidal_radius = ellipsoidal_radius;
          }
        }
      }
    }
  }
  if (diagnostic == 8 && taper_exchange_diagnostic_time_threshold > 0.0
      && total <= taper_exchange_diagnostic_time_threshold
      && minimum_i >= 0) {
    const int i = minimum_i;
    const int j = minimum_j;
    const int k = minimum_k;
    const Real density = block->phydro->u(IDN,k,j,i);
    const Real stellar_density = block->pscalars->s(0,k,j,i);
    const Real mx = block->phydro->u(IM1,k,j,i);
    const Real my = block->phydro->u(IM2,k,j,i);
    const Real mz = block->phydro->u(IM3,k,j,i);
    const Real gas_energy = block->phydro->u(IEN,k,j,i)
        -0.5*(mx*mx+my*my+mz*mz)/density;
    const Real radiation_energy = block->prfld->u_rad(k,j,i);
#if GENERAL_EOS
    const Real temperature = block->peos->TempFromRhoEg(
        density, gas_energy)*temperature_unit;
#else
    const Real temperature = (gamma_gas-1.0)*gas_energy/density
        *temperature_unit;
#endif
    const Real radiation_temperature = std::pow(
        radiation_energy*pressure_unit/scf_map::kRadiationConstantCgs, 0.25);
    const Real equilibrium = block->prfld->a_r
        *std::pow(temperature/temperature_unit, 4);
    const Real thermal_exchange =
        block->user_out_var(kThermalExchange,k,j,i);
    const Real volume = block->pcoord->GetCellVolume(k,j,i);
    const Real effective_planck_opacity = block->prfld->sigma_p(k,j,i)
        /(density*rho_unit*length_unit);
    std::stringstream message;
    message << std::scientific << std::setprecision(16)
            << "[SCF_TAPER_EXCHANGE_CELL] rank=" << Globals::my_rank
            << " cycle=" << block->pmy_mesh->ncycle
            << " time=" << block->pmy_mesh->time
            << " gid=" << block->gid
            << " level=" << block->loc.level
            << " i=" << i << " j=" << j << " k=" << k
            << " x_code=" << block->pcoord->x1v(i)
            << " y_code=" << block->pcoord->x2v(j)
            << " z_code=" << block->pcoord->x3v(k)
            << " ellipsoidal_radius=" << minimum_ellipsoidal_radius
            << " rho_cgs=" << density*rho_unit
            << " stellar_rho_cgs=" << stellar_density*rho_unit
            << " stellar_fraction=" << stellar_density/density
            << " pressure_cgs=" << block->phydro->w(IPR,k,j,i)*pressure_unit
            << " gas_energy_erg_cm3=" << gas_energy*pressure_unit
            << " radiation_energy_erg_cm3=" << radiation_energy*pressure_unit
            << " gas_temperature_k=" << temperature
            << " radiation_temperature_k=" << radiation_temperature
            << " equilibrium_to_radiation="
            << equilibrium/std::max(radiation_energy, TINY_NUMBER)
            << " sigma_planck_code=" << block->prfld->sigma_p(k,j,i)
            << " effective_planck_opacity_cm2_g=" << effective_planck_opacity
            << " exchange_erg_cm3_s="
            << thermal_exchange*pressure_unit/time_unit
            << " cell_exchange_power_erg_s="
            << thermal_exchange*volume*power_scale
            << " exchange_time_s=" << total
            << std::endl;
    std::cout << message.str();
  }
  return total;
}
#endif

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
#if NRMGFLD_ENABLED
        if (output == kBaseHistoryCount+2) total += stellar_density
            *SQR(block->user_out_var(17,k,j,i))*volume;
        if (output == kBaseHistoryCount+3) total += stellar_density
            *SQR(block->user_out_var(18,k,j,i))*volume;
        if (output == kBaseHistoryCount+4 && stellar_density/density > 0.999
            && block->user_out_var(18,k,j,i) > 0.0) {
          total = std::max(total, block->user_out_var(17,k,j,i)
                                    /block->user_out_var(18,k,j,i));
        }
#endif
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

Real HistoryRelaxationDiagnostic(MeshBlock* block, int output) {
  Real total = 0.0;
  const Real mass_scale = rho_unit*std::pow(length_unit,3);
  const Real momentum_scale = mass_scale*velocity_unit;
  const Real energy_scale = pressure_unit*std::pow(length_unit,3);
  const Real radius_scale = length_unit;
  for (int k = block->ks; k <= block->ke; ++k) {
    const Real z = block->pcoord->x3v(k);
    for (int j = block->js; j <= block->je; ++j) {
      const Real y = block->pcoord->x2v(j);
      for (int i = block->is; i <= block->ie; ++i) {
        const Real x = block->pcoord->x1v(i);
        const Real radius = std::sqrt(x*x+y*y+z*z);
        const Real cylindrical_radius = std::hypot(x,y);
        const Real volume = block->pcoord->GetCellVolume(k,j,i);
        const Real density = block->phydro->u(IDN,k,j,i);
        const Real stellar_density = block->pscalars->s(0,k,j,i);
        const Real vx = block->phydro->u(IM1,k,j,i)/density;
        const Real vy = block->phydro->u(IM2,k,j,i)/density;
        const Real vz = block->phydro->u(IM3,k,j,i)/density;
        const Real radial_velocity = radius > 0.0
            ? (x*vx+y*vy+z*vz)/radius : 0.0;
        const Real target_vx = -angular_velocity_code*y;
        const Real target_vy = angular_velocity_code*x;
        const Real dvx = vx-target_vx;
        const Real dvy = vy-target_vy;
        const Real dvz = vz;
        const Real perturbation_speed2 = dvx*dvx+dvy*dvy+dvz*dvz;
        const Real pressure = block->phydro->w(IPR,k,j,i);
        const Real sound_speed2 = AdiabaticSoundSpeedSquared(
            block, density, pressure);
        const Real stellar_fraction = std::min(
            static_cast<Real>(1.0), std::max(static_cast<Real>(0.0),
            stellar_density/density));
        const Real ellipsoidal_radius = std::sqrt(
            SQR(cylindrical_radius*length_unit/atmosphere_reference_radius)
            +SQR(z*length_unit/surface_polar_radius));

        if (output == 14) total += stellar_density*radial_velocity*volume;
        if (output == 15) total += 0.5*stellar_density
            *radial_velocity*radial_velocity*volume;
        if (output == 16) total += 0.5*stellar_density*std::max(
            static_cast<Real>(0.0), perturbation_speed2
                                      -radial_velocity*radial_velocity)*volume;
        if (output == 17 && sound_speed2 > 0.0) total += stellar_density
            *radial_velocity*radial_velocity/sound_speed2*volume;
        if (output == 18 && sound_speed2 > 0.0 && stellar_fraction > 0.5) {
          total = std::max(total, std::abs(radial_velocity)/std::sqrt(sound_speed2));
        }
        if (output == 19) total += stellar_density*radius*volume;
        if (output == 20) total += stellar_density*radius*radius*volume;
        if (output == 21) total += 3.0*pressure*stellar_fraction*volume;
        if (output == 22 || output == 23) {
          Real ax = 0.0, ay = 0.0, az = 0.0;
          if (output == 22) {
            ax = -(block->pgrav->phi(k,j,i+1)-block->pgrav->phi(k,j,i-1))
                /(block->pcoord->x1v(i+1)-block->pcoord->x1v(i-1));
            ay = -(block->pgrav->phi(k,j+1,i)-block->pgrav->phi(k,j-1,i))
                /(block->pcoord->x2v(j+1)-block->pcoord->x2v(j-1));
            az = -(block->pgrav->phi(k+1,j,i)-block->pgrav->phi(k-1,j,i))
                /(block->pcoord->x3v(k+1)-block->pcoord->x3v(k-1));
          } else {
            CoreAccelerationCode(block->pcoord, k, j, i, &ax, &ay, &az);
          }
          total += stellar_density*(x*ax+y*ay+z*az)*volume;
        }
        if (output == 24) total += 0.5*stellar_density
            *(vx*vx+vy*vy+vz*vz)*volume;
        if (output == 25) total += 0.5*stellar_density
            *perturbation_speed2*volume;
        if (output == 26 && ellipsoidal_radius > 1.0) {
          total += stellar_density*volume;
        }
        if (output == 27) total += stellar_density*ellipsoidal_radius*volume;
        if (output == 28 && stellar_fraction > 0.5) {
          total = std::max(total, ellipsoidal_radius);
        }
      }
    }
  }
  if (output == 14) return total*momentum_scale;
  if (output >= 15 && output <= 16) return total*energy_scale;
  if (output == 17 || output == 26) return total*mass_scale;
  if (output == 19) return total*mass_scale*radius_scale;
  if (output == 20) return total*mass_scale*radius_scale*radius_scale;
  if (output >= 21 && output <= 25) return total*energy_scale;
  if (output == 27) return total*mass_scale;
  return total;
}

Real HistorySurfaceMachDiagnostic(MeshBlock* block, int output) {
  const int diagnostic = output-kSurfaceMachHistoryStart;
  Real total = 0.0;
  Real max_x = 0.0, max_y = 0.0, max_z = 0.0;
  Real max_density = 0.0, max_stellar_density = 0.0, max_pressure = 0.0;
  Real max_radial_velocity = 0.0, max_sound_speed = 0.0;
  Real max_ellipsoidal_radius = 0.0;
  int max_i = 0, max_j = 0, max_k = 0;
  Real minimum_pressure_ratio = std::numeric_limits<Real>::infinity();
  Real minimum_pressure_x = 0.0, minimum_pressure_y = 0.0;
  Real minimum_pressure_z = 0.0, minimum_pressure_density = 0.0;
  Real minimum_pressure_stellar_density = 0.0;
  Real minimum_gas_pressure = 0.0, minimum_radiation_pressure = 0.0;
  Real minimum_total_pressure = 0.0, minimum_source_density = 0.0;
  Real minimum_source_pressure = 0.0, minimum_pressure_ellipsoid = 0.0;
  for (int k = block->ks; k <= block->ke; ++k) {
    const Real z = block->pcoord->x3v(k);
    for (int j = block->js; j <= block->je; ++j) {
      const Real y = block->pcoord->x2v(j);
      for (int i = block->is; i <= block->ie; ++i) {
        const Real x = block->pcoord->x1v(i);
        const Real radius = std::sqrt(x*x+y*y+z*z);
        const Real cylindrical_radius = std::hypot(x,y);
        const Real density = block->phydro->u(IDN,k,j,i);
        const Real stellar_density = block->pscalars->s(0,k,j,i);
        if (!(density > 0.0) || stellar_density/density <= 0.5) continue;
        const Real pressure = block->phydro->w(IPR,k,j,i);
        Real radiation_pressure = 0.0;
#if NRMGFLD_ENABLED
        radiation_pressure = block->prfld->u_rad(k,j,i)/3.0;
#endif
        const Real total_pressure = pressure+radiation_pressure;
        const Real sound_speed2 = AdiabaticSoundSpeedSquared(
            block, density, pressure);
        if (!(sound_speed2 > 0.0)) continue;
        const Real vx = block->phydro->u(IM1,k,j,i)/density;
        const Real vy = block->phydro->u(IM2,k,j,i)/density;
        const Real vz = block->phydro->u(IM3,k,j,i)/density;
        const Real radial_velocity = radius > 0.0
            ? (x*vx+y*vy+z*vz)/radius : 0.0;
        const Real mach = std::abs(radial_velocity)/std::sqrt(sound_speed2);
        const Real volume = block->pcoord->GetCellVolume(k,j,i);
        const Real match_ellipsoidal_radius = std::sqrt(
            SQR(cylindrical_radius*length_unit/surface_match_equatorial_radius)
            +SQR(z*length_unit/surface_match_polar_radius));
        if (diagnostic == 3
            && surface_pressure_ratio_diagnostic_threshold > 0.0
            && match_ellipsoidal_radius > 1.0) {
          const scf_map::State source = profile.Interpolate(
              cylindrical_radius*length_unit, z*length_unit);
          if (source.density > 5.0*atmosphere_density
              && source.density < surface_density_code*rho_unit
              && source.pressure > 0.0) {
            const Real ratio = total_pressure*pressure_unit/source.pressure;
            if (ratio < minimum_pressure_ratio) {
              minimum_pressure_ratio = ratio;
              minimum_pressure_x = x;
              minimum_pressure_y = y;
              minimum_pressure_z = z;
              minimum_pressure_density = density;
              minimum_pressure_stellar_density = stellar_density;
              minimum_gas_pressure = pressure;
              minimum_radiation_pressure = radiation_pressure;
              minimum_total_pressure = total_pressure;
              minimum_source_density = source.density;
              minimum_source_pressure = source.pressure;
              minimum_pressure_ellipsoid = match_ellipsoidal_radius;
            }
          }
        }
        if (diagnostic == 0 && match_ellipsoidal_radius <= 1.0) {
          total = std::max(total, mach);
        }
        if (diagnostic == 1 && match_ellipsoidal_radius > 1.0) {
          total = std::max(total, mach);
        }
        if (diagnostic == 2 && stellar_density >= surface_density_code) {
          total = std::max(total, mach);
        }
        if (diagnostic == 3 && stellar_density < surface_density_code) {
          if (mach > total) {
            total = mach;
            max_x = x;
            max_y = y;
            max_z = z;
            max_density = density;
            max_stellar_density = stellar_density;
            max_pressure = pressure;
            max_radial_velocity = radial_velocity;
            max_sound_speed = std::sqrt(sound_speed2);
            max_ellipsoidal_radius = match_ellipsoidal_radius;
            max_i = i;
            max_j = j;
            max_k = k;
          }
        }
        if (mach > 0.1) {
          if (diagnostic == 4) total += stellar_density*volume;
          if (diagnostic == 5) total += stellar_density*radius*volume;
          if (diagnostic == 6) {
            const Real density_cgs = std::max(
                stellar_density*rho_unit, std::numeric_limits<Real>::min());
            total += stellar_density*std::log(density_cgs)*volume;
          }
        }
        if (diagnostic == 7 && mach > 0.3) {
          total += stellar_density*volume;
        }
      }
    }
  }
  if (diagnostic == 4 || diagnostic == 6 || diagnostic == 7) {
    return total*mass_unit;
  }
  if (diagnostic == 5) return total*mass_unit*length_unit;
  if (diagnostic == 3
      && surface_pressure_ratio_diagnostic_threshold > 0.0
      && minimum_pressure_ratio <= surface_pressure_ratio_diagnostic_threshold) {
    std::stringstream message;
    message << std::scientific << std::setprecision(16)
            << "[SCF_SURFACE_PRESSURE_CELL] rank=" << Globals::my_rank
            << " cycle=" << block->pmy_mesh->ncycle
            << " time=" << block->pmy_mesh->time
            << " gid=" << block->gid
            << " level=" << block->loc.level
            << " x_code=" << minimum_pressure_x
            << " y_code=" << minimum_pressure_y
            << " z_code=" << minimum_pressure_z
            << " ellipsoidal_radius=" << minimum_pressure_ellipsoid
            << " rho_cgs=" << minimum_pressure_density*rho_unit
            << " stellar_rho_cgs="
            << minimum_pressure_stellar_density*rho_unit
            << " gas_pressure_cgs=" << minimum_gas_pressure*pressure_unit
            << " radiation_pressure_cgs="
            << minimum_radiation_pressure*pressure_unit
            << " total_pressure_cgs=" << minimum_total_pressure*pressure_unit
            << " source_rho_cgs=" << minimum_source_density
            << " source_pressure_cgs=" << minimum_source_pressure
            << " current_to_source_rho="
            << minimum_pressure_density*rho_unit/minimum_source_density
            << " current_to_source_gas_pressure="
            << minimum_gas_pressure*pressure_unit/minimum_source_pressure
            << " current_to_source_total_pressure=" << minimum_pressure_ratio
            << std::endl;
    std::cout << message.str();
  }
  if (diagnostic == 3 && surface_mach_diagnostic_threshold > 0.0
      && total >= surface_mach_diagnostic_threshold) {
#if NRMGFLD_ENABLED
    // The optional surface-cell report is used to distinguish a mechanical
    // surface mismatch from radiative cooling/heating at the same cell.
    // Refresh the lagged FLD terms here because surface histories precede the
    // aggregate taper histories in the output ordering.
    UpdateGravityDiagnostics(block);
    UpdateTaperRadiationDiagnostics(block);
    const Real diffusion_heating = block->user_out_var(19,max_k,max_j,max_i);
    const Real thermal_exchange = block->user_out_var(20,max_k,max_j,max_i);
    const Real density = block->phydro->u(IDN,max_k,max_j,max_i);
    const Real mx = block->phydro->u(IM1,max_k,max_j,max_i);
    const Real my = block->phydro->u(IM2,max_k,max_j,max_i);
    const Real mz = block->phydro->u(IM3,max_k,max_j,max_i);
    const Real gas_energy = block->phydro->u(IEN,max_k,max_j,max_i)
        -0.5*(mx*mx+my*my+mz*mz)/density;
    const Real radiation_energy = block->prfld->u_rad(max_k,max_j,max_i);
    const Real radiation_pressure = radiation_energy/3.0;
    const Real total_pressure = max_pressure+radiation_pressure;
    const Real local_energy = gas_energy+radiation_energy;
    const Real diffusion_time = std::abs(diffusion_heating) > 0.0
        ? local_energy/std::abs(diffusion_heating)*time_unit
        : std::numeric_limits<Real>::infinity();
    const Real exchange_time = std::abs(thermal_exchange) > 0.0
        ? local_energy/std::abs(thermal_exchange)*time_unit
        : std::numeric_limits<Real>::infinity();
    const Real acceleration_scale = velocity_unit*velocity_unit/length_unit;
    const Real pressure_acceleration =
        block->user_out_var(9,max_k,max_j,max_i)*acceleration_scale;
    const Real gravity_acceleration =
        block->user_out_var(10,max_k,max_j,max_i)*acceleration_scale;
    const Real radiation_acceleration =
        block->user_out_var(16,max_k,max_j,max_i)*acceleration_scale;
    const Real full_force_residual =
        block->user_out_var(17,max_k,max_j,max_i)*acceleration_scale;
    const Real full_force_scale =
        block->user_out_var(18,max_k,max_j,max_i)*acceleration_scale;
    const Real radius = std::sqrt(max_x*max_x+max_y*max_y+max_z*max_z);
    const Real dpx = (block->phydro->w(IPR,max_k,max_j,max_i+1)
                     -block->phydro->w(IPR,max_k,max_j,max_i-1))
        /(block->pcoord->x1v(max_i+1)-block->pcoord->x1v(max_i-1));
    const Real dpy = (block->phydro->w(IPR,max_k,max_j+1,max_i)
                     -block->phydro->w(IPR,max_k,max_j-1,max_i))
        /(block->pcoord->x2v(max_j+1)-block->pcoord->x2v(max_j-1));
    const Real dpz = (block->phydro->w(IPR,max_k+1,max_j,max_i)
                     -block->phydro->w(IPR,max_k-1,max_j,max_i))
        /(block->pcoord->x3v(max_k+1)-block->pcoord->x3v(max_k-1));
    const Real apx = -dpx/density;
    const Real apy = -dpy/density;
    const Real apz = -dpz/density;
    const Real asx = -(block->pgrav->phi(max_k,max_j,max_i+1)
                      -block->pgrav->phi(max_k,max_j,max_i-1))
        /(block->pcoord->x1v(max_i+1)-block->pcoord->x1v(max_i-1));
    const Real asy = -(block->pgrav->phi(max_k,max_j+1,max_i)
                      -block->pgrav->phi(max_k,max_j-1,max_i))
        /(block->pcoord->x2v(max_j+1)-block->pcoord->x2v(max_j-1));
    const Real asz = -(block->pgrav->phi(max_k+1,max_j,max_i)
                      -block->pgrav->phi(max_k-1,max_j,max_i))
        /(block->pcoord->x3v(max_k+1)-block->pcoord->x3v(max_k-1));
    Real acx, acy, acz;
    CoreAccelerationCode(block->pcoord, max_k, max_j, max_i,
                         &acx, &acy, &acz);
    Real arx = 0.0, ary = 0.0, arz = 0.0;
    block->prfld->RadiationAcceleration(
        max_k, max_j, max_i, density, &arx, &ary, &arz);
    const Real centrifugal_x = angular_velocity_code*angular_velocity_code*max_x;
    const Real centrifugal_y = angular_velocity_code*angular_velocity_code*max_y;
    const Real inverse_radius = radius > 0.0 ? 1.0/radius : 0.0;
    const auto radial_projection = [=](Real ax, Real ay, Real az) {
      return (max_x*ax+max_y*ay+max_z*az)*inverse_radius*acceleration_scale;
    };
    const Real pressure_acceleration_radial = radial_projection(apx, apy, apz);
    const Real gravity_acceleration_radial = radial_projection(
        asx+acx, asy+acy, asz+acz);
    const Real radiation_acceleration_radial = radial_projection(arx, ary, arz);
    const Real centrifugal_acceleration_radial = radial_projection(
        centrifugal_x, centrifugal_y, 0.0);
    const Real full_force_residual_radial = radial_projection(
        apx+asx+acx+arx+centrifugal_x,
        apy+asy+acy+ary+centrifugal_y, apz+asz+acz+arz);
    const scf_map::State source = profile.Interpolate(
        std::hypot(max_x,max_y)*length_unit, max_z*length_unit);
#endif
    std::stringstream message;
    message << std::scientific << std::setprecision(16)
            << "[SCF_SURFACE_MACH_CELL] rank=" << Globals::my_rank
            << " cycle=" << block->pmy_mesh->ncycle
            << " time=" << block->pmy_mesh->time
            << " gid=" << block->gid
            << " level=" << block->loc.level
            << " i=" << max_i << " j=" << max_j << " k=" << max_k
            << " x_code=" << max_x
            << " y_code=" << max_y
            << " z_code=" << max_z
            << " rho_code=" << max_density
            << " stellar_rho_code=" << max_stellar_density
            << " pressure_code=" << max_pressure
            << " vr_code=" << max_radial_velocity
            << " cs_code=" << max_sound_speed
            << " mach=" << total
            << " ellipsoidal_radius=" << max_ellipsoidal_radius
            << " rho_cgs=" << max_density*rho_unit
            << " stellar_rho_cgs=" << max_stellar_density*rho_unit
            << " pressure_cgs=" << max_pressure*pressure_unit
            << " vr_cgs=" << max_radial_velocity*velocity_unit
            << " cs_cgs=" << max_sound_speed*velocity_unit
#if NRMGFLD_ENABLED
            << " fld_diffusion_erg_cm3_s="
            << diffusion_heating*pressure_unit/time_unit
            << " fld_exchange_erg_cm3_s="
            << thermal_exchange*pressure_unit/time_unit
            << " fld_diffusion_time_s=" << diffusion_time
            << " fld_exchange_time_s=" << exchange_time
            << " pressure_accel_cgs=" << pressure_acceleration
            << " gravity_accel_cgs=" << gravity_acceleration
            << " radiation_accel_cgs=" << radiation_acceleration
            << " full_force_residual_cgs=" << full_force_residual
            << " full_force_relative="
            << full_force_residual/std::max(full_force_scale, TINY_NUMBER)
            << " pressure_accel_radial_cgs=" << pressure_acceleration_radial
            << " gravity_accel_radial_cgs=" << gravity_acceleration_radial
            << " radiation_accel_radial_cgs=" << radiation_acceleration_radial
            << " centrifugal_accel_radial_cgs="
            << centrifugal_acceleration_radial
            << " full_force_residual_radial_cgs="
            << full_force_residual_radial
            << " source_rho_cgs=" << source.density
            << " source_pressure_cgs=" << source.pressure
            << " current_to_source_rho="
            << max_density*rho_unit/std::max(source.density,
                                             std::numeric_limits<Real>::min())
            << " radiation_pressure_cgs="
            << radiation_pressure*pressure_unit
            << " total_pressure_cgs=" << total_pressure*pressure_unit
            << " current_to_source_gas_pressure="
            << max_pressure*pressure_unit/std::max(
                source.pressure, std::numeric_limits<Real>::min())
            << " current_to_source_total_pressure="
            << total_pressure*pressure_unit/std::max(
                source.pressure, std::numeric_limits<Real>::min())
#endif
            << std::endl;
    std::cout << message.str();
  }
  return total;
}

#if EOS_TABLE_ENABLED
Real HistoryEosTableMargin(MeshBlock* block, int output) {
  const int diagnostic = output-kEosTableHistoryStart;
  EosTable *table = block->peos->ptable;
  Require(table != nullptr && table->nVar > 3,
          "EOS table margin diagnostic requires a temperature field");
  Real minimum_margin = std::numeric_limits<Real>::infinity();
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const Real density = block->phydro->u(IDN,k,j,i);
        const Real mx = block->phydro->u(IM1,k,j,i);
        const Real my = block->phydro->u(IM2,k,j,i);
        const Real mz = block->phydro->u(IM3,k,j,i);
        const Real kinetic = 0.5*(mx*mx+my*my+mz*mz)
            /std::max(density, TINY_NUMBER);
        const Real gas_energy = block->phydro->u(IEN,k,j,i)-kinetic;
        if (!(density > 0.0) || !(gas_energy > 0.0)
            || !std::isfinite(density) || !std::isfinite(gas_energy)) {
          return -std::numeric_limits<Real>::infinity();
        }
        const Real x1 = std::log10(density*table->rhoUnit);
        // The temperature field uses the gas-energy table coordinate.  Its
        // EosRatios entry is index 3 in the standard Athena EOS table layout.
        const Real x2 = std::log10(gas_energy*table->EosRatios(3)*table->eUnit)
            +eos_dens_pow*x1;
        Real margin = 0.0;
        if (diagnostic == 0) margin = x1-table->logRhoMin;
        if (diagnostic == 1) margin = table->logRhoMax-x1;
        if (diagnostic == 2) margin = x2-table->logEgasMin;
        if (diagnostic == 3) margin = table->logEgasMax-x2;
        minimum_margin = std::min(minimum_margin, margin);
      }
    }
  }
  return minimum_margin;
}
#endif

Real HistoryRelaxationDamping(MeshBlock* block, int output) {
  const int local_output = output-kRelaxationHistoryStart;
  const Real value = block->ruser_meshblock_data[0](local_output);
  const Real energy_scale = pressure_unit*std::pow(length_unit, 3);
  return local_output % 2 == 0
      ? value*energy_scale : value*energy_scale/time_unit;
}
}  // namespace

void Mesh::InitUserMeshData(ParameterInput* pin) {
  root_logical_level = root_level;
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
#if EOS_TABLE_ENABLED
  eos_dens_pow = pin->GetOrAddReal("hydro", "dens_pow", -1.0);
#endif
  mean_molecular_weight = pin->GetReal("hydro", "mu");
#if GENERAL_EOS
  temperature_unit = pin->GetReal("hydro", "T_unit");
#else
  temperature_unit = 1.0;
#endif
  lte_temperature_min = pin->GetOrAddReal(
      "problem", "lte_temperature_min_k", 1.0e2);
  lte_temperature_max = pin->GetOrAddReal(
      "problem", "lte_temperature_max_k", 1.0e9);
  angular_velocity = pin->GetReal("problem", "angular_velocity_rad_s");
  relaxation_velocity_tau = pin->GetOrAddReal(
      "problem", "relaxation_velocity_tau", -1.0);
  relaxation_taper_start = pin->GetOrAddReal(
      "problem", "relaxation_taper_start", -1.0);
  relaxation_taper_end = pin->GetOrAddReal(
      "problem", "relaxation_taper_end", -1.0);
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
  core_luminosity_lsun = pin->GetOrAddReal(
      "problem", "core_luminosity_lsun", 0.0);
  luminosity_heating_radius = pin->GetOrAddReal(
      "problem", "luminosity_heating_radius_cm", core_support_radius);
  const Real gas_mass = pin->GetReal("problem", "gas_mass_g");
  atmosphere_reference_radius = pin->GetReal(
      "problem", "atmosphere_reference_radius_cm");
  surface_polar_radius = pin->GetOrAddReal(
      "problem", "surface_polar_radius_cm", atmosphere_reference_radius);
  surface_match_equatorial_radius = pin->GetOrAddReal(
      "problem", "surface_match_radius_cm", atmosphere_reference_radius);
  surface_match_polar_radius = pin->GetOrAddReal(
      "problem", "surface_match_polar_radius_cm",
      surface_match_equatorial_radius);
  quadrature_order = pin->GetOrAddInteger("problem", "cell_quadrature_order", 2);
  rotate_atmosphere = pin->GetOrAddBoolean(
      "problem", "rotate_atmosphere", false);
  Require(rho_unit > 0.0 && pressure_unit > 0.0 && length_unit > 0.0,
          "hydro units must be positive");
  Require(gamma_gas > 1.0 && mean_molecular_weight > 0.0
              && temperature_unit > 0.0,
          "gamma, mean molecular weight, and temperature unit must be valid");
  Require(lte_temperature_min > 0.0
              && lte_temperature_max > lte_temperature_min,
          "LTE temperature bracket must be positive and ordered");
  Require(angular_velocity >= 0.0 && atmosphere_density > 0.0
          && atmosphere_temperature > 0.0,
          "rotation and atmosphere parameters must be nonnegative/positive");
  Require((relaxation_velocity_tau < 0.0
              && relaxation_taper_start < 0.0
              && relaxation_taper_end < 0.0)
          || (relaxation_velocity_tau > 0.0
              && relaxation_taper_start >= 0.0
              && relaxation_taper_end > relaxation_taper_start),
          "velocity relaxation must be disabled or have tau>0 and 0<=start<end");
  Require(core_mass > 0.0 && core_support_radius > 0.0 && gas_mass > 0.0
          && core_luminosity_lsun >= 0.0 && luminosity_heating_radius > 0.0
          && atmosphere_reference_radius > 0.0 && surface_polar_radius > 0.0
          && surface_match_equatorial_radius > 0.0
          && surface_match_polar_radius > 0.0,
          "core, gas, and atmosphere reference parameters must be positive");
  Require(quadrature_order >= 1 && quadrature_order <= 3,
          "cell_quadrature_order must be 1, 2, or 3");
  const Real ideal_atmosphere_pressure = atmosphere_density
      *scf_map::kGasConstantCgs*atmosphere_temperature
      /mean_molecular_weight;
  atmosphere_pressure = pin->GetOrAddReal(
      "problem", "atmosphere_pressure_dyn_cm2", ideal_atmosphere_pressure);
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
  constexpr Real kSolarLuminosityCgs = 3.828e33;
  const Real power_unit = pressure_unit*std::pow(length_unit,3)/time_unit;
  core_luminosity_code = core_luminosity_lsun*kSolarLuminosityCgs/power_unit;
  luminosity_heating_radius_code = luminosity_heating_radius/length_unit;
  luminosity_kernel_normalization_code =
      scf_luminosity::VolumeNormalization(luminosity_heating_radius_code);
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
#if NRMGFLD_ENABLED
  Require(pin->GetOrAddBoolean("fld", "is_couple", true),
          "FLD gas-radiation coupling must be enabled");
  Require(pin->GetOrAddBoolean("fld", "include_radiation_force", true),
          "FLD radiation force must be enabled for the pressure split");
  Require(pin->GetOrAddBoolean("fld", "use_opacity_table", true),
          "the radiation-supported model requires an opacity table");
  opacity_table = new UserOpacityTable(pin);
  const Real external_temperature = pin->GetOrAddReal(
      "problem", "radiation_external_temperature", 0.0);
  Require(external_temperature >= 0.0,
          "radiation_external_temperature must be nonnegative");
  external_radiation_energy_code = scf_map::kRadiationConstantCgs
      *std::pow(external_temperature, 4)/pressure_unit;
  planck_taper_radius = pin->GetOrAddReal(
      "problem", "atmosphere_planck_taper_radius_cm",
      surface_match_equatorial_radius);
  planck_taper_polar_radius = pin->GetOrAddReal(
      "problem", "atmosphere_planck_taper_polar_radius_cm",
      planck_taper_radius);
  planck_taper_width = pin->GetOrAddReal(
      "problem", "atmosphere_planck_taper_width_cm", 0.0);
  marshak_alpha = pin->GetOrAddReal("problem", "marshak_alpha", 0.5);
  radiation_startup_dt_cap = pin->GetOrAddReal(
      "problem", "radiation_startup_dt_cap", -1.0);
  radiation_startup_dt_end = pin->GetOrAddReal(
      "problem", "radiation_startup_dt_end", 0.0);
  taper_exchange_diagnostic_time_threshold = pin->GetOrAddReal(
      "problem", "taper_exchange_diagnostic_time_threshold_s", -1.0);
  Require(planck_taper_radius > 0.0 && planck_taper_polar_radius > 0.0
              && planck_taper_width >= 0.0
              && marshak_alpha > 0.0 && radiation_startup_dt_end >= 0.0,
          "invalid Planck taper or Marshak boundary parameter");
  Require(taper_exchange_diagnostic_time_threshold != 0.0,
          "taper exchange diagnostic threshold must be negative or positive");
  if (radiation_startup_dt_cap > 0.0) {
    EnrollUserTimeStepFunction(RadiationStartupTimeStep);
  }
#endif

  SetGravitationalConstant(gravity_code);
  EnrollUserMGGravitySourceMaskFunction(AtmosphereGravityMask);
  EnrollUserExplicitSourceFunction(CoreGravitySource);
  surface_refinement_level = pin->GetOrAddInteger(
      "problem", "surface_refinement_level", 3);
  surface_tracer_lower = pin->GetOrAddReal(
      "problem", "surface_tracer_lower", 1.0e-6);
  surface_tracer_upper = pin->GetOrAddReal(
      "problem", "surface_tracer_upper", 1.0-1.0e-6);
  surface_refinement_use_tracer = pin->GetOrAddBoolean(
      "problem", "surface_refinement_use_tracer", true);
  const Real surface_density = pin->GetOrAddReal(
      "problem", "surface_refinement_density_g_cm3", 1.0e-9);
  surface_density_code = surface_density/rho_unit;
  surface_mach_diagnostic_threshold = pin->GetOrAddReal(
      "problem", "surface_mach_diagnostic_threshold", -1.0);
  surface_pressure_ratio_diagnostic_threshold = pin->GetOrAddReal(
      "problem", "surface_pressure_ratio_diagnostic_threshold", -1.0);
  Require(surface_refinement_level >= 0
              && surface_refinement_level <= max_level-root_level,
          "surface_refinement_level must lie within the configured mesh levels");
  Require(surface_tracer_lower >= 0.0
              && surface_tracer_lower < surface_tracer_upper
              && surface_tracer_upper <= 1.0,
          "invalid surface tracer refinement interval");
  Require(surface_density > 0.0,
          "surface_refinement_density_g_cm3 must be positive");
  Require(surface_pressure_ratio_diagnostic_threshold != 0.0,
          "surface pressure ratio diagnostic threshold must be negative or positive");
  if (adaptive) EnrollUserRefinementCondition(SurfaceRefinementCondition);
  const char* boundary_names[6] = {
      "ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"};
  BValFunc boundary_functions[6] = {
      InnerX1, OuterX1, InnerX2, OuterX2, InnerX3, OuterX3};
#if NRMGFLD_ENABLED
  FLDBoundaryFunc fld_boundaries[6] = {
      FLDIX1, FLDOX1, FLDIX2, FLDOX2, FLDIX3, FLDOX3};
  NRBoundaryFunc nr_boundaries[6] = {
      NRIX1, NROX1, NRIX2, NROX2, NRIX3, NROX3};
  MGBoundaryFunc linear_boundaries[6] = {
      MGIX1, MGOX1, MGIX2, MGOX2, MGIX3, MGOX3};
#endif
  for (int face = 0; face < 6; ++face) {
    Require(pin->GetString("mesh", boundary_names[face]) == "user",
            "all hydrodynamic boundaries must be user");
    Require(pin->GetString("gravity", boundary_names[face]) == "multipole",
            "all gravity boundaries must be isolated multipole");
    EnrollUserBoundaryFunction(static_cast<BoundaryFace>(face),
                               boundary_functions[face]);
#if NRMGFLD_ENABLED
    Require(pin->GetString("nrfld", boundary_names[face]) == "user"
                && pin->GetString(
                    "nrfld", std::string("linearMG_")+boundary_names[face])
                    == "user",
            "all NR-FLD and linear-MG boundaries must be user");
    const BoundaryFace boundary_face = static_cast<BoundaryFace>(face);
    EnrollUserFLDBoundaryFunction(boundary_face, fld_boundaries[face]);
    EnrollUserNRBoundaryFunction(boundary_face, nr_boundaries[face]);
    EnrollUserLinearMGBoundaryFunction(boundary_face,
                                       linear_boundaries[face]);
#endif
  }
  Require(pin->GetOrAddBoolean("gravity", "fill_ghost", true),
          "gravity/fill_ghost must be true for force diagnostics");

  AllocateUserHistoryOutput(kHistoryCount);
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
  EnrollUserHistoryOutput(14, HistoryRelaxationDiagnostic,
                          "stellar_radial_momentum_g_cm_s");
  EnrollUserHistoryOutput(15, HistoryRelaxationDiagnostic,
                          "stellar_radial_ke_erg");
  EnrollUserHistoryOutput(16, HistoryRelaxationDiagnostic,
                          "stellar_nonradial_perturbation_ke_erg");
  EnrollUserHistoryOutput(17, HistoryRelaxationDiagnostic,
                          "stellar_radial_mach2_mass_g");
  EnrollUserHistoryOutput(18, HistoryRelaxationDiagnostic,
                          "stellar_radial_mach_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(19, HistoryRelaxationDiagnostic,
                          "stellar_r_mass_g_cm");
  EnrollUserHistoryOutput(20, HistoryRelaxationDiagnostic,
                          "stellar_r2_mass_g_cm2");
  EnrollUserHistoryOutput(21, HistoryRelaxationDiagnostic,
                          "stellar_pressure_virial_erg");
  EnrollUserHistoryOutput(22, HistoryRelaxationDiagnostic,
                          "stellar_self_gravity_virial_erg");
  EnrollUserHistoryOutput(23, HistoryRelaxationDiagnostic,
                          "stellar_core_gravity_virial_erg");
  EnrollUserHistoryOutput(24, HistoryRelaxationDiagnostic,
                          "stellar_kinetic_energy_erg");
  EnrollUserHistoryOutput(25, HistoryRelaxationDiagnostic,
                          "stellar_perturbation_ke_erg");
  EnrollUserHistoryOutput(26, HistoryRelaxationDiagnostic,
                          "stellar_mass_outside_initial_surface_g");
  EnrollUserHistoryOutput(27, HistoryRelaxationDiagnostic,
                          "stellar_ellipsoidal_radius_mass_g");
  EnrollUserHistoryOutput(28, HistoryRelaxationDiagnostic,
                          "stellar_ellipsoidal_radius_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(kWenozHistoryStart, HistoryWenozCells,
                          "wenoz_available_cells");
  EnrollUserHistoryOutput(kWenozHistoryStart+1, HistoryWenozCells,
                          "wenoz_sensor_fallback_cells");
  EnrollUserHistoryOutput(kWenozHistoryStart+2, HistoryWenozCells,
                          "wenoz_amr_fallback_cells");
  EnrollUserHistoryOutput(kWenozHistoryStart+3, HistoryWenozCells,
                          "wenoz_plm_fallback_cells");
  for (int level = 0; level < kTrackedMeshLevels; ++level) {
    std::stringstream name;
    name << "wenoz_plm_fallback_cells_L" << level;
    EnrollUserHistoryOutput(kWenozLevelHistoryStart+level, HistoryWenozCells,
                            name.str().c_str());
  }
  EnrollUserHistoryOutput(kWenozFallbackMassHistory, HistoryWenozFallbackMass,
                          "wenoz_plm_fallback_stellar_mass_g");
  for (int level = 0; level < kTrackedMeshLevels; ++level) {
    std::stringstream name;
    name << "meshblocks_L" << level;
    EnrollUserHistoryOutput(kMeshBlockHistoryStart+level, HistoryMeshLevel,
                            name.str().c_str());
  }
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart,
                          HistorySurfaceMachDiagnostic,
                          "stellar_radial_mach_max_interior",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+1,
                          HistorySurfaceMachDiagnostic,
                          "stellar_radial_mach_max_surface_taper",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+2,
                          HistorySurfaceMachDiagnostic,
                          "stellar_radial_mach_max_rho_ge_surface",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+3,
                          HistorySurfaceMachDiagnostic,
                          "stellar_radial_mach_max_rho_lt_surface",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+4,
                          HistorySurfaceMachDiagnostic,
                          "stellar_mass_radial_mach_gt_0p1_g");
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+5,
                          HistorySurfaceMachDiagnostic,
                          "stellar_r_mass_radial_mach_gt_0p1_g_cm");
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+6,
                          HistorySurfaceMachDiagnostic,
                          "stellar_logrho_mass_radial_mach_gt_0p1_g");
  EnrollUserHistoryOutput(kSurfaceMachHistoryStart+7,
                          HistorySurfaceMachDiagnostic,
                          "stellar_mass_radial_mach_gt_0p3_g");
#if NRMGFLD_ENABLED
  EnrollUserHistoryOutput(kBaseHistoryCount, HistoryRadiationEnergy,
                          "radiation_energy_erg");
  EnrollUserHistoryOutput(kBaseHistoryCount+1, HistoryGasInternalEnergy,
                          "gas_internal_energy_erg");
  EnrollUserHistoryOutput(kBaseHistoryCount+2, HistoryGravityDiagnostic,
                          "full_force_error2");
  EnrollUserHistoryOutput(kBaseHistoryCount+3, HistoryGravityDiagnostic,
                          "full_force_norm2");
  EnrollUserHistoryOutput(kBaseHistoryCount+4, HistoryGravityDiagnostic,
                          "full_force_relative_max",
                          UserHistoryOperation::max);
#endif
#if EOS_TABLE_ENABLED
  EnrollUserHistoryOutput(kEosTableHistoryStart, HistoryEosTableMargin,
                          "eos_logrho_lower_margin_dex",
                          UserHistoryOperation::min);
  EnrollUserHistoryOutput(kEosTableHistoryStart+1, HistoryEosTableMargin,
                          "eos_logrho_upper_margin_dex",
                          UserHistoryOperation::min);
  EnrollUserHistoryOutput(kEosTableHistoryStart+2, HistoryEosTableMargin,
                          "eos_logu_lower_margin_dex",
                          UserHistoryOperation::min);
  EnrollUserHistoryOutput(kEosTableHistoryStart+3, HistoryEosTableMargin,
                          "eos_logu_upper_margin_dex",
                          UserHistoryOperation::min);
#endif
#if NRMGFLD_ENABLED
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart,
                          HistoryTaperRadiationDiagnostic,
                          "taper_stellar_mass_g");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+1,
                          HistoryTaperRadiationDiagnostic,
                          "taper_radiation_energy_erg");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+2,
                          HistoryTaperRadiationDiagnostic,
                          "taper_gas_internal_energy_erg");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+3,
                          HistoryTaperRadiationDiagnostic,
                          "taper_diffusion_power_erg_s");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+4,
                          HistoryTaperRadiationDiagnostic,
                          "taper_abs_diffusion_power_erg_s");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+5,
                          HistoryTaperRadiationDiagnostic,
                          "taper_exchange_power_erg_s");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+6,
                          HistoryTaperRadiationDiagnostic,
                          "taper_abs_exchange_power_erg_s");
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+7,
                          HistoryTaperRadiationDiagnostic,
                          "taper_diffusion_time_min_s",
                          UserHistoryOperation::min);
  EnrollUserHistoryOutput(kTaperRadiationHistoryStart+8,
                          HistoryTaperRadiationDiagnostic,
                          "taper_exchange_time_min_s",
                          UserHistoryOperation::min);
#endif
  EnrollUserHistoryOutput(kRelaxationHistoryStart,
                          HistoryRelaxationDamping,
                          "relaxation_inertial_ke_delta_erg");
  EnrollUserHistoryOutput(kRelaxationHistoryStart+1,
                          HistoryRelaxationDamping,
                          "relaxation_inertial_power_erg_s");
  EnrollUserHistoryOutput(kRelaxationHistoryStart+2,
                          HistoryRelaxationDamping,
                          "relaxation_corot_ke_removed_erg");
  EnrollUserHistoryOutput(kRelaxationHistoryStart+3,
                          HistoryRelaxationDamping,
                          "relaxation_corot_power_erg_s");
  EnrollUserHistoryOutput(kLuminosityHistory, HistoryCoreLuminosity,
                          "core_luminosity_injected_erg_s");
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
              << " core_luminosity_lsun=" << core_luminosity_lsun
              << " luminosity_heating_radius_cm="
              << luminosity_heating_radius
              << " surface_refinement_level=" << surface_refinement_level
              << " surface_refinement_use_tracer="
              << surface_refinement_use_tracer
              << " surface_refinement_density_g_cm3=" << surface_density
              << " relaxation_velocity_tau=" << relaxation_velocity_tau
              << " relaxation_taper_start=" << relaxation_taper_start
              << " relaxation_taper_end=" << relaxation_taper_end
              << " G_code=" << gravity_code
              << " frame=inertial\n";
  }
}

namespace {

int SurfaceRefinementCondition(MeshBlock* block) {
  const int relative_level = block->loc.level-root_logical_level;

  // The level-4--6 compact-core hierarchy is seeded by the static refinement
  // regions in the input.  Pin those blocks while AMR follows the surface.
  if (relative_level > surface_refinement_level) return 0;

  Real minimum_fraction = 1.0;
  Real maximum_fraction = 0.0;
  bool density_below_target = false;
  bool density_above_target = false;
  for (int k = block->ks; k <= block->ke; ++k) {
    for (int j = block->js; j <= block->je; ++j) {
      for (int i = block->is; i <= block->ie; ++i) {
        const Real density = block->phydro->u(IDN,k,j,i);
        if (density <= 0.0) continue;
        const Real fraction = std::max(static_cast<Real>(0.0),
            std::min(static_cast<Real>(1.0),
                     block->pscalars->s(0,k,j,i)/density));
        const Real stellar_density = std::max(
            static_cast<Real>(0.0), block->pscalars->s(0,k,j,i));
        minimum_fraction = std::min(minimum_fraction, fraction);
        maximum_fraction = std::max(maximum_fraction, fraction);
        density_below_target = density_below_target
            || stellar_density <= surface_density_code;
        density_above_target = density_above_target
            || stellar_density >= surface_density_code;
      }
    }
  }
  // The tracer interval deliberately follows all partially mixed stellar
  // material and can therefore select a thick volume when rho_atm/rho is not
  // negligible.  MESA-fit models may instead request the much narrower
  // density isosurface alone.
  const bool crosses_surface = surface_refinement_use_tracer
      && maximum_fraction > surface_tracer_lower
      && minimum_fraction < surface_tracer_upper;
  const bool crosses_density = density_below_target && density_above_target;
  const bool resolves_surface = crosses_surface || crosses_density;
  if (resolves_surface && relative_level < surface_refinement_level) return 1;
  if (resolves_surface) return 0;
  return -1;
}

}  // namespace

void MeshBlock::InitUserMeshBlockData(ParameterInput*) {
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(kRelaxationHistoryCount);
  ruser_meshblock_data[0].ZeroClear();
#if NRMGFLD_ENABLED
  prfld->EnrollOpacityFunction(Opacity);
  for (int face = 0; face < 6; ++face) {
    prfld->marshak_face_boundary[face] = true;
  }
  prfld->marshak_top_alpha = marshak_alpha;
  prfld->marshak_top_erad_ext = external_radiation_energy_code;
  AllocateUserOutputVariables(21);
#else
  AllocateUserOutputVariables(13);
#endif
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
#if NRMGFLD_ENABLED
  SetUserOutputVariableName(13, "Erad_cgs");
  SetUserOutputVariableName(14, "Tgas_fld");
  SetUserOutputVariableName(15, "Trad_fld");
  SetUserOutputVariableName(16, "radiation_accel");
  SetUserOutputVariableName(17, "full_force_residual");
  SetUserOutputVariableName(18, "full_force_scale");
  SetUserOutputVariableName(19, "fld_diffusion_heating");
  SetUserOutputVariableName(20, "fld_thermal_exchange");
#endif
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
#if NRMGFLD_ENABLED
        Real cell_temperature = 0.0;
        Real cell_gas_internal_code = 0.0;
        Real cell_radiation_energy_code = external_radiation_energy_code;
        if (mapped.stellar_density > 0.0) {
          const LteSplit split = SplitTotalPressureLte(
              this, mapped.density, mapped.pressure);
          cell_temperature = split.temperature;
          cell_gas_internal_code = split.gas_energy;
          cell_radiation_energy_code = split.radiation_energy;
        } else {
          const Real density_code = mapped.density/density_scale;
          const Real pressure_code = mapped.pressure/energy_scale;
#if GENERAL_EOS
          cell_gas_internal_code = peos->EgasFromRhoP(
              density_code, pressure_code);
#else
          cell_gas_internal_code = pressure_code/(gamma_gas-1.0);
#endif
          cell_temperature = GasTemperatureCgs(
              this, density_code, pressure_code);
        }
        Require(std::isfinite(cell_gas_internal_code)
                    && cell_gas_internal_code > 0.0,
                "EOS returned invalid initial gas energy");
        phydro->u(IEN,k,j,i) =
            cell_gas_internal_code+mapped.kinetic_energy/energy_scale;
        prfld->u_gas(k,j,i) = cell_gas_internal_code;
        prfld->u_rad(k,j,i) = cell_radiation_energy_code;
#else
        phydro->u(IEN,k,j,i) = mapped.total_energy/energy_scale;
#endif
#if NSCALARS > 0
        pscalars->s(0,k,j,i) = mapped.stellar_density/density_scale;
#endif
#if NSCALARS > 1
        pscalars->s(1,k,j,i) = mapped.radiative_density/density_scale;
#endif
        const Real rho = std::max(mapped.density, TINY_NUMBER);
#if NRMGFLD_ENABLED
        user_out_var(0,k,j,i) = cell_temperature;
#else
        user_out_var(0,k,j,i) = mapped.density_temperature/rho;
#endif
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
#if NRMGFLD_ENABLED
        user_out_var(13,k,j,i) = prfld->u_rad(k,j,i)*energy_scale;
        user_out_var(14,k,j,i) = cell_temperature;
        user_out_var(15,k,j,i) = std::pow(
            std::max(user_out_var(13,k,j,i), TINY_NUMBER)
              /scf_map::kRadiationConstantCgs, 0.25);
#endif
      }
    }
  }
}

void MeshBlock::UserWorkInLoop() {
  ruser_meshblock_data[0].ZeroClear();
  if (relaxation_velocity_tau <= 0.0) return;
  const Real strength = scf_relaxation::TaperStrength(
      pmy_mesh->time, relaxation_taper_start, relaxation_taper_end);
  if (strength <= 0.0) return;
  const Real dt = pmy_mesh->dt;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      const Real y = pcoord->x2v(j);
      for (int i = is; i <= ie; ++i) {
        const Real density = phydro->u(IDN,k,j,i);
        if (!(density > 0.0)) continue;
        const Real stellar_fraction = std::max(static_cast<Real>(0.0),
            std::min(static_cast<Real>(1.0),
                     pscalars->s(0,k,j,i)/density));
        if (stellar_fraction <= 0.0) continue;
        const Real vx = phydro->u(IM1,k,j,i)/density;
        const Real vy = phydro->u(IM2,k,j,i)/density;
        const Real vz = phydro->u(IM3,k,j,i)/density;
        const Real dvx = vx+angular_velocity_code*y;
        const Real dvy = vy-angular_velocity_code*pcoord->x1v(i);
        const Real corot_kinetic = 0.5*density*(dvx*dvx+dvy*dvy+vz*vz);
        const Real factor = std::exp(
            -strength*stellar_fraction*dt/relaxation_velocity_tau);
        const auto velocity = scf_relaxation::DampTowardCorotation(
            vx, vy, vz, pcoord->x1v(i), y, angular_velocity_code, factor);
        const Real old_kinetic = 0.5*density*(vx*vx+vy*vy+vz*vz);
        const Real new_kinetic = 0.5*density*(velocity.x*velocity.x
            +velocity.y*velocity.y+velocity.z*velocity.z);
        const Real kinetic_delta = new_kinetic-old_kinetic;
        phydro->u(IM1,k,j,i) = density*velocity.x;
        phydro->u(IM2,k,j,i) = density*velocity.y;
        phydro->u(IM3,k,j,i) = density*velocity.z;
        // Change total energy by exactly the kinetic-energy change so the
        // gas internal energy and the implicit FLD cache remain untouched.
        phydro->u(IEN,k,j,i) += kinetic_delta;
        phydro->w(IVX,k,j,i) = velocity.x;
        phydro->w(IVY,k,j,i) = velocity.y;
        phydro->w(IVZ,k,j,i) = velocity.z;
        const Real volume = pcoord->GetCellVolume(k,j,i);
        const Real corot_kinetic_removed = corot_kinetic*(1.0-factor*factor);
        ruser_meshblock_data[0](0) += kinetic_delta*volume;
        ruser_meshblock_data[0](1) += kinetic_delta*volume
            /std::max(dt, TINY_NUMBER);
        ruser_meshblock_data[0](2) += corot_kinetic_removed*volume;
        ruser_meshblock_data[0](3) += corot_kinetic_removed*volume
            /std::max(dt, TINY_NUMBER);
      }
    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput*) {
  UpdateGravityDiagnostics(this);
#if NRMGFLD_ENABLED
  UpdateTaperRadiationDiagnostics(this);
#endif
}
