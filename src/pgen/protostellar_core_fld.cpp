//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file protostellar_core_fld.cpp
//! \brief Simplified rotating, magnetized protostellar-core collapse with grey FLD.
//!
//! This intentionally omits the non-ideal MHD, chemistry, tabulated EOS, sink particles,
//! and H2 dissociation used by Mayer et al. (2025/2026).  In particular, a fixed-gamma
//! ideal gas cannot follow second collapse or form a second Larson core.

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
#include "../field/field.hpp"
#include "../fld/fld.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../nr_multigrid/NRFLD.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#if !NRMGFLD_ENABLED
#error "protostellar_core_fld requires the implicit FLD solver (-nrmgfld)."
#endif
#if SELF_GRAVITY_ENABLED != 2
#error "protostellar_core_fld requires multigrid self-gravity (--grav=mg)."
#endif
#if !MAGNETIC_FIELDS_ENABLED
#error "protostellar_core_fld requires ideal MHD (-b). " \
       "Use problem/enable_magnetic_field=false to initialize B=0."
#endif
#if !NON_BAROTROPIC_EOS
#error "protostellar_core_fld requires an adiabatic EOS (--eos=adiabatic)."
#endif

namespace {

constexpr Real kPi = 3.141592653589793238462643383279502884;
constexpr Real kG = 6.67430e-8;             // cm^3 g^-1 s^-2
constexpr Real kAu = 1.495978707e13;        // cm
constexpr Real kMsun = 1.98847e33;          // g
constexpr Real kYear = 3.15576e7;           // s
constexpr Real kRadiationConst = 7.5657e-15;  // erg cm^-3 K^-4
// Keep this bit-for-bit consistent with FLD::FLD so Er=aT^4 is an exact
// fixed point of the stiff matter-radiation exchange.
constexpr Real kGasConstant = 8.3144621e7;    // erg K^-1 mol^-1

Real rho_unit, egas_unit, length_unit, velocity_unit, time_unit, magnetic_unit;
Real mass_core, radius_core, temperature_core, mean_molecular_weight;
Real density_core, density_ambient, omega_phys, b0_phys;
Real kappa_planck, kappa_rosseland, transition_width;
Real radius_core_code, transition_width_code;
Real gravity_code, jeans_number;
Real background_temperature, background_tracer_threshold;
Real target_cell_mass, mass_refine_factor, mass_derefine_factor;
bool enable_rotation, enable_magnetic_field, uniform_equilibrium;
bool fix_background_temperature, use_mass_refinement;

enum InitialDataIndex {INITIAL_RHO=0, INITIAL_EGAS=1, INITIAL_ERAD=2,
                       NINITIAL_DATA=3};

Real CoreWeight(Real radius_code) {
  if (uniform_equilibrium) return 1.0;
  if (transition_width_code <= 0.0) return radius_code <= radius_core_code ? 1.0 : 0.0;
  const Real inner = radius_core_code - 0.5*transition_width_code;
  const Real outer = radius_core_code + 0.5*transition_width_code;
  if (radius_code <= inner) return 1.0;
  if (radius_code >= outer) return 0.0;
  const Real q = (radius_code - inner)/transition_width_code;
  return 0.5*(1.0 + std::cos(kPi*q));
}

Real GasTemperature(const MeshBlock *pmb, int k, int j, int i) {
  const Real rho_phys = std::max(pmb->phydro->w(IDN,k,j,i)*rho_unit, TINY_NUMBER);
  const Real pressure_phys = std::max(pmb->phydro->w(IPR,k,j,i)*egas_unit, 0.0);
  return pressure_phys*mean_molecular_weight/(rho_phys*kGasConstant);
}

Real RadiationTemperature(const MeshBlock *pmb, int k, int j, int i) {
  const Real erad = std::max(pmb->prfld->u_rad(k,j,i)*egas_unit, 0.0);
  return std::pow(erad/kRadiationConst, 0.25);
}

void CoreGravityMask(AthenaArray<Real> &src, int is, int ie, int js, int je,
                     int ks, int ke, const MGCoordinates &coord) {
  if (uniform_equilibrium) {
    // Retain the uniform source shape so multipole auto-centering has a
    // well-defined nonzero monopole.  Its coupling constant is suppressed in
    // Mesh::InitUserMeshData, making the resulting acceleration negligible.
    return;
  }
  const Real outer = radius_core_code + 0.5*transition_width_code;
  for (int k=ks; k<=ke; ++k) {
    const Real z = coord.x3v(k);
    for (int j=js; j<=je; ++j) {
      const Real y = coord.x2v(j);
      for (int i=is; i<=ie; ++i) {
        const Real x = coord.x1v(i);
        if (std::sqrt(SQR(x)+SQR(y)+SQR(z)) > outer) src(k,j,i) = 0.0;
      }
    }
  }
}

void DensityOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
                    AthenaArray<Real> &prim) {
  (void)u_fld;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  int jl=pmb->js, ju=pmb->je, kl=pmb->ks, ku=pmb->ke;
  if (pmb->block_size.nx2 > 1) { jl -= NGHOST; ju += NGHOST; }
  if (pmb->block_size.nx3 > 1) { kl -= NGHOST; ku += NGHOST; }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        const Real rho_phys = std::max(prim(IDN,k,j,i)*rho_unit, TINY_NUMBER);
        // FLD stores inverse-length coefficients in code units: sigma=kappa*rho*L.
        pmb->prfld->sigma_p(k,j,i) = kappa_planck*rho_phys*length_unit;
        pmb->prfld->sigma_r(k,j,i) = std::max(
            kappa_rosseland*rho_phys*length_unit, TINY_NUMBER);
      }
    }
  }
}

Real HistoryRhoMax(MeshBlock *pmb, int) {
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,pmb->phydro->w(IDN,k,j,i)*rho_unit);
  return out;
}

Real HistoryPMax(MeshBlock *pmb, int) {
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,pmb->phydro->w(IPR,k,j,i)*egas_unit);
  return out;
}

Real HistoryRhoMin(MeshBlock *pmb, int) {
  Real out=std::numeric_limits<Real>::max();
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::min(out,pmb->phydro->w(IDN,k,j,i)*rho_unit);
  return out;
}

Real HistoryPMin(MeshBlock *pmb, int) {
  Real out=std::numeric_limits<Real>::max();
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::min(out,pmb->phydro->w(IPR,k,j,i)*egas_unit);
  return out;
}

Real HistoryErMax(MeshBlock *pmb, int) {
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,pmb->prfld->u_rad(k,j,i)*egas_unit);
  return out;
}

Real HistoryErMin(MeshBlock *pmb, int) {
  Real out=std::numeric_limits<Real>::max();
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::min(out,pmb->prfld->u_rad(k,j,i)*egas_unit);
  return out;
}

Real HistoryTgasMax(MeshBlock *pmb, int) {
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,GasTemperature(pmb,k,j,i));
  return out;
}

Real HistoryTradMax(MeshBlock *pmb, int) {
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,RadiationTemperature(pmb,k,j,i));
  return out;
}

Real HistoryBMax(MeshBlock *pmb, int) {
  pmb->pfield->CalculateCellCenteredField(pmb->pfield->b,pmb->pfield->bcc,
      pmb->pcoord,pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke);
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,std::sqrt(SQR(pmb->pfield->bcc(IB1,k,j,i))
            +SQR(pmb->pfield->bcc(IB2,k,j,i))+SQR(pmb->pfield->bcc(IB3,k,j,i)))
            *magnetic_unit);
  return out;
}

Real HistoryMass(MeshBlock *pmb, int) {
  Real out=0.0;
  const Real volume_unit=std::pow(length_unit,3);
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out += pmb->phydro->w(IDN,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i)
               *rho_unit*volume_unit;
  return out;
}

Real HistoryInternalEnergy(MeshBlock *pmb, int) {
  Real out=0.0;
  const Real scale=egas_unit*std::pow(length_unit,3);
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out += pmb->prfld->u_gas(k,j,i)*pmb->pcoord->GetCellVolume(k,j,i)*scale;
  return out;
}

Real HistoryKineticEnergy(MeshBlock *pmb, int) {
  Real out=0.0;
  const Real scale=egas_unit*std::pow(length_unit,3);
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real rho=pmb->phydro->w(IDN,k,j,i);
        const Real vsq=SQR(pmb->phydro->w(IVX,k,j,i))
                       +SQR(pmb->phydro->w(IVY,k,j,i))
                       +SQR(pmb->phydro->w(IVZ,k,j,i));
        out += 0.5*rho*vsq*pmb->pcoord->GetCellVolume(k,j,i)*scale;
      }
  return out;
}

Real HistoryMagneticEnergy(MeshBlock *pmb, int) {
  pmb->pfield->CalculateCellCenteredField(pmb->pfield->b,pmb->pfield->bcc,
      pmb->pcoord,pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke);
  Real out=0.0;
  const Real scale=egas_unit*std::pow(length_unit,3);
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real bsq=SQR(pmb->pfield->bcc(IB1,k,j,i))
                        +SQR(pmb->pfield->bcc(IB2,k,j,i))
                        +SQR(pmb->pfield->bcc(IB3,k,j,i));
        out += 0.5*bsq*pmb->pcoord->GetCellVolume(k,j,i)*scale;
      }
  return out;
}

Real HistoryRadiationEnergy(MeshBlock *pmb, int) {
  Real out=0.0;
  const Real scale=egas_unit*std::pow(length_unit,3);
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out += pmb->prfld->u_rad(k,j,i)*pmb->pcoord->GetCellVolume(k,j,i)*scale;
  return out;
}

Real JeansCells(const MeshBlock *pmb, int k, int j, int i) {
  Real prim[NWAVE] = {};
  prim[IDN]=std::max(pmb->phydro->w(IDN,k,j,i),TINY_NUMBER);
  prim[IPR]=std::max(pmb->phydro->w(IPR,k,j,i),TINY_NUMBER);
  const Real cs=pmb->peos->SoundSpeed(prim);
  const Real lambda=std::sqrt(kPi*SQR(cs)/(gravity_code*prim[IDN]));
  Real dx=pmb->pcoord->dx1f(i);
  if (pmb->block_size.nx2 > 1) dx=std::min(dx,pmb->pcoord->dx2f(j));
  if (pmb->block_size.nx3 > 1) dx=std::min(dx,pmb->pcoord->dx3f(k));
  return lambda/dx;
}

Real HistoryJeansMin(MeshBlock *pmb, int) {
  Real out=std::numeric_limits<Real>::max();
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::min(out,JeansCells(pmb,k,j,i));
  return out;
}

Real HistoryLevelMax(MeshBlock *pmb, int) {
  return static_cast<Real>(pmb->loc.level);
}

Real HistoryCellMassMax(MeshBlock *pmb, int) {
  Real out=0.0;
  const Real scale=rho_unit*std::pow(length_unit,3)/kMsun;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        out=std::max(out,pmb->phydro->w(IDN,k,j,i)
                         *pmb->pcoord->GetCellVolume(k,j,i)*scale);
  return out;
}

Real HistoryBackgroundTgasError(MeshBlock *pmb, int) {
  Real out=0.0;
#if NSCALARS > 0
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        if (pmb->pscalars->s(0,k,j,i)
            /std::max(pmb->phydro->u(IDN,k,j,i),TINY_NUMBER)
            >= background_tracer_threshold)
          out=std::max(out,std::abs(GasTemperature(pmb,k,j,i)/background_temperature-1.0));
#endif
  return out;
}

Real HistoryBackgroundTradError(MeshBlock *pmb, int) {
  Real out=0.0;
#if NSCALARS > 0
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i)
        if (pmb->pscalars->s(0,k,j,i)
            /std::max(pmb->phydro->u(IDN,k,j,i),TINY_NUMBER)
            >= background_tracer_threshold)
          out=std::max(out,std::abs(RadiationTemperature(pmb,k,j,i)
                                    /background_temperature-1.0));
#endif
  return out;
}

int JeansRefinementCondition(MeshBlock *pmb) {
  const Real cells=HistoryJeansMin(pmb,0);
  Real max_cell_mass=0.0;
  if (use_mass_refinement) {
    const Real scale=rho_unit*std::pow(length_unit,3);
    for (int k=pmb->ks; k<=pmb->ke; ++k)
      for (int j=pmb->js; j<=pmb->je; ++j)
        for (int i=pmb->is; i<=pmb->ie; ++i)
          max_cell_mass=std::max(max_cell_mass,pmb->phydro->w(IDN,k,j,i)
              *pmb->pcoord->GetCellVolume(k,j,i)*scale);
  }
  if (cells < jeans_number
      || (use_mass_refinement && max_cell_mass > mass_refine_factor*target_cell_mass))
    return 1;
  if (cells > 2.5*jeans_number
      && (!use_mass_refinement
          || max_cell_mass < mass_derefine_factor*target_cell_mass)) return -1;
  return 0;
}

Real HistoryLz(MeshBlock *pmb, int) {
  Real out=0.0;
  const Real scale=rho_unit*std::pow(length_unit,4)*velocity_unit;
  for (int k=pmb->ks; k<=pmb->ke; ++k)
    for (int j=pmb->js; j<=pmb->je; ++j)
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        out += (pmb->pcoord->x1v(i)*pmb->phydro->u(IM2,k,j,i)
              -pmb->pcoord->x2v(j)*pmb->phydro->u(IM1,k,j,i))
              *pmb->pcoord->GetCellVolume(k,j,i)*scale;
      }
  return out;
}

Real HistoryDivBMax(MeshBlock *pmb, int) {
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real div=(pmb->pfield->b.x1f(k,j,i+1)-pmb->pfield->b.x1f(k,j,i))
                           /pmb->pcoord->dx1f(i)
                      +(pmb->pfield->b.x2f(k,j+1,i)-pmb->pfield->b.x2f(k,j,i))
                           /pmb->pcoord->dx2f(j)
                      +(pmb->pfield->b.x3f(k+1,j,i)-pmb->pfield->b.x3f(k,j,i))
                           /pmb->pcoord->dx3f(k);
        out=std::max(out,std::abs(div));
      }
    }
  }
  return out;
}

Real InitialRelativeChange(MeshBlock *pmb, int component) {
  const AthenaArray<Real> &initial=pmb->ruser_meshblock_data[0];
  Real out=0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real value;
        if (component == INITIAL_RHO) {
          value=pmb->phydro->w(IDN,k,j,i);
        } else if (component == INITIAL_EGAS) {
          value=pmb->prfld->u_gas(k,j,i);
        } else {
          value=pmb->prfld->u_rad(k,j,i);
        }
        const Real reference=initial(component,k,j,i);
        out=std::max(out,std::abs(value-reference)
                         /std::max(std::abs(reference),TINY_NUMBER));
      }
    }
  }
  return out;
}

Real HistoryRhoRel(MeshBlock *pmb, int) {
  return InitialRelativeChange(pmb,INITIAL_RHO);
}
Real HistoryEgasRel(MeshBlock *pmb, int) {
  return InitialRelativeChange(pmb,INITIAL_EGAS);
}
Real HistoryErRel(MeshBlock *pmb, int) { return InitialRelativeChange(pmb,INITIAL_ERAD); }

}  // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!pin->GetBoolean("fld","is_couple") || pin->GetBoolean("fld","only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld\n"
        << "fld/is_couple must be true and fld/only_rad must be false.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetOrAddString("fld","pressure_coupling","flux") != "flux") {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld\n"
        << "protostellar_core_fld requires the HLLD-FLD flux coupling mode.";
    ATHENA_ERROR(msg);
  }
  const Real gamma=pin->GetReal("hydro","gamma");
  if (std::abs(gamma-5.0/3.0) > 1.0e-10) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: hydro/gamma must be 5/3.";
    ATHENA_ERROR(msg);
  }

  rho_unit=pin->GetReal("hydro","rho_unit");
  egas_unit=pin->GetReal("hydro","egas_unit");
  length_unit=pin->GetReal("hydro","leng_unit");
  velocity_unit=std::sqrt(egas_unit/rho_unit);
  time_unit=length_unit/velocity_unit;
  magnetic_unit=std::sqrt(4.0*kPi*egas_unit);

  mass_core=pin->GetOrAddReal("problem","M_core",1.0)*kMsun;
  radius_core=pin->GetOrAddReal("problem","R_core",3000.0)*kAu;
  temperature_core=pin->GetOrAddReal("problem","T_core",14.0);
  mean_molecular_weight=pin->GetOrAddReal("problem","mu",2.381);
  omega_phys=pin->GetOrAddReal("problem","Omega",1.48e-13);
  b0_phys=pin->GetOrAddReal("problem","B0",129.0e-6);
  const Real ambient_factor=pin->GetOrAddReal("problem","rho_ambient_factor",1.0e-2);
  if (!(ambient_factor > 0.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: rho_ambient_factor must be positive.";
    ATHENA_ERROR(msg);
  }
  transition_width=pin->GetOrAddReal("problem","transition_width",0.0)*kAu;
  enable_rotation=pin->GetOrAddBoolean("problem","enable_rotation",true);
  enable_magnetic_field=pin->GetOrAddBoolean("problem","enable_magnetic_field",true);
  const std::string test_mode=pin->GetOrAddString("problem","test_mode","collapse");
  if (test_mode != "collapse" && test_mode != "uniform_equilibrium") {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: problem/test_mode must be "
        << "collapse or uniform_equilibrium.";
    ATHENA_ERROR(msg);
  }
  uniform_equilibrium=(test_mode == "uniform_equilibrium");
  if (uniform_equilibrium) enable_rotation=false;

  fix_background_temperature=pin->GetOrAddBoolean(
      "problem","fix_background_temperature",false);
  background_tracer_threshold=pin->GetOrAddReal(
      "problem","background_tracer_threshold",0.5);
  const Real background_pressure_ratio=pin->GetOrAddReal(
      "problem","background_pressure_ratio",ambient_factor);
  background_temperature=temperature_core*background_pressure_ratio/ambient_factor;
  if (!(background_temperature > 0.0)
      || background_tracer_threshold < 0.0 || background_tracer_threshold > 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: invalid background heat-sink "
        << "temperature or tracer threshold.";
    ATHENA_ERROR(msg);
  }
  if (fix_background_temperature && NSCALARS < 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: the Mayer background heat sink "
        << "requires --nscalars=1.";
    ATHENA_ERROR(msg);
  }

  kappa_planck=pin->GetOrAddReal("radiation","kappa_P",1.0);
  kappa_rosseland=pin->GetOrAddReal("radiation","kappa_R",1.0);
  density_core=3.0*mass_core/(4.0*kPi*std::pow(radius_core,3));
  density_ambient=ambient_factor*density_core;
  radius_core_code=radius_core/length_unit;
  transition_width_code=transition_width/length_unit;
  gravity_code=kG*rho_unit*SQR(time_unit);
  jeans_number=pin->GetOrAddReal("problem","N_J",16.0);
  const std::string refinement_mode=pin->GetOrAddString(
      "problem","refinement_mode","jeans");
  if (refinement_mode != "jeans" && refinement_mode != "jeans_and_mass") {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: problem/refinement_mode must be "
        << "jeans or jeans_and_mass.";
    ATHENA_ERROR(msg);
  }
  use_mass_refinement=(refinement_mode == "jeans_and_mass");
  target_cell_mass=pin->GetOrAddReal("problem","target_cell_mass_msun",3.33e-7)*kMsun;
  mass_refine_factor=pin->GetOrAddReal("problem","mass_refine_factor",2.0);
  mass_derefine_factor=pin->GetOrAddReal("problem","mass_derefine_factor",0.25);
  if (!(target_cell_mass > 0.0) || !(mass_refine_factor > 0.0)
      || !(mass_derefine_factor > 0.0)
      || mass_derefine_factor >= mass_refine_factor) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: invalid mass-refinement parameters.";
    ATHENA_ERROR(msg);
  }
  if (!(jeans_number > 0.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in protostellar_core_fld: problem/N_J must be positive.";
    ATHENA_ERROR(msg);
  }

  if (uniform_equilibrium) {
    // Gravity::Gravity rejects exactly zero, while an exactly zero masked source makes
    // automatic multipole centering evaluate 0/0.  A 1e-30 coupling is numerically
    // negligible over this test but keeps the isolated-gravity machinery well-defined.
    SetGravitationalConstant(gravity_code*1.0e-30);
  } else {
    SetGravitationalConstant(gravity_code);
  }
  EnrollUserMGGravitySourceMaskFunction(CoreGravityMask);
  if (adaptive) EnrollUserRefinementCondition(JeansRefinementCondition);

  AllocateUserHistoryOutput(24);
  EnrollUserHistoryOutput(0,HistoryRhoMax,"rho_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(1,HistoryPMax,"P_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(2,HistoryErMax,"Er_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(3,HistoryErMin,"Er_min",UserHistoryOperation::min);
  EnrollUserHistoryOutput(4,HistoryTgasMax,"Tgas_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(5,HistoryTradMax,"Trad_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(6,HistoryBMax,"B_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(7,HistoryMass,"gas_mass",UserHistoryOperation::sum);
  EnrollUserHistoryOutput(8,HistoryDivBMax,"divB_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(9,HistoryLz,"Lz",UserHistoryOperation::sum);
  EnrollUserHistoryOutput(10,HistoryRhoRel,"rho_rel",UserHistoryOperation::max);
  EnrollUserHistoryOutput(11,HistoryEgasRel,"egas_rel",UserHistoryOperation::max);
  EnrollUserHistoryOutput(12,HistoryErRel,"Er_rel",UserHistoryOperation::max);
  EnrollUserHistoryOutput(13,HistoryRhoMin,"rho_min",UserHistoryOperation::min);
  EnrollUserHistoryOutput(14,HistoryPMin,"P_min",UserHistoryOperation::min);
  EnrollUserHistoryOutput(15,HistoryInternalEnergy,"Eint_gas_cgs",UserHistoryOperation::sum);
  EnrollUserHistoryOutput(16,HistoryKineticEnergy,"Ekin_gas_cgs",UserHistoryOperation::sum);
  EnrollUserHistoryOutput(17,HistoryMagneticEnergy,"Emag_cgs",UserHistoryOperation::sum);
  EnrollUserHistoryOutput(18,HistoryRadiationEnergy,"Erad_tot_cgs",UserHistoryOperation::sum);
  EnrollUserHistoryOutput(19,HistoryJeansMin,"jeans_min",UserHistoryOperation::min);
  EnrollUserHistoryOutput(20,HistoryLevelMax,"level_max",UserHistoryOperation::max);
  EnrollUserHistoryOutput(21,HistoryBackgroundTgasError,"Tbg_rel",UserHistoryOperation::max);
  EnrollUserHistoryOutput(22,HistoryBackgroundTradError,"Trad_bg_rel",UserHistoryOperation::max);
  EnrollUserHistoryOutput(23,HistoryCellMassMax,"cell_mass_max_msun",
                          UserHistoryOperation::max);

  // Measure the cell-centred core representation on the uniform Cartesian root grid.
  const Real dx=(mesh_size.x1max-mesh_size.x1min)/mesh_size.nx1;
  const Real dy=(mesh_size.x2max-mesh_size.x2min)/mesh_size.nx2;
  const Real dz=(mesh_size.x3max-mesh_size.x3min)/mesh_size.nx3;
  Real measured_mass=0.0, vmax=0.0;
  for (int k=0; k<mesh_size.nx3; ++k) {
    const Real z=mesh_size.x3min+(k+0.5)*dz;
    for (int j=0; j<mesh_size.nx2; ++j) {
      const Real y=mesh_size.x2min+(j+0.5)*dy;
      for (int i=0; i<mesh_size.nx1; ++i) {
        const Real x=mesh_size.x1min+(i+0.5)*dx;
        const Real w=CoreWeight(std::sqrt(SQR(x)+SQR(y)+SQR(z)));
        measured_mass += density_core*w*dx*dy*dz*std::pow(length_unit,3);
        vmax=std::max(vmax,enable_rotation ? omega_phys*length_unit
                          *std::sqrt(SQR(x)+SQR(y))*w : 0.0);
      }
    }
  }
  if (Globals::my_rank == 0 && ncycle == 0) {
    const Real pressure=density_core*kGasConstant*temperature_core
                        /mean_molecular_weight;
    const Real erad=kRadiationConst*std::pow(temperature_core,4);
    const Real tff=std::sqrt(3.0*kPi/(32.0*kG*density_core));
    const Real tau_core=kappa_rosseland*density_core*radius_core;
    const Real ethermal=mass_core*kGasConstant*temperature_core
                        /(mean_molecular_weight*(gamma-1.0));
    const Real egrav=3.0*kG*SQR(mass_core)/(5.0*radius_core);
    const Real erot=0.2*mass_core*SQR(radius_core)*SQR(omega_phys);
    const int numlevel=pin->GetOrAddInteger("mesh","numlevel",1);
    const Real root_dx=std::min(dx,std::min(dy,dz))*length_unit/kAu;
    const Real finest_dx=root_dx/std::pow(2.0,numlevel-1);
    std::cout << std::setprecision(10)
      << "\n--- simplified protostellar core FLD initial condition ---\n"
      << "mode                     = " << test_mode << "\n"
      << "rho_core                 = " << density_core << " g cm^-3\n"
      << (uniform_equilibrium ? "uniform box mass         = "
                              : "core mass on grid        = ")
      << measured_mass/kMsun << " Msun\n";
    if (!uniform_equilibrium) std::cout
      << "expected M_core          = " << mass_core/kMsun << " Msun\n"
      << "relative core mass error = " << (measured_mass-mass_core)/mass_core << "\n";
    std::cout
      << "initial gas temperature  = " << pressure*mean_molecular_weight
                                             /(density_core*kGasConstant) << " K\n"
      << "initial rad temperature  = " << std::pow(erad/kRadiationConst,0.25) << " K\n"
      << "max core rotation speed  = " << vmax << " cm s^-1\n"
      << "B0                       = " << (enable_magnetic_field ? b0_phys : 0.0)
                                             << " G\n"
      << "kappa_P                  = " << kappa_planck << " cm^2 g^-1\n"
      << "kappa_R                  = " << kappa_rosseland << " cm^2 g^-1\n"
      << "tau_core                 = " << tau_core << "\n"
      << "alpha=Ethermal/|Egrav|   = " << ethermal/egrav << "\n"
      << "beta=Erot/|Egrav|        = " << (enable_rotation ? erot/egrav : 0.0) << "\n"
      << "free-fall time           = " << tff/kYear << " yr\n"
      << "Jeans target N_J         = " << jeans_number << " cells\n"
      << "refinement mode          = " << refinement_mode << "\n"
      << "target cell mass         = " << target_cell_mass/kMsun << " Msun\n"
      << "background temperature   = " << background_temperature << " K\n"
      << "background heat sink     = " << fix_background_temperature << "\n"
      << "root/minimum dx          = " << root_dx << " / " << finest_dx << " au\n"
      << "hydro/FLD boundaries     = periodic\n"
      << "gravity boundaries       = isolated multipole (set in athinput)\n"
      << "---------------------------------------------------------\n";
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  (void)pin;
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(NINITIAL_DATA,ncells3,ncells2,ncells1);
  AllocateUserOutputVariables(7);
  SetUserOutputVariableName(0,"rho_cgs");
  SetUserOutputVariableName(1,"Pgas_cgs");
  SetUserOutputVariableName(2,"Tgas_K");
  SetUserOutputVariableName(3,"Erad_cgs");
  SetUserOutputVariableName(4,"Trad_K");
  SetUserOutputVariableName(5,"B_G");
  SetUserOutputVariableName(6,"background_tracer");
  prfld->EnrollOpacityFunction(DensityOpacity);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  (void)pin;
  const Real gm1=peos->GetGamma()-1.0;
  const Real omega_code=omega_phys*time_unit;
  const Real b_code=enable_magnetic_field ? b0_phys/magnetic_unit : 0.0;

  for (int k=ks; k<=ke; ++k) {
    const Real z=pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      const Real y=pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        const Real x=pcoord->x1v(i);
        const Real weight=CoreWeight(std::sqrt(SQR(x)+SQR(y)+SQR(z)));
        const Real rho_phys=uniform_equilibrium ? density_core
            : density_ambient+(density_core-density_ambient)*weight;
        const Real temperature=uniform_equilibrium ? temperature_core
            : background_temperature+(temperature_core-background_temperature)*weight;
        const Real rho=rho_phys/rho_unit;
        const Real pressure=rho_phys*kGasConstant*temperature
                            /mean_molecular_weight/egas_unit;
        const Real vx=enable_rotation ? -omega_code*y*weight : 0.0;
        const Real vy=enable_rotation ?  omega_code*x*weight : 0.0;
        phydro->u(IDN,k,j,i)=rho;
        phydro->u(IM1,k,j,i)=rho*vx;
        phydro->u(IM2,k,j,i)=rho*vy;
        phydro->u(IM3,k,j,i)=0.0;
        const Real egas=pressure/gm1;
        phydro->u(IEN,k,j,i)=egas+0.5*rho*(SQR(vx)+SQR(vy))+0.5*SQR(b_code);
        prfld->u_gas(k,j,i)=egas;
        prfld->u_rad(k,j,i)=kRadiationConst*std::pow(temperature,4)/egas_unit;
#if NSCALARS > 0
        pscalars->s(0,k,j,i)=rho*(1.0-weight);
#endif
        ruser_meshblock_data[0](INITIAL_RHO,k,j,i)=rho;
        ruser_meshblock_data[0](INITIAL_EGAS,k,j,i)=egas;
        ruser_meshblock_data[0](INITIAL_ERAD,k,j,i)=prfld->u_rad(k,j,i);
      }
    }
  }

  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie+1; ++i) pfield->b.x1f(k,j,i)=0.0;
  for (int k=ks; k<=ke; ++k)
    for (int j=js; j<=je+1; ++j)
      for (int i=is; i<=ie; ++i) pfield->b.x2f(k,j,i)=0.0;
  for (int k=ks; k<=ke+1; ++k)
    for (int j=js; j<=je; ++j)
      for (int i=is; i<=ie; ++i) pfield->b.x3f(k,j,i)=b_code;
}

void MeshBlock::UserWorkInLoop() {
  if (!fix_background_temperature) return;
#if NSCALARS > 0
  const Real gm1=peos->GetGamma()-1.0;
  const Real erad_target=kRadiationConst*std::pow(background_temperature,4)/egas_unit;
  pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,
                                      is,ie,js,je,ks,ke);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const Real rho=std::max(phydro->u(IDN,k,j,i),TINY_NUMBER);
        const Real background_fraction=pscalars->s(0,k,j,i)/rho;
        if (background_fraction < background_tracer_threshold) continue;
        const Real pressure=rho*rho_unit*kGasConstant*background_temperature
                            /mean_molecular_weight/egas_unit;
        const Real eint=pressure/gm1;
        const Real kinetic=0.5*(SQR(phydro->u(IM1,k,j,i))
                               +SQR(phydro->u(IM2,k,j,i))
                               +SQR(phydro->u(IM3,k,j,i)))/rho;
        const Real magnetic=0.5*(SQR(pfield->bcc(IB1,k,j,i))
                                +SQR(pfield->bcc(IB2,k,j,i))
                                +SQR(pfield->bcc(IB3,k,j,i)));
        phydro->u(IEN,k,j,i)=eint+kinetic+magnetic;
        phydro->w(IPR,k,j,i)=pressure;
        prfld->u_gas(k,j,i)=eint;
        prfld->u_rad(k,j,i)=erad_target;
      }
    }
  }
#endif
}
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  (void)pin;
  pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,
                                      is,ie,js,je,ks,ke);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const Real bmag=std::sqrt(SQR(pfield->bcc(IB1,k,j,i))
            +SQR(pfield->bcc(IB2,k,j,i))+SQR(pfield->bcc(IB3,k,j,i)));
        user_out_var(0,k,j,i)=phydro->w(IDN,k,j,i)*rho_unit;
        user_out_var(1,k,j,i)=phydro->w(IPR,k,j,i)*egas_unit;
        user_out_var(2,k,j,i)=GasTemperature(this,k,j,i);
        user_out_var(3,k,j,i)=prfld->u_rad(k,j,i)*egas_unit;
        user_out_var(4,k,j,i)=RadiationTemperature(this,k,j,i);
        user_out_var(5,k,j,i)=bmag*magnetic_unit;
#if NSCALARS > 0
        user_out_var(6,k,j,i)=pscalars->s(0,k,j,i)
            /std::max(phydro->u(IDN,k,j,i),TINY_NUMBER);
#else
        user_out_var(6,k,j,i)=0.0;
#endif
      }
    }
  }
}
void Mesh::UserWorkInLoop() {}
void Mesh::UserWorkAfterLoop(ParameterInput *pin) { (void)pin; }
