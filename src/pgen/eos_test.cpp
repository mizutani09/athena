//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos_test.cpp
//! \brief Problem generator to test/verify equation of state.
//!
//! Problem generator designed to test, debug, and verify EOS tables and functions
//
//========================================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <fstream>
#include <iomanip>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/interp_table.hpp"

namespace {

constexpr Real kRadiationConstantCgs = 7.5657e-15;

void GenerateLteRadiationTable(EquationOfState *eos, EosTable *material,
                               ParameterInput *pin) {
  const std::string filename = pin->GetString("problem", "lte_table_output");
  const bool binary_output = pin->GetOrAddBoolean(
      "problem", "lte_table_binary", false);
  const int n_energy = pin->GetOrAddInteger(
      "problem", "lte_table_n_energy", material->nEgas);
  const int n_density = pin->GetOrAddInteger(
      "problem", "lte_table_n_density", material->nRho);
  const Real log_density_min = pin->GetOrAddReal(
      "problem", "lte_table_logrho_min", -12.0);
  const Real log_density_max = pin->GetOrAddReal(
      "problem", "lte_table_logrho_max", -3.0);
  const Real dens_pow = pin->GetOrAddReal("hydro", "dens_pow", -1.0);
  if (filename.empty() || n_energy < 2 || n_density < 2
      || !(log_density_max > log_density_min) || material->nVar < 7) {
    std::stringstream msg;
    msg << "### FATAL ERROR in GenerateLteRadiationTable" << std::endl
        << "invalid output path, shape, density interval, or source schema";
    ATHENA_ERROR(msg);
  }

  auto material_state_at_x2 = [&](const Real logrho, const Real x2,
                                  Real *energy, Real *pressure,
                                  Real *temperature) {
    const Real log_energy_code = x2-dens_pow*logrho
        -std::log10(material->EosRatios(3)*material->eUnit);
    *energy = std::pow(10.0, log_energy_code);
    const Real pressure_x2 = std::log10(
        *energy*material->EosRatios(0)*material->eUnit)+dens_pow*logrho;
    *pressure = *energy*std::pow(10.0,
        material->table.interpolate(0, pressure_x2, logrho));
    *temperature = std::pow(10.0,
        material->table.interpolate(3, x2, logrho));
  };

  Real energy_lower = -std::numeric_limits<Real>::infinity();
  Real energy_upper = std::numeric_limits<Real>::infinity();
  Real pressure_lower = -std::numeric_limits<Real>::infinity();
  Real pressure_upper = std::numeric_limits<Real>::infinity();
  for (int irho = 0; irho < n_density; ++irho) {
    const Real fraction = static_cast<Real>(irho)/(n_density-1);
    const Real logrho = log_density_min
        +fraction*(log_density_max-log_density_min);
    const Real rho_cgs = std::pow(10.0, logrho);
    Real material_energy_low, material_pressure_low, temperature_low;
    Real material_energy_high, material_pressure_high, temperature_high;
    material_state_at_x2(logrho, material->logEgasMin,
        &material_energy_low, &material_pressure_low, &temperature_low);
    material_state_at_x2(logrho, material->logEgasMax,
        &material_energy_high, &material_pressure_high, &temperature_high);
    const Real radiation_low = kRadiationConstantCgs
        *std::pow(temperature_low, 4)/material->eUnit;
    const Real radiation_high = kRadiationConstantCgs
        *std::pow(temperature_high, 4)/material->eUnit;
    energy_lower = std::max(energy_lower, std::log10(
        (material_energy_low+radiation_low)*material->eUnit/rho_cgs));
    energy_upper = std::min(energy_upper, std::log10(
        (material_energy_high+radiation_high)*material->eUnit/rho_cgs));
    pressure_lower = std::max(pressure_lower, std::log10(
        (material_pressure_low+radiation_low/3.0)
        *material->eUnit/rho_cgs));
    pressure_upper = std::min(pressure_upper, std::log10(
        (material_pressure_high+radiation_high/3.0)
        *material->eUnit/rho_cgs));
  }
  // Stay off the source table's closed endpoints.  Decimal serialization of
  // the output coordinate can otherwise place a target a few ulps beyond the
  // material bracket when the generated table is read back.
  constexpr Real kDomainMarginDex = 1.0e-4;
  energy_lower += kDomainMarginDex;
  energy_upper -= kDomainMarginDex;
  pressure_lower += kDomainMarginDex;
  pressure_upper -= kDomainMarginDex;
  const Real common_width = std::min(
      energy_upper-energy_lower, pressure_upper-pressure_lower);
  if (!std::isfinite(common_width) || common_width <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in GenerateLteRadiationTable" << std::endl
        << "no common total-energy/pressure domain: E=[" << energy_lower
        << ',' << energy_upper << "] P=[" << pressure_lower << ','
        << pressure_upper << ']';
    ATHENA_ERROR(msg);
  }
  // Center the common-width subinterval in both physical domains.  A ratio
  // shifts the pressure coordinate onto the energy coordinate without
  // changing its logarithmic width, matching the standard EOS-table schema.
  const Real output_x2_min = 0.5*(energy_lower+energy_upper-common_width);
  const Real output_x2_max = output_x2_min+common_width;
  const Real pressure_min =
      0.5*(pressure_lower+pressure_upper-common_width);
  const Real pressure_ratio = std::pow(10.0, output_x2_min-pressure_min);
  const Real ratios[7] = {
      1.0, pressure_ratio, pressure_ratio, 1.0,
      1.0, pressure_ratio, 1.0};

  std::vector<Real> table(
      static_cast<std::size_t>(7)*n_energy*n_density, 0.0);
  auto at = [&](const int field, const int ienergy, const int irho) -> Real& {
    return table[(static_cast<std::size_t>(field)*n_energy+ienergy)
                 *n_density+irho];
  };
  Real maximum_energy_roundtrip = 0.0;
  Real maximum_pressure_roundtrip = 0.0;
  for (int ienergy = 0; ienergy < n_energy; ++ienergy) {
    const Real energy_fraction = static_cast<Real>(ienergy)/(n_energy-1);
    const Real x2 = output_x2_min
        +energy_fraction*(output_x2_max-output_x2_min);
    for (int irho = 0; irho < n_density; ++irho) {
      const Real density_fraction = static_cast<Real>(irho)/(n_density-1);
      const Real logrho = log_density_min
          +density_fraction*(log_density_max-log_density_min);
      const Real rho_cgs = std::pow(10.0, logrho);
      const Real rho = rho_cgs/material->rhoUnit;

      const Real total_energy =
          std::pow(10.0, x2)*rho_cgs/material->eUnit;
      const Real total_pressure = eos->PresFromRhoEg(rho, total_energy);
      const Real temperature = eos->TempFromRhoEg(rho, total_energy)
          *material->tUnit;
      const Real entropy_forward = eos->EntropyFromRhoEg(rho, total_energy);
      const Real nabla_ad = eos->NablaAdFromRhoP(rho, total_pressure);
      const Real energy_roundtrip = eos->EgasFromRhoP(rho, total_pressure);

      const Real inverse_pressure = std::pow(10.0, x2)
          *rho_cgs/(pressure_ratio*material->eUnit);
      const Real inverse_energy = eos->EgasFromRhoP(rho, inverse_pressure);
      const Real sound_speed_squared =
          eos->AsqFromRhoP(rho, inverse_pressure);
      const Real entropy_inverse = eos->EntropyFromRhoP(rho, inverse_pressure);
      const Real pressure_roundtrip =
          eos->PresFromRhoEg(rho, inverse_energy);

      const Real values[7] = {
          total_pressure/total_energy,
          inverse_energy/inverse_pressure,
          sound_speed_squared*rho/inverse_pressure,
          temperature,
          entropy_forward,
          entropy_inverse,
          nabla_ad};
      for (int field = 0; field < 7; ++field) {
        if (!std::isfinite(values[field]) || values[field] <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in GenerateLteRadiationTable" << std::endl
              << "invalid field " << field << " at x2=" << x2
              << " logrho=" << logrho << " value=" << values[field];
          ATHENA_ERROR(msg);
        }
        at(field, ienergy, irho) = std::log10(values[field]);
      }
      maximum_energy_roundtrip = std::max(maximum_energy_roundtrip,
          std::abs(energy_roundtrip/total_energy-1.0));
      maximum_pressure_roundtrip = std::max(maximum_pressure_roundtrip,
          std::abs(pressure_roundtrip/inverse_pressure-1.0));
    }
  }

  std::ofstream output(filename, binary_output
      ? (std::ios::out | std::ios::binary) : std::ios::out);
  if (!output) {
    std::stringstream msg;
    msg << "### FATAL ERROR in GenerateLteRadiationTable" << std::endl
        << "cannot open " << filename;
    ATHENA_ERROR(msg);
  }
  if (binary_output) {
    const int n_fields = 7;
    output.write(reinterpret_cast<const char *>(&n_fields), sizeof(int));
    output.write(reinterpret_cast<const char *>(&n_energy), sizeof(int));
    output.write(reinterpret_cast<const char *>(&n_density), sizeof(int));
    output.write(reinterpret_cast<const char *>(&output_x2_min), sizeof(Real));
    output.write(reinterpret_cast<const char *>(&output_x2_max), sizeof(Real));
    output.write(reinterpret_cast<const char *>(&log_density_min), sizeof(Real));
    output.write(reinterpret_cast<const char *>(&log_density_max), sizeof(Real));
    output.write(reinterpret_cast<const char *>(ratios), 7*sizeof(Real));
    output.write(reinterpret_cast<const char *>(table.data()),
                 table.size()*sizeof(Real));
  } else {
    output << "# Tomida et al. 2013 material EOS plus one-temperature LTE photons\n"
           << "# E=E_material+aT^4; P=P_material+aT^4/3\n"
           << "# generated by eos_test/lte_table_output\n"
           << "# n_var n_x2 n_rho\n"
           << 7 << ' ' << n_energy << ' ' << n_density << '\n'
           << "# log10 x2 limits\n" << std::setprecision(16)
           << output_x2_min << ' ' << output_x2_max << '\n'
           << "# log10 rho limits\n"
           << log_density_min << ' ' << log_density_max << '\n'
           << "# coordinate ratios\n";
    for (int field = 0; field < 7; ++field)
      output << ratios[field] << (field == 6 ? '\n' : ' ');
    const char *labels[7] = {
        "log10(P_total/E_total) on (rho,E_total/rho)",
        "log10(E_total/P_total) on (rho,P_total/rho)",
        "log10(asq*rho/P_total) on (rho,P_total/rho)",
        "log10(T) on (rho,E_total/rho)",
        "log10(s_total) on (rho,E_total/rho)",
        "log10(s_total) on (rho,P_total/rho)",
        "log10(nabla_ad_total) on (rho,E_total/rho)"};
    output << std::scientific << std::setprecision(12);
    for (int field = 0; field < 7; ++field) {
      output << "# " << labels[field] << '\n';
      for (int ienergy = 0; ienergy < n_energy; ++ienergy) {
        for (int irho = 0; irho < n_density; ++irho) {
          output << at(field, ienergy, irho)
                 << (irho+1 == n_density ? '\n' : ' ');
        }
      }
    }
  }
  output.close();
  std::cout << std::scientific << std::setprecision(12)
            << "LTE_TABLE_GENERATED file=" << filename
            << " format=" << (binary_output ? "binary" : "ascii")
            << " x2_min=" << output_x2_min
            << " x2_max=" << output_x2_max
            << " pressure_ratio=" << pressure_ratio
            << " max_energy_roundtrip=" << maximum_energy_roundtrip
            << " max_pressure_roundtrip=" << maximum_pressure_roundtrip
            << std::endl;
}

void ValidateTableRoundtrips(EquationOfState *eos, EosTable *table,
                             ParameterInput *pin) {
  const int samples = pin->GetOrAddInteger(
      "problem", "roundtrip_grid_samples", 64);
  const Real density_margin = pin->GetOrAddReal(
      "problem", "roundtrip_density_margin_dex", 0.0);
  const Real energy_margin = pin->GetOrAddReal(
      "problem", "roundtrip_energy_margin_dex", 0.0);
  const Real dens_pow = pin->GetOrAddReal("hydro", "dens_pow", -1.0);
  if (density_margin < 0.0 || 2.0*density_margin
          >= table->logRhoMax-table->logRhoMin
      || energy_margin < 0.0 || 2.0*energy_margin
          >= table->logEgasMax-table->logEgasMin) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ValidateTableRoundtrips" << std::endl
        << "invalid validation margins";
    ATHENA_ERROR(msg);
  }
  Real maximum_energy_error = 0.0;
  Real maximum_pressure_error = 0.0;
  Real maximum_energy_logrho = 0.0;
  Real maximum_energy_x2 = 0.0;
  Real maximum_pressure_logrho = 0.0;
  Real maximum_pressure_x2 = 0.0;
  Real minimum_sound_speed_squared = std::numeric_limits<Real>::infinity();
  Real minimum_nabla_ad = std::numeric_limits<Real>::infinity();
  for (int j = 0; j < samples; ++j) {
    const Real fj = (j+0.5)/samples;
    const Real logrho = table->logRhoMin+density_margin
        +fj*(table->logRhoMax-table->logRhoMin-2.0*density_margin);
    const Real rho_cgs = std::pow(10.0, logrho);
    const Real rho = rho_cgs/table->rhoUnit;
    for (int i = 0; i < samples; ++i) {
      const Real fi = (i+0.5)/samples;
      const Real x2 = table->logEgasMin+energy_margin
          +fi*(table->logEgasMax-table->logEgasMin-2.0*energy_margin);
      const Real energy = std::pow(10.0,
          x2-dens_pow*logrho
          -std::log10(table->EosRatios(3)*table->eUnit));
      const Real pressure = eos->PresFromRhoEg(rho, energy);
      const Real energy_back = eos->EgasFromRhoP(rho, pressure);
      const Real nabla_ad = eos->NablaAdFromRhoP(rho, pressure);
      const Real energy_error = std::abs(energy_back/energy-1.0);
      if (energy_error > maximum_energy_error) {
        maximum_energy_error = energy_error;
        maximum_energy_logrho = logrho;
        maximum_energy_x2 = x2;
      }
      minimum_nabla_ad = std::min(minimum_nabla_ad, nabla_ad);

      const Real inverse_pressure = std::pow(10.0,
          x2-dens_pow*logrho
          -std::log10(table->EosRatios(1)*table->eUnit));
      const Real inverse_energy = eos->EgasFromRhoP(rho, inverse_pressure);
      const Real pressure_back = eos->PresFromRhoEg(rho, inverse_energy);
      const Real sound_speed_squared =
          eos->AsqFromRhoP(rho, inverse_pressure);
      const Real pressure_error = std::abs(pressure_back/inverse_pressure-1.0);
      if (pressure_error > maximum_pressure_error) {
        maximum_pressure_error = pressure_error;
        maximum_pressure_logrho = logrho;
        maximum_pressure_x2 = x2;
      }
      minimum_sound_speed_squared = std::min(
          minimum_sound_speed_squared, sound_speed_squared);
    }
  }
  std::cout << std::scientific << std::setprecision(12)
            << "EOS_TABLE_ROUNDTRIP max_energy_error="
            << maximum_energy_error
            << " at_logrho=" << maximum_energy_logrho
            << " at_x2=" << maximum_energy_x2
            << " max_pressure_error=" << maximum_pressure_error
            << " at_logrho=" << maximum_pressure_logrho
            << " at_x2=" << maximum_pressure_x2
            << " min_asq=" << minimum_sound_speed_squared
            << " min_nabla_ad=" << minimum_nabla_ad << std::endl;
  if (!std::isfinite(maximum_energy_error)
      || !std::isfinite(maximum_pressure_error)
      || maximum_energy_error > 5.0e-3
      || maximum_pressure_error > 5.0e-3
      || !std::isfinite(minimum_sound_speed_squared)
      || minimum_sound_speed_squared <= 0.0
      || !std::isfinite(minimum_nabla_ad) || minimum_nabla_ad <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ValidateTableRoundtrips" << std::endl
        << "generated EOS table failed its roundtrip/positivity gate";
    ATHENA_ERROR(msg);
  }
}

}  // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for eos tests
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  EosTable *ptable = pmy_mesh->peos_table;
  bool print_table = pin->GetOrAddBoolean("problem","print_table",false);
  bool exp_table = pin->GetOrAddBoolean("problem","exponentiate_table",false);
  bool eos_loop = pin->GetOrAddBoolean("problem","eos_loop",false);

  // Print EOS info
  std::cout << "Equation of state (EOS) diagnostics:" << '\n';
  if (GENERAL_EOS) {
    std::cout << "General EOS enabled with " << EQUATION_OF_STATE << '\n';
    if (EOS_TABLE_ENABLED) {
      // Print table info
      std::cout << "Using table '" << pin->GetString("hydro", "eos_file_name") << "'.\n";
      std::cout << "Shape (nVar, nEgas, nRho): " << ptable->nVar << ", " << ptable->nEgas
                <<  ", " << ptable->nRho << '\n';
      std::cout << "logEgasMin, logEgasMax: " << ptable->logEgasMin << ", "
                << ptable->logEgasMax << '\n';
      std::cout << "logRhoMin, logRhoMax: " << ptable->logRhoMin << ", "
                << ptable->logRhoMax << '\n';
      std::cout << "eUnit, rhoUnit, hUnit: " << ptable->eUnit << ", " << ptable->rhoUnit
                << ", " << ptable->hUnit << '\n';
      std::cout << "EosRatios: ";
      for (int i=0; i < ptable->nVar; ++i) std::cout << ptable->EosRatios(i) << ", ";
      std::cout << '\n';
    }
  } else if (NON_BAROTROPIC_EOS) {
    std::cout << "Adiabatic/ideal EOS enabled." << '\n';
  } else {
    std::cout << "Isothermal EOS enabled." << '\n';
  }
  std::cout << std::endl;

  static bool lte_table_generated = false;
  const std::string lte_table_output = pin->GetOrAddString(
      "problem", "lte_table_output", "none");
  if (!lte_table_generated && Globals::my_rank == 0
      && lte_table_output != "none") {
    GenerateLteRadiationTable(peos, ptable, pin);
    lte_table_generated = true;
  }
  if (pin->GetOrAddBoolean("problem", "roundtrip_grid_test", false))
    ValidateTableRoundtrips(peos, ptable, pin);

  if (print_table) {
    if (EOS_TABLE_ENABLED) {
      // Print EOS table
      for (int i=0; i<ptable->nVar; i++) {
        std::cout << "var = " << i << std::endl;
        if (exp_table) {
          for (int j=0; j<ptable->nEgas; j++) {
            for (int k=0; k<ptable->nRho; k++) {
              std::cout << std::pow((Real) 10, ptable->table.data(i,j,k)) << " ";
            }
            std::cout << '\n';
          }
          std::cout << '\n';
        } else {
          for (int j=0; j<ptable->nEgas; j++) {
            for (int k=0; k<ptable->nRho; k++) {
              std::cout << ptable->table.data(i,j,k) << " ";
            }
            std::cout << '\n';
          }
          std::cout << '\n';
        }
      }
    } else {
      std::cout << "Warning: print table specified, but no table to print." << '\n';
    }
    std::cout << std::endl;
  }

  if (eos_loop) {
    if (!NON_BAROTROPIC_EOS) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Problem Generator" << std::endl
          << "Isothermal EOS incompatible with eos_loop."<< std::endl;
      ATHENA_ERROR(msg);
    }
    Real rho=0.0, egas=0.0;
    AthenaArray<Real> u, w, zeros;
    u.NewAthenaArray(NHYDRO,1,1,1);
    w.NewAthenaArray(NHYDRO,1,1,1);
    zeros.NewAthenaArray(NHYDRO,1,1,1);
    Real w2[(NHYDRO)];
    for (int i=0; i<NHYDRO; i++) {
      u(i)=0.0;
      zeros(i)=0.0;
      w2[i]=0.0;
    }
    std::cout << "Input fluid parameters and retrieve EOS parameters." << '\n'
              << "Non-positive inputs will exit loop." << '\n';
    std::cout << "Input density (mass/volume): ";
    std::cin >> rho;
    std::cout << "Input internal energy (energy/volume): ";
    std::cin >> egas;
    while (rho > 0 && std::isfinite(rho) && egas >0 && std::isfinite(egas)) {
      Real p, asq, perr;  //, h;
      FaceField f;
      std::cout << "Density, internal energy: " << rho << ", " << egas << '\n';
      u(IDN,0,0,0) = rho;
      u(IEN,0,0,0) = egas;
      peos->ConservedToPrimitive(u, zeros, f, w, zeros, pcoord, 0, 0, 0, 0, 0, 0);
      peos->PrimitiveToConserved(w, zeros, u, pcoord, 0, 0, 0, 0, 0, 0);
      p = w(IPR,0,0,0);
      w2[IPR]=p;
      w2[IDN]=rho;
      asq = SQR(peos->SoundSpeed(w2));
      //h = p + egas;
      perr = 1.0 - u(IEN)/egas;
      std::cout << "P(d, e)    , ASq(d, P)  , PErr\n";
      std::cout << p << ", " << asq  << ", " << perr << '\n' << std::endl;
      std::cout << "Input density (mass/volume): ";
      std::cin >> rho;
      std::cout << "Input internal energy (energy/volume): ";
      std::cin >> egas;
    }
    std::cout << std::endl;
  }

  // Set all fluid parameters to zero
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN,k,j,i) = 0.0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) phydro->u(IEN,k,j,i) = 0.0;
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}
