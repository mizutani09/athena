//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/
//! \file nrfld_conv_sun.cpp
//! \brief Problem generator for convection in the Sun
//! REFERENCE: M. Rempel, Numerical simulations of quiet sun magnetism: On the contribution from a small-scale dynamo. Astrophys. J. 789, 22 (2014).

//======================================================================================

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <random>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>


// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../fld/fld.hpp"
#include "../fld/opacity_table.hpp"
#include "../fld/fld.hpp"


#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif


namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit, grav_unit;
  Real T_unit, time_unit, vel_unit, opacity_unit;
  Real a_r_dim, Rgas, mu, gamma_gas;
  Real a_r_sim, c_light_sim;
  Real dt_initial;
  Real dt_min_factor, dt_max_factor;
  bool abort_on_dt_excursion;
  bool dt_excursion_reported;
#if EOS_TABLE_ENABLED
  EosTable *pglobal_eos_table = nullptr;
  Real eos_dens_pow = -1.0;
  constexpr int kTablePresOverEgas = 0;
  constexpr int kTableEgasOverPres = 1;
  constexpr int kTableAsqRhoOverPres = 2;
  constexpr int kTableTemperature = 3;
  InterpTable2D entropy_table;
  bool has_entropy_table = false;
#endif
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  Real HistoryFradTop(MeshBlock *pmb, int iout);
  Real HistoryFtotTop(MeshBlock *pmb, int iout);
  Real HistoryEinBottom(MeshBlock *pmb, int iout);
  Real HistoryMass(MeshBlock *pmb, int iout);
  Real HistoryMassDrift(MeshBlock *pmb, int iout);
  Real HistoryVzRms(MeshBlock *pmb, int iout);
  Real HistoryEkinConv(MeshBlock *pmb, int iout);
  // Real HistoryL1norm(MeshBlock *pmb, int iout);
  Real nabla, nabla_eps;
  Real grav_acc;
  Real z_ref, rho_ref, T_ref, press_ref;
  Real T_ex;
  Real stellar_r_base, stellar_r_surface, stellar_r_photosphere;
  Real profile_r_min, profile_r_max;
  Real profile_height_above_photo, profile_depth_below_photo;
  Real z_photosphere;
  std::vector<Real> stellar_r_profile;
  std::vector<Real> stellar_rho_profile;
  std::vector<Real> stellar_press_profile;
  std::vector<Real> stellar_grav_profile;
  Real bottom_press_ref, bottom_rho_ref, bottom_temp_ref, bottom_grav_ref;
  Real bottom_inflow_eint, bottom_pressure_scale;
  Real target_top_flux_cgs, measured_top_flux_cgs, top_flux_rad_cgs;
  Real flux_relax_time;
  bool flux_feedback_on;
  Real total_mass_initial, total_mass_current;
  bool mass_correction_on;
  Real mass_relax_time;
  Real vz_perturb_frac;

//   Real igm1;
  Real sigma_P, sigma_R;
  int rk_cycle;
  bool use_opacity_table;
  UserOpacityTable *puser_table = nullptr;
  int iuov_max;

  // for fixed boundaries
  enum BIDX {
    RHO=0,
    PRESS=1,
    EGAS=2,
    ERAD=3,
    NBIDX,
  };
  Real bottom_boundary[NGHOST*NBIDX];
  Real top_boundary[NGHOST*NBIDX];

  // for ruser_meshblock
  int UBTOP = 0;
  int UBBOTTOM = 1;

  // for iuser_meshblock
  int TSTEP_COUNTER = 0;

  void ReadStellarProfileScaleFile(const std::string &filename, Real &r_b_cgs,
                                   Real &r_b_rsun, Real &rho_b_cgs,
                                   Real &p_b_cgs, Real &g_b_cgs) {
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ReadStellarProfileScaleFile]" << std::endl
          << "File " << filename << " does not exist" << std::endl;
      ATHENA_ERROR(msg);
    }

    bool found_r_b = false;
    bool found_r_b_rsun = false;
    bool found_rho_b = false;
    bool found_p_b = false;
    bool found_g_b = false;
    std::string line;
    while (std::getline(ifs, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream ss(line);
      std::string key, eq;
      Real value;
      if (!(ss >> key >> eq >> value)) continue;
      if (key == "r_b_cgs") {
        r_b_cgs = value;
        found_r_b = true;
      } else if (key == "r_b_rsun") {
        r_b_rsun = value;
        found_r_b_rsun = true;
      } else if (key == "rho_b_cgs") {
        rho_b_cgs = value;
        found_rho_b = true;
      } else if (key == "p_b_cgs") {
        p_b_cgs = value;
        found_p_b = true;
      } else if (key == "g_b_cgs") {
        g_b_cgs = value;
        found_g_b = true;
      }
    }

    if (!found_r_b || !found_r_b_rsun || !found_rho_b || !found_p_b || !found_g_b) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ReadStellarProfileScaleFile]" << std::endl
          << "Failed to find one or more scale factors "
          << "(r_b_cgs, r_b_rsun, rho_b_cgs, p_b_cgs, g_b_cgs) in "
          << filename << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  void ReadStellarProfileDataFile(const std::string &filename, const Real r_b_code,
                                  const Real rho_b_code, const Real p_b_code,
                                  const Real g_b_code) {
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ReadStellarProfileDataFile]" << std::endl
          << "File " << filename << " does not exist" << std::endl;
      ATHENA_ERROR(msg);
    }

    stellar_r_profile.clear();
    stellar_rho_profile.clear();
    stellar_press_profile.clear();
    stellar_grav_profile.clear();

    std::string line;
    while (std::getline(ifs, line)) {
      if (line.empty() || line[0] == '#') continue;

      std::stringstream ss(line);
      Real x_hat, rho_hat, press_hat, grav_hat;
      if (!(ss >> x_hat >> rho_hat >> press_hat >> grav_hat)) continue;

      stellar_r_profile.push_back(x_hat * r_b_code);
      stellar_rho_profile.push_back(rho_hat * rho_b_code);
      stellar_press_profile.push_back(press_hat * p_b_code);
      stellar_grav_profile.push_back(grav_hat * g_b_code);
    }

    if (stellar_r_profile.size() < 2) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ReadStellarProfileDataFile]" << std::endl
          << "At least two profile points are required in " << filename << std::endl;
      ATHENA_ERROR(msg);
    }

    profile_r_min = stellar_r_profile.front();
    profile_r_max = stellar_r_profile.back();
  }

  void LoadStellarProfile(ParameterInput *pin) {
    std::string profile_file =
        pin->GetOrAddString("problem", "stellar_profile_file", "");
    std::string scale_file =
        pin->GetOrAddString("problem", "stellar_profile_scale_file", "");
    if (profile_file.empty() || scale_file.empty()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadStellarProfile]" << std::endl
          << "Both problem/stellar_profile_file and "
          << "problem/stellar_profile_scale_file must be specified." << std::endl;
      ATHENA_ERROR(msg);
    }

    Real r_b_cgs = -1.0;
    Real r_b_rsun = -1.0;
    Real rho_b_cgs = -1.0;
    Real p_b_cgs = -1.0;
    Real g_b_cgs = -1.0;
    ReadStellarProfileScaleFile(scale_file, r_b_cgs, r_b_rsun,
                                rho_b_cgs, p_b_cgs, g_b_cgs);
    stellar_r_base = r_b_cgs/leng_unit;
    stellar_r_photosphere = stellar_r_base/r_b_rsun;
    Real rho_b_code = rho_b_cgs/rho_unit;
    Real p_b_code = p_b_cgs/egas_unit;
    Real g_b_code = g_b_cgs/grav_unit;

    ReadStellarProfileDataFile(profile_file, stellar_r_base, rho_b_code, p_b_code,
                               g_b_code);
    stellar_r_surface = stellar_r_profile.back();
  }

  void SetProfileWindow(ParameterInput *pin, const RegionSize &mesh_size) {
    profile_height_above_photo =
        pin->GetReal("problem", "photosphere_height_above_cgs")/leng_unit;
    profile_depth_below_photo =
        pin->GetReal("problem", "photosphere_depth_below_cgs")/leng_unit;

    Real domain_height = mesh_size.x3max - mesh_size.x3min;
    Real requested_height = profile_height_above_photo + profile_depth_below_photo;
    Real rel_err = std::abs(domain_height - requested_height)
                 / std::max(std::abs(requested_height), TINY_NUMBER);
    if (rel_err > 1.0e-10) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SetProfileWindow]" << std::endl
          << "mesh x3 extent must match photosphere window." << std::endl
          << "domain height = " << domain_height << ", requested = " << requested_height
          << std::endl;
      ATHENA_ERROR(msg);
    }

    z_photosphere = mesh_size.x3min + profile_depth_below_photo;

    Real rmin = stellar_r_photosphere - profile_depth_below_photo;
    Real rmax = stellar_r_photosphere + profile_height_above_photo;
    if (rmin < profile_r_min || rmax > profile_r_max) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SetProfileWindow]" << std::endl
          << "Requested photosphere window lies outside loaded profile range."
          << std::endl
          << "Requested radius range = [" << rmin << ", " << rmax << "]" << std::endl
          << "Profile radius range   = [" << profile_r_min << ", " << profile_r_max << "]"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  Real HeightFromPhotosphere(const Real z) {
    return z - z_photosphere;
  }

  Real RadiusFromHeight(const Real z) {
    return stellar_r_photosphere + HeightFromPhotosphere(z);
  }

#if EOS_TABLE_ENABLED
  bool ReadNextEntropyTableLine(std::ifstream &file, std::string &line) {
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        if (line.find("END ENTROPY_POVER_RHO") != std::string::npos) break;
        continue;
      }
      return true;
    }
    return false;
  }

  void LoadEntropyTableFromAscii(const std::string &filename) {
    has_entropy_table = false;
    std::ifstream file(filename.c_str(), std::ios::in);
    if (!file.is_open()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Unable to open EOS table file: " << filename << std::endl;
      ATHENA_ERROR(msg);
    }

    std::string line;
    bool found = false;
    while (std::getline(file, line)) {
      if (line.find("BEGIN ENTROPY_POVER_RHO") != std::string::npos) {
        found = true;
        break;
      }
    }
    if (!found) return;

    int nvar = 0, nx2 = 0, nx1 = 0;
    if (!ReadNextEntropyTableLine(file, line)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Unexpected EOF while reading entropy table shape from "
          << filename << std::endl;
      ATHENA_ERROR(msg);
    }
    std::stringstream stream(line);
    if (!(stream >> nvar >> nx2 >> nx1) || nvar < 1 || nx2 < 2 || nx1 < 2) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Invalid entropy table shape line: \"" << line << "\"" << std::endl
          << "file: " << filename << std::endl;
      ATHENA_ERROR(msg);
    }

    entropy_table.SetSize(nvar, nx2, nx1);

    Real min_val, max_val;
    if (!ReadNextEntropyTableLine(file, line)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Unexpected EOF while reading entropy x2 limits from "
          << filename << std::endl;
      ATHENA_ERROR(msg);
    }
    stream.clear();
    stream.str(line);
    if (!(stream >> min_val >> max_val) || min_val >= max_val) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Invalid entropy x2 limits line: \"" << line << "\"" << std::endl
          << "file: " << filename << std::endl;
      ATHENA_ERROR(msg);
    }
    entropy_table.SetX2lim(min_val, max_val);

    if (!ReadNextEntropyTableLine(file, line)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Unexpected EOF while reading entropy x1 limits from "
          << filename << std::endl;
      ATHENA_ERROR(msg);
    }
    stream.clear();
    stream.str(line);
    if (!(stream >> min_val >> max_val) || min_val >= max_val) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Invalid entropy x1 limits line: \"" << line << "\"" << std::endl
          << "file: " << filename << std::endl;
      ATHENA_ERROR(msg);
    }
    entropy_table.SetX1lim(min_val, max_val);

    if (!ReadNextEntropyTableLine(file, line)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
          << "Unexpected EOF while reading entropy ratios from "
          << filename << std::endl;
      ATHENA_ERROR(msg);
    }

    for (int row = 0; row < nx2*nvar; ++row) {
      if (!ReadNextEntropyTableLine(file, line)) {
        std::stringstream msg;
        msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
            << "Unexpected EOF while reading entropy table row " << row
            << " from " << filename << std::endl;
        ATHENA_ERROR(msg);
      }
      std::stringstream row_stream(line);
      for (int col = 0; col < nx1; ++col) {
        if (!(row_stream >> entropy_table.data(row, col))) {
          std::stringstream msg;
          msg << "### FATAL ERROR in function [LoadEntropyTableFromAscii]" << std::endl
              << "Failed to parse entropy table value at row=" << row
              << ", col=" << col << std::endl
              << "line: \"" << line << "\"" << std::endl
              << "file: " << filename << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
    has_entropy_table = true;
  }

  inline void GetTableEosCoordinates(const int k_out, const Real var, const Real rho,
                                     Real &x2, Real &x1) {
    x1 = std::log10(rho * pglobal_eos_table->rhoUnit);
    x2 = std::log10(var * pglobal_eos_table->EosRatios(k_out) * pglobal_eos_table->eUnit)
       + eos_dens_pow*x1;
  }

  inline Real GetTableEosData(const int k_out, const Real var, const Real rho) {
    Real x1, x2;
    GetTableEosCoordinates(k_out, var, rho, x2, x1);
    return std::pow(10.0, pglobal_eos_table->table.interpolate(k_out, x2, x1));
  }

  Real TablePresFromRhoEg(const Real rho, const Real egas) {
    return GetTableEosData(kTablePresOverEgas, egas, rho) * egas;
  }

  Real TableEgasFromRhoP(const Real rho, const Real pres) {
    return GetTableEosData(kTableEgasOverPres, pres, rho) * pres;
  }

  Real TableAsqFromRhoP(const Real rho, const Real pres) {
    return GetTableEosData(kTableAsqRhoOverPres, pres, rho) * pres / rho;
  }

  Real TableTempFromRhoEg(const Real rho, const Real egas) {
    Real x1, x2;
    GetTableEosCoordinates(kTableTemperature, egas, rho, x2, x1);
    Real temp_phys = std::pow(10.0, pglobal_eos_table->table.interpolate(kTableTemperature,
                                                                          x2, x1));
    return std::max(temp_phys/pglobal_eos_table->tUnit, TINY_NUMBER);
  }

  Real TableTempFromRhoP(const Real rho, const Real pres) {
    return TableTempFromRhoEg(rho, TableEgasFromRhoP(rho, pres));
  }

  Real TableEntropyFromRhoP(const Real rho, const Real pres) {
    if (!has_entropy_table || rho <= TINY_NUMBER || pres <= TINY_NUMBER) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    Real x1 = std::log10(rho * pglobal_eos_table->rhoUnit);
    Real poverrho_phys = pres * pglobal_eos_table->eUnit
                       / (rho * pglobal_eos_table->rhoUnit);
    if (poverrho_phys <= TINY_NUMBER) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    Real x2 = std::log10(poverrho_phys);
    return entropy_table.interpolate(0, x2, x1);
  }
#endif

  Real EgasFromRhoP(const Real rho, const Real pres) {
#if EOS_TABLE_ENABLED
    return TableEgasFromRhoP(rho, pres);
#else
    return pres/(gamma_gas - 1.0);
#endif
  }

  Real TempFromRhoP(const Real rho, const Real pres) {
#if EOS_TABLE_ENABLED
    return TableTempFromRhoP(rho, pres);
#else
    return pres/rho;
#endif
  }

  Real AsqFromRhoP(const Real rho, const Real pres) {
#if EOS_TABLE_ENABLED
    return TableAsqFromRhoP(rho, pres);
#else
    return gamma_gas*pres/rho;
#endif
  }

  Real SolveRhoFromPressureAndEint(const Real pres, const Real eint_spec,
                                   const Real rho_guess) {
#if EOS_TABLE_ENABLED
    auto residual = [&](const Real rho) {
      return TablePresFromRhoEg(rho, rho*eint_spec) - pres;
    };

    Real guess = std::max(rho_guess, TINY_NUMBER);
    Real lo = guess;
    Real hi = guess;
    Real flo = residual(lo);
    Real fhi = flo;

    for (int n = 0; n < 40 && flo > 0.0; ++n) {
      lo *= 0.5;
      flo = residual(lo);
    }
    for (int n = 0; n < 40 && fhi < 0.0; ++n) {
      hi *= 2.0;
      fhi = residual(hi);
    }
    if (flo > 0.0 || fhi < 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [SolveRhoFromPressureAndEint]" << std::endl
          << "Failed to bracket density for pres=" << pres
          << ", eint_spec=" << eint_spec
          << ", guess=" << rho_guess << std::endl;
      ATHENA_ERROR(msg);
    }

    for (int n = 0; n < 80; ++n) {
      Real mid = 0.5*(lo + hi);
      Real fmid = residual(mid);
      if (std::abs(fmid) < 1.0e-12*std::max(pres, 1.0)) return mid;
      if (fmid > 0.0) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return 0.5*(lo + hi);
#else
    return pres/((gamma_gas - 1.0)*eint_spec);
#endif
  }

  void ComputeBottomBoundaryState(Real &rho, Real &pres, Real &egas, Real &temp) {
    pres = bottom_press_ref*bottom_pressure_scale;
    rho = SolveRhoFromPressureAndEint(pres, bottom_inflow_eint, bottom_rho_ref);
    egas = rho*bottom_inflow_eint;
#if EOS_TABLE_ENABLED
    temp = TableTempFromRhoEg(rho, egas);
#else
    temp = (gamma_gas - 1.0)*bottom_inflow_eint;
#endif
  }

  int FindProfileIndex(const Real radius) {
    if (radius < stellar_r_profile.front() || radius > stellar_r_profile.back()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [FindProfileIndex]" << std::endl
          << "Requested radius = " << radius
          << " is outside loaded profile range ["
          << stellar_r_profile.front() << ", " << stellar_r_profile.back() << "]."
          << std::endl;
      ATHENA_ERROR(msg);
    }

    auto it = std::lower_bound(stellar_r_profile.begin() + 1,
                               stellar_r_profile.end(), radius);
    int mm = static_cast<int>(it - stellar_r_profile.begin());
    if (mm >= static_cast<int>(stellar_r_profile.size())) {
      mm = static_cast<int>(stellar_r_profile.size()) - 1;
    }
    return mm;
  }

  Real InterpolateProfileValue(const std::vector<Real> &values, const Real radius) {
    int mm = FindProfileIndex(radius);
    Real dr = stellar_r_profile[mm] - stellar_r_profile[mm-1];
    if (dr <= 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [InterpolateProfileValue]" << std::endl
          << "Profile radius must be strictly increasing." << std::endl;
      ATHENA_ERROR(msg);
    }
    Real wL = (stellar_r_profile[mm] - radius)/dr;
    return wL*values[mm-1] + (1.0 - wL)*values[mm];
  }

  void GetProfileAtHeight(const Real z, Real &rho, Real &press, Real &temp, Real &grav) {
    Real radius = RadiusFromHeight(z);
    rho = InterpolateProfileValue(stellar_rho_profile, radius);
    press = InterpolateProfileValue(stellar_press_profile, radius);
    grav = InterpolateProfileValue(stellar_grav_profile, radius);
    temp = TempFromRhoP(rho, press);
  }

  bool IsBottomUpflow(const AthenaArray<Real> &prim, const int k, const int j, const int i) {
    return prim(IVZ, k, j, i) > 0.0;
  }

  Real GetBottomBoundaryPressure() {
    return bottom_press_ref*bottom_pressure_scale;
  }

  Real GetBottomBoundaryTemperature() {
    Real rho, pres, egas, temp;
    ComputeBottomBoundaryState(rho, pres, egas, temp);
    return temp;
  }

  Real GetBottomBoundaryDensity() {
    Real rho, pres, egas, temp;
    ComputeBottomBoundaryState(rho, pres, egas, temp);
    return rho;
  }

  Real ComputeRadiativeFluxZ(const Real erad, const Real erad_km1, const Real erad_kp1,
                             const Real sigma_r_loc, const Real dz) {
    if (sigma_r_loc <= 0.0) return 0.0;
    Real gradE = (erad_kp1 - erad_km1)/(2.0*dz);
    Real R = std::abs(gradE)/(sigma_r_loc*std::max(erad, TINY_NUMBER));
    Real lambda = RadFLD2::FluxLimiter(R, false);
    return -c_light_sim*lambda*gradE/sigma_r_loc;
  }

  void ComputeFluxesAtCell(MeshBlock *pmb, const int k, const int j, const int i,
                           Real &frad, Real &fenth, Real &fkin, Real &ftot) {
    Real rho = pmb->phydro->w(IDN, k, j, i);
    Real vx = pmb->phydro->w(IVX, k, j, i);
    Real vy = pmb->phydro->w(IVY, k, j, i);
    Real vz = pmb->phydro->w(IVZ, k, j, i);
    Real pres = pmb->phydro->w(IPR, k, j, i);
    Real egas = pmb->prfld2->u_gas(k, j, i);
    Real erad = pmb->prfld2->u_rad(k, j, i);
    Real sigma_r_loc = pmb->prfld2->sigma_r(k, j, i);

    frad = ComputeRadiativeFluxZ(erad, pmb->prfld2->u_rad(k-1, j, i),
                                 pmb->prfld2->u_rad(k+1, j, i), sigma_r_loc,
                                 pmb->pcoord->dx3f(k));
    fenth = (egas + pres)*vz;
    fkin = 0.5*rho*(SQR(vx) + SQR(vy) + SQR(vz))*vz;
    ftot = frad + fenth + fkin;
  }

  void ComputeTopFluxAndMass(Mesh *pm, Real &ftop_rad, Real &ftop_tot, Real &mass_total) {
    ftop_rad = 0.0;
    ftop_tot = 0.0;
    mass_total = 0.0;
    Real top_area = 0.0;

    for (int n = 0; n < pm->nblocal; ++n) {
      MeshBlock *pmb = pm->my_blocks(n);
      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        for (int j = pmb->js; j <= pmb->je; ++j) {
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            mass_total += pmb->phydro->w(IDN, k, j, i)*pmb->pcoord->GetCellVolume(k, j, i);
          }
        }
      }

      if (std::abs(pmb->block_size.x3max - pm->mesh_size.x3max) > 1.0e-12) continue;
      int k = pmb->ke;
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real frad, fenth, fkin, ftot;
          ComputeFluxesAtCell(pmb, k, j, i, frad, fenth, fkin, ftot);
          Real area = pmb->pcoord->GetFace3Area(k+1, j, i);
          ftop_rad += frad*area;
          ftop_tot += ftot*area;
          top_area += area;
        }
      }
    }

#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &ftop_rad, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ftop_tot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mass_total, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &top_area, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

    if (top_area > 0.0) {
      ftop_rad /= top_area;
      ftop_tot /= top_area;
    }
  }

  void HydrostaticExtrapolateFromState(const Real rho_anchor, const Real pres_anchor,
                                       const Real eint_spec, const Real z_anchor,
                                       const Real z_target, Real &rho_target,
                                       Real &pres_target, Real &egas_target,
                                       Real &temp_target) {
    Real rho_tmp, pres_tmp, temp_tmp, grav_anchor;
    GetProfileAtHeight(z_anchor, rho_tmp, pres_tmp, temp_tmp, grav_anchor);
    Real grav_target;
    GetProfileAtHeight(z_target, rho_tmp, pres_tmp, temp_tmp, grav_target);
    Real grav_avg = 0.5*(grav_anchor + grav_target);
    Real dz = z_target - z_anchor;

    pres_target = std::max(pres_anchor - rho_anchor*grav_avg*dz, TINY_NUMBER);
    rho_target = SolveRhoFromPressureAndEint(pres_target, eint_spec, rho_anchor);

    for (int n = 0; n < 8; ++n) {
      Real pres_new = std::max(pres_anchor
                         - 0.5*(rho_anchor + rho_target)*grav_avg*dz, TINY_NUMBER);
      if (std::abs(pres_new - pres_target) <= 1.0e-12*std::max(pres_target, 1.0)) {
        pres_target = pres_new;
        break;
      }
      pres_target = pres_new;
      rho_target = SolveRhoFromPressureAndEint(pres_target, eint_spec, rho_anchor);
    }

    egas_target = rho_target*eint_spec;
    temp_target = TempFromRhoP(rho_target, pres_target);
  }
}


void NRInnerX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      bool is_upflow = IsBottomUpflow(w, ks, j, i);
      Real rho_anchor = w(IDN, ks, j, i);
      Real pres_anchor = w(IPR, ks, j, i);
      Real eint_spec = is_upflow ? bottom_inflow_eint
                                 : EgasFromRhoP(rho_anchor, pres_anchor)/rho_anchor;
      Real z_anchor = pco->x3v(ks);

      for (int k=1; k<=ngh; k++) {
        Real rho_ghost, pres_ghost, egas_ghost, temp_ghost;
        Real z_ghost = pco->x3v(ks-k);
        HydrostaticExtrapolateFromState(rho_anchor, pres_anchor, eint_spec,
                                        z_anchor, z_ghost,
                                        rho_ghost, pres_ghost, egas_ghost, temp_ghost);
        u_gas(ks-k,j,i) = egas_ghost;
        u_rad(ks-k,j,i) = a_r_sim*std::pow(temp_ghost, 4);
        rho_anchor = rho_ghost;
        pres_anchor = pres_ghost;
        z_anchor = z_ghost;
      }
    }
  }
  return;
}

void NRFixedOuterX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      Real rho_anchor = w(IDN, ke, j, i);
      Real pres_anchor = w(IPR, ke, j, i);
      Real eint_spec = EgasFromRhoP(rho_anchor, pres_anchor)/rho_anchor;
      Real z_anchor = pco->x3v(ke);
      for (int k=1; k<=ngh; k++) {
        Real rho_ghost, pres_ghost, egas_ghost, temp_ghost;
        Real z_ghost = pco->x3v(ke+k);
        HydrostaticExtrapolateFromState(rho_anchor, pres_anchor, eint_spec,
                                        z_anchor, z_ghost,
                                        rho_ghost, pres_ghost, egas_ghost, temp_ghost);
        u_gas(ke+k,j,i) = egas_ghost;
        u_rad(ke+k,j,i) = a_r_sim*std::pow(T_ex, 4);
        rho_anchor = rho_ghost;
        pres_anchor = pres_ghost;
        z_anchor = z_ghost;
      }
    }
  }
  return;
}


void NROpenOuterX3(MeshBlock *pmb,
               AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  NRFixedOuterX3(pmb, u_rad, u_gas, pco, w, time, dt, is, ie, js, je, ks, ke, ngh);
}

void FLDFixedInnerX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      bool is_upflow = IsBottomUpflow(w, ks, j, i);
      Real rho_anchor = w(IDN, ks, j, i);
      Real pres_anchor = w(IPR, ks, j, i);
      Real eint_spec = is_upflow ? bottom_inflow_eint
                                 : EgasFromRhoP(rho_anchor, pres_anchor)/rho_anchor;
      Real z_anchor = pco->x3v(ks);
      for (int k=1; k<=ngh; k++) {
        Real rho_ghost, pres_ghost, egas_ghost, temp_ghost;
        Real z_ghost = pco->x3v(ks-k);
        HydrostaticExtrapolateFromState(rho_anchor, pres_anchor, eint_spec,
                                        z_anchor, z_ghost,
                                        rho_ghost, pres_ghost, egas_ghost, temp_ghost);
        u_rad_fld(ks-k,j,i) = a_r_sim*std::pow(temp_ghost, 4);
        rho_anchor = rho_ghost;
        pres_anchor = pres_ghost;
        z_anchor = z_ghost;
      }
    }
  }
  return;
}

void FLDFixedOuterX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=1; k<=ngh; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        u_rad_fld(ke+k,j,i) = a_r_sim*std::pow(T_ex, 4);
      }
    }
  }
  return;
}

void FLDOpenOuterX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  FLDFixedOuterX3(pmb, pco, pfld, w, u_rad_fld, time, dt, is, ie, js, je, ks, ke, ngh);
}

void HydroReflectInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      bool is_upflow = IsBottomUpflow(prim, ks, j, i);
      Real rho_anchor = prim(IDN, ks, j, i);
      Real pres_anchor = prim(IPR, ks, j, i);
      Real vx_anchor = prim(IVX, ks, j, i);
      Real vy_anchor = prim(IVY, ks, j, i);
      Real vz_anchor = prim(IVZ, ks, j, i);
      Real eint_spec = is_upflow ? bottom_inflow_eint
                                 : EgasFromRhoP(rho_anchor, pres_anchor)/rho_anchor;
      Real z_anchor = pco->x3v(ks);

      for (int k=1; k<=ngh; k++) {
        Real rho_ghost, pres_ghost, egas_ghost, temp_ghost;
        Real z_ghost = pco->x3v(ks-k);
        HydrostaticExtrapolateFromState(rho_anchor, pres_anchor, eint_spec,
                                        z_anchor, z_ghost,
                                        rho_ghost, pres_ghost, egas_ghost, temp_ghost);
        prim(IDN,ks-k,j,i) = rho_ghost;
        prim(IPR,ks-k,j,i) = pres_ghost;
        prim(IVX,ks-k,j,i) = is_upflow ? 0.0 : vx_anchor;
        prim(IVY,ks-k,j,i) = is_upflow ? 0.0 : vy_anchor;
        prim(IVZ,ks-k,j,i) = vz_anchor;

        rho_anchor = rho_ghost;
        pres_anchor = pres_ghost;
        vx_anchor = prim(IVX,ks-k,j,i);
        vy_anchor = prim(IVY,ks-k,j,i);
        vz_anchor = prim(IVZ,ks-k,j,i);
        z_anchor = z_ghost;
      }
    }
  }
  return;
}

void HydroFixedOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      Real rho_anchor = prim(IDN, ke, j, i);
      Real pres_anchor = prim(IPR, ke, j, i);
      Real vx_anchor = prim(IVX, ke, j, i);
      Real vy_anchor = prim(IVY, ke, j, i);
      Real eint_spec = EgasFromRhoP(rho_anchor, pres_anchor)/rho_anchor;
      Real z_anchor = pco->x3v(ke);

      for (int k=1; k<=ngh; k++) {
        Real rho_ghost, pres_ghost, egas_ghost, temp_ghost;
        Real z_ghost = pco->x3v(ke+k);
        HydrostaticExtrapolateFromState(rho_anchor, pres_anchor, eint_spec,
                                        z_anchor, z_ghost,
                                        rho_ghost, pres_ghost, egas_ghost, temp_ghost);
        prim(IDN,ke+k,j,i) = rho_ghost;
        prim(IVX,ke+k,j,i) = vx_anchor;
        prim(IVY,ke+k,j,i) = vy_anchor;
        prim(IVZ,ke+k,j,i) = 0.0;
        prim(IPR,ke+k,j,i) = pres_ghost;
        rho_anchor = rho_ghost;
        pres_anchor = pres_ghost;
        z_anchor = z_ghost;
      }
    }
  }
  return;
}

void HydroOpenOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  HydroFixedOuterX3(pmb, pco, prim, b, time, dt, is, ie, js, je, ks, ke, ngh);
}

void GetOpacityFromUserTable(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD2 *prfld = pmb->prfld2;
  int kl=pmb->ks, ku=pmb->ke;
  int jl=pmb->js, ju=pmb->je;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        Real rho = prim(IDN,k,j,i);
        Real press = prim(IPR,k,j,i);
        Real temp = TempFromRhoP(rho, press);
        Real rho_phys = rho*rho_unit;
        Real temp_phys = temp*T_unit;
        prfld->sigma_p(k,j,i) =
            puser_table->GetOpacity(RadFLD2::SIGMA_P, rho_phys, temp_phys)/opacity_unit*rho;
        prfld->sigma_r(k,j,i) =
            puser_table->GetOpacity(RadFLD2::SIGMA_R, rho_phys, temp_phys)/opacity_unit*rho;
      }
    }
  }
}

void ConstantOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
              AthenaArray<Real> &prim) {
  FLD2 *prfld = pmb->prfld2;
  int kl=pmb->ks-NGHOST, ku=pmb->ke+NGHOST;
  int jl=pmb->js-NGHOST, ju=pmb->je+NGHOST;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  for(int k=kl; k<=ku; ++k) {
    for(int j=jl; j<=ju; ++j) {
#pragma omp simd
      for(int i=il; i<=iu; ++i) {
        prfld->sigma_p(k,j,i) = sigma_P;
        prfld->sigma_r(k,j,i) = sigma_R;
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho_unit = pin->GetReal("hydro", "rho_unit");
  egas_unit = pin->GetReal("hydro", "egas_unit");
  time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }
  Real pres_unit = egas_unit;
  // Rgas in cgs
  Rgas = 8.31451e+7; // erg/(mol*K)
  gamma_gas = pin->GetOrAddReal("hydro", "gamma", 5.0/3.0);
  mu = pin->GetOrAddReal("hydro", "mu", 0.6);
#if EOS_TABLE_ENABLED
  pglobal_eos_table = peos_table;
  eos_dens_pow = pin->GetOrAddReal("hydro", "dens_pow", -1.0);
  T_unit = pin->GetOrAddReal("hydro", "eos_temp_unit", pres_unit/rho_unit*mu/Rgas);
  LoadEntropyTableFromAscii(pin->GetString("hydro", "eos_file_name"));
#else
  T_unit = pres_unit/rho_unit*mu/Rgas;
#endif
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  a_r_sim = a_r_dim/(egas_unit/std::pow(T_unit, 4));
  // a_r_sim = 1.0;

  vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;
  grav_unit = vel_unit / time_unit;
  opacity_unit = 1.0/(rho_unit*leng_unit); // opacity unit in cm^2/g

  // Real const_opasity = pin->GetReal("fld", "const_opacity");
  // sigma_P = pin->GetReal("fld", "const_opacity_P") * (leng_unit);
  // sigma_R = pin->GetReal("fld", "const_opacity_R") * (leng_unit);
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  c_light_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;

  grav_acc = pin->GetReal("problem", "grav_acc");
  nabla_eps = pin->GetReal("problem", "nabla_eps");
  z_ref = mesh_size.x3max; // reference is top
  LoadStellarProfile(pin);
  SetProfileWindow(pin, mesh_size);

  Real rho_top, press_top, temp_top, grav_top;
  GetProfileAtHeight(mesh_size.x3max, rho_top, press_top, temp_top, grav_top);
  rho_ref = rho_top;
  press_ref = press_top;
  T_ref = temp_top;
  grav_acc = grav_top;

  GetProfileAtHeight(mesh_size.x3min, bottom_rho_ref, bottom_press_ref,
                     bottom_temp_ref, bottom_grav_ref);
  bottom_inflow_eint = pin->GetOrAddReal("problem", "bottom_inflow_eint", -1.0);
  if (bottom_inflow_eint < 0.0) {
    bottom_inflow_eint = EgasFromRhoP(bottom_rho_ref, bottom_press_ref)/bottom_rho_ref;
  }
  bottom_pressure_scale = 1.0;

  T_ex = pin->GetReal("problem", "top_radiation_temperature")/T_unit;
  target_top_flux_cgs = pin->GetOrAddReal("problem", "target_solar_flux_cgs", 6.34e10);
  flux_feedback_on = pin->GetOrAddBoolean("problem", "flux_feedback_on", false);
  flux_relax_time = pin->GetOrAddReal("problem", "flux_relax_time_cgs", 1.0e5)/time_unit;
  mass_correction_on = pin->GetOrAddBoolean("problem", "mass_correction_on", false);
  mass_relax_time = pin->GetOrAddReal("problem", "mass_relax_time_cgs", 30.0)/time_unit;
  vz_perturb_frac = pin->GetOrAddReal("problem", "vz_perturb_frac", 1.0e-3);
  dt_min_factor = pin->GetOrAddReal("problem", "dt_min_factor", 1.0e-4);
  dt_max_factor = pin->GetOrAddReal("problem", "dt_max_factor", 1.0e2);
  abort_on_dt_excursion = pin->GetOrAddBoolean("problem", "abort_on_dt_excursion", false);
  dt_excursion_reported = false;
  measured_top_flux_cgs = 0.0;
  top_flux_rad_cgs = 0.0;
  total_mass_initial = -1.0;
  total_mass_current = -1.0;

  std::string ix3_bc = pin->GetString("mesh", "ix3_bc");
  std::string ox3_bc = pin->GetString("mesh", "ox3_bc");
  if (ix3_bc == "user") {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroReflectInnerX3);
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, FLDFixedInnerX3);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x3, NRInnerX3);
  }
  if (ox3_bc == "user"){
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroFixedOuterX3);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, FLDFixedOuterX3);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x3, NRFixedOuterX3);
  }


  std::string integrator = pin->GetString("time","integrator");
  if (integrator == "rk3") {
    rk_cycle = 3;
  } else if (integrator == "rk2") {
    rk_cycle = 2;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR about integrator " << std::endl
        << "now only support the rk2 or rk3 integrator" << std::endl;
    ATHENA_ERROR(msg);
  }

  AllocateUserHistoryOutput(14);
  EnrollUserHistoryOutput(0, HistoryTg, "Tgas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "Trad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "egas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "Erad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, HistoryFradTop, "Frad_top", UserHistoryOperation::max);
  EnrollUserHistoryOutput(8, HistoryFtotTop, "Ftot_top", UserHistoryOperation::max);
  EnrollUserHistoryOutput(9, HistoryEinBottom, "ein_bottom", UserHistoryOperation::max);
  EnrollUserHistoryOutput(10, HistoryMass, "mass", UserHistoryOperation::max);
  EnrollUserHistoryOutput(11, HistoryMassDrift, "mass_drift", UserHistoryOperation::max);
  EnrollUserHistoryOutput(12, HistoryVzRms, "vz2_mean", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(13, HistoryEkinConv, "ekin_conv", UserHistoryOperation::sum);

}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  int idata_size = 0;
  idata_size += 1; // for test counter
  AllocateIntUserMeshBlockDataField(idata_size);

  iuser_meshblock_data[TSTEP_COUNTER].NewAthenaArray(1);
  iuser_meshblock_data[TSTEP_COUNTER](0) = 0;

  // user output variables
  int iuov = 0; // initialize
  iuov_max = 0;
  iuov_max += 2; // for e_gas, E_rad
  iuov_max += 2; // for T_gas, T_rad
  iuov_max += 1; // for P_tot
  iuov_max += 1; // for ent
  iuov_max += 1; // for sound speed
  iuov_max += 1; // for Mach number
  iuov_max += 2; // for opacity
  iuov_max += 4; // for vertical fluxes

  AllocateUserOutputVariables(iuov_max);
  SetUserOutputVariableName(iuov, "e_gas"), iuov++;
  SetUserOutputVariableName(iuov, "E_rad"), iuov++;
  SetUserOutputVariableName(iuov, "T_gas"), iuov++;
  SetUserOutputVariableName(iuov, "T_rad"), iuov++;
  SetUserOutputVariableName(iuov, "P_tot"), iuov++;
  SetUserOutputVariableName(iuov, "ent"), iuov++;
  SetUserOutputVariableName(iuov, "sound"), iuov++;
  SetUserOutputVariableName(iuov, "Mach"), iuov++;
  SetUserOutputVariableName(iuov, "sigma_P"), iuov++;
  SetUserOutputVariableName(iuov, "sigma_R"), iuov++;
  SetUserOutputVariableName(iuov, "Frad_z"), iuov++;
  SetUserOutputVariableName(iuov, "Fenth_z"), iuov++;
  SetUserOutputVariableName(iuov, "Fkin_z"), iuov++;
  SetUserOutputVariableName(iuov, "Ftot_z"), iuov++;

  use_opacity_table = pin->GetBoolean("fld", "use_opacity_table");
  if (use_opacity_table) {
    puser_table = new UserOpacityTable(pin);
    prfld2->EnrollOpacityFunction(GetOpacityFromUserTable);
  } else {
    sigma_P = pin->GetReal("fld", "const_opacity_P"); // in code unit
    sigma_R = pin->GetReal("fld", "const_opacity_R"); // in code unit
    prfld2->EnrollOpacityFunction(ConstantOpacity);
  }

  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
#if EOS_TABLE_ENABLED
  nabla = nabla_eps;
#else
  Real gamma = peos->GetGamma();
  Real igm1 = 1.0/(gamma-1.0);
  nabla = (gamma - 1.0)/gamma + nabla_eps;
#endif
  Real dx1 = pcoord->dx1f(4);
  Real courant = pin->GetReal("time", "cfl_number");
  Real z = pmy_mesh->mesh_size.x3min;
  Real rho_bottom, press_bottom, T_bottom, grav_bottom;
  GetProfileAtHeight(z, rho_bottom, press_bottom, T_bottom, grav_bottom);
  Real Er_bottom = a_r_sim*std::pow(T_bottom, 4);
  Real egas_bottom = EgasFromRhoP(rho_bottom, press_bottom);
  Real Cs_bottom = std::sqrt(AsqFromRhoP(rho_bottom, press_bottom));
  Real dt_exp = courant*dx1/Cs_bottom;
  dt_initial = dt_exp;
  // Real const_opasity = pin->GetReal("fld", "const_opacity");
  // Real const_opasity_sim = const_opasity*leng_unit*rho_unit;
  Real c_ph_dim = 2.99792458e10; // speed of light in cm s^-1
  Real c_ph_sim = c_ph_dim/(leng_unit/time_unit);
  // Real mfp_sim = 1.0/(const_opasity*rho_unit)/leng_unit;
  Real L = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real t_sc = L/Cs_bottom;
  Real exp_cycle = t_sc/dt_exp;

  Real optical_depth = sigma_P*L;
  if (gid == 0) {
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "vel_unit = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    std::cout << "T_unit = " << T_unit << " K" << std::endl;
#if EOS_TABLE_ENABLED
    std::cout << "entropy_table = "
              << (has_entropy_table ? "loaded" : "not found") << std::endl;
#endif
    std::cout << "c_ph_sim = " << c_ph_sim << " cm s^-1" << std::endl;
    std::cout << "dx = " << dx1*leng_unit << " cm" << std::endl;
    std::cout << "dt = " << dt_exp*time_unit << " s" << std::endl;
    std::cout << "dt_sim = " << dt_exp << std::endl;
    std::cout << "t_sc = " << L/Cs_bottom*time_unit << " s" << std::endl;
    std::cout << "t_sc_sim = " << L/Cs_bottom << std::endl;
    std::cout << "T_bottom = " << T_bottom << std::endl;
    std::cout << "T_top = " << T_ref << std::endl;
    std::cout << "press_bottom = " << press_bottom << std::endl;
    std::cout << "Er_bottom = " << Er_bottom << std::endl;
    std::cout << "expected cycle for t_sc = " << exp_cycle << std::endl;
    std::cout << "sigma_P = " << sigma_P << std::endl;
    std::cout << "sigma_R = " << sigma_R << std::endl;
    std::cout << "optical depth = " << optical_depth << std::endl;

    // also output the upper values in txt file
    std::ofstream ofs("problem_parameters.txt");
    ofs << ">>> Problem parameters <<<" << std::endl;
    ofs << "- Units" << std::endl;
    ofs << "rhoUnit        = " << rho_unit << " g cm^-3" << std::endl;
    ofs << "egasUnit       = " << egas_unit << " erg cm^-3" << std::endl;
    ofs << "timeUnit       = " << time_unit << " s" << std::endl;
    ofs << "lengUnit       = " << leng_unit << " cm" << std::endl;
    ofs << "velUnit        = " << leng_unit/time_unit << " cm s^-1" << std::endl;
    ofs << "TUnit          = " << T_unit << " K" << std::endl;
    ofs << std::endl;

    ofs << "- Simulation parameters" << std::endl;
    ofs << "c_ph_sim           = " << c_ph_sim << std::endl;
    ofs << "dx_dim             = " << dx1*leng_unit << " cm" << std::endl;
    ofs << "dt_dim             = " << dt_exp*time_unit << " s" << std::endl;
    ofs << "dt_sim             = " << dt_exp << std::endl;
    ofs << "t_sc               = " << t_sc*time_unit << " s" << std::endl;
    ofs << "t_sc_sim           = " << t_sc << std::endl;
    ofs << "exp_cycle for t_sc = " << exp_cycle << std::endl;
    ofs << "grav_top           = " << grav_acc << std::endl;
    ofs << "grav_bottom        = " << grav_bottom << std::endl;
    ofs << "nabla              = " << nabla << std::endl;
    // ofs << "poly_n             = " << poly_n << std::endl;
    ofs << "rho_bottom         = " << rho_bottom << std::endl;
    ofs << "rho_top            = " << rho_ref << std::endl;
    ofs << "T_bottom           = " << T_bottom << std::endl;
    ofs << "T_top              = " << T_ref << std::endl;
    ofs << "press_bottom       = " << press_bottom << std::endl;
    Real press_top = press_ref;
    Real egas_top = EgasFromRhoP(rho_ref, press_ref);
    Real Er_top = a_r_sim*std::pow(T_ref, 4);
    ofs << "press_top          = " << press_top << std::endl;
    ofs << "egas_bottom        = " << egas_bottom << std::endl;
    ofs << "egas_top           = " << egas_top << std::endl;
    ofs << "Er_bottom          = " << Er_bottom << std::endl;
    ofs << "Er_top             = " << Er_top << std::endl;
    ofs << "sigma_P            = " << sigma_P << std::endl;
    ofs << "sigma_R            = " << sigma_R << std::endl;
    ofs << "optical_depth      = " << optical_depth << std::endl;
    ofs << "vz_perturb_frac    = " << vz_perturb_frac << std::endl;
    ofs.close();
  }

  std::mt19937 rng(gid);  // seed
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  int kl = ks;
  int ku = ke;
  int jl = js;
  int ju = je;
  int il = is;
  int iu = ie;

  for(int k=kl; k<=ku; ++k) {
    Real z = pcoord->x3v(k);
    Real rho, pres, T, grav;
    GetProfileAtHeight(z, rho, pres, T, grav);
    Real cs_local = std::sqrt(AsqFromRhoP(rho, pres));
    Real weight = 0.0;
    if (z < z_photosphere && z_photosphere > pmy_mesh->mesh_size.x3min) {
      Real xi = (z - pmy_mesh->mesh_size.x3min)
              /(z_photosphere - pmy_mesh->mesh_size.x3min);
      xi = std::max(0.0, std::min(1.0, xi));
      weight = std::pow(std::sin(M_PI*xi), 2);
    }

    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = rho*vz_perturb_frac*weight*dist(rng)*cs_local;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = EgasFromRhoP(rho, pres);

        // for FLD
        prfld2->u_gas(k,j,i) = EgasFromRhoP(rho, pres);
        prfld2->u_rad(k,j,i) = a_r_sim*std::pow(T, 4);
      }
    }
  }
  std::cout << "ProblemGenerator completed." << std::endl;
  return;
}

void Mesh::UserWorkInLoop() {
  if (dt_initial > 0.0 && (dt > dt_max_factor*dt_initial || dt < dt_min_factor*dt_initial)) {
    if (abort_on_dt_excursion) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::UserWorkInLoop]" << std::endl;
      msg << "Timestep moved outside the configured range." << std::endl;
      msg << "dt = " << dt << ", dt_initial = " << dt_initial
          << ", allowed range = [" << dt_min_factor*dt_initial << ", "
          << dt_max_factor*dt_initial << "]." << std::endl;
      ATHENA_ERROR(msg);
    } else if (!dt_excursion_reported && Globals::my_rank == 0) {
      std::cout << "### Warning in function [Mesh::UserWorkInLoop]" << std::endl
                << "Timestep moved outside the configured range." << std::endl
                << "dt = " << dt << ", dt_initial = " << dt_initial
                << ", allowed range = [" << dt_min_factor*dt_initial << ", "
                << dt_max_factor*dt_initial << "]." << std::endl
                << "Continuing because problem/abort_on_dt_excursion = false."
                << std::endl;
      dt_excursion_reported = true;
    }
  }

  Real ftop_rad_sim, ftop_tot_sim, mass_sim;
  ComputeTopFluxAndMass(this, ftop_rad_sim, ftop_tot_sim, mass_sim);
  top_flux_rad_cgs = ftop_rad_sim*egas_unit*vel_unit;
  measured_top_flux_cgs = ftop_tot_sim*egas_unit*vel_unit;
  total_mass_current = mass_sim*rho_unit*std::pow(leng_unit, 3);
  if (total_mass_initial < 0.0) {
    total_mass_initial = total_mass_current;
  }

  if (flux_feedback_on && flux_relax_time > 0.0) {
    Real corr = 1.0 + (dt/flux_relax_time)
                     *(target_top_flux_cgs - measured_top_flux_cgs)/target_top_flux_cgs;
    corr = std::max(0.9, std::min(1.1, corr));
    bottom_inflow_eint *= corr;
  }

  if (mass_correction_on && mass_relax_time > 0.0 && total_mass_initial > 0.0) {
    Real delta_m = (total_mass_current - total_mass_initial)/total_mass_initial;
    Real corr = 1.0 - (dt/mass_relax_time)*delta_m;
    corr = std::max(0.95, std::min(1.05, corr));
    bottom_pressure_scale *= corr;
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        int iuov = 0; // initialize
        // assume cal in E
        user_out_var(iuov,k,j,i) = prfld2->u_gas(k,j,i)*egas_unit; iuov++;
        user_out_var(iuov,k,j,i) = prfld2->u_rad(k,j,i)*egas_unit; iuov++;
        Real vx = phydro->w(IVX,k,j,i);
        Real vy = phydro->w(IVY,k,j,i);
        Real vz = phydro->w(IVZ,k,j,i);
        Real dens = phydro->w(IDN,k,j,i);
        Real idens = 1.0 / dens;
        Real pres = phydro->w(IPR,k,j,i);
        Real egas = prfld2->u_gas(k,j,i);
        Real temp = peos->TempFromRhoEg(dens, egas)*T_unit;

        user_out_var(iuov,k,j,i) = temp; iuov++;
        user_out_var(iuov,k,j,i) = std::pow(prfld2->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25); iuov++;
        user_out_var(iuov,k,j,i) = pres*egas_unit + ONE_3RD*user_out_var(1,k,j,i); iuov++;

        // for entropy
#if EOS_TABLE_ENABLED
        user_out_var(iuov,k,j,i) = TableEntropyFromRhoP(dens, pres); iuov++;
#else
        user_out_var(iuov,k,j,i) = std::log(pres/std::pow(dens, gamma_gas)); iuov++;
#endif

        // for sound speed
        Real sound = std::sqrt(peos->AsqFromRhoP(dens, pres));
        user_out_var(iuov,k,j,i) = sound*vel_unit; iuov++; // cm/s

        // for Mach number
        Real v_sq = SQR(vx) + SQR(vy) + SQR(vz);
        Real mach = std::sqrt(v_sq) / sound;
        user_out_var(iuov,k,j,i) = mach; iuov++;

        // for opacity
        user_out_var(iuov,k,j,i) = prfld2->sigma_p(k,j,i); iuov++;
        user_out_var(iuov,k,j,i) = prfld2->sigma_r(k,j,i); iuov++;

        Real frad=0.0, fenth=0.0, fkin=0.0, ftot=0.0;
        if (k > kl && k < ku) {
          ComputeFluxesAtCell(this, k, j, i, frad, fenth, fkin, ftot);
        }
        user_out_var(iuov,k,j,i) = frad*egas_unit*vel_unit; iuov++;
        user_out_var(iuov,k,j,i) = fenth*egas_unit*vel_unit; iuov++;
        user_out_var(iuov,k,j,i) = fkin*egas_unit*vel_unit; iuov++;
        user_out_var(iuov,k,j,i) = ftot*egas_unit*vel_unit; iuov++;
      }
    }
  }
  return;
}



namespace {

Real HistoryTg(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += pmb->peos->TempFromRhoEg(pmb->phydro->w(IDN,k,j,i),
                                      pmb->prfld2->u_gas(k,j,i))*T_unit;
        num++;
      }
    }
  }
  T /= num;
  return T;
}

Real HistoryTr(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += std::pow(pmb->prfld2->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
        num++;
      }
    }
  }
  T /= num;
  return T;
}

// caution! this is for a mean of gas energy density.
Real HistoryEg(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real e = 0;
  // AthenaArray<Real> vol;
  // vol.NewAthenaArray((ie-is)+2*NGHOST);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      // pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        e += pmb->prfld2->u_gas(k,j,i);//*vol(i);
        num++;
      }
    }
  }
  e /= num;
  return e*egas_unit;
}

// caution! this is for a mean of radiation energy density.
Real HistoryEr(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real E = 0;
  // AthenaArray<Real> vol;
  // vol.NewAthenaArray((ie-is)+2*NGHOST);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      // pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        E += pmb->prfld2->u_rad(k,j,i);//*vol(i);
        num++;
      }
    }
  }
  E /= num;
  return E*egas_unit;
}

Real HistoryaTg4(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real aT4 = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real temp = pmb->peos->TempFromRhoEg(pmb->phydro->w(IDN,k,j,i),
                                             pmb->prfld2->u_gas(k,j,i))*T_unit;
        aT4 += std::pow(temp, 4);
        num++;
      }
    }
  }
  aT4 *= a_r_dim;
  aT4 /= num;
  return aT4;
}

Real HistoryRtime(MeshBlock *pmb, int iout) {
  return pmb->pmy_mesh->time*time_unit;
}

// caution! this is for a sum of all energy.
Real HistoryEall(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((ie-is)+2*NGHOST);
  int num = 0;
  Real E = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        E += pmb->prfld2->u_gas(k,j,i)*vol(i);
        E += pmb->prfld2->u_rad(k,j,i)*vol(i);
      }
    }
  }
  return E*egas_unit;
}

Real HistoryFradTop(MeshBlock *pmb, int iout) {
  return top_flux_rad_cgs;
}

Real HistoryFtotTop(MeshBlock *pmb, int iout) {
  return measured_top_flux_cgs;
}

Real HistoryEinBottom(MeshBlock *pmb, int iout) {
  return bottom_inflow_eint*egas_unit/rho_unit;
}

Real HistoryMass(MeshBlock *pmb, int iout) {
  return total_mass_current;
}

Real HistoryMassDrift(MeshBlock *pmb, int iout) {
  if (total_mass_initial <= 0.0) return 0.0;
  return (total_mass_current - total_mass_initial)/total_mass_initial;
}

Real HistoryVzRms(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  Real vz2 = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real vz = pmb->phydro->w(IVZ,k,j,i);
        vz2 += vz*vz;
      }
    }
  }
  int nx1 = pmb->pmy_mesh->mesh_size.nx1;
  int nx2 = pmb->pmy_mesh->mesh_size.nx2;
  int nx3 = pmb->pmy_mesh->mesh_size.nx3;
  Real ncells = static_cast<Real>(nx1)*static_cast<Real>(nx2)*static_cast<Real>(nx3);
  if (ncells <= 0.0) return 0.0;
  return vz2*SQR(vel_unit)/ncells;
}

Real HistoryEkinConv(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> vol;
  vol.NewAthenaArray((ie-is)+2*NGHOST);
  Real ekin = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; ++i) {
        Real rho = pmb->phydro->w(IDN,k,j,i);
        Real vx = pmb->phydro->w(IVX,k,j,i);
        Real vy = pmb->phydro->w(IVY,k,j,i);
        Real vz = pmb->phydro->w(IVZ,k,j,i);
        ekin += 0.5*rho*(vx*vx + vy*vy + vz*vz)*vol(i);
      }
    }
  }
  return ekin*egas_unit;
}

// Real HistoryL1norm(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   Real L1norm = 0;
//   Real x_L = pmb->pmy_mesh->mesh_size.x1min - pmb->pcoord->dx1f(0)/2.0;
//   Real x_R = pmb->pmy_mesh->mesh_size.x1max + pmb->pcoord->dx1f(0)/2.0;
//   Real slope = (Er0_R-Er0_L)/(x_R-x_L);
//   Real cons = Er0_L - slope*x_L;
//   for (int k=ks; k<=ke; k++) {
//     for (int j=js; j<=je; j++) {
//       for (int i=is; i<=ie; i++) {
//         Real x = pmb->pcoord->x1v(i);
//         Real an = slope*x + cons;
//         L1norm += std::abs(pmb->prfld2->u_rad(k,j,i) - an)/std::abs(an);
//       }
//     }
//   }
//   int nbtotal = pmb->pmy_mesh->nbtotal;
//   int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
//   L1norm /= ncells*nbtotal;
//   return L1norm;
// }

} // namespace
