//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/
//! \file nrfld_thermal_wave.cpp
//! \brief Problem generator for the Zhang et al. (2011) thermal-wave NR-FLD test.

// C++ headers
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fld/fld.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
constexpr Real kRadiationConst = 7.5657e-15;  // erg cm^-3 K^-4
constexpr Real kLightSpeed = 2.99792458e10;   // cm s^-1
constexpr Real kSimilarityM = 2.5;
constexpr Real kSimilarityD = 3.0;

Real rho_unit, egas_unit, leng_unit, time_unit, t_unit;
Real rho0, rho_cv_phys, kappa_p_phys, chi_r_coeff_phys;
Real t_ambient_phys, hot_radius_phys, hot_energy_total_phys;
Real egas_floor, erad_floor;
Real sigma_r_floor_phys;
Real hot_volume_phys, hot_temperature_phys;
int hot_cell_count_root;
Real dt0_phys, dt_growth, dt0_amr_phys, dt_growth_amr;
Real amr_grad_factor, amr_temp_floor_phys;
Real hydro_p_floor;
Real analytic_init_ratio, analytic_init_time_phys, analytic_time_ref_phys;
Real analytic_init_front_radius_phys;
Real similarity_alpha, similarity_beta, similarity_A, similarity_B, similarity_K;
bool use_discrete_hot_volume, use_amr_dt, allow_debug_toggles;
bool use_analytic_profile_init, analytic_init_lte;
std::string hot_init_mode;

Real FixedTimeStep(MeshBlock *pmb);
int RefinementCondition(MeshBlock *pmb);
Real HistoryEall(MeshBlock *pmb, int iout);
Real HistoryTmax(MeshBlock *pmb, int iout);
Real HistoryTmin(MeshBlock *pmb, int iout);
Real HistoryErMax(MeshBlock *pmb, int iout);
Real HistoryVmax(MeshBlock *pmb, int iout);
constexpr int kHotspotSubsample = 4;

Real GasEnergyDensityPhysFromTemp(Real temp_phys) {
  return rho_cv_phys * temp_phys;
}

Real GasEnergyCodeFromTemp(Real temp_phys) {
  return GasEnergyDensityPhysFromTemp(temp_phys) / egas_unit;
}

Real RadiationEnergyDensityPhysFromTemp(Real temp_phys) {
  return kRadiationConst * std::pow(temp_phys, 4);
}

Real RadiationEnergyCodeFromTemp(Real temp_phys) {
  return RadiationEnergyDensityPhysFromTemp(temp_phys) / egas_unit;
}

Real TemperaturePhysFromGasEnergyCode(Real egas_code) {
  return std::max(egas_code, 0.0) * egas_unit / rho_cv_phys;
}

Real RosselandOpacityPhysFromTemp(Real temp_phys) {
  return chi_r_coeff_phys * std::sqrt(std::max(temp_phys, 0.0));
}

Real LocalDtPhys(const Mesh *pm) {
  // Zhang et al. (2011) define the growing timestep with step index n starting
  // from 1. Athena++ starts ncycle from 0, so the correct exponent here is
  // simply ncycle. Select the AMR schedule only when explicitly requested.
  if (use_amr_dt) {
    return dt0_amr_phys * std::pow(dt_growth_amr, static_cast<Real>(pm->ncycle));
  }
  return dt0_phys * std::pow(dt_growth, static_cast<Real>(pm->ncycle));
}

Real SolveHotTemperature(Real energy_density_phys) {
  Real t_lo = std::max(t_ambient_phys, 0.0);
  Real t_hi = std::max(1.0, t_lo);
  auto total_energy_density = [](Real temp) {
    return GasEnergyDensityPhysFromTemp(temp) + RadiationEnergyDensityPhysFromTemp(temp);
  };
  while (total_energy_density(t_hi) < energy_density_phys) {
    t_hi *= 2.0;
  }
  for (int n = 0; n < 200; ++n) {
    Real t_mid = 0.5 * (t_lo + t_hi);
    if (total_energy_density(t_mid) < energy_density_phys) {
      t_lo = t_mid;
    } else {
      t_hi = t_mid;
    }
  }
  return 0.5 * (t_lo + t_hi);
}

Real SolveGasDominatedHotTemperature(Real energy_density_phys) {
  return std::max(energy_density_phys / rho_cv_phys, t_ambient_phys);
}

Real BetaFunction(Real a, Real b) {
  return std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);
}

void ComputeSimilarityConstants() {
  similarity_beta = 1.0 / (kSimilarityD * kSimilarityM + 2.0);
  similarity_alpha = kSimilarityD * similarity_beta;
  similarity_K = 4.0 * kRadiationConst * kLightSpeed / (3.0 * chi_r_coeff_phys * rho_cv_phys);
  similarity_B = kSimilarityM * similarity_beta / (2.0 * similarity_K);
  Real i_m = 0.5 * BetaFunction(1.5, 1.0 + 1.0 / kSimilarityM);
  Real exponent = 1.0 / kSimilarityM + 1.5;
  similarity_A = std::pow(((hot_energy_total_phys / rho_cv_phys) * std::pow(similarity_B, 1.5))
                          / (4.0 * PI * i_m), 1.0 / exponent);
}

Real ThermalWaveAnalyticTemperature(Real radius_phys, Real time_phys) {
  if (time_phys <= 0.0) {
    return t_ambient_phys;
  }
  Real eta2 = radius_phys * radius_phys * std::pow(time_phys, -2.0 * similarity_beta);
  Real core = std::max(similarity_A - similarity_B * eta2, 0.0);
  if (core <= 0.0) {
    return t_ambient_phys;
  }
  Real temp = std::pow(time_phys, -similarity_alpha) * std::pow(core, 1.0 / kSimilarityM);
  return std::max(temp, t_ambient_phys);
}

Real ThermalWaveFrontRadius(Real time_phys) {
  if (time_phys <= 0.0) {
    return 0.0;
  }
  return std::sqrt(similarity_A / similarity_B) * std::pow(time_phys, similarity_beta);
}

Real ThermalWaveTimeFromFrontRadius(Real radius_phys) {
  if (radius_phys <= 0.0) {
    return 0.0;
  }
  Real front_coef = std::sqrt(similarity_A / similarity_B);
  return std::pow(radius_phys / front_coef, 1.0 / similarity_beta);
}

Real HotspotVolumeFractionInCell(Real x_center_phys, Real y_center_phys, Real z_center_phys,
                                 Real dx_phys, Real dy_phys, Real dz_phys) {
  int nhot = 0;
  int ntot = kHotspotSubsample * kHotspotSubsample * kHotspotSubsample;
  for (int kk = 0; kk < kHotspotSubsample; ++kk) {
    Real z = z_center_phys
        + (static_cast<Real>(kk) + 0.5) * dz_phys / static_cast<Real>(kHotspotSubsample)
        - 0.5 * dz_phys;
    for (int jj = 0; jj < kHotspotSubsample; ++jj) {
      Real y = y_center_phys
          + (static_cast<Real>(jj) + 0.5) * dy_phys / static_cast<Real>(kHotspotSubsample)
          - 0.5 * dy_phys;
      for (int ii = 0; ii < kHotspotSubsample; ++ii) {
        Real x = x_center_phys
            + (static_cast<Real>(ii) + 0.5) * dx_phys / static_cast<Real>(kHotspotSubsample)
            - 0.5 * dx_phys;
        if (std::sqrt(x * x + y * y + z * z) < hot_radius_phys) {
          ++nhot;
        }
      }
    }
  }
  return static_cast<Real>(nhot) / static_cast<Real>(ntot);
}

Real ComputeDiscreteHotVolumeRootGrid(const RegionSize &rs) {
  if (rs.nx1 <= 0 || rs.nx2 <= 0 || rs.nx3 <= 0) {
    return 0.0;
  }
  if (rs.x1rat != 1.0 || rs.x2rat != 1.0 || rs.x3rat != 1.0) {
    return -1.0;
  }

  Real dx = (rs.x1max - rs.x1min) / static_cast<Real>(rs.nx1);
  Real dy = (rs.x2max - rs.x2min) / static_cast<Real>(rs.nx2);
  Real dz = (rs.x3max - rs.x3min) / static_cast<Real>(rs.nx3);
  Real cell_volume_phys = dx * dy * dz * std::pow(leng_unit, 3);
  Real volume = 0.0;

  for (int k = 0; k < rs.nx3; ++k) {
    Real z = (rs.x3min + (static_cast<Real>(k) + 0.5) * dz) * leng_unit;
    for (int j = 0; j < rs.nx2; ++j) {
      Real y = (rs.x2min + (static_cast<Real>(j) + 0.5) * dy) * leng_unit;
      for (int i = 0; i < rs.nx1; ++i) {
        Real x = (rs.x1min + (static_cast<Real>(i) + 0.5) * dx) * leng_unit;
        Real frac = HotspotVolumeFractionInCell(x, y, z, dx * leng_unit, dy * leng_unit,
                                                dz * leng_unit);
        volume += frac * cell_volume_phys;
      }
    }
  }
  return volume;
}

int CountHotCellsRootGrid(const RegionSize &rs) {
  if (rs.nx1 <= 0 || rs.nx2 <= 0 || rs.nx3 <= 0) {
    return 0;
  }
  if (rs.x1rat != 1.0 || rs.x2rat != 1.0 || rs.x3rat != 1.0) {
    return -1;
  }

  Real dx = (rs.x1max - rs.x1min) / static_cast<Real>(rs.nx1);
  Real dy = (rs.x2max - rs.x2min) / static_cast<Real>(rs.nx2);
  Real dz = (rs.x3max - rs.x3min) / static_cast<Real>(rs.nx3);
  int count = 0;
  for (int k = 0; k < rs.nx3; ++k) {
    Real z = (rs.x3min + (static_cast<Real>(k) + 0.5) * dz) * leng_unit;
    for (int j = 0; j < rs.nx2; ++j) {
      Real y = (rs.x2min + (static_cast<Real>(j) + 0.5) * dy) * leng_unit;
      for (int i = 0; i < rs.nx1; ++i) {
        Real x = (rs.x1min + (static_cast<Real>(i) + 0.5) * dx) * leng_unit;
        Real frac = HotspotVolumeFractionInCell(x, y, z, dx * leng_unit, dy * leng_unit,
                                                dz * leng_unit);
        if (frac > 0.0) {
          ++count;
        }
      }
    }
  }
  return count;
}

void CopyNRBoundary(AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas, int axis, bool inner,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (axis == 1) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        int ir = inner ? is : ie;
        for (int i = 1; i <= ngh; ++i) {
          int ii = inner ? is - i : ie + i;
          u_rad(k, j, ii) = u_rad(k, j, ir);
          u_gas(k, j, ii) = u_gas(k, j, ir);
        }
      }
    }
  } else if (axis == 2) {
    for (int k = ks; k <= ke; ++k) {
      int jr = inner ? js : je;
      for (int j = 1; j <= ngh; ++j) {
        int jj = inner ? js - j : je + j;
        for (int i = is; i <= ie; ++i) {
          u_rad(k, jj, i) = u_rad(k, jr, i);
          u_gas(k, jj, i) = u_gas(k, jr, i);
        }
      }
    }
  } else {
    int kr = inner ? ks : ke;
    for (int k = 1; k <= ngh; ++k) {
      int kk = inner ? ks - k : ke + k;
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          u_rad(kk, j, i) = u_rad(kr, j, i);
          u_gas(kk, j, i) = u_gas(kr, j, i);
        }
      }
    }
  }
}

void CopyFLDBoundary(AthenaArray<Real> &u_rad_fld, int axis, bool inner,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (axis == 1) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        int ir = inner ? is : ie;
        for (int i = 1; i <= ngh; ++i) {
          int ii = inner ? is - i : ie + i;
          u_rad_fld(k, j, ii) = u_rad_fld(k, j, ir);
        }
      }
    }
  } else if (axis == 2) {
    for (int k = ks; k <= ke; ++k) {
      int jr = inner ? js : je;
      for (int j = 1; j <= ngh; ++j) {
        int jj = inner ? js - j : je + j;
        for (int i = is; i <= ie; ++i) {
          u_rad_fld(k, jj, i) = u_rad_fld(k, jr, i);
        }
      }
    }
  } else {
    int kr = inner ? ks : ke;
    for (int k = 1; k <= ngh; ++k) {
      int kk = inner ? ks - k : ke + k;
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          u_rad_fld(kk, j, i) = u_rad_fld(kr, j, i);
        }
      }
    }
  }
}

void CopyHydroBoundary(AthenaArray<Real> &prim, int axis, bool inner,
                       int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (axis == 1) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        int ir = inner ? is : ie;
        for (int i = 1; i <= ngh; ++i) {
          int ii = inner ? is - i : ie + i;
          prim(IDN, k, j, ii) = prim(IDN, k, j, ir);
          prim(IVX, k, j, ii) = 0.0;
          prim(IVY, k, j, ii) = 0.0;
          prim(IVZ, k, j, ii) = 0.0;
          prim(IPR, k, j, ii) = prim(IPR, k, j, ir);
        }
      }
    }
  } else if (axis == 2) {
    for (int k = ks; k <= ke; ++k) {
      int jr = inner ? js : je;
      for (int j = 1; j <= ngh; ++j) {
        int jj = inner ? js - j : je + j;
        for (int i = is; i <= ie; ++i) {
          prim(IDN, k, jj, i) = prim(IDN, k, jr, i);
          prim(IVX, k, jj, i) = 0.0;
          prim(IVY, k, jj, i) = 0.0;
          prim(IVZ, k, jj, i) = 0.0;
          prim(IPR, k, jj, i) = prim(IPR, k, jr, i);
        }
      }
    }
  } else {
    int kr = inner ? ks : ke;
    for (int k = 1; k <= ngh; ++k) {
      int kk = inner ? ks - k : ke + k;
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          prim(IDN, kk, j, i) = prim(IDN, kr, j, i);
          prim(IVX, kk, j, i) = 0.0;
          prim(IVY, kk, j, i) = 0.0;
          prim(IVZ, kk, j, i) = 0.0;
          prim(IPR, kk, j, i) = prim(IPR, kr, j, i);
        }
      }
    }
  }
}

void ThermalWaveOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld, AthenaArray<Real> &prim) {
  (void)u_fld;
  (void)prim;
  FLD2 *prfld = pmb->prfld2;
  int kl = pmb->ks;
  int ku = pmb->ke;
  int jl = pmb->js;
  int ju = pmb->je;
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  if (pmb->block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  Real sigma_p_code = kappa_p_phys * leng_unit;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        Real temp_phys = TemperaturePhysFromGasEnergyCode(prfld->u_gas(k, j, i));
        Real sigma_r_code = std::max(RosselandOpacityPhysFromTemp(temp_phys),
                                     sigma_r_floor_phys) * leng_unit;
        prfld->sigma_p(k, j, i) = (prfld->is_couple ? sigma_p_code : 0.0);
        prfld->sigma_r(k, j, i) = std::max(sigma_r_code, TINY_NUMBER);
      }
    }
  }
}
}  // namespace

void FLDInnerX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyFLDBoundary(u_rad_fld, 1, true, is, ie, js, je, ks, ke, ngh);
}

void FLDOuterX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyFLDBoundary(u_rad_fld, 1, false, is, ie, js, je, ks, ke, ngh);
}

void FLDInnerX2(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyFLDBoundary(u_rad_fld, 2, true, is, ie, js, je, ks, ke, ngh);
}

void FLDOuterX2(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyFLDBoundary(u_rad_fld, 2, false, is, ie, js, je, ks, ke, ngh);
}

void FLDInnerX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyFLDBoundary(u_rad_fld, 3, true, is, ie, js, je, ks, ke, ngh);
}

void FLDOuterX3(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyFLDBoundary(u_rad_fld, 3, false, is, ie, js, je, ks, ke, ngh);
}

void NRInnerX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyNRBoundary(u_rad, u_gas, 1, true, is, ie, js, je, ks, ke, ngh);
}

void NROuterX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyNRBoundary(u_rad, u_gas, 1, false, is, ie, js, je, ks, ke, ngh);
}

void NRInnerX2(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyNRBoundary(u_rad, u_gas, 2, true, is, ie, js, je, ks, ke, ngh);
}

void NROuterX2(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyNRBoundary(u_rad, u_gas, 2, false, is, ie, js, je, ks, ke, ngh);
}

void NRInnerX3(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyNRBoundary(u_rad, u_gas, 3, true, is, ie, js, je, ks, ke, ngh);
}

void NROuterX3(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyNRBoundary(u_rad, u_gas, 3, false, is, ie, js, je, ks, ke, ngh);
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyHydroBoundary(prim, 1, true, is, ie, js, je, ks, ke, ngh);
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyHydroBoundary(prim, 1, false, is, ie, js, je, ks, ke, ngh);
}

void HydroInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyHydroBoundary(prim, 2, true, is, ie, js, je, ks, ke, ngh);
}

void HydroOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyHydroBoundary(prim, 2, false, is, ie, js, je, ks, ke, ngh);
}

void HydroInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyHydroBoundary(prim, 3, true, is, ie, js, je, ks, ke, ngh);
}

void HydroOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  CopyHydroBoundary(prim, 3, false, is, ie, js, je, ks, ke, ngh);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  allow_debug_toggles = pin->GetOrAddBoolean("problem", "allow_debug_toggles", false);

  if (!pin->GetBoolean("fld", "is_couple") && !allow_debug_toggles) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "is_couple must be true for this problem.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "only_rad must be true for this problem.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "cut_diff") && !allow_debug_toggles) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_diff must be false for this problem.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "is_couple") && !allow_debug_toggles) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "is_couple must be true for this problem.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "include_radiation_force")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "include_radiation_force must be false for this problem.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "fixed_flux_limiter")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "fixed_flux_limiter must be true for this problem.";
    ATHENA_ERROR(msg);
  }
  FluidFormulation fluid = GetFluidFormulation(pin->GetOrAddString("hydro", "active", "true"));
  if (fluid != FluidFormulation::background) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "hydro/active must be set to background for this problem.";
    ATHENA_ERROR(msg);
  }

  rho_unit = pin->GetReal("hydro", "rho_unit");
  egas_unit = pin->GetReal("hydro", "egas_unit");
  t_unit = pin->GetReal("hydro", "T_unit");
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
  Real vel_unit = std::sqrt(egas_unit / rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit / vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit * time_unit;

  rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  rho_cv_phys = pin->GetOrAddReal("problem", "rho_cv", 0.05);
  kappa_p_phys = pin->GetOrAddReal("problem", "kappa_p", 1.0e6);
  chi_r_coeff_phys = pin->GetOrAddReal("problem", "chi_r_coeff", 1.0e-3);
  sigma_r_floor_phys = pin->GetOrAddReal("problem", "sigma_r_floor", 0.0);
  t_ambient_phys = pin->GetOrAddReal("problem", "t_ambient", 1.0e-6);
  hot_radius_phys = pin->GetOrAddReal("problem", "r_hot", 3.125);
  hot_energy_total_phys = pin->GetOrAddReal("problem", "e_hot_total", 3.0e7);
  egas_floor = pin->GetOrAddReal("problem", "egas_floor", 1.0e-40);
  erad_floor = pin->GetOrAddReal("problem", "erad_floor", 1.0e-60);
  use_discrete_hot_volume = pin->GetOrAddBoolean("problem", "use_discrete_hot_volume", true);
  use_amr_dt = pin->GetOrAddBoolean("problem", "use_amr_dt", multilevel);
  use_analytic_profile_init =
      pin->GetOrAddBoolean("problem", "use_analytic_profile_init", false);
  analytic_init_lte = pin->GetOrAddBoolean("problem", "analytic_init_lte", true);
  hot_init_mode = pin->GetOrAddString("problem", "hot_init_mode", "gas_total");
  analytic_init_front_radius_phys =
      pin->GetOrAddReal("problem", "init_front_radius", 50.0);
  analytic_init_ratio = pin->GetOrAddReal("problem", "init_ratio", -1.0);
  analytic_init_time_phys = pin->GetOrAddReal("problem", "init_time", -1.0);
  analytic_time_ref_phys = pin->GetOrAddReal("problem", "analytic_time_ref", -1.0);
  dt0_phys = pin->GetOrAddReal("problem", "dt0", 5.0e-16);
  dt_growth = pin->GetOrAddReal("problem", "dt_growth", 1.03);
  dt0_amr_phys = pin->GetOrAddReal("problem", "dt0_amr", 1.015e-15);
  dt_growth_amr = pin->GetOrAddReal("problem", "dt_growth_amr", 1.0609);
  amr_grad_factor = pin->GetOrAddReal("problem", "amr_grad_factor", 0.4);
  amr_temp_floor_phys = pin->GetOrAddReal("problem", "amr_temp_floor", 1.0e-5);

  Real analytic_hot_volume = 4.0 * PI * std::pow(hot_radius_phys, 3) / 3.0;
  hot_volume_phys = analytic_hot_volume;
  hot_cell_count_root = CountHotCellsRootGrid(mesh_size);
  if (use_discrete_hot_volume) {
    Real discrete_root_volume = ComputeDiscreteHotVolumeRootGrid(mesh_size);
    if (discrete_root_volume > 0.0) {
      hot_volume_phys = discrete_root_volume;
    }
  }
  if (hot_init_mode == "lte_total") {
    hot_temperature_phys = SolveHotTemperature(hot_energy_total_phys / hot_volume_phys);
  } else if (hot_init_mode == "gas_total") {
    hot_temperature_phys = SolveGasDominatedHotTemperature(hot_energy_total_phys / hot_volume_phys);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "problem/hot_init_mode must be either 'lte_total' or 'gas_total'.";
    ATHENA_ERROR(msg);
  }
  ComputeSimilarityConstants();
  if (use_analytic_profile_init) {
    if (analytic_init_time_phys <= 0.0) {
      if (analytic_init_front_radius_phys > 0.0) {
        analytic_init_time_phys = ThermalWaveTimeFromFrontRadius(analytic_init_front_radius_phys);
      } else {
        if (analytic_time_ref_phys <= 0.0) {
          analytic_time_ref_phys = pin->GetReal("time", "tlim");
        }
        if (analytic_init_ratio <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
          msg << "analytic initial profile requires problem/init_front_radius > 0, "
                 "problem/init_time > 0, or problem/init_ratio > 0.";
          ATHENA_ERROR(msg);
        }
        analytic_init_time_phys = analytic_init_ratio * analytic_time_ref_phys;
      }
    }
    if (analytic_init_time_phys <= 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
      msg << "problem/init_time must be positive after applying the chosen analytic initializer.";
      ATHENA_ERROR(msg);
    }
  }

  Real gamma_gas = pin->GetReal("hydro", "gamma");
  hydro_p_floor = std::max((gamma_gas - 1.0) * egas_floor, TINY_NUMBER);

  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDInnerX1);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDOuterX1);
  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x2, FLDInnerX2);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x2, FLDOuterX2);
  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, FLDInnerX3);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, FLDOuterX3);

  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x2, NRInnerX2);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x2, NROuterX2);
  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x3, NRInnerX3);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x3, NROuterX3);

  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, HydroInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, HydroOuterX2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, HydroInnerX3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, HydroOuterX3);

  EnrollUserTimeStepFunction(FixedTimeStep);
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
  }

  AllocateUserHistoryOutput(5);
  EnrollUserHistoryOutput(0, HistoryEall, "all_E", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(1, HistoryTmax, "T_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryTmin, "T_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(3, HistoryErMax, "Er_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryVmax, "vmax", UserHistoryOperation::max);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(6);
  SetUserOutputVariableName(0, "Tgas");
  SetUserOutputVariableName(1, "Erad");
  SetUserOutputVariableName(2, "egas");
  SetUserOutputVariableName(3, "etherm");
  SetUserOutputVariableName(4, "chiR");
  SetUserOutputVariableName(5, "amr_metric");
  prfld2->EnrollOpacityFunction(ThermalWaveOpacity);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int kl = ks - NGHOST;
  int ku = ke + NGHOST;
  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  Real egas_ambient = GasEnergyCodeFromTemp(t_ambient_phys);
  Real erad_ambient = RadiationEnergyCodeFromTemp(t_ambient_phys);

  for (int k = kl; k <= ku; ++k) {
    Real z_phys = pcoord->x3v(k) * leng_unit;
    for (int j = jl; j <= ju; ++j) {
      Real y_phys = pcoord->x2v(j) * leng_unit;
      for (int i = il; i <= iu; ++i) {
        Real x_phys = pcoord->x1v(i) * leng_unit;
        Real egas_code = egas_ambient;
        Real erad_code = erad_ambient;
        if (use_analytic_profile_init) {
          Real radius_phys = std::sqrt(x_phys * x_phys + y_phys * y_phys + z_phys * z_phys);
          Real temp_phys = ThermalWaveAnalyticTemperature(radius_phys, analytic_init_time_phys);
          egas_code = std::max(GasEnergyCodeFromTemp(temp_phys), egas_floor);
          if (analytic_init_lte) {
            erad_code = std::max(RadiationEnergyCodeFromTemp(temp_phys), erad_floor);
          }
        } else {
          Real frac = HotspotVolumeFractionInCell(x_phys, y_phys, z_phys,
                                                  pcoord->dx1v(i) * leng_unit,
                                                  pcoord->dx2v(j) * leng_unit,
                                                  pcoord->dx3v(k) * leng_unit);
          Real egas_hot = GasEnergyCodeFromTemp(hot_temperature_phys);
          Real erad_hot = erad_ambient;
          if (hot_init_mode == "lte_total") {
            erad_hot = RadiationEnergyCodeFromTemp(hot_temperature_phys);
          }
          egas_code = std::max(egas_ambient + frac * (egas_hot - egas_ambient), egas_floor);
          erad_code = std::max(erad_ambient + frac * (erad_hot - erad_ambient), erad_floor);
        }
        Real pres = std::max((pin->GetReal("hydro", "gamma") - 1.0) * egas_code,
                             hydro_p_floor);

        phydro->u(IDN, k, j, i) = rho0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        if (NON_BAROTROPIC_EOS) phydro->u(IEN, k, j, i) = egas_code;
        prfld2->u_gas(k, j, i) = egas_code;
        prfld2->u_rad(k, j, i) = erad_code;

        phydro->w(IDN, k, j, i) = rho0;
        phydro->w(IVX, k, j, i) = 0.0;
        phydro->w(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = 0.0;
        phydro->w(IPR, k, j, i) = pres;
      }
    }
  }

  if (gid == 0) {
    Real analytic_hot_volume = 4.0 * PI * std::pow(hot_radius_phys, 3) / 3.0;
    Real egas_hot = GasEnergyCodeFromTemp(hot_temperature_phys);
    Real erad_hot = RadiationEnergyCodeFromTemp(t_ambient_phys);
    Real analytic_front = 0.0;
    Real analytic_center_temp = 0.0;
    if (hot_init_mode == "lte_total") {
      erad_hot = RadiationEnergyCodeFromTemp(hot_temperature_phys);
    }
    if (use_analytic_profile_init) {
      analytic_front = ThermalWaveFrontRadius(analytic_init_time_phys);
      analytic_center_temp = ThermalWaveAnalyticTemperature(0.0, analytic_init_time_phys);
    }
    Real sigma_p_code = kappa_p_phys * leng_unit;
    Real sigma_r_hot = RosselandOpacityPhysFromTemp(hot_temperature_phys) * leng_unit;
    Real sigma_r_amb = RosselandOpacityPhysFromTemp(t_ambient_phys) * leng_unit;
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "T_unit = " << t_unit << " K" << std::endl;
    std::cout << "hot_volume = " << hot_volume_phys << " cm^3" << std::endl;
    std::cout << "hot_volume_analytic = " << analytic_hot_volume << " cm^3" << std::endl;
    std::cout << "hot_cell_count_root = " << hot_cell_count_root << std::endl;
    std::cout << "hot_init_mode = " << hot_init_mode << std::endl;
    std::cout << "T_hot = " << hot_temperature_phys << " K" << std::endl;
    std::cout << "egas_hot = " << egas_hot << " [code]" << std::endl;
    std::cout << "erad_hot = " << erad_hot << " [code]" << std::endl;
    std::cout << "sigma_p = " << sigma_p_code << " [code]" << std::endl;
    std::cout << "sigma_r_hot = " << sigma_r_hot << " [code]" << std::endl;
    std::cout << "sigma_r_ambient = " << sigma_r_amb << " [code]" << std::endl;
    std::cout << "sigma_r_floor = " << sigma_r_floor_phys * leng_unit << " [code]" << std::endl;
    std::cout << "use_analytic_profile_init = " << use_analytic_profile_init << std::endl;
    if (use_analytic_profile_init) {
      std::cout << "analytic_init_lte = " << analytic_init_lte << std::endl;
      std::cout << "init_front_radius = " << analytic_init_front_radius_phys << " cm" << std::endl;
      std::cout << "analytic_time_ref = " << analytic_time_ref_phys << " s" << std::endl;
      std::cout << "init_ratio = " << analytic_init_ratio << std::endl;
      std::cout << "init_time = " << analytic_init_time_phys << " s" << std::endl;
      std::cout << "analytic_alpha = " << similarity_alpha << std::endl;
      std::cout << "analytic_beta = " << similarity_beta << std::endl;
      std::cout << "analytic_front = " << analytic_front << " cm" << std::endl;
      std::cout << "analytic_T_center = " << analytic_center_temp << " K" << std::endl;
    }
    std::cout << "dt0 = " << dt0_phys << " s" << std::endl;
    std::cout << "dt_growth = " << dt_growth << std::endl;
    std::cout << "dt0_amr = " << dt0_amr_phys << " s" << std::endl;
    std::cout << "dt_growth_amr = " << dt_growth_amr << std::endl;

    std::ofstream ofs("problem_parameters.txt");
    ofs << ">>> Thermal-wave problem parameters <<<" << std::endl;
    ofs << "rho_unit          = " << rho_unit << std::endl;
    ofs << "egas_unit         = " << egas_unit << std::endl;
    ofs << "time_unit         = " << time_unit << std::endl;
    ofs << "leng_unit         = " << leng_unit << std::endl;
    ofs << "T_unit            = " << t_unit << std::endl;
    ofs << "rho0              = " << rho0 << std::endl;
    ofs << "rho_cv            = " << rho_cv_phys << std::endl;
    ofs << "kappa_p           = " << kappa_p_phys << std::endl;
    ofs << "chi_r_coeff       = " << chi_r_coeff_phys << std::endl;
    ofs << "t_ambient         = " << t_ambient_phys << std::endl;
    ofs << "r_hot             = " << hot_radius_phys << std::endl;
    ofs << "e_hot_total       = " << hot_energy_total_phys << std::endl;
    ofs << "hot_volume        = " << hot_volume_phys << std::endl;
    ofs << "hot_cell_count_root = " << hot_cell_count_root << std::endl;
    ofs << "hot_init_mode     = " << hot_init_mode << std::endl;
    ofs << "T_hot             = " << hot_temperature_phys << std::endl;
    ofs << "egas_hot          = " << egas_hot << std::endl;
    ofs << "erad_hot          = " << erad_hot << std::endl;
    ofs << "sigma_p_code      = " << sigma_p_code << std::endl;
    ofs << "sigma_r_hot_code  = " << sigma_r_hot << std::endl;
    ofs << "sigma_r_amb_code  = " << sigma_r_amb << std::endl;
    ofs << "sigma_r_floor_code = " << sigma_r_floor_phys * leng_unit << std::endl;
    ofs << "use_analytic_profile_init = " << use_analytic_profile_init << std::endl;
    if (use_analytic_profile_init) {
      ofs << "analytic_init_lte = " << analytic_init_lte << std::endl;
      ofs << "init_front_radius = " << analytic_init_front_radius_phys << std::endl;
      ofs << "analytic_time_ref = " << analytic_time_ref_phys << std::endl;
      ofs << "init_ratio        = " << analytic_init_ratio << std::endl;
      ofs << "init_time         = " << analytic_init_time_phys << std::endl;
      ofs << "analytic_alpha    = " << similarity_alpha << std::endl;
      ofs << "analytic_beta     = " << similarity_beta << std::endl;
      ofs << "analytic_front    = " << analytic_front << std::endl;
      ofs << "analytic_T_center = " << analytic_center_temp << std::endl;
    }
    ofs << "dt0               = " << dt0_phys << std::endl;
    ofs << "dt_growth         = " << dt_growth << std::endl;
    ofs << "dt0_amr           = " << dt0_amr_phys << std::endl;
    ofs << "dt_growth_amr     = " << dt_growth_amr << std::endl;
    ofs.close();
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  for (int k = ks; k <= ke; ++k) {
    Real dz_phys = pcoord->dx3v(k) * leng_unit;
    for (int j = js; j <= je; ++j) {
      Real dy_phys = pcoord->dx2v(j) * leng_unit;
      for (int i = is; i <= ie; ++i) {
        Real dx_phys = pcoord->dx1v(i) * leng_unit;
        Real temp_phys = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k, j, i));
        Real erad_phys = prfld2->u_rad(k, j, i) * egas_unit;
        Real egas_phys = prfld2->u_gas(k, j, i) * egas_unit;
        Real chi_r_phys = RosselandOpacityPhysFromTemp(temp_phys);

        Real temp_xm = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k, j, i - 1));
        Real temp_xp = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k, j, i + 1));
        Real temp_ym = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k, j - 1, i));
        Real temp_yp = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k, j + 1, i));
        Real temp_zm = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k - 1, j, i));
        Real temp_zp = TemperaturePhysFromGasEnergyCode(prfld2->u_gas(k + 1, j, i));
        Real dtdx = 0.5 * (temp_xp - temp_xm) / std::max(dx_phys, TINY_NUMBER);
        Real dtdy = 0.5 * (temp_yp - temp_ym) / std::max(dy_phys, TINY_NUMBER);
        Real dtdz = 0.5 * (temp_zp - temp_zm) / std::max(dz_phys, TINY_NUMBER);
        Real dx_min = std::min(dx_phys, std::min(dy_phys, dz_phys));
        Real metric = 0.0;
        if (temp_phys > amr_temp_floor_phys) {
          metric = std::sqrt(dtdx * dtdx + dtdy * dtdy + dtdz * dtdz) * dx_min
                   / std::max(temp_phys, TINY_NUMBER);
        }

        user_out_var(0, k, j, i) = temp_phys;
        user_out_var(1, k, j, i) = erad_phys;
        user_out_var(2, k, j, i) = egas_phys;
        user_out_var(3, k, j, i) = egas_phys + erad_phys;
        user_out_var(4, k, j, i) = chi_r_phys;
        user_out_var(5, k, j, i) = metric;
      }
    }
  }
}

namespace {
Real FixedTimeStep(MeshBlock *pmb) {
  return LocalDtPhys(pmb->pmy_mesh) / time_unit;
}

int RefinementCondition(MeshBlock *pmb) {
  Real max_metric = 0.0;
  Real max_temp = 0.0;

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    Real dz_phys = pmb->pcoord->dx3v(k) * leng_unit;
    for (int j = pmb->js; j <= pmb->je; ++j) {
      Real dy_phys = pmb->pcoord->dx2v(j) * leng_unit;
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real dx_phys = pmb->pcoord->dx1v(i) * leng_unit;
        Real temp_phys = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j, i));
        max_temp = std::max(max_temp, temp_phys);
        if (temp_phys <= amr_temp_floor_phys) {
          continue;
        }
        Real temp_xm = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j, i - 1));
        Real temp_xp = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j, i + 1));
        Real temp_ym = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j - 1, i));
        Real temp_yp = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j + 1, i));
        Real temp_zm = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k - 1, j, i));
        Real temp_zp = TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k + 1, j, i));
        Real dtdx = 0.5 * (temp_xp - temp_xm) / std::max(dx_phys, TINY_NUMBER);
        Real dtdy = 0.5 * (temp_yp - temp_ym) / std::max(dy_phys, TINY_NUMBER);
        Real dtdz = 0.5 * (temp_zp - temp_zm) / std::max(dz_phys, TINY_NUMBER);
        Real dx_min = std::min(dx_phys, std::min(dy_phys, dz_phys));
        Real metric = std::sqrt(dtdx * dtdx + dtdy * dtdy + dtdz * dtdz) * dx_min
                      / std::max(temp_phys, TINY_NUMBER);
        max_metric = std::max(max_metric, metric);
      }
    }
  }

  if (max_metric > amr_grad_factor && max_temp > amr_temp_floor_phys) return 1;
  if (max_temp <= amr_temp_floor_phys || max_metric < 0.25 * amr_grad_factor) return -1;
  return 0;
}

Real HistoryEall(MeshBlock *pmb, int iout) {
  AthenaArray<Real> vol;
  vol.NewAthenaArray((pmb->ie - pmb->is) + 2 * NGHOST);
  Real eall = 0.0;
  Real eambient = GasEnergyCodeFromTemp(t_ambient_phys)
                + RadiationEnergyCodeFromTemp(t_ambient_phys);
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        eall += (pmb->prfld2->u_gas(k, j, i) + pmb->prfld2->u_rad(k, j, i) - eambient)
            * vol(i);
      }
    }
  }
  return eall * egas_unit;
}

Real HistoryTmax(MeshBlock *pmb, int iout) {
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j, i)));
      }
    }
  }
  return out;
}

Real HistoryTmin(MeshBlock *pmb, int iout) {
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::min(out, TemperaturePhysFromGasEnergyCode(pmb->prfld2->u_gas(k, j, i)));
      }
    }
  }
  return out;
}

Real HistoryErMax(MeshBlock *pmb, int iout) {
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, pmb->prfld2->u_rad(k, j, i) * egas_unit);
      }
    }
  }
  return out;
}

Real HistoryVmax(MeshBlock *pmb, int iout) {
  Real vmax = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real vmag = std::sqrt(SQR(pmb->phydro->w(IVX, k, j, i))
                            + SQR(pmb->phydro->w(IVY, k, j, i))
                            + SQR(pmb->phydro->w(IVZ, k, j, i)));
        vmax = std::max(vmax, vmag);
      }
    }
  }
  return vmax;
}
}  // namespace
