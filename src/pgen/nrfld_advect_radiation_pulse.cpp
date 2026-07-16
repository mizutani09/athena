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
 * You should have received a copy of GNU General Public License in the file LICENSE
 * included in the code distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/
//! \file nrfld_advect_radiation_pulse.cpp
//! \brief Problem generator for the Zhang et al. (2011) Section 6.7 advecting
//! radiation pulse and Section 6.8 static-equilibrium tests in NR-FLD.

// C++ headers
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fld/fld.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
constexpr Real kRadiationConst = 7.5657e-15;   // erg cm^-3 K^-4
// Keep this identical to FLD2::FLD2 so that E_rad=aT_gas^4 is also an
// exact fixed point of the stiff matter-radiation coupling solve.
constexpr Real kGasConstant = 8.3144621e7;      // erg mol^-1 K^-1

Real rho_unit, egas_unit, leng_unit, time_unit, vel_unit, t_unit;
Real mu, gamma_gas, gm1, igm1;
Real t0_phys, t1_phys, rho0_phys, width_phys, v0_phys;
Real opacity_mass_coeff_phys;
Real rgas_over_mu_phys, total_pressure0_phys;
bool force_lambda_one_third;
bool static_equilibrium_2d;
int blocks_initialized_local;
Real diag_t_min, diag_t_max;
Real diag_rho_min, diag_rho_max;
Real diag_pgas_min, diag_pgas_max;
Real diag_prad_min, diag_prad_max;
Real diag_ptot_min, diag_ptot_max;
Real diag_erad_min, diag_erad_max;
Real diag_ptot_rel_var;

Real TemperatureProfile(Real radius_phys) {
  return t0_phys + (t1_phys - t0_phys)
      * std::exp(-0.5 * SQR(radius_phys / std::max(width_phys, TINY_NUMBER)));
}

Real RadiationEnergyPhysFromTemp(Real temperature) {
  return kRadiationConst * std::pow(temperature, 4);
}

Real GasPressurePhys(Real density, Real temperature) {
  return density * rgas_over_mu_phys * temperature;
}

Real DensityFromPressureBalance(Real temperature) {
  Real prad = RadiationEnergyPhysFromTemp(temperature) * ONE_3RD;
  return (total_pressure0_phys - prad) / (rgas_over_mu_phys * temperature);
}

Real TemperatureFromGasState(Real density_code, Real egas_code) {
  Real density_phys = std::max(density_code * rho_unit, TINY_NUMBER);
  Real pgas_phys = std::max(gm1 * egas_code * egas_unit, 0.0);
  return pgas_phys / (density_phys * rgas_over_mu_phys);
}

Real HistoryTgasMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = -std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, TemperatureFromGasState(pmb->phydro->w(IDN, k, j, i),
                                                    pmb->prfld2->u_gas(k, j, i)));
      }
    }
  }
  return out;
}

Real HistoryTgasMin(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::min(out, TemperatureFromGasState(pmb->phydro->w(IDN, k, j, i),
                                                    pmb->prfld2->u_gas(k, j, i)));
      }
    }
  }
  return out;
}

Real HistoryTradMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = -std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real erad_phys = std::max(pmb->prfld2->u_rad(k, j, i) * egas_unit, 0.0);
        out = std::max(out, std::pow(erad_phys / kRadiationConst, 0.25));
      }
    }
  }
  return out;
}

Real HistoryVelocityMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real vx = pmb->phydro->w(IVX, k, j, i);
        const Real vy = pmb->phydro->w(IVY, k, j, i);
        const Real vz = pmb->phydro->w(IVZ, k, j, i);
        out = std::max(out, std::sqrt(SQR(vx) + SQR(vy) + SQR(vz)) * vel_unit);
      }
    }
  }
  return out;
}

Real HistoryPtotMax(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = -std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real pgas = gm1 * pmb->prfld2->u_gas(k, j, i) * egas_unit;
        const Real prad = ONE_3RD * pmb->prfld2->u_rad(k, j, i) * egas_unit;
        out = std::max(out, pgas + prad);
      }
    }
  }
  return out;
}

Real HistoryPtotMin(MeshBlock *pmb, int iout) {
  (void)iout;
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real pgas = gm1 * pmb->prfld2->u_gas(k, j, i) * egas_unit;
        const Real prad = ONE_3RD * pmb->prfld2->u_rad(k, j, i) * egas_unit;
        out = std::min(out, pgas + prad);
      }
    }
  }
  return out;
}

Real HistoryTransverseTgasSpread(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.nx2 == 1 && pmb->block_size.nx3 == 1) return 0.0;

  Real out = 0.0;
  for (int i = pmb->is; i <= pmb->ie; ++i) {
    Real tmin = std::numeric_limits<Real>::max();
    Real tmax = -std::numeric_limits<Real>::max();
    Real tsum = 0.0;
    int n = 0;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        const Real tgas = TemperatureFromGasState(pmb->phydro->w(IDN, k, j, i),
                                                  pmb->prfld2->u_gas(k, j, i));
        tmin = std::min(tmin, tgas);
        tmax = std::max(tmax, tgas);
        tsum += tgas;
        ++n;
      }
    }
    const Real tmean = tsum / std::max(n, 1);
    out = std::max(out, (tmax - tmin) / std::max(std::abs(tmean), TINY_NUMBER));
  }
  return out;
}

void AdvectPulseOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld, AthenaArray<Real> &prim) {
  (void)u_fld;
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

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        Real rho_phys = std::max(prim(IDN, k, j, i) * rho_unit, TINY_NUMBER);
        // This NR-FLD implementation expects inverse-length coefficients in code
        // units, so sigma = (kappa_mass * rho_phys) * leng_unit.
        Real sigma_code = opacity_mass_coeff_phys * rho_phys * leng_unit;
        prfld->sigma_p(k, j, i) = sigma_code;
        prfld->sigma_r(k, j, i) = std::max(sigma_code, TINY_NUMBER);
      }
    }
  }
}

void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
                              const AthenaArray<Real> &prim,
                              const AthenaArray<Real> &prim_scalar,
                              const AthenaArray<Real> &bcc,
                              AthenaArray<Real> &cons,
                              AthenaArray<Real> &cons_scalar) {
  (void)time;
  (void)prim_scalar;
  (void)bcc;
  (void)cons_scalar;
  FLD2 *prfld = pmb->prfld2;
  AthenaArray<Real> &erad = prfld->u_rad;
  const Real idx1 = 1.0 / pmb->pcoord->dx1f(pmb->is);
  const Real idy = (pmb->block_size.nx2 > 1
                        ? 1.0 / pmb->pcoord->dx2f(pmb->js) : 0.0);
  const Real idz = (pmb->block_size.nx3 > 1
                        ? 1.0 / pmb->pcoord->dx3f(pmb->ks) : 0.0);

  // The static-diffusion algorithm treats the mixed-frame v dot grad(E)
  // exchange explicitly, alongside the force and radiation advection terms.
  // The gas and radiation updates are equal and opposite.
  if (prfld->include_mixed_frame_terms && prfld->mixed_frame_terms_explicit) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          const Real grad_ex = 0.5 * idx1
              * (erad(k, j, i + 1) - erad(k, j, i - 1));
          const Real grad_ey = (pmb->block_size.nx2 > 1
              ? 0.5 * idy * (erad(k, j + 1, i) - erad(k, j - 1, i)) : 0.0);
          const Real grad_ez = (pmb->block_size.nx3 > 1
              ? 0.5 * idz * (erad(k + 1, j, i) - erad(k - 1, j, i)) : 0.0);
          const Real opacity_ratio = prfld->sigma_p(k, j, i)
              / std::max(prfld->sigma_r(k, j, i), TINY_NUMBER);
          const Real mixed_factor = ONE_3RD * (2.0 * opacity_ratio - 1.0);
          const Real mixed_gas = mixed_factor
              * (prim(IVX, k, j, i) * grad_ex
                 + prim(IVY, k, j, i) * grad_ey
                 + prim(IVZ, k, j, i) * grad_ez);
          if (NON_BAROTROPIC_EOS) cons(IEN, k, j, i) += dt * mixed_gas;
          erad(k, j, i) -= dt * mixed_gas;
          erad(k, j, i) = std::max(erad(k, j, i), TINY_NUMBER);
        }
      }
    }
  }

#if NRMGFLD_ENABLED
  // HLLC-FLD includes div(P_rad) in momentum.  This test uses the
  // well-balanced enthalpy-flux mode with implicit_pnablav=false, so the
  // legacy centered force and work below would double count those terms.
  // Keep only the separately derived mixed-frame O(v/c) exchange above.
  return;
#endif

  // P:nabla-v is an explicit radiation-energy source when the implicit NR
  // operator is configured with fld/implicit_pnablav=false.  Applying it here
  // keeps its velocity gradient at the same RK stage as the hydro source.
  if (!prfld->implicit_pnablav) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real div_v = 0.5 * idx1 *
              (prim(IVX, k, j, i + 1) - prim(IVX, k, j, i - 1));
          if (pmb->block_size.nx2 > 1) {
            div_v += 0.5 * idy *
                (prim(IVY, k, j + 1, i) - prim(IVY, k, j - 1, i));
          }
          if (pmb->block_size.nx3 > 1) {
            div_v += 0.5 * idz *
                (prim(IVZ, k + 1, j, i) - prim(IVZ, k - 1, j, i));
          }
          erad(k, j, i) -= dt * ONE_3RD * erad(k, j, i) * div_v;
          erad(k, j, i) = std::max(erad(k, j, i), TINY_NUMBER);
        }
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        // The test fixes lambda=1/3 and uses the post-NR radiation state.
        const Real grad_prad = ONE_3RD * 0.5 * idx1
            * (erad(k, j, i + 1) - erad(k, j, i - 1));
        const Real force_x = -grad_prad;
        const Real force_y = (pmb->block_size.nx2 > 1
            ? -ONE_3RD * 0.5 * idy
                * (erad(k, j + 1, i) - erad(k, j - 1, i)) : 0.0);
        const Real force_z = (pmb->block_size.nx3 > 1
            ? -ONE_3RD * 0.5 * idz
                * (erad(k + 1, j, i) - erad(k - 1, j, i)) : 0.0);

        const Real vx_old = prim(IVX, k, j, i);
        const Real vy_old = prim(IVY, k, j, i);
        const Real vz_old = prim(IVZ, k, j, i);
        cons(IM1, k, j, i) += dt * force_x;
        if (pmb->block_size.nx2 > 1) cons(IM2, k, j, i) += dt * force_y;
        if (pmb->block_size.nx3 > 1) cons(IM3, k, j, i) += dt * force_z;
        if (NON_BAROTROPIC_EOS) {
          const Real rho = std::max(prim(IDN, k, j, i), TINY_NUMBER);
          const Real vx_new = cons(IM1, k, j, i) / rho;
          const Real vy_new = (pmb->block_size.nx2 > 1
                                   ? cons(IM2, k, j, i) / rho : vy_old);
          const Real vz_new = (pmb->block_size.nx3 > 1
                                   ? cons(IM3, k, j, i) / rho : vz_old);
          const Real work = 0.5 * (force_x * (vx_old + vx_new)
                                   + force_y * (vy_old + vy_new)
                                   + force_z * (vz_old + vz_new));
          // Use the same momentum kick in the work term. This is the exact
          // discrete kinetic-energy change for a constant force over dt.
          cons(IEN, k, j, i) += dt * work;
        }
      }
    }
  }
}

void GlobalMinMax(Real &min_val, Real &max_val) {
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &min_val, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
}

void GlobalMax(Real &val) {
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
}
}  // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  static_equilibrium_2d =
      pin->GetOrAddString("problem", "problem_type", "advecting_pulse_1d")
      == "static_equilibrium_2d";
  if (static_equilibrium_2d && mesh_size.nx2 <= 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "static_equilibrium_2d requires nx2 > 1.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "is_couple")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "is_couple must be true for the advecting radiation pulse test.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "only_rad")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "only_rad must be false for the advecting radiation pulse test.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "cut_diff")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "cut_diff must be false for the advecting radiation pulse test.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "cut_Pnablav")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "cut_Pnablav must be false for the advecting radiation pulse test.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetString("mesh", "ix1_bc") != "periodic"
      || pin->GetString("mesh", "ox1_bc") != "periodic"
      || pin->GetString("nrfld", "ix1_bc") != "periodic"
      || pin->GetString("nrfld", "ox1_bc") != "periodic") {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "Hydro and NR-FLD x1 boundaries must both be periodic.";
    ATHENA_ERROR(msg);
  }
  if ((mesh_size.nx2 > 1 || mesh_size.nx3 > 1)
      && (pin->GetString("mesh", "ix2_bc") != "periodic"
          || pin->GetString("mesh", "ox2_bc") != "periodic"
          || pin->GetString("mesh", "ix3_bc") != "periodic"
          || pin->GetString("mesh", "ox3_bc") != "periodic"
          || pin->GetString("nrfld", "ix2_bc") != "periodic"
          || pin->GetString("nrfld", "ox2_bc") != "periodic"
          || pin->GetString("nrfld", "ix3_bc") != "periodic"
          || pin->GetString("nrfld", "ox3_bc") != "periodic")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "Transverse directions must also be periodic when present.";
    ATHENA_ERROR(msg);
  }

  force_lambda_one_third = pin->GetOrAddBoolean("problem", "force_lambda_one_third", true);
  if (!force_lambda_one_third || !pin->GetBoolean("fld", "fixed_flux_limitter")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "This test requires force_lambda_one_third=true and "
        << "fixed_flux_limitter=true so that lambda=1/3 everywhere.";
    ATHENA_ERROR(msg);
  }
  if (pin->GetBoolean("fld", "implicit_pnablav")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "This test requires fld/implicit_pnablav=false for its "
        << "well-balanced radiation-enthalpy flux.";
    ATHENA_ERROR(msg);
  }
  if (!pin->GetBoolean("fld", "include_mixed_frame_terms")
      || !pin->GetBoolean("fld", "mixed_frame_terms_explicit")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "This test requires explicit mixed-frame terms in the fld block.";
    ATHENA_ERROR(msg);
  }

  rho_unit = pin->GetReal("hydro", "rho_unit");
  egas_unit = pin->GetReal("hydro", "egas_unit");
  time_unit = pin->GetOrAddReal("hydro", "time_unit", -1.0);
  leng_unit = pin->GetOrAddReal("hydro", "leng_unit", -1.0);
  if (time_unit < 0.0 && leng_unit < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "time_unit or leng_unit must be specified in block 'hydro'.";
    ATHENA_ERROR(msg);
  } else if (time_unit > 0.0 && leng_unit > 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "time_unit and leng_unit cannot be specified at the same time.";
    ATHENA_ERROR(msg);
  }

  mu = pin->GetOrAddReal("hydro", "mu", 2.33);
  gamma_gas = pin->GetOrAddReal("hydro", "gamma", 5.0 / 3.0);
  gm1 = gamma_gas - 1.0;
  igm1 = 1.0 / gm1;
  vel_unit = std::sqrt(egas_unit / rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit / vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit * time_unit;
  t_unit = egas_unit / rho_unit * mu / kGasConstant;

  t0_phys = pin->GetOrAddReal("problem", "T0", 1.0e7);
  t1_phys = pin->GetOrAddReal("problem", "T1", 2.0e7);
  rho0_phys = pin->GetOrAddReal("problem", "rho0", 1.2);
  width_phys = pin->GetOrAddReal("problem", "w", 24.0);
  v0_phys = pin->GetOrAddReal("problem", "v0", 0.0);
  opacity_mass_coeff_phys = pin->GetOrAddReal("problem", "opacity_mass_coeff",
                                              static_equilibrium_2d ? 1.0e20 : 100.0);

  rgas_over_mu_phys = kGasConstant / mu;
  total_pressure0_phys = GasPressurePhys(rho0_phys, t0_phys)
      + RadiationEnergyPhysFromTemp(t0_phys) * ONE_3RD;
  blocks_initialized_local = 0;
  diag_t_min = std::numeric_limits<Real>::max();
  diag_t_max = -std::numeric_limits<Real>::max();
  diag_rho_min = std::numeric_limits<Real>::max();
  diag_rho_max = -std::numeric_limits<Real>::max();
  diag_pgas_min = std::numeric_limits<Real>::max();
  diag_pgas_max = -std::numeric_limits<Real>::max();
  diag_prad_min = std::numeric_limits<Real>::max();
  diag_prad_max = -std::numeric_limits<Real>::max();
  diag_ptot_min = std::numeric_limits<Real>::max();
  diag_ptot_max = -std::numeric_limits<Real>::max();
  diag_erad_min = std::numeric_limits<Real>::max();
  diag_erad_max = -std::numeric_limits<Real>::max();
  diag_ptot_rel_var = 0.0;

  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTgasMax, "Tgas_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTgasMin, "Tgas_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(2, HistoryTradMax, "Trad_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryVelocityMax, "vel_abs_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryPtotMax, "Ptot_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryPtotMin, "Ptot_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(6, HistoryTransverseTgasSpread, "Tgas_trans_rel",
                          UserHistoryOperation::max);

  EnrollUserExplicitSourceFunction(AddRadiativeForceAndWork);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  (void)pin;
  AllocateUserOutputVariables(14);
  SetUserOutputVariableName(0, "rho");
  SetUserOutputVariableName(1, "vel1");
  SetUserOutputVariableName(2, "vel2");
  SetUserOutputVariableName(3, "vel3");
  SetUserOutputVariableName(4, "vel_mag");
  SetUserOutputVariableName(5, "Pgas");
  SetUserOutputVariableName(6, "Prad");
  SetUserOutputVariableName(7, "Ptot");
  SetUserOutputVariableName(8, "Tgas");
  SetUserOutputVariableName(9, "Trad");
  SetUserOutputVariableName(10, "Erad");
  SetUserOutputVariableName(11, "lambda");
  SetUserOutputVariableName(12, "sigma_P");
  SetUserOutputVariableName(13, "sigma_R");
  prfld2->EnrollOpacityFunction(AdvectPulseOpacity);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  (void)pin;
  Real vx0_code = v0_phys / vel_unit;

  Real t_min = std::numeric_limits<Real>::max();
  Real t_max = -std::numeric_limits<Real>::max();
  Real rho_min = std::numeric_limits<Real>::max();
  Real rho_max = -std::numeric_limits<Real>::max();
  Real pgas_min = std::numeric_limits<Real>::max();
  Real pgas_max = -std::numeric_limits<Real>::max();
  Real prad_min = std::numeric_limits<Real>::max();
  Real prad_max = -std::numeric_limits<Real>::max();
  Real ptot_min = std::numeric_limits<Real>::max();
  Real ptot_max = -std::numeric_limits<Real>::max();
  Real erad_min = std::numeric_limits<Real>::max();
  Real erad_max = -std::numeric_limits<Real>::max();
  Real ptot_rel_var = 0.0;

  int kl = ks;
  int ku = ke;
  int jl = js;
  int ju = je;
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real x_phys = pcoord->x1v(i) * leng_unit;
        Real y_phys = pcoord->x2v(j) * leng_unit;
        // Section 6.8 replaces the Section 6.7 coordinate x by cylindrical
        // radius sqrt(x^2+y^2); the 1-D mode remains unchanged.
        Real profile_coordinate = static_equilibrium_2d
            ? std::sqrt(SQR(x_phys) + SQR(y_phys)) : x_phys;
        Real temp_phys = TemperatureProfile(profile_coordinate);
        Real rho_phys = DensityFromPressureBalance(temp_phys);
        Real erad_phys = RadiationEnergyPhysFromTemp(temp_phys);
        Real pgas_phys = GasPressurePhys(rho_phys, temp_phys);
        Real ptot_phys = pgas_phys + erad_phys * ONE_3RD;

        if (!(rho_phys > 0.0) || !(pgas_phys > 0.0) || !(erad_phys > 0.0)) {
          std::stringstream msg;
          msg << "### FATAL ERROR in MeshBlock::ProblemGenerator" << std::endl
              << "Non-positive initial state at (x,y) = (" << x_phys << ", "
              << y_phys << ") cm: "
              << "rho=" << rho_phys << ", pgas=" << pgas_phys
              << ", Er=" << erad_phys;
          ATHENA_ERROR(msg);
        }

        Real rho_code = rho_phys / rho_unit;
        Real pgas_code = pgas_phys / egas_unit;
        Real egas_code = pgas_code * igm1;
        Real erad_code = erad_phys / egas_unit;

        phydro->u(IDN, k, j, i) = rho_code;
        phydro->u(IM1, k, j, i) = rho_code * vx0_code;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = egas_code + 0.5 * rho_code * SQR(vx0_code);
        }
        prfld2->u_gas(k, j, i) = egas_code;
        prfld2->u_rad(k, j, i) = erad_code;

        if (i >= is && i <= ie) {
          t_min = std::min(t_min, temp_phys);
          t_max = std::max(t_max, temp_phys);
          rho_min = std::min(rho_min, rho_phys);
          rho_max = std::max(rho_max, rho_phys);
          pgas_min = std::min(pgas_min, pgas_phys);
          pgas_max = std::max(pgas_max, pgas_phys);
          prad_min = std::min(prad_min, erad_phys * ONE_3RD);
          prad_max = std::max(prad_max, erad_phys * ONE_3RD);
          ptot_min = std::min(ptot_min, ptot_phys);
          ptot_max = std::max(ptot_max, ptot_phys);
          erad_min = std::min(erad_min, erad_phys);
          erad_max = std::max(erad_max, erad_phys);
          ptot_rel_var = std::max(ptot_rel_var,
                                  std::abs(ptot_phys - total_pressure0_phys)
                                      / std::max(std::abs(total_pressure0_phys), TINY_NUMBER));
        }
      }
    }
  }

  diag_t_min = std::min(diag_t_min, t_min);
  diag_t_max = std::max(diag_t_max, t_max);
  diag_rho_min = std::min(diag_rho_min, rho_min);
  diag_rho_max = std::max(diag_rho_max, rho_max);
  diag_pgas_min = std::min(diag_pgas_min, pgas_min);
  diag_pgas_max = std::max(diag_pgas_max, pgas_max);
  diag_prad_min = std::min(diag_prad_min, prad_min);
  diag_prad_max = std::max(diag_prad_max, prad_max);
  diag_ptot_min = std::min(diag_ptot_min, ptot_min);
  diag_ptot_max = std::max(diag_ptot_max, ptot_max);
  diag_erad_min = std::min(diag_erad_min, erad_min);
  diag_erad_max = std::max(diag_erad_max, erad_max);
  diag_ptot_rel_var = std::max(diag_ptot_rel_var, ptot_rel_var);
  ++blocks_initialized_local;

  if (blocks_initialized_local == pmy_mesh->nblocal) {
    GlobalMinMax(diag_t_min, diag_t_max);
    GlobalMinMax(diag_rho_min, diag_rho_max);
    GlobalMinMax(diag_pgas_min, diag_pgas_max);
    GlobalMinMax(diag_prad_min, diag_prad_max);
    GlobalMinMax(diag_ptot_min, diag_ptot_max);
    GlobalMinMax(diag_erad_min, diag_erad_max);
    GlobalMax(diag_ptot_rel_var);

    Real sigma_min = opacity_mass_coeff_phys * diag_rho_min;
    Real sigma_max = opacity_mass_coeff_phys * diag_rho_max;
    std::cout << std::setprecision(16);
    if (Globals::my_rank == 0) {
      std::cout << ">>> " << (static_equilibrium_2d ? "Static equilibrium"
                                                     : "Advecting radiation pulse")
                << " initial diagnostics <<<" << std::endl;
      std::cout << "rho_unit            = " << rho_unit << " g cm^-3" << std::endl;
      std::cout << "egas_unit           = " << egas_unit << " erg cm^-3" << std::endl;
      std::cout << "leng_unit           = " << leng_unit << " cm" << std::endl;
      std::cout << "time_unit           = " << time_unit << " s" << std::endl;
      std::cout << "vel_unit            = " << vel_unit << " cm s^-1" << std::endl;
      std::cout << "T_unit              = " << t_unit << " K" << std::endl;
      std::cout << "T_min               = " << diag_t_min << " K" << std::endl;
      std::cout << "T_max               = " << diag_t_max << " K" << std::endl;
      std::cout << "rho_min             = " << diag_rho_min << " g cm^-3" << std::endl;
      std::cout << "rho_max             = " << diag_rho_max << " g cm^-3" << std::endl;
      std::cout << "Pgas_min            = " << diag_pgas_min << " erg cm^-3" << std::endl;
      std::cout << "Pgas_max            = " << diag_pgas_max << " erg cm^-3" << std::endl;
      std::cout << "Prad_min            = " << diag_prad_min << " erg cm^-3" << std::endl;
      std::cout << "Prad_max            = " << diag_prad_max << " erg cm^-3" << std::endl;
      std::cout << "Ptot_min            = " << diag_ptot_min << " erg cm^-3" << std::endl;
      std::cout << "Ptot_max            = " << diag_ptot_max << " erg cm^-3" << std::endl;
      std::cout << "Ptot_rel_variation  = " << diag_ptot_rel_var << std::endl;
      std::cout << "Erad_min            = " << diag_erad_min << " erg cm^-3" << std::endl;
      std::cout << "Erad_max            = " << diag_erad_max << " erg cm^-3" << std::endl;
      std::cout << "v0                  = " << v0_phys << " cm s^-1" << std::endl;
      std::cout << "opacity_convention  = sigma = (opacity_mass_coeff * rho) [cm^-1]" << std::endl;
      std::cout << "sigma_min           = " << sigma_min << " cm^-1" << std::endl;
      std::cout << "sigma_max           = " << sigma_max << " cm^-1" << std::endl;
      std::cout << "lambda_mode         = fixed 1/3" << std::endl;

    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  (void)pin;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real rho_code = phydro->w(IDN, k, j, i);
        Real vx_code = phydro->w(IVX, k, j, i);
        Real vy_code = phydro->w(IVY, k, j, i);
        Real vz_code = phydro->w(IVZ, k, j, i);
        Real egas_code = prfld2->u_gas(k, j, i);
        Real erad_code = prfld2->u_rad(k, j, i);
        Real rho_phys = rho_code * rho_unit;
        Real pgas_phys = gm1 * egas_code * egas_unit;
        Real prad_phys = erad_code * egas_unit * ONE_3RD;
        Real ptot_phys = pgas_phys + prad_phys;
        Real tgas_phys = TemperatureFromGasState(rho_code, egas_code);
        Real trad_phys = std::pow(std::max(erad_code * egas_unit, 0.0) / kRadiationConst, 0.25);
        Real lambda = ONE_3RD;

        user_out_var(0, k, j, i) = rho_phys;
        user_out_var(1, k, j, i) = vx_code * vel_unit;
        user_out_var(2, k, j, i) = vy_code * vel_unit;
        user_out_var(3, k, j, i) = vz_code * vel_unit;
        user_out_var(4, k, j, i) =
            std::sqrt(SQR(vx_code) + SQR(vy_code) + SQR(vz_code)) * vel_unit;
        user_out_var(5, k, j, i) = pgas_phys;
        user_out_var(6, k, j, i) = prad_phys;
        user_out_var(7, k, j, i) = ptot_phys;
        user_out_var(8, k, j, i) = tgas_phys;
        user_out_var(9, k, j, i) = trad_phys;
        user_out_var(10, k, j, i) = erad_code * egas_unit;
        user_out_var(11, k, j, i) = lambda;
        user_out_var(12, k, j, i) = prfld2->sigma_p(k, j, i) / leng_unit;
        user_out_var(13, k, j, i) = prfld2->sigma_r(k, j, i) / leng_unit;
      }
    }
  }
}
