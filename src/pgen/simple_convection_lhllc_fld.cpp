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

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdint>
#include <fstream>
#include <limits>
#include <iostream>   // endl
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
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../fld/fld.hpp"
#include "../fld/opacity_table.hpp"
#include "../globals.hpp"


#if !NRMGFLD_ENABLED
#error "The implicit FLD solver must be enabled (-nrmgfld)."
#endif

namespace {
  // Real HistoryRtime(MeshBlock *pmb, int iout);
  Real rho_unit, egas_unit, leng_unit;
  Real T_unit, time_unit;
  Real a_r_dim, Rgas, mu;
  UserOpacityTable *puser_table = nullptr;
  Real T0;
  Real a_r_sim, poly_n, grav_acc, z_ref, rho_ref, T_ref, gamma_gas;
  Real noise_lx, noise_ly;
  Real bottom_inflow_speed;
  std::string profile_output;
  Real profile_zmax;
  Real profile_tau_target_z;
  Real profile_tau_target;
  bool profile_written = false;
  ParameterInput *profile_pin = nullptr;
  std::string bottom_bc_mode;
  Real bottom_pbnd = 0.0;
  Real bottom_s_in = 0.0;
  Real bottom_cdmp = 0.95;
  Real hd2_pbar1 = 0.0;
  Real hd2_pbar2 = 0.0;
  unsigned long long hd2_pressure_floors = 0;
  unsigned long long hd2_density_floors = 0;
  unsigned long long hd2_energy_floors = 0;
  bool hd2_diagnostic_printed = false;
  bool hd2_mass_balance_on = true;
  Real hd2_mass_balance_relax_time = 10.0;
  Real hd2_mass_balance_gain = 10.0;
  Real hd2_mass_balance_rate_gain = 10.0;
  Real hd2_mass_balance_min_scale = 0.7;
  Real hd2_mass_balance_max_scale = 1.5;
  Real hd2_pressure_scale = 1.0;
  Real hd2_initial_mass = -1.0;
  Real hd2_current_mass = -1.0;
  Real hd2_mass_rate = 0.0;
  bool top_outflow_only;
  bool top_impenetrable;
  std::string top_bc_mode;
  Real rad_flux_cgs;
  Real rad_top_alpha;
  Real rad_top_erad_ext;
  unsigned long long marshak_floor_count = 0;
  unsigned long long marshak_aux_fallback_count = 0;
  bool marshak_diagnostic_printed = false;
  Real HistoryTg(MeshBlock *pmb, int iout);
  Real HistoryTr(MeshBlock *pmb, int iout);
  Real HistoryEg(MeshBlock *pmb, int iout);
  Real HistoryEr(MeshBlock *pmb, int iout);
  Real HistoryaTg4(MeshBlock *pmb, int iout);
  Real HistoryRtime(MeshBlock *pmb, int iout);
  Real HistoryEall(MeshBlock *pmb, int iout);
  Real HistoryVzRms(MeshBlock *pmb, int iout);
  Real HistoryVzRms2Mean(MeshBlock *pmb, int iout);
  Real HistoryMaxSpeed(MeshBlock *pmb, int iout);
  Real HistoryTopFlux(MeshBlock *pmb, int iout);
  Real HistoryTopLuminosity(MeshBlock *pmb, int iout);
  Real HistoryBottomInflow(MeshBlock *pmb, int iout);
  Real HistoryTopOutflow(MeshBlock *pmb, int iout);
  Real HistoryMassDrift(MeshBlock *pmb, int iout);
  Real HistoryPBNDScale(MeshBlock *pmb, int iout);
  Real HistoryMassRate(MeshBlock *pmb, int iout);
  Real HistoryBottomUpflow(MeshBlock *pmb, int iout);
  Real HistoryBottomDownflow(MeshBlock *pmb, int iout);
  Real HistoryBottomEnergyFlux(MeshBlock *pmb, int iout);
  Real HistoryBottomEntropyExcess(MeshBlock *pmb, int iout);
  Real HistoryConservedEnergy(MeshBlock *pmb, int iout);
  Real HistoryBottomHydroFluxCgs(MeshBlock *pmb, int iout);
  Real HistoryBottomRadiationFluxCgs(MeshBlock *pmb, int iout);
  Real HistoryTopHydroFluxCgs(MeshBlock *pmb, int iout);
  Real HistoryTopTotalFluxCgs(MeshBlock *pmb, int iout);
  // Real HistoryL1norm(MeshBlock *pmb, int iout);
}

// User boundary callbacks can be invoked on every meshblock face.  The
// Rempel upper condition belongs only to the physical global upper boundary;
// an internal z-meshblock face must remain an ordinary zero-gradient
// interface for the local ghost fill.
static bool IsPhysicalUpperBoundary(const MeshBlock *pmb);

void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar);

static void SimplePolytrope(const Real z, Real &rho, Real &pres, Real &erad) {
  const Real temp = T_ref - grav_acc*(z - z_ref)/(poly_n + 1.0);
  const Real theta = temp/std::max(T_ref, TINY_NUMBER);
  rho = rho_ref*std::pow(std::max(theta, 1.0e-6), poly_n);
  pres = rho*temp;
  erad = a_r_sim*std::pow(std::max(temp, 1.0e-6), 4);
}

// Deterministic cell noise keeps the same initial condition independent of
// MPI decomposition.  The vertical envelope is applied separately below.
static Real SimpleCellNoise(const Real x, const Real y, const Real z) {
  const std::int64_t ix = static_cast<std::int64_t>(std::llround(1.0e6*x));
  const std::int64_t iy = static_cast<std::int64_t>(std::llround(1.0e6*y));
  const std::int64_t iz = static_cast<std::int64_t>(std::llround(1.0e6*z));
  std::uint64_t h = 1469598103934665603ULL;
  h ^= static_cast<std::uint64_t>(ix); h *= 1099511628211ULL;
  h ^= static_cast<std::uint64_t>(iy); h *= 1099511628211ULL;
  h ^= static_cast<std::uint64_t>(iz); h *= 1099511628211ULL;
  h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27; h *= 0x94d049bb133111ebULL;
  h ^= h >> 31;
  return 2.0*(static_cast<Real>(h)/18446744073709551615.0) - 1.0;
}

// Smooth periodic low-wavenumber seed; cell-wise white noise can leave a
// checkerboard imprint at coarse resolution.
static Real SimpleSmoothNoise(const Real x, const Real y) {
  const Real ax = 2.0*M_PI*x/std::max(noise_lx, TINY_NUMBER);
  const Real ay = 2.0*M_PI*y/std::max(noise_ly, TINY_NUMBER);
  return 0.55*std::sin(ax + 0.31)*std::sin(ay + 1.17)
       + 0.25*std::cos(2.0*ax - 0.73)*std::cos(ay + 0.44)
       + 0.20*std::sin(3.0*ax + 1.21)*std::cos(2.0*ay - 0.19);
}

static Real SimpleEntropyProxy(const Real rho, const Real pres) {
  if (rho <= TINY_NUMBER || pres <= TINY_NUMBER) {
    return -std::numeric_limits<Real>::infinity();
  }
  return std::log(pres) - gamma_gas*std::log(rho);
}

static void SetHydroGhostConserved(MeshBlock *pmb, const int k, const int j,
                                   const int i, const Real rho, const Real pres,
                                   const Real vx, const Real vy, const Real vz,
                                   const Real egas) {
  pmb->phydro->u(IDN, k, j, i) = rho;
  pmb->phydro->u(IM1, k, j, i) = rho*vx;
  pmb->phydro->u(IM2, k, j, i) = rho*vy;
  pmb->phydro->u(IM3, k, j, i) = rho*vz;
  pmb->phydro->u(IEN, k, j, i) =
      egas + 0.5*rho*(vx*vx + vy*vy + vz*vz);
  pmb->prfld->u_gas(k, j, i) = egas;
  // This is the FLD-specific LTE extension; HD2 itself is hydrodynamic.
  pmb->prfld->u_rad(k, j, i) = a_r_sim*std::pow((gamma_gas - 1.0)*egas/
                                                 std::max(rho, TINY_NUMBER), 4);
  (void)pres;
}

static void SimpleHD2Bottom(MeshBlock *pmb, AthenaArray<Real> &prim,
                            int is, int ie, int js, int je, int ks, int ke,
                            int ngh) {
  (void)ke;
  const Real pbar1 = std::max(hd2_pbar1, TINY_NUMBER);
  const Real pbar2 = std::max(hd2_pbar2, TINY_NUMBER);
  const Real pbnd = std::max(bottom_pbnd*hd2_pressure_scale, TINY_NUMBER);
  const Real denom = std::max(std::sqrt(std::max(pbar1*pbar2, TINY_NUMBER)),
                              TINY_NUMBER);
  const Real pbar_g1 = pbar1*pbnd/denom;
  const Real pbar_g2 = pbar1*pbnd*pbnd/
                       std::max(pbar1*pbar2, TINY_NUMBER);
  const Real p_floor = std::max(pmb->peos->GetPressureFloor(), TINY_NUMBER);
  const Real rho_floor = std::max(pmb->peos->GetDensityFloor(), TINY_NUMBER);
  const Real gm1 = std::max(gamma_gas - 1.0, TINY_NUMBER);

  for (int j = js; j <= je; ++j) {
    for (int i = is; i <= ie; ++i) {
      const Real rho1 = std::max(prim(IDN, ks, j, i), rho_floor);
      const Real rho2 = std::max(prim(IDN, ks + 1, j, i), rho_floor);
      const Real p1 = std::max(prim(IPR, ks, j, i), p_floor);
      const Real p2 = std::max(prim(IPR, ks + 1, j, i), p_floor);
      const Real s1 = SimpleEntropyProxy(rho1, p1);
      const Real s2 = SimpleEntropyProxy(rho2, p2);
      const Real vz1 = prim(IVZ, ks, j, i);
      const bool upflow = vz1 > 0.0;
      for (int n = 1; n <= ngh; ++n) {
        const int kg = ks - n;
        const int ka = (n == 1) ? ks : ks + 1;
        const Real rho_a = (n == 1) ? rho1 : rho2;
        const Real pprime1 = p1 - pbar1;
        const Real pbar_g = (n == 1) ? pbar_g1 : pbar_g2;
        Real p_raw = pbar_g + std::pow(bottom_cdmp, static_cast<Real>(n))*pprime1;
        Real p_g = p_raw;
        if (!std::isfinite(p_g) || p_g <= p_floor) {
          ++hd2_pressure_floors;
          p_g = p_floor;
        }
        const Real s_target = upflow ? bottom_s_in : ((n == 1) ? s1 : s2);
        const Real log_rho = (std::log(p_g) - s_target)/gamma_gas;
        Real rho_g = std::exp(log_rho);
        if (!std::isfinite(rho_g) || rho_g <= rho_floor) {
          ++hd2_density_floors;
          rho_g = rho_floor;
        }
        const Real egas_g = p_g/gm1;
        if (!std::isfinite(egas_g) || egas_g <= TINY_NUMBER) {
          ++hd2_energy_floors;
          std::stringstream msg;
          msg << "### FATAL ERROR in Rempel HD2 lower boundary: invalid "
              << "internal energy at i=" << i << " j=" << j << " k=" << kg
              << " P=" << p_g << " s=" << s_target;
          ATHENA_ERROR(msg);
        }
        const Real vx_a = prim(IVX, ka, j, i);
        const Real vy_a = prim(IVY, ka, j, i);
        const Real vz_a = prim(IVZ, ka, j, i);
        // Rempel HD2: symmetrize mass fluxes, not primitive velocities.
        const Real velocity_ratio = rho_a/std::max(rho_g, rho_floor);
        const Real vx_g = velocity_ratio*vx_a;
        const Real vy_g = velocity_ratio*vy_a;
        const Real vz_g = velocity_ratio*vz_a;
        const Real temp_g = gm1*egas_g/std::max(rho_g, rho_floor);
        if (!std::isfinite(temp_g) || temp_g <= TINY_NUMBER ||
            !std::isfinite(rho_g) || rho_g <= 0.0 ||
            !std::isfinite(p_g) || p_g <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in Rempel HD2 lower boundary: invalid "
              << "thermodynamic reconstruction at i=" << i << " j=" << j
              << " k=" << kg << " rho=" << rho_g << " P=" << p_g
              << " T=" << temp_g;
          ATHENA_ERROR(msg);
        }
        prim(IDN, kg, j, i) = rho_g;
        prim(IPR, kg, j, i) = p_g;
        prim(IVX, kg, j, i) = vx_g;
        prim(IVY, kg, j, i) = vy_g;
        prim(IVZ, kg, j, i) = vz_g;
        SetHydroGhostConserved(pmb, kg, j, i, rho_g, p_g, vx_g, vy_g,
                               vz_g, egas_g);
      }
    }
  }
}

static void SimpleHydroBoundary(MeshBlock *pmb, Coordinates *pco,
                                AthenaArray<Real> &prim, FaceField &b,
                                Real time, Real dt, int is, int ie, int js,
                                int je, int ks, int ke, int ngh, bool inner) {
  (void)pco; (void)b; (void)time; (void)dt; (void)ke;
  if (inner && bottom_bc_mode == "rempel_hd2") {
    SimpleHD2Bottom(pmb, prim, is, ie, js, je, ks, ke, ngh);
    return;
  }
  for (int n = 1; n <= ngh; ++n) {
    const int kg = inner ? ks - n : ke + n;
    const Real z = pmb->pcoord->x3v(kg);
    Real rho, pres, erad;
    SimplePolytrope(z, rho, pres, erad);
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        prim(IDN, kg, j, i) = rho;
        prim(IPR, kg, j, i) = pres;
        prim(IVX, kg, j, i) = 0.0;
        prim(IVY, kg, j, i) = 0.0;
        // A positive value provides a controlled reservoir inflow at the
        // lower boundary.  The default remains zero, i.e. the original
        // fixed hydrostatic boundary.
        prim(IVZ, kg, j, i) = inner ? bottom_inflow_speed : 0.0;
      }
    }
  }
}

static void UpdateHD2PressureMeans(Mesh *pm) {
  if (bottom_bc_mode != "rempel_hd2") return;
  Real sum1 = 0.0, sum2 = 0.0, count = 0.0;
  for (int nb = 0; nb < pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    if (pmb->block_size.x3min != pm->mesh_size.x3min) continue;
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        sum1 += pmb->phydro->w(IPR, pmb->ks, j, i);
        sum2 += pmb->phydro->w(IPR, pmb->ks + 1, j, i);
        count += 1.0;
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sum1, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &sum2, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (count > 0.0) {
    hd2_pbar1 = std::max(sum1/count, TINY_NUMBER);
    hd2_pbar2 = std::max(sum2/count, TINY_NUMBER);
  }
}

static Real TotalDomainMassCode(Mesh *pm) {
  Real mass = 0.0;
  for (int nb = 0; nb < pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          mass += pmb->phydro->w(IDN,k,j,i)
                *pmb->pcoord->GetCellVolume(k,j,i);
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  return mass;
}

static Real TotalDomainMassRateCode(Mesh *pm) {
  Real rate = 0.0;
  for (int nb = 0; nb < pm->nblocal; ++nb) {
    MeshBlock *pmb = pm->my_blocks(nb);
    const bool at_bottom = pmb->block_size.x3min == pm->mesh_size.x3min;
    const bool at_top = pmb->block_size.x3max == pm->mesh_size.x3max;
    if (!at_bottom && !at_top) continue;
    if (at_bottom) {
      const int k = pmb->ks;
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          rate += pmb->phydro->flux[X3DIR](IDN,k,j,i)
                  *pmb->pcoord->GetFace3Area(k,j,i);
        }
      }
    }
    if (at_top) {
      const int k = pmb->ke + 1;
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          rate -= pmb->phydro->flux[X3DIR](IDN,k,j,i)
                  *pmb->pcoord->GetFace3Area(k,j,i);
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &rate, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  return rate;
}

static void UpdateHD2MassBalance(Mesh *pm) {
  if (bottom_bc_mode != "rempel_hd2") return;
  hd2_current_mass = TotalDomainMassCode(pm);
  hd2_mass_rate = TotalDomainMassRateCode(pm);
  if (!(hd2_initial_mass > 0.0)) {
    hd2_initial_mass = hd2_current_mass;
    return;
  }
  if (!hd2_mass_balance_on || !(hd2_mass_balance_relax_time > 0.0)) return;

  // Rempel HD2 preserves local mass-flux symmetry. This optional extension
  // automatically calibrates the otherwise fixed PBND. Proportional mass
  // feedback plus a measured-flux damping term avoids integral wind-up.
  const Real drift = (hd2_current_mass - hd2_initial_mass)/hd2_initial_mass;
  const Real fractional_rate = hd2_mass_rate/hd2_initial_mass;
  Real target = 1.0 - hd2_mass_balance_gain*drift
                      - hd2_mass_balance_rate_gain*fractional_rate;
  target = std::max(hd2_mass_balance_min_scale,
                    std::min(hd2_mass_balance_max_scale, target));
  const Real relax = 1.0 - std::exp(-pm->dt/hd2_mass_balance_relax_time);
  hd2_pressure_scale += relax*(target - hd2_pressure_scale);
}

static void WriteInitialProfiles(ParameterInput *pin) {
  if (profile_output.empty()) return;
  const int nz = pin->GetInteger("mesh", "nx3");
  const Real zmin = pin->GetReal("mesh", "x3min");
  const Real mesh_zmax = pin->GetReal("mesh", "x3max");
  Real zmax = profile_zmax > mesh_zmax ? profile_zmax : mesh_zmax;
  const Real target_z = profile_tau_target_z;
  const Real target_tau = profile_tau_target;
  const Real dz = (zmax - zmin)/static_cast<Real>(nz);
  const Real nabla_ad = (gamma_gas - 1.0)/gamma_gas;
  std::vector<Real> z(nz), rho(nz), press(nz), temp(nz), sigma(nz);
  std::vector<Real> tau(nz), nabla(nz), entropy(nz);
  for (int k = 0; k < nz; ++k) {
    z[k] = zmin + (static_cast<Real>(k) + 0.5)*dz;
    Real erad;
    SimplePolytrope(z[k], rho[k], press[k], erad);
    temp[k] = press[k]/std::max(rho[k], TINY_NUMBER);
    sigma[k] = 0.0;
    if (puser_table != nullptr) {
      const Real kap = puser_table->GetOpacity(
          RadFLD::SIGMA_R, rho[k]*rho_unit, temp[k]*T_unit);
      sigma[k] = std::max(kap*rho[k]*rho_unit*leng_unit, 0.0);
    }
    entropy[k] = std::log(std::max(press[k], TINY_NUMBER))
               - gamma_gas*std::log(std::max(rho[k], TINY_NUMBER));
  }
  tau[nz-1] = 0.5*sigma[nz-1]*dz;
  for (int k = nz - 2; k >= 0; --k)
    tau[k] = tau[k+1] + 0.5*(sigma[k] + sigma[k+1])*dz;
  for (int k = 0; k < nz; ++k) {
    const int km = std::max(k - 1, 0);
    const int kp = std::min(k + 1, nz - 1);
    const Real dlnT = std::log(std::max(temp[kp], TINY_NUMBER))
                    - std::log(std::max(temp[km], TINY_NUMBER));
    const Real dlnP = std::log(std::max(press[kp], TINY_NUMBER))
                    - std::log(std::max(press[km], TINY_NUMBER));
    nabla[k] = (std::abs(dlnP) > TINY_NUMBER) ? dlnT/dlnP : 0.0;
  }

  // Recompute optical depth using an extended 1D atmosphere when requested.
  // The hydro/radiation state is still sampled only on the actual domain;
  // the extension represents the overlying atmosphere used for tau(z).
  if (target_z >= zmin && target_z <= mesh_zmax && target_tau > 0.0) {
    auto tau_from_top = [&](const Real ztop) {
      const int ne = std::max(1024, 8*nz);
      const Real de = (ztop - target_z)/static_cast<Real>(ne);
      Real t = 0.0;
      for (int n = 0; n < ne; ++n) {
        const Real za = ztop - (static_cast<Real>(n) + 0.5)*de;
        Real rr, pp, ee;
        SimplePolytrope(za, rr, pp, ee);
        const Real tt = std::max(pp/std::max(rr, TINY_NUMBER), TINY_NUMBER);
        const Real kap = puser_table != nullptr
            ? puser_table->GetOpacity(RadFLD::SIGMA_R,
                                      rr*rho_unit, tt*T_unit) : 0.0;
        t += std::max(kap*rr*rho_unit*leng_unit, 0.0)*de;
      }
      return t;
    };
    Real lo = mesh_zmax;
    Real hi = (profile_zmax > mesh_zmax) ? profile_zmax : mesh_zmax + 1.0;
    while (tau_from_top(hi) < target_tau && hi < mesh_zmax + 32.0)
      hi = mesh_zmax + 2.0*(hi - mesh_zmax);
    if (tau_from_top(hi) >= target_tau) {
      for (int n = 0; n < 50; ++n) {
        const Real mid = 0.5*(lo + hi);
        if (tau_from_top(mid) < target_tau) lo = mid;
        else hi = mid;
      }
      zmax = hi;
      const Real de = (zmax - target_z)/static_cast<Real>(std::max(1024, 8*nz));
      (void)de;
      for (int k = 0; k < nz; ++k) {
        const Real zk = z[k];
        const int ne = std::max(1024, 8*nz);
        const Real dzs = (zmax - zk)/static_cast<Real>(ne);
        Real t = 0.0;
        for (int n = 0; n < ne; ++n) {
          const Real za = zmax - (static_cast<Real>(n) + 0.5)*dzs;
          Real rr, pp, ee;
          SimplePolytrope(za, rr, pp, ee);
          const Real tt = std::max(pp/std::max(rr, TINY_NUMBER), TINY_NUMBER);
          const Real kap = puser_table != nullptr
              ? puser_table->GetOpacity(RadFLD::SIGMA_R,
                                        rr*rho_unit, tt*T_unit) : 0.0;
          t += std::max(kap*rr*rho_unit*leng_unit, 0.0)*dzs;
        }
        tau[k] = t;
      }
    }
  }
  std::ofstream out(profile_output);
  if (!out) {
    std::cerr << "### WARNING: could not write profile file '"
              << profile_output << "'\n";
    return;
  }
  out << "# Initial 1D profile; optical_depth is measured downward from the top\n";
  out << "# z rho pressure temperature optical_depth nabla nabla_ad "
         "superadiabaticity entropy_proxy sigma_R\n";
  out << "# profile_zmax = " << zmax << " target_z = " << target_z
      << " target_tau = " << target_tau << "\n";
  out.setf(std::ios::scientific);
  out.precision(16);
  for (int k = 0; k < nz; ++k) {
    out << z[k] << " " << rho[k] << " " << press[k] << " " << temp[k]
        << " " << tau[k] << " " << nabla[k] << " " << nabla_ad << " "
        << nabla[k] - nabla_ad << " " << entropy[k] << " " << sigma[k] << "\n";
  }
}

static void SimpleHydroInner(MeshBlock *pmb, Coordinates *pco,
                             AthenaArray<Real> &prim, FaceField &b,
                             Real time, Real dt, int is, int ie, int js,
                             int je, int ks, int ke, int ngh) {
  SimpleHydroBoundary(pmb, pco, prim, b, time, dt, is, ie, js, je, ks, ke,
                      ngh, true);
}

static void SimpleHydroOuter(MeshBlock *pmb, Coordinates *pco,
                             AthenaArray<Real> &prim, FaceField &b,
                             Real time, Real dt, int is, int ie, int js,
                             int je, int ks, int ke, int ngh) {
  (void)pco; (void)b; (void)time; (void)dt;
  if (!IsPhysicalUpperBoundary(pmb)) {
    for (int n = 1; n <= ngh; ++n) {
      const int kg = ke + n;
      // Zero-normal-gradient extrapolation uses the uppermost active state
      // for every ghost layer.  Mirroring deeper active layers would create
      // an artificial gradient in the second ghost layer.
      const int ka = ke;
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          for (int v = 0; v < NHYDRO; ++v)
            prim(v, kg, j, i) = prim(v, ka, j, i);
        }
      }
    }
    return;
  }
  if (top_bc_mode == "rempel_outflow") {
    const Real rho_floor = std::max(pmb->peos->GetDensityFloor(), TINY_NUMBER);
    const Real p_floor = std::max(pmb->peos->GetPressureFloor(), TINY_NUMBER);
    const Real e_floor = TINY_NUMBER;
    for (int n = 1; n <= ngh; ++n) {
      const int kg = ke + n;
      const int ka = std::max(ke - (n - 1), pmb->ks);
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real rho = std::max(prim(IDN, ka, j, i), rho_floor);
          const Real pres = std::max(prim(IPR, ka, j, i), p_floor);
          const Real egas = std::max(pmb->prfld->u_gas(ka, j, i), e_floor);
          const Real vx = prim(IVX, ka, j, i);
          const Real vy = prim(IVY, ka, j, i);
          const Real vz = std::max(prim(IVZ, ka, j, i), 0.0);
          if (!std::isfinite(rho) || !std::isfinite(pres) ||
              !std::isfinite(egas)) {
            std::stringstream msg;
            msg << "### FATAL ERROR in Rempel upper boundary: invalid "
                << "thermodynamic state at i=" << i << " j=" << j
                << " k=" << kg << " rho=" << rho << " P=" << pres
                << " egas=" << egas;
            ATHENA_ERROR(msg);
          }
          prim(IDN, kg, j, i) = rho;
          prim(IPR, kg, j, i) = pres;
          prim(IVX, kg, j, i) = vx;
          prim(IVY, kg, j, i) = vy;
          prim(IVZ, kg, j, i) = vz;
          // Rebuild conserved variables consistently with the modified v3.
          pmb->phydro->u(IDN, kg, j, i) = rho;
          pmb->phydro->u(IM1, kg, j, i) = rho*vx;
          pmb->phydro->u(IM2, kg, j, i) = rho*vy;
          pmb->phydro->u(IM3, kg, j, i) = rho*vz;
          pmb->phydro->u(IEN, kg, j, i) = egas
              + 0.5*rho*(vx*vx + vy*vy + vz*vz);
          pmb->prfld->u_gas(kg, j, i) = egas;
        }
      }
    }
    return;
  }
  if (!top_outflow_only) {
    SimpleHydroBoundary(pmb, pco, prim, b, time, dt, is, ie, js, je, ks, ke,
                        ngh, false);
    return;
  }
  if (top_impenetrable) {
    for (int n = 1; n <= ngh; ++n) {
      const int kg = ke + n;
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          prim(IDN, kg, j, i) = prim(IDN, ke, j, i);
          prim(IPR, kg, j, i) = prim(IPR, ke, j, i);
          prim(IVX, kg, j, i) = prim(IVX, ke, j, i);
          prim(IVY, kg, j, i) = prim(IVY, ke, j, i);
          prim(IVZ, kg, j, i) = -prim(IVZ, ke, j, i);
        }
      }
    }
    return;
  }
  for (int n = 1; n <= ngh; ++n) {
    const int kg = ke + n;
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        prim(IDN, kg, j, i) = prim(IDN, ke, j, i);
        prim(IPR, kg, j, i) = prim(IPR, ke, j, i);
        prim(IVX, kg, j, i) = prim(IVX, ke, j, i);
        prim(IVY, kg, j, i) = prim(IVY, ke, j, i);
        prim(IVZ, kg, j, i) = std::max(0.0, prim(IVZ, ke, j, i));
      }
    }
  }
}

static Real SimpleFluxForGhost(const Real ec, const Real eg, const Real sigma,
                               const Real dz, const Real c_ph, const bool inner,
                               const bool fixed_limiter) {
  if (sigma <= TINY_NUMBER || dz <= TINY_NUMBER) return 0.0;
  const Real grad = inner ? (ec - eg)/dz : (eg - ec)/dz;
  const Real r = std::abs(grad)/(sigma*std::max(ec, TINY_NUMBER));
  const Real lambda = fixed_limiter ? ONE_3RD
      : (2.0 + r)/(6.0 + 2.0*r + r*r);
  // Positive flux is in the +x3 direction at both boundaries.
  return -c_ph*lambda*grad/sigma;
}

static Real SimpleFluxGhost(MeshBlock *pmb, const Real ec, const Real sigma,
                            const Real dz, const bool inner) {
  const Real flux_unit = egas_unit*(leng_unit/time_unit);
  const Real target = std::max(rad_flux_cgs/flux_unit, 0.0);
  if (target <= 0.0) return ec;
  const bool fixed_limiter = pmb->prfld->fixed_flux_limiter;
  if (inner) {
    Real lo = ec;
    Real hi = std::max(2.0*ec, 1.0e-20);
    for (int n = 0; n < 80 && SimpleFluxForGhost(ec, hi, sigma, dz,
                                                   pmb->prfld->c_ph, true,
                                                   fixed_limiter) < target; ++n) {
      hi *= 2.0;
    }
    for (int n = 0; n < 60; ++n) {
      const Real mid = 0.5*(lo + hi);
      if (SimpleFluxForGhost(ec, mid, sigma, dz, pmb->prfld->c_ph,
                             true, fixed_limiter) < target) lo = mid;
      else hi = mid;
    }
    return std::max(0.5*(lo + hi), TINY_NUMBER);
  }
  Real lo = TINY_NUMBER;
  Real hi = std::max(ec, TINY_NUMBER);
  if (SimpleFluxForGhost(ec, lo, sigma, dz, pmb->prfld->c_ph, false,
                         fixed_limiter) < target) return lo;
  for (int n = 0; n < 60; ++n) {
    const Real mid = 0.5*(lo + hi);
    if (SimpleFluxForGhost(ec, mid, sigma, dz, pmb->prfld->c_ph, false,
                           fixed_limiter) > target) lo = mid;
    else hi = mid;
  }
  return std::max(0.5*(lo + hi), TINY_NUMBER);
}

static void SimpleRadiationBoundary(MeshBlock *pmb, AthenaArray<Real> &u_rad,
                                     AthenaArray<Real> &u_gas, int is, int ie,
                                     int js, int je, int ks, int ke, int ngh,
                                     bool inner) {
  (void)u_gas;
  for (int n = 1; n <= ngh; ++n) {
    const int kg = inner ? ks - n : ke + n;
    const int kc = inner ? ks : ke;
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real ec = std::max(u_rad(kc, j, i), TINY_NUMBER);
        const Real sigma = std::max(pmb->prfld->sigma_r(kc, j, i), TINY_NUMBER);
        const Real dz = pmb->pcoord->dx3f(kc);
        const Real eg1 = SimpleFluxGhost(pmb, ec, sigma, dz, inner);
        u_rad(kg, j, i) = std::max(ec + static_cast<Real>(n)*(eg1 - ec),
                                   TINY_NUMBER);
      }
    }
  }
}

// A user boundary callback can be dispatched for the outer face of every
// z-meshblock.  Only the block touching the global upper boundary owns the
// physical radiation boundary; interior z-block interfaces must remain
// ordinary inter-block interfaces.
static bool IsPhysicalUpperBoundary(const MeshBlock *pmb) {
  return pmb->block_size.x3max == pmb->pmy_mesh->mesh_size.x3max;
}

static void CopyUpperRadiationGhost(MeshBlock *pmb, AthenaArray<Real> &u_rad,
                                    int is, int ie, int js, int je, int ke,
                                    int ngh) {
  for (int n = 1; n <= ngh; ++n) {
    const int kg = ke + n;
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i)
        u_rad(kg, j, i) = u_rad(ke, j, i);
  }
}

// FLD-specific Marshak boundary.  The hydrodynamic Rempel upper condition is
// separate from this radiation condition.  The NR operator consumes the
// reconstructed ghost value when assembling its lagged face coefficient.
static Real MarshakBoundaryEnergy(MeshBlock *pmb, AthenaArray<Real> &u_rad,
                                  int ke, int j, int i) {
  const Real ei = std::max(u_rad(ke, j, i), TINY_NUMBER);
  const Real em = std::max(u_rad(std::max(ke - 1, pmb->ks), j, i), TINY_NUMBER);
  const Real dz = std::max(pmb->pcoord->dx3f(ke), TINY_NUMBER);
  const Real sigma_i = std::max(pmb->prfld->sigma_r(ke, j, i), TINY_NUMBER);
  const Real sigma_m = std::max(
      pmb->prfld->sigma_r(std::max(ke - 1, pmb->ks), j, i), TINY_NUMBER);
  const Real sigma_face = std::min(0.5*(sigma_i + sigma_m),
      std::max(2.0*sigma_i*sigma_m/(sigma_i + sigma_m),
               2.0*TWO_3RD/dz));
  const Real r = std::abs(ei - em)/(dz*std::max(sigma_face*ei, TINY_NUMBER));
  const Real lambda = RadFLD::FluxLimiter(r, pmb->prfld->fixed_flux_limiter);
  const Real D = pmb->prfld->c_ph*lambda/std::max(sigma_face, TINY_NUMBER);
  const Real dx_half = 0.5*dz;
  const Real denom = D + rad_top_alpha*pmb->prfld->c_ph*dx_half;
  Real eb = (D*ei + rad_top_alpha*pmb->prfld->c_ph*dx_half*rad_top_erad_ext)
            / std::max(denom, TINY_NUMBER);
  if (!std::isfinite(eb) || eb <= TINY_NUMBER) {
    ++marshak_floor_count;
    eb = TINY_NUMBER;
  }
  return eb;
}

static void SimpleRadiationMarshak(MeshBlock *pmb, AthenaArray<Real> &u_rad,
                                    int is, int ie, int js, int je, int ks,
                                    int ke, int ngh) {
  Real eb_min = std::numeric_limits<Real>::max();
  Real eb_max = 0.0;
  for (int n = 1; n <= ngh; ++n) {
    const int kg = ke + n;
    const int kc = std::max(ke - (n - 1), ks);
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real eb = MarshakBoundaryEnergy(pmb, u_rad, ke, j, i);
        const Real eg = 2.0*eb - u_rad(ke, j, i);
        if (!std::isfinite(eg) || eg <= TINY_NUMBER) {
          // The face value is authoritative in the implicit NR operator.
          // If the linear auxiliary extrapolation is non-positive, use the
          // positive face value rather than injecting a TINY_NUMBER spike
          // into individual top meshblocks.
          ++marshak_aux_fallback_count;
          u_rad(kg, j, i) = std::max(eb, TINY_NUMBER);
        } else {
          u_rad(kg, j, i) = (n == 1) ? eg : 2.0*u_rad(kg-1, j, i) - u_rad(kg-2, j, i);
          if (!std::isfinite(u_rad(kg, j, i)) || u_rad(kg, j, i) <= TINY_NUMBER) {
            ++marshak_aux_fallback_count;
            u_rad(kg, j, i) = std::max(eb, TINY_NUMBER);
          }
        }
        eb_min = std::min(eb_min, eb);
        eb_max = std::max(eb_max, eb);
      }
    }
  }
  if (Globals::my_rank == 0 && pmb->gid == 0 &&
      !marshak_diagnostic_printed) {
    marshak_diagnostic_printed = true;
    std::cout << "### Marshak upper radiation boundary alpha=" << rad_top_alpha
              << " Erad_ext=" << rad_top_erad_ext
              << " Erad_b_min=" << eb_min << " Erad_b_max=" << eb_max
              << " radiation_floors=" << marshak_floor_count
              << " auxiliary_fallbacks=" << marshak_aux_fallback_count << "\n";
  }
}

static void SimpleNRInner(MeshBlock *pmb, AthenaArray<Real> &u_rad,
                          AthenaArray<Real> &u_gas, Coordinates *pco,
                          const AthenaArray<Real> &w, Real time, Real dt,
                          int is, int ie, int js, int je, int ks, int ke,
                          int ngh) {
  (void)pco; (void)w; (void)time; (void)dt;
  SimpleRadiationBoundary(pmb, u_rad, u_gas, is, ie, js, je, ks, ke, ngh,
                          true);
}

static void SimpleNROuter(MeshBlock *pmb, AthenaArray<Real> &u_rad,
                          AthenaArray<Real> &u_gas, Coordinates *pco,
                          const AthenaArray<Real> &w, Real time, Real dt,
                          int is, int ie, int js, int je, int ks, int ke,
                          int ngh) {
  (void)pco; (void)w; (void)time; (void)dt;
  if (!IsPhysicalUpperBoundary(pmb)) {
    CopyUpperRadiationGhost(pmb, u_rad, is, ie, js, je, ke, ngh);
  } else if (top_bc_mode == "rempel_outflow") {
    SimpleRadiationMarshak(pmb, u_rad, is, ie, js, je, ks, ke, ngh);
  } else {
    SimpleRadiationBoundary(pmb, u_rad, u_gas, is, ie, js, je, ks, ke, ngh,
                            false);
  }
}

static void SimpleFLDInner(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                           const AthenaArray<Real> &w,
                           AthenaArray<Real> &u_rad, Real time, Real dt,
                           int is, int ie, int js, int je, int ks, int ke,
                           int ngh) {
  (void)pfld; (void)w; (void)time; (void)dt;
  SimpleRadiationBoundary(pmb, u_rad, pmb->prfld->u_gas, is, ie, js, je,
                          ks, ke, ngh, true);
}

static void SimpleFLDOuter(MeshBlock *pmb, Coordinates *pco, FLD *pfld,
                           const AthenaArray<Real> &w,
                           AthenaArray<Real> &u_rad, Real time, Real dt,
                           int is, int ie, int js, int je, int ks, int ke,
                           int ngh) {
  (void)pfld; (void)w; (void)time; (void)dt;
  if (!IsPhysicalUpperBoundary(pmb)) {
    CopyUpperRadiationGhost(pmb, u_rad, is, ie, js, je, ke, ngh);
  } else if (top_bc_mode == "rempel_outflow") {
    SimpleRadiationMarshak(pmb, u_rad, is, ie, js, je, ks, ke, ngh);
  } else {
    SimpleRadiationBoundary(pmb, u_rad, pmb->prfld->u_gas, is, ie, js, je,
                            ks, ke, ngh, false);
  }
}


void TableOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld,
                  AthenaArray<Real> &prim) {
  FLD *prfld = pmb->prfld;
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
        const Real rho_code = std::max(prim(IDN,k,j,i), TINY_NUMBER);
        const Real temp_code = std::max(prim(IPR,k,j,i)/rho_code,
                                         TINY_NUMBER);
        const Real rho_cgs = rho_code*rho_unit;
        const Real temp_cgs = temp_code*T_unit;
        const Real kap_p = puser_table->GetOpacity(RadFLD::SIGMA_P,
                                                    rho_cgs, temp_cgs);
        const Real kap_r = puser_table->GetOpacity(RadFLD::SIGMA_R,
                                                    rho_cgs, temp_cgs);
        prfld->sigma_p(k,j,i) = kap_p*rho_cgs*leng_unit;
        prfld->sigma_r(k,j,i) = kap_r*rho_cgs*leng_unit;
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  /*
  is_couple       = true
  only_rad        = false
  cut_diff        = false
  cut_Pnablav     = true
  */
  // // check input
  // if (!pin->GetBoolean("fld", "is_couple")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "is_couple must be true for this problem.";
  //   ATHENA_ERROR(msg);
  // }

  // if (pin->GetBoolean("fld", "only_rad")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "only_rad must be false for this problem.";
  //   ATHENA_ERROR(msg);
  // }

  // if (pin->GetBoolean("fld", "cut_diff")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "cut_diff must be false for this problem.";
  //   ATHENA_ERROR(msg);
  // }

  // if (!pin->GetBoolean("fld", "cut_Pnablav")) {
  //   std::stringstream msg;
  //   msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
  //   msg << "cut_Pnablav must be true for this problem.";
  //   ATHENA_ERROR(msg);
  // }

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
  Real vel_unit = std::sqrt(pres_unit/rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit/vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit*time_unit;

  // Rgas in cgs
  Rgas = 8.31451e+7; // erg/(mol*K)
  mu = pin->GetReal("hydro", "mu");
  T_unit = pres_unit/rho_unit*mu/Rgas;
  a_r_dim = 7.5657e-15; // radiation constant in erg cm^-3 K^-4
  a_r_sim = a_r_dim/(egas_unit/std::pow(T_unit, 4));

  poly_n = pin->GetOrAddReal("problem", "poly_n", 1.0);
  gamma_gas = pin->GetReal("hydro", "gamma");
  rho_ref = pin->GetReal("problem", "rho_top")/rho_unit;
  T_ref = pin->GetReal("problem", "T_top")/T_unit;
  grav_acc = pin->GetReal("problem", "grav_acc")/(leng_unit/(time_unit*time_unit));
  z_ref = pin->GetOrAddReal("problem", "z_ref", 0.0);
  top_outflow_only = pin->GetOrAddBoolean("problem", "top_outflow_only", true);
  top_impenetrable = pin->GetOrAddBoolean("problem", "top_impenetrable", false);
  top_bc_mode = pin->GetOrAddString("problem", "top_bc_mode", "rempel_outflow");
  bottom_inflow_speed = pin->GetOrAddReal("problem", "bottom_inflow_speed", 0.0);
  bottom_bc_mode = pin->GetOrAddString("problem", "bottom_bc_mode", "rempel_hd2");
  profile_output = pin->GetOrAddString("problem", "profile_output",
                                       "simple_convection_initial_profiles.txt");
  profile_zmax = pin->GetOrAddReal("problem", "profile_zmax", 0.0);
  profile_tau_target_z = pin->GetOrAddReal("problem", "profile_tau_target_z", -1.0);
  profile_tau_target = pin->GetOrAddReal("problem", "profile_tau_target", 1.0);
  profile_pin = pin;
  bottom_cdmp = pin->GetOrAddReal("problem", "Cdmp", 0.95);
  const bool s_in_from_profile = pin->GetOrAddBoolean(
      "problem", "s_in_from_profile", true);
  bottom_s_in = pin->GetOrAddReal("problem", "s_in", 0.0);
  bottom_pbnd = pin->GetOrAddReal("problem", "PBND", 0.0);
  hd2_mass_balance_on = pin->GetOrAddBoolean("problem", "mass_balance_on", true);
  hd2_mass_balance_relax_time = pin->GetOrAddReal(
      "problem", "mass_balance_relax_time", 10.0);
  hd2_mass_balance_gain = pin->GetOrAddReal(
      "problem", "mass_balance_gain", 10.0);
  hd2_mass_balance_rate_gain = pin->GetOrAddReal(
      "problem", "mass_balance_rate_gain", 10.0);
  hd2_mass_balance_min_scale = pin->GetOrAddReal(
      "problem", "mass_balance_min_scale", 0.7);
  hd2_mass_balance_max_scale = pin->GetOrAddReal(
      "problem", "mass_balance_max_scale", 1.5);
  if (!(hd2_mass_balance_min_scale > 0.0) ||
      !(hd2_mass_balance_max_scale > hd2_mass_balance_min_scale)) {
    std::stringstream msg;
    msg << "### FATAL ERROR: invalid mass-balance PBND scale limits.";
    ATHENA_ERROR(msg);
  }
  Real rho_b, p_b, erad_b;
  SimplePolytrope(pin->GetReal("mesh", "x3min"), rho_b, p_b, erad_b);
  if (!std::isfinite(bottom_pbnd) || bottom_pbnd <= 0.0) bottom_pbnd = p_b;
  if (s_in_from_profile)
    bottom_s_in = SimpleEntropyProxy(rho_b, bottom_pbnd);
  if (!std::isfinite(bottom_s_in)) {
    std::stringstream msg;
    msg << "### FATAL ERROR: problem/s_in must be finite.";
    ATHENA_ERROR(msg);
  }
  hd2_pbar1 = bottom_pbnd;
  hd2_pbar2 = bottom_pbnd;
  rad_flux_cgs = pin->GetOrAddReal("problem", "rad_flux_cgs", 6.3e10);
  rad_top_alpha = pin->GetOrAddReal("problem", "rad_top_alpha", 0.5);
  rad_top_erad_ext = pin->GetOrAddReal("problem", "rad_top_erad_ext", 0.0);
  if (!(rad_top_alpha > 0.0) || !std::isfinite(rad_top_alpha)) {
    std::stringstream msg;
    msg << "### FATAL ERROR: rad_top_alpha must be positive and finite.";
    ATHENA_ERROR(msg);
  }
  noise_lx = pin->GetReal("mesh", "x1max") - pin->GetReal("mesh", "x1min");
  noise_ly = pin->GetReal("mesh", "x2max") - pin->GetReal("mesh", "x2min");

  if (!pin->GetOrAddBoolean("fld", "use_opacity_table", true)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "simple_convection_lhllc_fld requires use_opacity_table=true.";
    ATHENA_ERROR(msg);
  }

  if (pin->GetString("mesh", "ix3_bc") == "user") {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, SimpleHydroInner);
    EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x3, SimpleFLDInner);
    EnrollUserNRBoundaryFunction(BoundaryFace::inner_x3, SimpleNRInner);
  }
  if (pin->GetString("mesh", "ox3_bc") == "user") {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, SimpleHydroOuter);
    EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x3, SimpleFLDOuter);
    EnrollUserNRBoundaryFunction(BoundaryFace::outer_x3, SimpleNROuter);
  }

  AllocateUserHistoryOutput(26);
  EnrollUserHistoryOutput(0, HistoryTg, "T_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryTr, "T_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEg, "e_gas", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryEr, "E_rad", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryaTg4, "aTgas^4", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, HistoryRtime, "Rtime", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all-E", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(7, HistoryVzRms, "VzRms", UserHistoryOperation::max);
  EnrollUserHistoryOutput(8, HistoryMaxSpeed, "MaxSpeed", UserHistoryOperation::max);
  EnrollUserHistoryOutput(9, HistoryTopFlux, "FradTop", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(10, HistoryTopLuminosity, "LradTop", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(11, HistoryVzRms2Mean, "VzRms2Mean", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(12, HistoryBottomInflow, "BottomMdotNet", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(13, HistoryTopOutflow, "TopMdotOut", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(14, HistoryMassDrift, "MassDrift", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(15, HistoryPBNDScale, "PBNDScale", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(16, HistoryMassRate, "MassRate", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(17, HistoryBottomUpflow, "BottomMdotUp", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(18, HistoryBottomDownflow, "BottomMdotDown", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(19, HistoryBottomEnergyFlux, "BottomEflux", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(20, HistoryBottomEntropyExcess, "BottomSminusSin", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(21, HistoryConservedEnergy, "EtotGrav", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(22, HistoryBottomHydroFluxCgs, "FhydBottom", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(23, HistoryBottomRadiationFluxCgs, "FradBottom", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(24, HistoryTopHydroFluxCgs, "FhydTop", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(25, HistoryTopTotalFluxCgs, "FtotTop", UserHistoryOperation::sum);
  // EnrollUserHistoryOutput(7, HistoryL1norm, "L1norm", UserHistoryOperation::sum);

  EnrollUserExplicitSourceFunction(AddRadiativeForceAndWork);
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(5);
  SetUserOutputVariableName(0, "e_gas");
  SetUserOutputVariableName(1, "E_rad");
  SetUserOutputVariableName(2, "T_gas");
  SetUserOutputVariableName(3, "T_rad");
  SetUserOutputVariableName(4, "radiative_cooling");

  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(2, ncells3, ncells2, ncells1);
  if (puser_table == nullptr) puser_table = new UserOpacityTable(pin);
  prfld->EnrollOpacityFunction(TableOpacity);
  prfld->hydro_top_outflow_diode = (top_bc_mode == "rempel_outflow");
  prfld->marshak_top_boundary = (top_bc_mode == "rempel_outflow");
  prfld->marshak_top_alpha = rad_top_alpha;
  prfld->marshak_top_erad_ext = rad_top_erad_ext;
  return;
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief FLD test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real temp_noise_amp = pin->GetOrAddReal("problem", "temp_noise_amp", 1.0e-3);
  const bool smooth_temp_noise = pin->GetOrAddBoolean("problem", "smooth_temp_noise", true);
  const Real igm1 = 1.0/(pin->GetReal("hydro", "gamma") - 1.0);
  if (gid == 0) {
    std::cout << "### NR-FLD simple convection (LHLLC-FLD, opacity table)\n"
              << "poly_n=" << poly_n << " rho_top=" << rho_ref
              << " T_top=" << T_ref << " grav=" << grav_acc
              << " opacity=external_table"
              << " temp_noise_amp=" << temp_noise_amp
              << " smooth_temp_noise=" << smooth_temp_noise
              << " top_outflow_only=" << top_outflow_only
              << " top_impenetrable=" << top_impenetrable
              << " top_bc_mode=" << top_bc_mode
              << " bottom_inflow_speed=" << bottom_inflow_speed
              << " bottom_bc_mode=" << bottom_bc_mode
              << " PBND=" << bottom_pbnd
              << " s_in=" << bottom_s_in
              << " Cdmp=" << bottom_cdmp
              << " mass_balance_on=" << hd2_mass_balance_on
              << " mass_balance_relax_time=" << hd2_mass_balance_relax_time
              << " mass_balance_gain=" << hd2_mass_balance_gain
              << " mass_balance_rate_gain=" << hd2_mass_balance_rate_gain
              << " rad_flux_cgs=" << rad_flux_cgs << "\n";
  }

  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;

  for(int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real rho, p, erad;
        SimplePolytrope(pcoord->x3v(k), rho, p, erad);
        const Real x = pcoord->x1v(i);
        const Real y = pcoord->x2v(j);
        const Real zmin = pmy_mesh->mesh_size.x3min;
        const Real zmax = pmy_mesh->mesh_size.x3max;
        const Real envelope = std::sin(M_PI*(pcoord->x3v(k)-zmin)/(zmax-zmin));
        const Real noise = smooth_temp_noise ? SimpleSmoothNoise(x, y)
                                             : SimpleCellNoise(x, y, pcoord->x3v(k));
        const Real delta_T = temp_noise_amp*noise
            *std::max(0.0, envelope);
        const Real T = (p/rho)*(1.0 + delta_T);
        p = rho*T;
        const Real egas = p*igm1;

        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = egas;
        prfld->u_gas(k,j,i) = egas;
        prfld->u_rad(k,j,i) = a_r_sim*std::pow(T, 4);
      }
    }
  }

  // record initial profile
  if (gid == 0) {
    std::cout << "simple initial sample rho=" << phydro->u(IDN,ks,js,is)
              << " egas=" << phydro->u(IEN,ks,js,is)
              << " u_gas=" << prfld->u_gas(ks,js,is)
              << " u_rad=" << prfld->u_rad(ks,js,is) << "\n";
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        ruser_meshblock_data[0](0,k,j,i) = phydro->u(IDN,k,j,i);
        ruser_meshblock_data[0](1,k,j,i) = phydro->u(IEN,k,j,i);
      }
    }
  }

  return;
}

void MeshBlock::UserWorkInLoop() {
  return;
}

void Mesh::UserWorkInLoop() {
  UpdateHD2PressureMeans(this);
  UpdateHD2MassBalance(this);
  if (!profile_written) {
    WriteInitialProfiles(profile_pin);
    profile_written = true;
  }
  if (Globals::my_rank == 0 && bottom_bc_mode == "rempel_hd2" &&
      !hd2_diagnostic_printed) {
    std::cout << "### Rempel HD2 lower boundary diagnostics"
              << " PBND=" << bottom_pbnd
              << " PBNDScale=" << hd2_pressure_scale
              << " Pbar1=" << hd2_pbar1
              << " Pbar2=" << hd2_pbar2
              << " s_in=" << bottom_s_in
              << " Cdmp=" << bottom_cdmp
              << " pressure_floors=" << hd2_pressure_floors
              << " density_floors=" << hd2_density_floors
              << " energy_floors=" << hd2_energy_floors << "\n";
    hd2_diagnostic_printed = true;
  }
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  Real gm1 = gamma_gas - 1.0;
  Real temp_coef = gm1*mu/Rgas*egas_unit/rho_unit;
  int kl = ks-NGHOST;
  int ku = ke+NGHOST;
  int jl = js-NGHOST;
  int ju = je+NGHOST;
  int il = is-NGHOST;
  int iu = ie+NGHOST;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        // assume cal in E
        user_out_var(0,k,j,i) = prfld->u_gas(k,j,i)*egas_unit;
        user_out_var(1,k,j,i) = prfld->u_rad(k,j,i)*egas_unit;
        user_out_var(2,k,j,i) = prfld->u_gas(k,j,i)/phydro->w(IDN,k,j,i)*temp_coef;
        user_out_var(3,k,j,i) = std::pow(prfld->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
        const Real temp = prfld->u_gas(k,j,i)
                        / std::max(phydro->w(IDN,k,j,i), TINY_NUMBER)
                        * (gamma_gas - 1.0);
        const Real qrad = prfld->c_ph*prfld->sigma_p(k,j,i)
                        * (a_r_sim*std::pow(std::max(temp, TINY_NUMBER), 4)
                           - prfld->u_rad(k,j,i));
        user_out_var(4,k,j,i) = qrad*egas_unit/time_unit;
      }
    }
  }
  return;
}


void AddRadiativeForceAndWork(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar) {
  // Radiation pressure work and a uniform downward gravitational source.
    int il = pmb->is - NGHOST, iu = pmb->ie + NGHOST;
    int jl = pmb->js - NGHOST, ju = pmb->je + NGHOST;
    int kl = pmb->ks - NGHOST, ku = pmb->ke + NGHOST;
    // The minimal benchmark omits the optional radiation-pressure force and
    // work source.  FLD still transports and thermally couples radiation;
    // retaining only gravity makes the convection trigger unambiguous and
    // avoids adding a second explicit spatial-difference operator here.
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          const Real rho = prim(IDN,k,j,i);
          cons(IM3,k,j,i) -= rho*grav_acc*dt;
          cons(IEN,k,j,i) -= rho*grav_acc*prim(IVZ,k,j,i)*dt;
        }
      }
    }
}


namespace {

Real HistoryTg(MeshBlock *pmb, int iout) {
  const Real gm1  = gamma_gas - 1.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real T = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        T += pmb->prfld->u_gas(k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit;
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
        T += std::pow(pmb->prfld->u_rad(k,j,i)*egas_unit/a_r_dim, 0.25);
        num++;
      }
    }
  }
  T /= num;
  return T;
}

// caution! this is for a mean of gas energy density.
Real HistoryEg(MeshBlock *pmb, int iout) {
  const Real gm1  = gamma_gas - 1.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real e = 0;
  // AthenaArray<Real> vol;
  // vol.NewAthenaArray((ie-is)+2*NGHOST);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      // pmb->pcoord->CellVolume(k, j, is, ie, vol);
      for (int i=is; i<=ie; i++) {
        e += pmb->prfld->u_gas(k,j,i);//*vol(i);
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
        E += pmb->prfld->u_rad(k,j,i);//*vol(i);
        num++;
      }
    }
  }
  E /= num;
  return E*egas_unit;
}

Real HistoryaTg4(MeshBlock *pmb, int iout) {
  const Real gm1  = gamma_gas - 1.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  int num = 0;
  Real aT4 = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        aT4 += std::pow(pmb->prfld->u_gas(k,j,i)*gm1/pmb->phydro->w(IDN,k,j,i)*T_unit, 4);
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
        E += pmb->prfld->u_gas(k,j,i)*vol(i);
        E += pmb->prfld->u_rad(k,j,i)*vol(i);
      }
    }
  }
  return E*egas_unit;
}

Real HistoryVzRms(MeshBlock *pmb, int iout) {
  Real sum = 0.0;
  int num = 0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real vz = pmb->phydro->w(IVZ,k,j,i);
        sum += vz*vz;
        ++num;
      }
    }
  }
  return std::sqrt(sum/static_cast<Real>(num));
}

// Unlike the legacy VzRms entry (a max of block-local RMS values), this
// returns the globally comparable volume-weighted mean of vz^2.  The history
// reduction is a sum, so take sqrt(VzRms2Mean) when plotting the global RMS.
Real HistoryVzRms2Mean(MeshBlock *pmb, int iout) {
  Real sum = 0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real vz = pmb->phydro->w(IVZ,k,j,i);
        sum += vz*vz;
      }
    }
  }
  const auto &ms = pmb->pmy_mesh->mesh_size;
  const Real ncell = static_cast<Real>(ms.nx1*ms.nx2*ms.nx3);
  return sum/std::max(ncell, 1.0);
}

Real HistoryMaxSpeed(MeshBlock *pmb, int iout) {
  Real vmax = 0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real vx = pmb->phydro->w(IVX,k,j,i);
        const Real vy = pmb->phydro->w(IVY,k,j,i);
        const Real vz = pmb->phydro->w(IVZ,k,j,i);
        vmax = std::max(vmax, std::sqrt(vx*vx + vy*vy + vz*vz));
      }
    }
  }
  return vmax;
}

static Real MarshakTopFluxCode(MeshBlock *pmb, int j, int i) {
  const int ke = pmb->ke;
  // Use the same lagged face coefficient as the implicit operator.  A
  // separately reconstructed coefficient can differ strongly in optically
  // thin cells and makes the history flux inconsistent with the solve.
  const Real dface = std::max(pmb->prfld->marshak_dface(ke,j,i), TINY_NUMBER);
  const Real dx_half = 0.5*std::max(pmb->pcoord->dx3f(ke), TINY_NUMBER);
  const Real acoef = rad_top_alpha*pmb->prfld->c_ph*dx_half;
  const Real eb = (dface*std::max(pmb->prfld->u_rad(ke,j,i), TINY_NUMBER)
                   + acoef*rad_top_erad_ext)
                  /std::max(dface + acoef, TINY_NUMBER);
  return rad_top_alpha*pmb->prfld->c_ph*(eb - rad_top_erad_ext);
}

Real HistoryTopFlux(MeshBlock *pmb, int iout) {
  if (!IsPhysicalUpperBoundary(pmb)) return 0.0;
  Real local = 0.0;
  for (int j = pmb->js; j <= pmb->je; ++j) {
    for (int i = pmb->is; i <= pmb->ie; ++i) {
      const Real area = pmb->pcoord->dx1f(i)*pmb->pcoord->dx2f(j);
      local += MarshakTopFluxCode(pmb, j, i)*area;
    }
  }
  const Real area_total = (pmb->pmy_mesh->mesh_size.x1max -
                           pmb->pmy_mesh->mesh_size.x1min)*
                          (pmb->pmy_mesh->mesh_size.x2max -
                           pmb->pmy_mesh->mesh_size.x2min);
  return local/std::max(area_total, TINY_NUMBER)*egas_unit*leng_unit/time_unit;
}

Real HistoryTopLuminosity(MeshBlock *pmb, int iout) {
  if (!IsPhysicalUpperBoundary(pmb)) return 0.0;
  Real local = 0.0;
  for (int j = pmb->js; j <= pmb->je; ++j) {
    for (int i = pmb->is; i <= pmb->ie; ++i) {
      local += MarshakTopFluxCode(pmb, j, i)
             *pmb->pcoord->dx1f(i)*pmb->pcoord->dx2f(j);
    }
  }
  return local*egas_unit*leng_unit/time_unit;
}

Real HistoryBottomInflow(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real local = 0.0;
  const int k = pmb->ks;
  for (int j = pmb->js; j <= pmb->je; ++j) {
    for (int i = pmb->is; i <= pmb->ie; ++i) {
      local += pmb->phydro->flux[X3DIR](IDN,k,j,i)
              *pmb->pcoord->GetFace3Area(k,j,i);
    }
  }
  const Real area = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)
                  * (pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min);
  return local/std::max(area, TINY_NUMBER);
}

Real HistoryTopOutflow(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3max != pmb->pmy_mesh->mesh_size.x3max) return 0.0;
  Real local = 0.0;
  const int k = pmb->ke + 1;
  for (int j = pmb->js; j <= pmb->je; ++j) {
    for (int i = pmb->is; i <= pmb->ie; ++i) {
      local += pmb->phydro->flux[X3DIR](IDN,k,j,i)
              *pmb->pcoord->GetFace3Area(k,j,i);
    }
  }
  const Real area = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)
                  * (pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min);
  return local/std::max(area, TINY_NUMBER);
}

Real HistoryMassDrift(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->gid != 0 || !(hd2_initial_mass > 0.0)) return 0.0;
  return (hd2_current_mass - hd2_initial_mass)/hd2_initial_mass;
}

Real HistoryPBNDScale(MeshBlock *pmb, int iout) {
  (void)iout;
  return (pmb->gid == 0) ? hd2_pressure_scale : 0.0;
}

Real HistoryMassRate(MeshBlock *pmb, int iout) {
  (void)iout;
  return (pmb->gid == 0 && hd2_initial_mass > 0.0)
      ? hd2_mass_rate/hd2_initial_mass : 0.0;
}

Real HistoryBottomUpflow(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real flux = 0.0;
  const int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; ++j) {
    for (int i=pmb->is; i<=pmb->ie; ++i) {
      const Real mdot = pmb->phydro->flux[X3DIR](IDN,k,j,i);
      if (mdot > 0.0) flux += mdot*pmb->pcoord->GetFace3Area(k,j,i);
    }
  }
  const auto &ms = pmb->pmy_mesh->mesh_size;
  const Real area = (ms.x1max-ms.x1min)*(ms.x2max-ms.x2min);
  return flux/std::max(area, TINY_NUMBER);
}

Real HistoryBottomDownflow(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real flux = 0.0;
  const int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; ++j) {
    for (int i=pmb->is; i<=pmb->ie; ++i) {
      const Real mdot = pmb->phydro->flux[X3DIR](IDN,k,j,i);
      if (mdot < 0.0) flux += mdot*pmb->pcoord->GetFace3Area(k,j,i);
    }
  }
  const auto &ms = pmb->pmy_mesh->mesh_size;
  const Real area = (ms.x1max-ms.x1min)*(ms.x2max-ms.x2min);
  return flux/std::max(area, TINY_NUMBER);
}

Real HistoryBottomEnergyFlux(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real flux = 0.0;
  const int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; ++j) {
    for (int i=pmb->is; i<=pmb->ie; ++i) {
      flux += pmb->phydro->flux[X3DIR](IEN,k,j,i)
             *pmb->pcoord->GetFace3Area(k,j,i);
    }
  }
  const auto &ms = pmb->pmy_mesh->mesh_size;
  const Real area = (ms.x1max-ms.x1min)*(ms.x2max-ms.x2min);
  return flux/std::max(area, TINY_NUMBER);
}

Real HistoryBottomEntropyExcess(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real sum = 0.0;
  const int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; ++j) {
    for (int i=pmb->is; i<=pmb->ie; ++i) {
      sum += SimpleEntropyProxy(pmb->phydro->w(IDN,k,j,i),
                                pmb->phydro->w(IPR,k,j,i)) - bottom_s_in;
    }
  }
  const auto &ms = pmb->pmy_mesh->mesh_size;
  return sum/std::max(static_cast<Real>(ms.nx1*ms.nx2), 1.0);
}

Real HistoryConservedEnergy(MeshBlock *pmb, int iout) {
  (void)iout;
  Real total = 0.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real volume = pmb->pcoord->GetCellVolume(k,j,i);
        const Real rho = pmb->phydro->u(IDN,k,j,i);
        const Real potential = rho*grav_acc*pmb->pcoord->x3v(k);
        total += (pmb->phydro->u(IEN,k,j,i)
                  + pmb->prfld->u_rad(k,j,i) + potential)*volume;
      }
    }
  }
  return total*egas_unit*leng_unit*leng_unit*leng_unit;
}

static Real HorizontalArea(const MeshBlock *pmb) {
  const auto &ms = pmb->pmy_mesh->mesh_size;
  return std::max((ms.x1max-ms.x1min)*(ms.x2max-ms.x2min), TINY_NUMBER);
}

Real HistoryBottomHydroFluxCgs(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real flux = 0.0;
  const int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; ++j)
    for (int i=pmb->is; i<=pmb->ie; ++i)
      flux += pmb->phydro->flux[X3DIR](IEN,k,j,i)
             *pmb->pcoord->GetFace3Area(k,j,i);
  return flux/HorizontalArea(pmb)*egas_unit*leng_unit/time_unit;
}

Real HistoryBottomRadiationFluxCgs(MeshBlock *pmb, int iout) {
  (void)iout;
  if (pmb->block_size.x3min != pmb->pmy_mesh->mesh_size.x3min) return 0.0;
  Real area = 0.0;
  const int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; ++j)
    for (int i=pmb->is; i<=pmb->ie; ++i)
      area += pmb->pcoord->GetFace3Area(k,j,i);
  // SimpleFluxGhost imposes this physical flux.  Ghost arrays need not still
  // contain the boundary-applied state when history output is evaluated.
  return rad_flux_cgs*area/HorizontalArea(pmb);
}

Real HistoryTopHydroFluxCgs(MeshBlock *pmb, int iout) {
  (void)iout;
  if (!IsPhysicalUpperBoundary(pmb)) return 0.0;
  Real flux = 0.0;
  const int k = pmb->ke + 1;
  for (int j=pmb->js; j<=pmb->je; ++j)
    for (int i=pmb->is; i<=pmb->ie; ++i)
      flux += pmb->phydro->flux[X3DIR](IEN,k,j,i)
             *pmb->pcoord->GetFace3Area(k,j,i);
  return flux/HorizontalArea(pmb)*egas_unit*leng_unit/time_unit;
}

Real HistoryTopTotalFluxCgs(MeshBlock *pmb, int iout) {
  return HistoryTopHydroFluxCgs(pmb, iout) + HistoryTopFlux(pmb, iout);
}

// Real HistoryL1norm(MeshBlock *pmb, int iout) {
//   int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
//   Real L1norm = 0;
//   Real chi_t = chi * (pmb->pmy_mesh->time+init_time);
//   if (dim == 1) {
//     Real coef = Er0/(2*std::sqrt(M_PI*chi_t));
//     for (int k=ks; k<=ke; k++) {
//       for (int j=js; j<=je; j++) {
//         for (int i=is; i<=ie; i++) {
//           Real x = pmb->pcoord->x1v(i);
//           Real r_sq = SQR(x-0.5);
//           Real an = coef*std::exp(-r_sq/(4*chi_t));
//           L1norm += std::abs(pmb->prfld->u_rad(k,j,i)-an)/an;
//         }
//       }
//     }
//   }
//   int nbtotal = pmb->pmy_mesh->nbtotal;
//   int ncells = (ie-is+1)*(je-js+1)*(ke-ks+1);
//   L1norm /= ncells*nbtotal;
//   return L1norm;
// }

} // namespace
