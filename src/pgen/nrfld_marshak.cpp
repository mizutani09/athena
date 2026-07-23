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
//! \file nrfld_marshak.cpp
//! \brief Problem generator for a non-equilibrium Marshak-wave setup for NR-FLD.

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
Real rho_unit, egas_unit, leng_unit, time_unit;
Real T_unit, a_r_dim, Rgas, mu;

Real rho0, kappa_phys, kappa_sim, beta_marshak, finc_phys;
Real egas_floor, erad_floor, egas_init, Tgas_init, fixed_dt_sim, c_light_sim;
Real hydro_p_floor, hydro_p_init;

Real FixedTimeStep(MeshBlock *pmb);
Real HistoryTau(MeshBlock *pmb, int iout);
Real HistoryErMax(MeshBlock *pmb, int iout);
Real HistoryEgMax(MeshBlock *pmb, int iout);
Real HistoryErMin(MeshBlock *pmb, int iout);
Real HistoryEgMin(MeshBlock *pmb, int iout);
Real HistoryVMax(MeshBlock *pmb, int iout);
Real HistoryEall(MeshBlock *pmb, int iout);

Real MarshakBoundaryGhostValue(const Real interior_erad,
                               const Coordinates *pco, int iref) {
  constexpr Real c_ph_dim = 2.99792458e10;
  Real dx_phys = pco->dx1v(iref) * leng_unit;
  // Su-Olson Marshak boundary condition at z=0:
  //   E - 2/(3 kappa) dE/dz = 4 F_inc/c.
  // Approximate E at the boundary by (E_0+E_g)/2 and dE/dz by
  // (E_0-E_g)/dx, then solve the resulting Robin condition for E_g.
  const Real a = 2.0/(3.0*kappa_phys*dx_phys);
  const Real b = 4.0*finc_phys/(c_ph_dim*egas_unit);
  return (b - (0.5-a)*interior_erad)/(0.5+a);
}

void SetInnerFluxBoundary(AthenaArray<Real> &u_rad, Coordinates *pco,
                          int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      const Real edge_val = std::max(u_rad(k, j, is), erad_floor);
      const Real first_ghost = std::max(
          MarshakBoundaryGhostValue(edge_val, pco, is), erad_floor);
      const Real ghost_delta = first_ghost - edge_val;
      for (int i = 1; i <= ngh; ++i) {
        u_rad(k, j, is - i) =
            std::max(edge_val + i*ghost_delta, erad_floor);
      }
    }
  }
}

void SetOuterColdBoundary(AthenaArray<Real> &u_rad,
                          int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        u_rad(k, j, ie + i) = erad_floor;
      }
    }
  }
}

void SetNRInnerX1(AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas, Coordinates *pco,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetInnerFluxBoundary(u_rad, pco, is, ie, js, je, ks, ke, ngh);
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        u_gas(k, j, is - i) = egas_floor;
      }
    }
  }
}

void SetNROuterX1(AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetOuterColdBoundary(u_rad, is, ie, js, je, ks, ke, ngh);
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        u_gas(k, j, ie + i) = egas_floor;
      }
    }
  }
}

void SetHydroStaticBoundary(AthenaArray<Real> &prim, int ngh, bool inner,
                            int is, int ie, int js, int je, int ks, int ke) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        int ii = inner ? is - i : ie + i;
        prim(IDN, k, j, ii) = rho0;
        prim(IVX, k, j, ii) = 0.0;
        prim(IVY, k, j, ii) = 0.0;
        prim(IVZ, k, j, ii) = 0.0;
        prim(IPR, k, j, ii) = hydro_p_init;
      }
    }
  }
}

void ConstantOpacity(MeshBlock *pmb, AthenaArray<Real> &u_fld, AthenaArray<Real> &prim) {
  FLD2 *prfld = pmb->prfld2;
  int kl = pmb->ks - NGHOST, ku = pmb->ke + NGHOST;
  int jl = pmb->js - NGHOST, ju = pmb->je + NGHOST;
  int il = pmb->is - NGHOST, iu = pmb->ie + NGHOST;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        prfld->sigma_p(k, j, i) = kappa_sim;
        prfld->sigma_r(k, j, i) = kappa_sim;
      }
    }
  }
}
}  // namespace

void FLDFixedInnerX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetInnerFluxBoundary(u_rad_fld, pco, is, ie, js, je, ks, ke, ngh);
}

void FLDFixedOuterX1(MeshBlock *pmb, Coordinates *pco, FLD2 *pfld,
                     const AthenaArray<Real> &w, AthenaArray<Real> &u_rad_fld,
                     Real time, Real dt,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetOuterColdBoundary(u_rad_fld, is, ie, js, je, ks, ke, ngh);
}

void NRInnerX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetNRInnerX1(u_rad, u_gas, pco, is, ie, js, je, ks, ke, ngh);
}

void NROuterX1(MeshBlock *pmb, AthenaArray<Real> &u_rad, AthenaArray<Real> &u_gas,
               Coordinates *pco, const AthenaArray<Real> &w, Real time, Real dt,
               int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetNROuterX1(u_rad, u_gas, is, ie, js, je, ks, ke, ngh);
}

void HydroInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetHydroStaticBoundary(prim, ngh, true, is, ie, js, je, ks, ke);
}

void HydroOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt,
                  int is, int ie, int js, int je, int ks, int ke, int ngh) {
  SetHydroStaticBoundary(prim, ngh, false, is, ie, js, je, ks, ke);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!pin->GetBoolean("fld", "is_couple")) {
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

  if (pin->GetBoolean("fld", "cut_diff")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "cut_diff must be false for this problem.";
    ATHENA_ERROR(msg);
  }

  if (pin->GetBoolean("fld", "include_radiation_force")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "include_radiation_force must be false for this problem.";
    ATHENA_ERROR(msg);
  }

  // In this codebase fixed_flux_limiter=true forces lambda=1/3 everywhere.
  if (!pin->GetBoolean("fld", "fixed_flux_limiter")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]" << std::endl;
    msg << "fixed_flux_limiter must be true for this problem.";
    ATHENA_ERROR(msg);
  }

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

  Rgas = 8.31451e+7;
  mu = pin->GetReal("hydro", "mu");
  Real pres_unit = egas_unit;
  T_unit = pres_unit / rho_unit * mu / Rgas;
  a_r_dim = 7.5657e-15;

  Real vel_unit = std::sqrt(pres_unit / rho_unit);
  if (time_unit < 0.0) time_unit = leng_unit / vel_unit;
  if (leng_unit < 0.0) leng_unit = vel_unit * time_unit;

  constexpr Real c_ph_dim = 2.99792458e10;
  c_light_sim = c_ph_dim / (leng_unit / time_unit);

  rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  kappa_phys = pin->GetOrAddReal("problem", "kappa", 1.0);
  beta_marshak = pin->GetOrAddReal("problem", "beta", 0.1);
  finc_phys = pin->GetOrAddReal("problem", "Finc", 1.0);
  egas_floor = pin->GetOrAddReal("problem", "egas_floor", 1.0e-50);
  erad_floor = pin->GetOrAddReal("problem", "erad_floor", 1.0e-50);
  Tgas_init = pin->GetOrAddReal("problem", "Tgas0", -1.0);
  if (Tgas_init > 0.0) {
    Real egas_init_phys = a_r_dim * std::pow(Tgas_init, 4) / beta_marshak;
    egas_init = std::max(egas_init_phys / egas_unit, egas_floor);
  } else {
    egas_init = egas_floor;
  }
  hydro_p_floor = std::max((pin->GetReal("hydro", "gamma") - 1.0) * egas_floor,
                           TINY_NUMBER);
  hydro_p_init = std::max((pin->GetReal("hydro", "gamma") - 1.0) * egas_init,
                          hydro_p_floor);

  kappa_sim = kappa_phys * leng_unit;
  // delta_tau = beta*c*kappa*dt = 3e-4, so 1000 cycles reach tau=0.3.
  fixed_dt_sim = 3.0e-4 / (beta_marshak * c_light_sim * kappa_sim);

  EnrollUserFLDBoundaryFunction(BoundaryFace::inner_x1, FLDFixedInnerX1);
  EnrollUserFLDBoundaryFunction(BoundaryFace::outer_x1, FLDFixedOuterX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::inner_x1, NRInnerX1);
  EnrollUserNRBoundaryFunction(BoundaryFace::outer_x1, NROuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, HydroInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, HydroOuterX1);
  EnrollUserTimeStepFunction(FixedTimeStep);

  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryTau, "tau", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryErMax, "Er_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, HistoryEgMax, "egas_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, HistoryErMin, "Er_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(4, HistoryEgMin, "egas_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(5, HistoryVMax, "vmax", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryEall, "all_E", UserHistoryOperation::sum);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(8);
  SetUserOutputVariableName(0, "z");
  SetUserOutputVariableName(1, "x_dimless");
  SetUserOutputVariableName(2, "E_rad");
  SetUserOutputVariableName(3, "e_gas");
  SetUserOutputVariableName(4, "u");
  SetUserOutputVariableName(5, "v");
  SetUserOutputVariableName(6, "T_gas");
  SetUserOutputVariableName(7, "T_rad");
  prfld2->EnrollOpacityFunction(ConstantOpacity);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  constexpr Real c_ph_dim = 2.99792458e10;
  int kl = ks - NGHOST, ku = ke + NGHOST;
  int jl = js - NGHOST, ju = je + NGHOST;
  int il = is - NGHOST, iu = ie + NGHOST;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        phydro->u(IDN, k, j, i) = rho0;
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        if (NON_BAROTROPIC_EOS) phydro->u(IEN, k, j, i) = egas_init;
        prfld2->u_gas(k, j, i) = egas_init;
        prfld2->u_rad(k, j, i) = erad_floor;
      }
    }
  }

  if (gid == 0) {
    Real dt_phys = fixed_dt_sim * time_unit;
    Real vel_unit = leng_unit / time_unit;
    Real dx1 = pcoord->dx1v(is);
    Real delta_tau = beta_marshak * c_ph_dim * kappa_phys * dt_phys;
    std::cout << "rho_unit = " << rho_unit << " g cm^-3" << std::endl;
    std::cout << "egas_unit = " << egas_unit << " erg cm^-3" << std::endl;
    std::cout << "time_unit = " << time_unit << " s" << std::endl;
    std::cout << "leng_unit = " << leng_unit << " cm" << std::endl;
    std::cout << "c_light_sim = " << c_light_sim << std::endl;
    std::cout << "kappa_phys = " << kappa_phys << " cm^-1" << std::endl;
    std::cout << "kappa_sim = " << kappa_sim << std::endl;
    std::cout << "beta = " << beta_marshak << std::endl;
    std::cout << "Finc = " << finc_phys << " erg cm^-2 s^-1" << std::endl;
    std::cout << "Tgas0 = " << Tgas_init << " K" << std::endl;
    std::cout << "egas_init = " << egas_init << " [code]" << std::endl;
    std::cout << "fixed_dt = " << fixed_dt_sim << " [code]" << std::endl;
    std::cout << "fixed_dt = " << dt_phys << " s" << std::endl;
    std::cout << "delta_tau = " << delta_tau << std::endl;
    if (block_size.nx2 > 1 || block_size.nx3 > 1) {
      std::cout << "Warning: Marshak test is intended for 1D; x2/x3 are passive."
                << std::endl;
    }

    std::ofstream ofs("problem_parameters.txt");
    ofs << ">>> Problem parameters <<<" << std::endl;
    ofs << "- Units" << std::endl;
    ofs << "rho_unit        = " << rho_unit << " g cm^-3" << std::endl;
    ofs << "egas_unit       = " << egas_unit << " erg cm^-3" << std::endl;
    ofs << "time_unit       = " << time_unit << " s" << std::endl;
    ofs << "leng_unit       = " << leng_unit << " cm" << std::endl;
    ofs << "vel_unit        = " << vel_unit << " cm s^-1" << std::endl;
    ofs << "T_unit          = " << T_unit << " K" << std::endl;
    ofs << "mu              = " << mu << std::endl;
    ofs << "Rgas            = " << Rgas << " erg K^-1 mol^-1" << std::endl;
    ofs << "a_r_dim         = " << a_r_dim << " erg cm^-3 K^-4" << std::endl;
    ofs << std::endl;

    ofs << "- Problem setup" << std::endl;
    ofs << "rho0            = " << rho0 << std::endl;
    ofs << "kappa_phys      = " << kappa_phys << " cm^-1" << std::endl;
    ofs << "kappa_sim       = " << kappa_sim << std::endl;
    ofs << "beta            = " << beta_marshak << std::endl;
    ofs << "Finc            = " << finc_phys << " erg cm^-2 s^-1" << std::endl;
    ofs << "Tgas0           = " << Tgas_init << " K" << std::endl;
    ofs << "egas_floor      = " << egas_floor << std::endl;
    ofs << "egas_init       = " << egas_init << std::endl;
    ofs << "erad_floor      = " << erad_floor << std::endl;
    ofs << "hydro_p_floor   = " << hydro_p_floor << std::endl;
    ofs << "hydro_p_init    = " << hydro_p_init << std::endl;
    ofs << std::endl;

    ofs << "- Derived parameters" << std::endl;
    ofs << "c_light_phys    = " << c_ph_dim << " cm s^-1" << std::endl;
    ofs << "c_light_sim     = " << c_light_sim << std::endl;
    ofs << "fixed_dt_sim    = " << fixed_dt_sim << std::endl;
    ofs << "fixed_dt_phys   = " << dt_phys << " s" << std::endl;
    ofs << "delta_tau       = " << delta_tau << std::endl;
    ofs << "dx1_sim         = " << dx1 << std::endl;
    ofs << "dx1_phys        = " << dx1 * leng_unit << " cm" << std::endl;
    ofs << "x1min_sim       = " << pmy_mesh->mesh_size.x1min << std::endl;
    ofs << "x1max_sim       = " << pmy_mesh->mesh_size.x1max << std::endl;
    ofs << "x1min_phys      = " << pmy_mesh->mesh_size.x1min * leng_unit << " cm" << std::endl;
    ofs << "x1max_phys      = " << pmy_mesh->mesh_size.x1max * leng_unit << " cm" << std::endl;
    ofs << "tau_factor      = " << beta_marshak * c_light_sim * kappa_sim << std::endl;
    ofs << std::endl;

    ofs << "- MeshBlock local extents (gid=0)" << std::endl;
    ofs << "is              = " << is << std::endl;
    ofs << "ie              = " << ie << std::endl;
    ofs << "js              = " << js << std::endl;
    ofs << "je              = " << je << std::endl;
    ofs << "ks              = " << ks << std::endl;
    ofs << "ke              = " << ke << std::endl;
    ofs.close();
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  constexpr Real c_ph_dim = 2.99792458e10;
  Real u_norm = 4.0 * std::max(finc_phys, TINY_NUMBER);
  int kl = ks - NGHOST, ku = ke + NGHOST;
  int jl = js - NGHOST, ju = je + NGHOST;
  int il = is - NGHOST, iu = ie + NGHOST;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real z_phys = pcoord->x1v(i) * leng_unit;
        Real erad_phys = prfld2->u_rad(k, j, i) * egas_unit;
        Real egas_phys = prfld2->u_gas(k, j, i) * egas_unit;
        user_out_var(0, k, j, i) = z_phys;
        user_out_var(1, k, j, i) = std::sqrt(3.0) * kappa_phys * z_phys;
        user_out_var(2, k, j, i) = erad_phys;
        user_out_var(3, k, j, i) = egas_phys;
        user_out_var(4, k, j, i) = c_ph_dim * erad_phys / u_norm;
        user_out_var(5, k, j, i) = c_ph_dim * beta_marshak * egas_phys / u_norm;
        user_out_var(6, k, j, i) = std::pow(beta_marshak * egas_phys / a_r_dim, 0.25);
        user_out_var(7, k, j, i) = std::pow(erad_phys / a_r_dim, 0.25);
      }
    }
  }
}

namespace {
Real FixedTimeStep(MeshBlock *pmb) {
  return fixed_dt_sim;
}

Real HistoryTau(MeshBlock *pmb, int iout) {
  return beta_marshak * c_light_sim * kappa_sim * pmb->pmy_mesh->time;
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

Real HistoryEgMax(MeshBlock *pmb, int iout) {
  Real out = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::max(out, pmb->prfld2->u_gas(k, j, i) * egas_unit);
      }
    }
  }
  return out;
}

Real HistoryErMin(MeshBlock *pmb, int iout) {
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::min(out, pmb->prfld2->u_rad(k, j, i) * egas_unit);
      }
    }
  }
  return out;
}

Real HistoryEgMin(MeshBlock *pmb, int iout) {
  Real out = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        out = std::min(out, pmb->prfld2->u_gas(k, j, i) * egas_unit);
      }
    }
  }
  return out;
}

Real HistoryVMax(MeshBlock *pmb, int iout) {
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

Real HistoryEall(MeshBlock *pmb, int iout) {
  AthenaArray<Real> vol;
  vol.NewAthenaArray((pmb->ie - pmb->is) + 2 * NGHOST);
  Real eall = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        eall += (pmb->prfld2->u_gas(k, j, i) + pmb->prfld2->u_rad(k, j, i)) * vol(i);
      }
    }
  }
  return eall * egas_unit;
}
}  // namespace
