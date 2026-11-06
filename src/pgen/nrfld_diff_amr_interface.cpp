//======================================================================================
//! \file nrfld_diff_amr_interface.cpp
//! \brief 1D-like variable-coefficient FLD/SMR multigrid diagnostic (Test C)
//!
//! This problem is intentionally separate from nrfld_diff.cpp.  It provides a
//! controlled matrix of tests for separating
//!   (1) a material-opacity interface,
//!   (2) a static mesh-refinement interface, and
//!   (3) their interaction.
//!
//! All physical fields vary only along x1.  x2 and x3 are retained so that the
//! same three-dimensional SMR and multigrid paths used by collapse calculations
//! are exercised.
//======================================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
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
Real density_low;
Real density_high;
Real slab_left;
Real slab_right;
Real interface_width;
Real erad_background;
Real erad_amplitude;
Real pulse_center;
Real pulse_width;
int profile_axis;
Real transverse_amplitude;
int test_regrid_cycle;
bool exact_cosine_mode;
int cosine_wavenumber;

int ControlledRefinement(MeshBlock *pmb) {
  // Deterministic topology change for restart/rank-invariance regression.
  // The pulse region is refined initially, then every fine block is removed.
  if (test_regrid_cycle < 0) return 0;
  if (pmb->pmy_mesh->ncycle >= test_regrid_cycle) return -1;
  const Real lower = profile_axis == 1 ? pmb->block_size.x1min
      : profile_axis == 2 ? pmb->block_size.x2min : pmb->block_size.x3min;
  const Real upper = profile_axis == 1 ? pmb->block_size.x1max
      : profile_axis == 2 ? pmb->block_size.x2max : pmb->block_size.x3max;
  return (upper > slab_left && lower < slab_right) ? 1 : -1;
}

// A periodic dense slab gives two interfaces without introducing a second
// discontinuity at the periodic domain boundary.  width=0 requests a sharp
// cell-centred step; width>0 uses a smooth pair of tanh transitions.
Real DensityProfile(const Real x) {
  Real window;
  if (interface_width > 0.0) {
    window = 0.5*(std::tanh((x - slab_left)/interface_width)
                  - std::tanh((x - slab_right)/interface_width));
  } else {
    window = (x >= slab_left && x < slab_right) ? 1.0 : 0.0;
  }
  return density_low + (density_high - density_low)*window;
}

Real RadiationProfile(const Real x) {
  const Real q = (x - pulse_center)/pulse_width;
  return erad_background + erad_amplitude*std::exp(-0.5*q*q);
}

Real HistoryErMax(MeshBlock *pmb, int iout) {
  Real result = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        result = std::max(result, pmb->prfld->u_rad(k,j,i));
      }
    }
  }
  return result;
}

Real HistoryErSum(MeshBlock *pmb, int iout) {
  Real result = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        result += pmb->prfld->u_rad(k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
      }
    }
  }
  return result;
}
}  // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  density_low = pin->GetOrAddReal("problem", "density_low", 1.0);
  density_high = pin->GetOrAddReal("problem", "density_high", 1.15);
  slab_left = pin->GetOrAddReal("problem", "slab_left", 0.25);
  slab_right = pin->GetOrAddReal("problem", "slab_right", 0.75);
  interface_width = pin->GetOrAddReal("problem", "interface_width", 0.0);
  erad_background = pin->GetOrAddReal("problem", "erad_background", 1.0);
  erad_amplitude = pin->GetOrAddReal("problem", "erad_amplitude", 0.1);
  pulse_center = pin->GetOrAddReal("problem", "pulse_center", slab_left);
  pulse_width = pin->GetOrAddReal("problem", "pulse_width", 0.08);
  profile_axis = pin->GetOrAddInteger("problem", "profile_axis", 1);
  transverse_amplitude = pin->GetOrAddReal("problem", "transverse_amplitude", 0.0);
  exact_cosine_mode = pin->GetOrAddBoolean("problem", "exact_cosine_mode", false);
  cosine_wavenumber = pin->GetOrAddInteger("problem", "cosine_wavenumber", 1);
  test_regrid_cycle = pin->GetOrAddInteger("problem", "test_regrid_cycle", -1);
  if (adaptive && test_regrid_cycle >= 0) {
    EnrollUserRefinementCondition(ControlledRefinement);
  }

  if (!(density_low > 0.0) || !(density_high > 0.0)
      || !(slab_left < slab_right) || interface_width < 0.0
      || !(erad_background > 0.0) || erad_amplitude < 0.0
      || !(pulse_width > 0.0) || profile_axis < 1 || profile_axis > 3
      || std::abs(transverse_amplitude) >= erad_background
      || cosine_wavenumber < 1
      || (exact_cosine_mode && (profile_axis != 1
          || density_low != density_high || transverse_amplitude != 0.0))) {
    std::stringstream msg;
    msg << "### FATAL ERROR in nrfld_diff_amr_interface\n"
        << "Require positive densities/background/pulse_width, non-negative "
        << "interface_width/amplitude, slab_left < slab_right, and "
        << "profile_axis in [1,3], and |transverse_amplitude| < background. "
        << "exact_cosine_mode additionally requires x1, uniform density, "
        << "zero transverse amplitude, and positive cosine_wavenumber.";
    ATHENA_ERROR(msg);
  }

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, HistoryErMax, "Er_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, HistoryErSum, "Er_volume", UserHistoryOperation::sum);

  if (Globals::my_rank == 0) {
    std::cout << "TEST_C_SETUP density_low=" << density_low
              << " density_high=" << density_high
              << " slab_left=" << slab_left
              << " slab_right=" << slab_right
              << " interface_width=" << interface_width
              << " profile_axis=" << profile_axis
              << " transverse_amplitude=" << transverse_amplitude
              << " pulse_center=" << pulse_center
              << " pulse_width=" << pulse_width << std::endl;
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(3);
  SetUserOutputVariableName(0, "Er");
  SetUserOutputVariableName(1, "rho_profile");
  SetUserOutputVariableName(2, "sigma_R");
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real gamma = peos->GetGamma();
  const Real pressure = pin->GetOrAddReal("problem", "pressure", 100.0);
  const Real gas_energy = pressure/(gamma - 1.0);

  const int il = is - NGHOST;
  const int iu = ie + NGHOST;
  const int jl = js - NGHOST;
  const int ju = je + NGHOST;
  const int kl = ks - NGHOST;
  const int ku = ke + NGHOST;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const Real position = profile_axis == 1 ? pcoord->x1v(i)
            : profile_axis == 2 ? pcoord->x2v(j) : pcoord->x3v(k);
        const Real rho = DensityProfile(position);
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) phydro->u(IEN,k,j,i) = gas_energy;
        prfld->u_gas(k,j,i) = gas_energy;
        const int transverse_axis = profile_axis == 3 ? 1 : profile_axis+1;
        const Real transverse_position = transverse_axis == 1 ? pcoord->x1v(i)
            : transverse_axis == 2 ? pcoord->x2v(j) : pcoord->x3v(k);
        const RegionSize &domain = pmy_mesh->mesh_size;
        const Real transverse_min = transverse_axis == 1 ? domain.x1min
            : transverse_axis == 2 ? domain.x2min : domain.x3min;
        const Real transverse_max = transverse_axis == 1 ? domain.x1max
            : transverse_axis == 2 ? domain.x2max : domain.x3max;
        const Real transverse_phase = 2.0*PI*(transverse_position-transverse_min)
            /(transverse_max-transverse_min);
        if (exact_cosine_mode) {
          // Cell average of a periodic diffusion eigenmode.  With uniform
          // opacity and lambda=1/3, the backward-Euler continuum solution
          // is known exactly for the same cell average at every later time.
          const Real length = domain.x1max-domain.x1min;
          const Real wave = 2.0*PI*cosine_wavenumber/length;
          const Real half_width = 0.5*pcoord->dx1f(i);
          const Real average = std::sin(wave*half_width)/(wave*half_width);
          prfld->u_rad(k,j,i) = erad_background
              + erad_amplitude*average
                  *std::cos(wave*(pcoord->x1v(i)-domain.x1min));
        } else {
          prfld->u_rad(k,j,i) = RadiationProfile(position)
              + transverse_amplitude*std::cos(transverse_phase);
        }
      }
    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0,k,j,i) = prfld->u_rad(k,j,i);
        user_out_var(1,k,j,i) = phydro->w(IDN,k,j,i);
        user_out_var(2,k,j,i) = prfld->sigma_r(k,j,i);
      }
    }
  }
}
