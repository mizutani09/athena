//========================================================================================
// Read-only split-step diagnostics. Disabled by default; one CSV per MeshBlock.
// Units are code units except the explicitly named *_K columns. The Mach-5
// slab input uses cgs base units. Never refresh boundaries or coefficients here:
// doing so would change the very update sequence being diagnosed.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "fld.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_multigrid/NRFLD.hpp"

void FLD::DumpStageDiagnostics(const char *phase, int stage, Real subdt) {
  if (!stage_diagnostics_) return;
  MeshBlock *mb = pmy_block;
  Mesh *mesh = mb->pmy_mesh;
  const bool early = mesh->ncycle < stage_diagnostic_first_cycles_;
  if (!early && (mesh->time < stage_diagnostic_start_
      || mesh->time > stage_diagnostic_end_
      || mesh->ncycle % stage_diagnostic_interval_ != 0)) return;

  // Only one transverse block line, selected with half-open bounds. Within
  // each block sample its central cell and record the full transverse spread.
  if (!(mb->block_size.x2min <= 0.0 && mb->block_size.x2max > 0.0
      && mb->block_size.x3min <= 0.0 && mb->block_size.x3max > 0.0)) return;
  const int j = (mb->js + mb->je + 1)/2;
  const int k = (mb->ks + mb->ke + 1)/2;
  const std::string name(phase);
  const bool implicit = name == "after_implicit";
  const bool faces = name == "before_transport" || name == "after_transport"
      || name == "before_source" || name == "after_source";
  std::ostringstream path;
  path << "fld_stages.r" << Globals::my_rank << ".b" << mb->gid << ".csv";
  std::ofstream out(path.str(), std::ios::app);
  if (!out) {
    std::stringstream msg;
    msg << "Cannot open FLD stage diagnostic CSV: " << path.str();
    ATHENA_ERROR(msg);
  }
  if (out.tellp() == 0) {
    out << "cycle,time,dt,phase,stage,subdt,gid,i,x,dx,rho,vx,egas,Er,Tgas_K,Trad_K,"
        << "Tgas_primitive_K,sigma_p,sigma_r,c_ph,a_r,S_couple,"
        << "D_xm_implicit,D_xp_implicit,lambda_xm_implicit,lambda_xp_implicit,"
        << "S_diff_implicit,Er_old_implicit,implicit_rate_residual,"
        << "F_transport_xm,F_transport_xp,S_transport_discrete,lambda_hydro,"
        << "Er_transverse_min,Er_transverse_max,mixed_frame_terms,implicit_after_hydro\n";
  }
  out << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10);
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real gm1 = mb->peos->GetGamma() - 1.0;
  const auto &u = mb->phydro->u;
  const auto &w = mb->phydro->w;
  for (int i = mb->is; i <= mb->ie; ++i) {
    const Real rho = u(IDN,k,j,i);
    const Real mx = u(IM1,k,j,i), my = u(IM2,k,j,i), mz = u(IM3,k,j,i);
    const Real egas = u(IEN,k,j,i) - 0.5*(mx*mx+my*my+mz*mz)/rho;
    const Real temp = gm1*egas/rho;
    const Real er = u_rad(k,j,i);
    const Real sc = c_ph*sigma_p(k,j,i)*(a_r*std::pow(temp,4)-er);
    const Real dx = mb->pcoord->dx1f(i);
    Real dm=nan, dp=nan, lm=nan, lp=nan, sd=nan, old=nan, residual=nan;
#if NRMGFLD_ENABLED
    if (implicit) {
      using namespace NewtonRaphsonFLD;
      const auto &der = mb->pnr->derivetive_;
      const auto &nr = mb->pnr->u_;
      dm = der(dFr_dEr_xm,k,j,i);
      dp = der(dFr_dEr_xp,k,j,i);
      // Recover the ACTUAL limiter used in each face coefficient. The fixed
      // limiter slab has constant sigma; use the same face-opacity rule anyway.
      lm = dm*RadFLD::FaceOpacity(sigma_r(k,j,i),sigma_r(k,j,i-1),1.0/dx)/c_ph;
      lp = dp*RadFLD::FaceOpacity(sigma_r(k,j,i),sigma_r(k,j,i+1),1.0/dx)/c_ph;
      sd = (dm*(nr(k,j,i-1)-nr(k,j,i)) + dp*(nr(k,j,i+1)-nr(k,j,i))
          + der(dFr_dEr_ym,k,j,i)*(nr(k,j-1,i)-nr(k,j,i))
          + der(dFr_dEr_yp,k,j,i)*(nr(k,j+1,i)-nr(k,j,i))
          + der(dFr_dEr_zm,k,j,i)*(nr(k-1,j,i)-nr(k,j,i))
          + der(dFr_dEr_zp,k,j,i)*(nr(k+1,j,i)-nr(k,j,i)))/(dx*dx);
      old = mb->pnr->uold_(k,j,i);
      residual = (er-old)/mesh->dt - sc - sd;
    }
#endif
    Real emin=er, emax=er;
    Real transport=nan;
    if (faces) {
      transport = -(u_rad_flux[X1DIR](k,j,i+1)-u_rad_flux[X1DIR](k,j,i))/dx;
      if (mesh->f2)
        transport -= (u_rad_flux[X2DIR](k,j+1,i)-u_rad_flux[X2DIR](k,j,i))
                     /mb->pcoord->dx2f(j);
      if (mesh->f3)
        transport -= (u_rad_flux[X3DIR](k+1,j,i)-u_rad_flux[X3DIR](k,j,i))
                     /mb->pcoord->dx3f(k);
    }
    for (int kk=mb->ks; kk<=mb->ke; ++kk)
      for (int jj=mb->js; jj<=mb->je; ++jj) {
        emin=std::min(emin,u_rad(kk,jj,i));
        emax=std::max(emax,u_rad(kk,jj,i));
      }
    out << mesh->ncycle << ',' << mesh->time << ',' << mesh->dt << ',' << phase
        << ',' << stage << ',' << subdt << ',' << mb->gid << ',' << i
        << ',' << mb->pcoord->x1v(i) << ',' << dx << ',' << rho << ',' << mx/rho
        << ',' << egas << ',' << er << ',' << temp*stage_diagnostic_temperature_unit_
        << ',' << std::pow(er/a_r,0.25)*stage_diagnostic_temperature_unit_
        << ',' << w(IPR,k,j,i)/w(IDN,k,j,i)*stage_diagnostic_temperature_unit_
        << ',' << sigma_p(k,j,i) << ',' << sigma_r(k,j,i) << ',' << c_ph << ',' << a_r
        << ',' << sc << ',' << dm << ',' << dp << ',' << lm << ',' << lp << ',' << sd
        << ',' << old << ',' << residual
        << ',' << (faces ? u_rad_flux[X1DIR](k,j,i) : nan)
        << ',' << (faces ? u_rad_flux[X1DIR](k,j,i+1) : nan)
        << ',' << transport
        << ',' << (faces ? rad_state_cc_(RadFLD::LAMBDA,k,j,i) : nan)
        << ',' << emin << ',' << emax << ',' << include_mixed_frame_terms
        << ',' << 1 << '\n';  // Historical CSV schema: implicit solve is always last.
  }
  out.flush();
  if (!out) {
    std::stringstream msg;
    msg << "Failed writing FLD stage diagnostic CSV: " << path.str();
    ATHENA_ERROR(msg);
  }
}
