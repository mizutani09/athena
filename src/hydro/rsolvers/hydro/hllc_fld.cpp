//========================================================================================
//! \file hllc_fld.cpp
//! \brief HLLC solver for the gas plus NR-FLD hyperbolic subsystem.
//!
//! Radiation energy and the FLD Eddington tensor are reconstructed on the
//! same faces as the gas.  The Riemann problem uses the normal total pressure
//! p_gas + P_rad,nn and writes both the Hydro flux and the advective radiation
//! energy flux.  Diffusion and matter-radiation exchange remain in NRFLD.

#include <algorithm>
#include <cmath>

#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"
#include "../../../fld/fld.hpp"
#include "../../hydro.hpp"

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, AthenaArray<Real> &wl,
                          AthenaArray<Real> &wr, AthenaArray<Real> &flx,
                          const AthenaArray<Real> &dxw) {
  (void)dxw;
  const int dir = ivx - IVX;
  const int t1dir = (dir + 1) % 3;
  const int t2dir = (dir + 2) % 3;
  const int ivy = IVX + t1dir;
  const int ivz = IVX + t2dir;
  FLD2 *pfld = pmy_block->prfld2;
  // Some NR-FLD test problems intentionally remove radiation pressure from the
  // hyperbolic subsystem.  In that case this solver must reduce to the
  // ordinary gas HLLC solver, with E_rad advected as a passive quantity.
  // Momentum coupling is independent of the P:grad(v) energy-work switch.
  // cut_Pnablav must never remove the radiation force from the momentum flux.
  const bool couple_rad_pressure = pfld->is_couple && !pfld->only_rad;
  AthenaArray<Real> &radl = pfld->rad_face_l[dir];
  AthenaArray<Real> &radr = pfld->rad_face_r[dir];
  AthenaArray<Real> &radflux = pfld->u_rad_flux[dir];

  Real wli[NHYDRO], wri[NHYDRO];
  Real flxi[NHYDRO], fl[NHYDRO], fr[NHYDRO];
  Real gamma;
  if (GENERAL_EOS) gamma = std::nan("");
  else gamma = pmy_block->peos->GetGamma();
  const Real gm1 = gamma - 1.0;
  const Real igm1 = 1.0/gm1;

#pragma omp simd private(wli,wri,flxi,fl,fr)
  for (int i = il; i <= iu; ++i) {
    wli[IDN] = wl(IDN,i);
    wli[IVX] = wl(ivx,i);
    wli[IVY] = wl(ivy,i);
    wli[IVZ] = wl(ivz,i);
    wli[IPR] = wl(IPR,i);
    wri[IDN] = wr(IDN,i);
    wri[IVX] = wr(ivx,i);
    wri[IVY] = wr(ivy,i);
    wri[IVZ] = wr(ivz,i);
    wri[IPR] = wr(IPR,i);

    const Real erl = std::max(radl(RadFLD2::ERAD,k,j,i), TINY_NUMBER);
    const Real err = std::max(radr(RadFLD2::ERAD,k,j,i), TINY_NUMBER);
    const int pnn_idx = 1 + 3*dir + dir;
    const int pnt1_idx = 1 + 3*dir + t1dir;
    const int pnt2_idx = 1 + 3*dir + t2dir;
    const Real pnnl = couple_rad_pressure
        ? std::max(radl(pnn_idx,k,j,i), 0.0) : 0.0;
    const Real pnnr = couple_rad_pressure
        ? std::max(radr(pnn_idx,k,j,i), 0.0) : 0.0;
    const Real pnt1l = couple_rad_pressure ? radl(pnt1_idx,k,j,i) : 0.0;
    const Real pnt1r = couple_rad_pressure ? radr(pnt1_idx,k,j,i) : 0.0;
    const Real pnt2l = couple_rad_pressure ? radl(pnt2_idx,k,j,i) : 0.0;
    const Real pnt2r = couple_rad_pressure ? radr(pnt2_idx,k,j,i) : 0.0;
    const Real ptl = couple_rad_pressure
        ? std::max(radl(RadFLD2::PTOT1 + dir,k,j,i), TINY_NUMBER) : wli[IPR];
    const Real ptr = couple_rad_pressure
        ? std::max(radr(RadFLD2::PTOT1 + dir,k,j,i), TINY_NUMBER) : wri[IPR];

    Real elgas, ergas;
    if (GENERAL_EOS) {
      elgas = pmy_block->peos->EgasFromRhoP(wli[IDN], wli[IPR]);
      ergas = pmy_block->peos->EgasFromRhoP(wri[IDN], wri[IPR]);
    } else {
      elgas = wli[IPR]*igm1;
      ergas = wri[IPR]*igm1;
    }
    const Real kel = 0.5*wli[IDN]
        *(SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
    const Real ker = 0.5*wri[IDN]
        *(SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
    const Real etotl = elgas + kel + (couple_rad_pressure ? erl : 0.0);
    const Real etotr = ergas + ker + (couple_rad_pressure ? err : 0.0);

    // Freeze the FLD closure during this Riemann solve.  Linearizing
    // dP_nn=f_nn dE and dE/dt+(E+P_nn) div(v)=0 gives the radiation
    // acoustic contribution f_nn(E+P_nn)/rho.
    const Real fnnl = std::max(0.0, std::min(1.0, pnnl/erl));
    const Real fnnr = std::max(0.0, std::min(1.0, pnnr/err));
    const Real cgasl = pmy_block->peos->SoundSpeed(wli);
    const Real cgasr = pmy_block->peos->SoundSpeed(wri);
    const Real cl = std::sqrt(SQR(cgasl) + fnnl*(erl+pnnl)/wli[IDN]);
    const Real cr = std::sqrt(SQR(cgasr) + fnnr*(err+pnnr)/wri[IDN]);

    const Real rhoa = 0.5*(wli[IDN] + wri[IDN]);
    const Real ca = 0.5*(cl + cr);
    const Real pmid = 0.5*(ptl + ptr
        + (wli[IVX]-wri[IVX])*rhoa*ca);
    const Real umid = 0.5*(wli[IVX] + wri[IVX]
        + (ptl-ptr)/(rhoa*ca));
    const Real rhol = wli[IDN] + (wli[IVX]-umid)*rhoa/ca;
    const Real rhor = wri[IDN] + (umid-wri[IVX])*rhoa/ca;

    Real ql, qr;
    if (GENERAL_EOS) {
      // The gas EOS supplies the nonlinear gas part; the radiation acoustic
      // contribution is already included in cl/cr above.  Use the local gas
      // gamma only for the shock correction factor.
      const Real gl = pmy_block->peos->AsqFromRhoP(rhol, wli[IPR])
          * rhol/std::max(wli[IPR], TINY_NUMBER);
      const Real gr = pmy_block->peos->AsqFromRhoP(rhor, wri[IPR])
          * rhor/std::max(wri[IPR], TINY_NUMBER);
      ql = (pmid <= ptl) ? 1.0
          : std::sqrt(1.0 + (gl+1.0)/(2.0*gl)*(pmid/ptl-1.0));
      qr = (pmid <= ptr) ? 1.0
          : std::sqrt(1.0 + (gr+1.0)/(2.0*gr)*(pmid/ptr-1.0));
    } else {
      ql = (pmid <= ptl) ? 1.0
          : std::sqrt(1.0 + (gamma+1.0)/(2.0*gamma)*(pmid/ptl-1.0));
      qr = (pmid <= ptr) ? 1.0
          : std::sqrt(1.0 + (gamma+1.0)/(2.0*gamma)*(pmid/ptr-1.0));
    }

    const Real al = wli[IVX] - cl*ql;
    const Real ar = wri[IVX] + cr*qr;
    const Real bp = ar > 0.0 ? ar : TINY_NUMBER;
    const Real bm = al < 0.0 ? al : -TINY_NUMBER;
    const Real vxl = wli[IVX] - al;
    const Real vxr = wri[IVX] - ar;
    const Real tl = ptl + vxl*wli[IDN]*wli[IVX];
    const Real tr = ptr + vxr*wri[IDN]*wri[IVX];
    const Real ml = wli[IDN]*vxl;
    const Real mr = -wri[IDN]*vxr;
    const Real am = (tl-tr)/(ml+mr);
    Real cp = (ml*tr + mr*tl)/(ml+mr);
    cp = std::max(cp, 0.0);

    const Real vl[3] = {wli[IVX], wli[IVY], wli[IVZ]};
    const Real vr[3] = {wri[IVX], wri[IVY], wri[IVZ]};
    const Real pradwork_l = pnnl*vl[0] + pnt1l*vl[1] + pnt2l*vl[2];
    const Real pradwork_r = pnnr*vr[0] + pnt1r*vr[1] + pnt2r*vr[2];
    const Real vlbm = wli[IVX] - bm;
    const Real vrbp = wri[IVX] - bp;
    fl[IDN] = wli[IDN]*vlbm;
    fr[IDN] = wri[IDN]*vrbp;
    fl[IVX] = wli[IDN]*wli[IVX]*vlbm + ptl;
    fr[IVX] = wri[IDN]*wri[IVX]*vrbp + ptr;
    fl[IVY] = wli[IDN]*wli[IVY]*vlbm + pnt1l;
    fr[IVY] = wri[IDN]*wri[IVY]*vrbp + pnt1r;
    fl[IVZ] = wli[IDN]*wli[IVZ]*vlbm + pnt2l;
    fr[IVZ] = wri[IDN]*wri[IVZ]*vrbp + pnt2r;
    fl[IEN] = etotl*vlbm + wli[IPR]*wli[IVX] + pradwork_l;
    fr[IEN] = etotr*vrbp + wri[IPR]*wri[IVX] + pradwork_r;

    Real sl, sr, sm;
    if (am >= 0.0) {
      sl = am/(am-bm); sr = 0.0; sm = -bm/(am-bm);
    } else {
      sl = 0.0; sr = -am/(bp-am); sm = bp/(bp-am);
    }
    flxi[IDN] = sl*fl[IDN] + sr*fr[IDN];
    flxi[IVX] = sl*fl[IVX] + sr*fr[IVX] + sm*cp;
    flxi[IVY] = sl*fl[IVY] + sr*fr[IVY];
    flxi[IVZ] = sl*fl[IVZ] + sr*fr[IVZ];
    flxi[IEN] = sl*fl[IEN] + sr*fr[IEN] + sm*cp*am;
    const Real vf = sl*wli[IVX] + sr*wri[IVX] + sm*am;

    // Split the total-energy flux into gas and radiation parts.  E_rad obeys
    // the advective part dE/dt + div(E v)=0 here; P_rad:grad(v) is applied with
    // opposite signs to gas and radiation in the NR solve.  Applying the same
    // HLLC weights as the density flux gives the Rankine-Hugoniot-consistent
    // radiation star state across either outer wave.  The remainder retains
    // P_rad.v in the gas-energy face flux, which combines with +P:grad(v) in
    // the gas update to give the physical -v.grad(P_rad) work.
    Real fer;
    if (!couple_rad_pressure) {
      fer = sl*erl*vlbm + sr*err*vrbp;
    } else if (pfld->implicit_pnablav) {
      fer = sl*erl*vlbm + sr*err*vrbp;
    } else if (am >= 0.0) {
      // The well-balanced static/advection mode places radiation enthalpy in
      // the radiation flux and omits the separate P:grad(v) NR work.
      fer = (erl+pnnl)*am + pnt1l*wli[IVY] + pnt2l*wli[IVZ];
    } else {
      fer = (err+pnnr)*am + pnt1r*wri[IVY] + pnt2r*wri[IVZ];
    }
    radflux(k,j,i) = fer;
    if (couple_rad_pressure) flxi[IEN] -= fer;

    flx(IDN,k,j,i) = flxi[IDN];
    flx(ivx,k,j,i) = flxi[IVX];
    flx(ivy,k,j,i) = flxi[IVY];
    flx(ivz,k,j,i) = flxi[IVZ];
    flx(IEN,k,j,i) = flxi[IEN];
    pmy_block->phydro->vf[dir](k,j,i) = couple_rad_pressure ? am : vf;
  }
}
