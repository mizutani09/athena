//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lhllc_fld.cpp
//! \brief Low-dissipation HLLC Riemann solver with scalar FLD coupling.

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"
#include "../../../fld/fld.hpp"
#include "../../hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief The Low-dissipation HLLC (LHLLC) solver with radiation pressure.

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, AthenaArray<Real> &wl,
                          AthenaArray<Real> &wr, AthenaArray<Real> &flx,
                          const AthenaArray<Real> &dxw) {
  (void)dxw;
  const int dir = ivx-IVX;
  const int ivy = IVX + (dir+1)%3;
  const int ivz = IVX + (dir+2)%3;
  FLD *pfld = pmy_block->prfld;
  const bool coupled = pfld->is_couple && !pfld->only_rad
                       && pfld->include_radiation_force;
  AthenaArray<Real> &radl = pfld->rad_face_l[dir];
  AthenaArray<Real> &radr = pfld->rad_face_r[dir];
  AthenaArray<Real> &radflux = pfld->u_rad_flux[dir];
  Real wli[(NHYDRO)],wri[(NHYDRO)];
  Real flxi[(NHYDRO)],fl[(NHYDRO)],fr[(NHYDRO)];
  Real gamma;
  if (GENERAL_EOS) {
    gamma = std::nan("");
  } else {
    gamma = pmy_block->peos->GetGamma();
  }
  Real igm1 = GENERAL_EOS ? 0.0 : 1.0/(gamma - 1.0);

  CalculateVelocityDifferences(k, j, il, iu, ivx, dvn, dvt);

#pragma distribute_point
#pragma omp simd private(wli,wri,flxi,fl,fr)
  for (int i=il; i<=iu; ++i) {
    //--- Step 1.  Load L/R states into local variables
    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    wli[IPR]=wl(IPR,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    wri[IPR]=wr(IPR,i);

    //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

    const Real erl = std::max(radl(RadFLD::ERAD,k,j,i), TINY_NUMBER);
    const Real err = std::max(radr(RadFLD::ERAD,k,j,i), TINY_NUMBER);
    const Real laml = std::max(radl(RadFLD::LAMBDA,k,j,i), 0.0);
    const Real lamr = std::max(radr(RadFLD::LAMBDA,k,j,i), 0.0);
    const Real arl = std::max(radl(RadFLD::ARAD,k,j,i), 0.0);
    const Real arr = std::max(radr(RadFLD::ARAD,k,j,i), 0.0);
    const Real prl = coupled ? laml*erl : 0.0;
    const Real prr = coupled ? lamr*err : 0.0;
    const Real ptl = wli[IPR] + prl;
    const Real ptr = wri[IPR] + prr;

    Real al, ar, el, er;
    const Real clg = pmy_block->peos->SoundSpeed(wli);
    const Real crg = pmy_block->peos->SoundSpeed(wri);
    Real vsql = SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]);
    Real vsqr = SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]);
    if (GENERAL_EOS) {
      el = pmy_block->peos->EgasFromRhoP(wli[IDN], wli[IPR]) + 0.5*wli[IDN]*vsql;
      er = pmy_block->peos->EgasFromRhoP(wri[IDN], wri[IPR]) + 0.5*wri[IDN]*vsqr;
    } else {
      el = wli[IPR]*igm1 + 0.5*wli[IDN]*vsql;
      er = wri[IPR]*igm1 + 0.5*wri[IDN]*vsqr;
    }
    const Real cl = std::sqrt(SQR(clg) + (coupled ? laml*(erl+prl)/wli[IDN] : 0.0));
    const Real cr = std::sqrt(SQR(crg) + (coupled ? lamr*(err+prr)/wri[IDN] : 0.0));
    Real rhoa = .5 * (wli[IDN] + wri[IDN]);
    Real ca = .5 * (cl + cr);
    Real pmid = .5 * (ptl + ptr + (wli[IVX]-wri[IVX]) * rhoa * ca);
    Real umid = .5 * (wli[IVX] + wri[IVX] + (ptl-ptr) / (rhoa * ca));
    Real rhol = wli[IDN] + (wli[IVX] - umid) * rhoa / ca;
    Real rhor = wri[IDN] + (umid - wri[IVX]) * rhoa / ca;

    //--- Step 3.  Compute sound speed in L,R

    Real ql, qr;
    if (GENERAL_EOS) {
      // The EOS still supplies the gas response; ptl/ptr additionally carry
      // the scalar FLD pressure in the Riemann wave construction.
      Real gl = pmy_block->peos->AsqFromRhoP(rhol, wli[IPR]) * rhol
                / std::max(wli[IPR], TINY_NUMBER);
      Real gr = pmy_block->peos->AsqFromRhoP(rhor, wri[IPR]) * rhor
                / std::max(wri[IPR], TINY_NUMBER);
      ql = (pmid <= ptl) ? 1.0 :
           std::sqrt(1.0 + (gl + 1) / (2 * gl) * (pmid / ptl-1.0));
      qr = (pmid <= ptr) ? 1.0 :
           std::sqrt(1.0 + (gr + 1) / (2 * gr) * (pmid / ptr-1.0));
    } else {
      ql = (pmid <= ptl) ? 1.0 :
           std::sqrt(1.0 + (gamma + 1) / (2 * gamma) * (pmid / ptl-1.0));
      qr = (pmid <= ptr) ? 1.0 :
           std::sqrt(1.0 + (gamma + 1) / (2 * gamma) * (pmid / ptr-1.0));
    }

    //--- Step 4.  Compute the max/min wave speeds based on L/R

    al = wli[IVX] - cl*ql;
    ar = wri[IVX] + cr*qr;

    Real bp = ar > 0.0 ? ar : (TINY_NUMBER);
    Real bm = al < 0.0 ? al : -(TINY_NUMBER);

    //--- Step 5. Compute the contact wave speed and pressure

    Real vxl = al - wli[IVX];
    Real vxr = ar - wri[IVX];

    Real ml = wli[IDN]*vxl;
    Real mr = wri[IDN]*vxr;

    // shock detector
    // Real cmax = std::max(cl*ql, cr*qr);
    Real cmax = std::max(cl, cr);
    Real th1 = std::min(1.0, (cmax-std::min(dvn(i),0.0)) / (cmax-std::min(dvt(i),0.0)));
    Real th = th1 * th1 * th1 * th1; // this 4th power is empirical (see Minoshima+)

    // Determine the contact wave speed...
    Real am = (mr*wri[IVX] - ml*wli[IVX] - th*(ptr-ptl)) / (mr - ml);

    // ...and the pressure at the contact surface
    Real chi = std::min(1.0, std::sqrt(std::max(vsql, vsqr)) / cmax);
    Real phi = chi * (2.0 - chi);

    Real cp = (mr*ptl - ml*ptr + phi*mr*ml*(wri[IVX]-wli[IVX])) / (mr - ml);
    cp = cp > 0.0 ? cp : 0.0;

    // No loop-carried dependencies anywhere in this loop
    //    #pragma distribute_point
    //--- Step 6. Compute L/R fluxes along the line bm, bp

    Real uxl = wli[IVX] - bm;
    Real uxr = wri[IVX] - bp;

    fl[IDN] = wli[IDN]*uxl;
    fr[IDN] = wri[IDN]*uxr;

    fl[IVX] = wli[IDN]*wli[IVX]*uxl + ptl;
    fr[IVX] = wri[IDN]*wri[IVX]*uxr + ptr;

    fl[IVY] = wli[IDN]*wli[IVY]*uxl;
    fr[IVY] = wri[IDN]*wri[IVY]*uxr;

    fl[IVZ] = wli[IDN]*wli[IVZ]*uxl;
    fr[IVZ] = wri[IDN]*wri[IVZ]*uxr;

    const Real etl = el + (coupled ? erl : 0.0);
    const Real etr = er + (coupled ? err : 0.0);
    fl[IEN] = etl*uxl + ptl*wli[IVX];
    fr[IEN] = etr*uxr + ptr*wri[IVX];

    //--- Step 8. Compute flux weights or scales

    Real sl,sr,sm;
    if (am >= 0.0) {
      sl =  am/(am - bm);
      sr = 0.0;
      sm = -bm/(am - bm);
    } else {
      sl =  0.0;
      sr = -am/(bp - am);
      sm =  bp/(bp - am);
    }

    //--- Step 9. Compute the HLLC flux at interface, including weighted contribution
    // of the flux along the contact

    flxi[IDN] = sl*fl[IDN] + sr*fr[IDN];
    flxi[IVX] = sl*fl[IVX] + sr*fr[IVX] + sm*cp;
    flxi[IVY] = sl*fl[IVY] + sr*fr[IVY];
    flxi[IVZ] = sl*fl[IVZ] + sr*fr[IVZ];
    flxi[IEN] = sl*fl[IEN] + sr*fr[IEN] + sm*cp*am;

    const bool left = am >= 0.0;
    const Real erg = left ? erl : err;
    const Real lambdag = left ? laml : lamr;
    const Real ag = left ? arl : arr;
    const Real fer = ag*erg*am;
    radflux(k,j,i) = fer;
    pfld->rad_face_g[dir](k,j,i) = erg;
    if (coupled) {
      flxi[IVX] -= lambdag*erg;
      flxi[IEN] -= fer;
    }

    // A ghost-state diode can still yield a small inward HLLC contact flux
    // when the uppermost active gas is moving away from the boundary.  For
    // an explicitly outflow-only physical upper face, remove only that
    // incoming advective flux.  Retain the normal momentum (pressure) flux;
    // implicit Marshak diffusion remains authoritative for radiation loss.
    const bool physical_top = ivx == IVZ && k == pmy_block->ke + 1
        && pmy_block->block_size.x3max
           == pmy_block->pmy_mesh->mesh_size.x3max;
    if (pfld->hydro_top_outflow_diode && physical_top && flxi[IDN] < 0.0) {
      flxi[IDN] = 0.0;
      flxi[IVY] = 0.0;
      flxi[IVZ] = 0.0;
      flxi[IEN] = 0.0;
      radflux(k,j,i) = 0.0;
      am = 0.0;
    }

    flx(IDN,k,j,i) = flxi[IDN];
    flx(ivx,k,j,i) = flxi[IVX];
    flx(ivy,k,j,i) = flxi[IVY];
    flx(ivz,k,j,i) = flxi[IVZ];
    flx(IEN,k,j,i) = flxi[IEN];
    pmy_block->phydro->vf[dir](k,j,i) = am;
  }
  return;
}
