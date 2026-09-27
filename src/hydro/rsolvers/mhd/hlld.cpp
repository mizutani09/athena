//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld.cpp
//! \brief HLLD Riemann solver for adiabatic MHD.
//!
//! REFERENCES:
//! - T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal
//!   MHD", JCP, 208, 315 (2005)

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()
#include <sstream>

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"
#if NRMGFLD_ENABLED
#include "../../../fld/fld.hpp"
#endif
#include "../../../mesh/mesh.hpp"
#include "../../hydro.hpp"

// container to store (density, momentum, total energy, tranverse magnetic field)
// minimizes changes required to adopt athena4.2 version of this solver
struct Cons1D {
  Real d, mx, my, mz, e, by, bz;
};

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief The HLLD Riemann solver for adiabatic MHD

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx, const AthenaArray<Real> &bx,
                          AthenaArray<Real> &wl, AthenaArray<Real> &wr,
                          AthenaArray<Real> &flx,
                          AthenaArray<Real> &ey, AthenaArray<Real> &ez,
                          AthenaArray<Real> &wct, const AthenaArray<Real> &dxw) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real flxi[(NWAVE)];             // temporary variable to store flux
  Real wli[(NWAVE)],wri[(NWAVE)]; // L/R states, primitive variables (input)
  Real spd[5];                    // signal speeds, left to right
  Real vf;                        // Velocity at the cell face for any advection term
  constexpr Real SMALL_NUMBER = 1.0e-4;

  Real igm1;
  EquationOfState *peos = pmy_block->peos;
  if (!GENERAL_EOS) igm1 = 1.0 / (peos->GetGamma() - 1.0);
  Real dt = pmy_block->pmy_mesh->dt;
#if NRMGFLD_ENABLED
  const int dir=ivx-IVX;
  FLD *pfld=pmy_block->prfld;
  const bool coupled=pfld->pressure_coupling_mode
      != RadFLD::PressureCouplingMode::kOff;
  const bool radiation_pressure_in_flux=pfld->pressure_coupling_mode
      == RadFLD::PressureCouplingMode::kFlux;
  AthenaArray<Real> &radl=pfld->rad_face_l[dir];
  AthenaArray<Real> &radr=pfld->rad_face_r[dir];
  AthenaArray<Real> &radflux=pfld->u_rad_flux[dir];
#endif

#if NRMGFLD_ENABLED
  int bad_rad_flux=0;
#pragma omp simd simdlen(SIMD_WIDTH) private(wli,wri,spd,flxi,vf) reduction(|:bad_rad_flux)
#else
#pragma omp simd simdlen(SIMD_WIDTH) private(wli,wri,spd,flxi,vf)
#endif
  for (int i=il; i<=iu; ++i) {
    Cons1D ul,ur;                   // L/R states, conserved variables (computed)
    Cons1D ulst,uldst,urdst,urst;   // Conserved variable for all states
    Cons1D fl,fr;                   // Fluxes for left & right states

    //--- Step 1.  Load L/R states into local variables

    wli[IDN]=wl(IDN,i);
    wli[IVX]=wl(ivx,i);
    wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i);
    wli[IPR]=wl(IPR,i);
    wli[IBY]=wl(IBY,i);
    wli[IBZ]=wl(IBZ,i);

    wri[IDN]=wr(IDN,i);
    wri[IVX]=wr(ivx,i);
    wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i);
    wri[IPR]=wr(IPR,i);
    wri[IBY]=wr(IBY,i);
    wri[IBZ]=wr(IBZ,i);

    Real bxi = bx(k,j,i);

#if NRMGFLD_ENABLED
    // MHD requests extra transverse Riemann rows for constrained transport.
    // Those corner ghost states are not radiation control-volume faces and can
    // legitimately be incomplete at a physical boundary.  Couple FLD only on
    // the active transverse rows whose radiation flux divergence is evolved.
    const bool active_rad_face=(dir == X1DIR)
        ? (j >= pmy_block->js && j <= pmy_block->je
           && k >= pmy_block->ks && k <= pmy_block->ke)
        : ((dir == X2DIR)
           ? (i >= pmy_block->is && i <= pmy_block->ie
              && k >= pmy_block->ks && k <= pmy_block->ke)
           : (i >= pmy_block->is && i <= pmy_block->ie
              && j >= pmy_block->js && j <= pmy_block->je));
    const bool face_coupled=coupled && active_rad_face;
    const Real erl=active_rad_face
        ? std::max(radl(RadFLD::ERAD,k,j,i),TINY_NUMBER) : 0.0;
    const Real err=active_rad_face
        ? std::max(radr(RadFLD::ERAD,k,j,i),TINY_NUMBER) : 0.0;
    const Real laml=active_rad_face
        ? std::max(radl(RadFLD::LAMBDA,k,j,i),0.0) : 0.0;
    const Real lamr=active_rad_face
        ? std::max(radr(RadFLD::LAMBDA,k,j,i),0.0) : 0.0;
    const Real arl=active_rad_face
        ? std::max(radl(RadFLD::ARAD,k,j,i),0.0) : 0.0;
    const Real arr=active_rad_face
        ? std::max(radr(RadFLD::ARAD,k,j,i),0.0) : 0.0;
    const Real prl=face_coupled ? laml*erl : 0.0;
    const Real prr=face_coupled ? lamr*err : 0.0;
#else
    const Real prl=0.0, prr=0.0;
#endif

#if NRMGFLD_ENABLED
    // At exactly zero magnetic field, use the same hydrodynamic HLLC-FLD
    // construction as hllc_fld.cpp.  Although the HLLD fan degenerates to a
    // hydrodynamic fan analytically, its default Davis wave-speed estimates
    // differ from Athena++'s HLLC-FLD estimates.  This explicit degeneracy
    // branch makes the B=0 numerical limit reproducible cell for cell and
    // provides a direct guard against double-counting radiation terms.
    const bool zero_magnetic = (bxi == 0.0 && wli[IBY] == 0.0 && wli[IBZ] == 0.0
                                && wri[IBY] == 0.0 && wri[IBZ] == 0.0);
    if (zero_magnetic) {
      const Real ptl=wli[IPR]+prl, ptr=wri[IPR]+prr;
      const Real eil=GENERAL_EOS ? peos->EgasFromRhoP(wli[IDN],wli[IPR])
                                 : wli[IPR]*igm1;
      const Real eir=GENERAL_EOS ? peos->EgasFromRhoP(wri[IDN],wri[IPR])
                                 : wri[IPR]*igm1;
      const Real kel=0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ]));
      const Real ker=0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ]));
      const Real etl=eil+kel+(face_coupled ? erl : 0.0);
      const Real etr=eir+ker+(face_coupled ? err : 0.0);
      const Real cgl=peos->SoundSpeed(wli), cgr=peos->SoundSpeed(wri);
      const Real cl=std::sqrt(SQR(cgl)+(face_coupled
                              ? laml*(erl+prl)/wli[IDN] : 0.0));
      const Real cr=std::sqrt(SQR(cgr)+(face_coupled
                              ? lamr*(err+prr)/wri[IDN] : 0.0));
      const Real rhoa=0.5*(wli[IDN]+wri[IDN]), ca=0.5*(cl+cr);
      const Real pmid=0.5*(ptl+ptr+(wli[IVX]-wri[IVX])*rhoa*ca);
      const Real umid=0.5*(wli[IVX]+wri[IVX]+(ptl-ptr)/(rhoa*ca));
      const Real rhol=wli[IDN]+(wli[IVX]-umid)*rhoa/ca;
      const Real rhor=wri[IDN]+(umid-wri[IVX])*rhoa/ca;
      Real ql, qr;
      if (GENERAL_EOS) {
        const Real gl=peos->AsqFromRhoP(rhol,wli[IPR])*rhol
                      /std::max(wli[IPR],TINY_NUMBER);
        const Real gr=peos->AsqFromRhoP(rhor,wri[IPR])*rhor
                      /std::max(wri[IPR],TINY_NUMBER);
        ql=(pmid<=ptl) ? 1.0 : std::sqrt(1.0+(gl+1.0)/(2.0*gl)*(pmid/ptl-1.0));
        qr=(pmid<=ptr) ? 1.0 : std::sqrt(1.0+(gr+1.0)/(2.0*gr)*(pmid/ptr-1.0));
      } else {
        const Real gamma=peos->GetGamma();
        ql=(pmid<=ptl) ? 1.0 : std::sqrt(1.0+(gamma+1.0)/(2.0*gamma)
                                              *(pmid/ptl-1.0));
        qr=(pmid<=ptr) ? 1.0 : std::sqrt(1.0+(gamma+1.0)/(2.0*gamma)
                                              *(pmid/ptr-1.0));
      }
      const Real al=wli[IVX]-cl*ql, ar=wri[IVX]+cr*qr;
      const Real bp=ar>0.0 ? ar : TINY_NUMBER;
      const Real bm=al<0.0 ? al : -TINY_NUMBER;
      const Real vlbm=wli[IVX]-bm, vrbp=wri[IVX]-bp;
      const Real ml=wli[IDN]*(wli[IVX]-al);
      const Real mr=-wri[IDN]*(wri[IVX]-ar);
      const Real tl=ptl+(wli[IVX]-al)*wli[IDN]*wli[IVX];
      const Real tr=ptr+(wri[IVX]-ar)*wri[IDN]*wri[IVX];
      const Real am=(tl-tr)/(ml+mr);
      const Real cp=std::max((ml*tr+mr*tl)/(ml+mr),0.0);
      Real hfl[NHYDRO], hfr[NHYDRO];
      hfl[IDN]=wli[IDN]*vlbm; hfr[IDN]=wri[IDN]*vrbp;
      hfl[IVX]=wli[IDN]*wli[IVX]*vlbm+ptl;
      hfr[IVX]=wri[IDN]*wri[IVX]*vrbp+ptr;
      hfl[IVY]=wli[IDN]*wli[IVY]*vlbm;
      hfr[IVY]=wri[IDN]*wri[IVY]*vrbp;
      hfl[IVZ]=wli[IDN]*wli[IVZ]*vlbm;
      hfr[IVZ]=wri[IDN]*wri[IVZ]*vrbp;
      hfl[IEN]=etl*vlbm+ptl*wli[IVX];
      hfr[IEN]=etr*vrbp+ptr*wri[IVX];
      Real a, b, c;
      if (am>=0.0) { a=am/(am-bm); b=0.0; c=-bm/(am-bm); }
      else { a=0.0; b=-am/(bp-am); c=bp/(bp-am); }
      flxi[IDN]=a*hfl[IDN]+b*hfr[IDN];
      flxi[IVX]=a*hfl[IVX]+b*hfr[IVX]+c*cp;
      flxi[IVY]=a*hfl[IVY]+b*hfr[IVY];
      flxi[IVZ]=a*hfl[IVZ]+b*hfr[IVZ];
      flxi[IEN]=a*hfl[IEN]+b*hfr[IEN]+c*cp*am;
      const bool left=am>=0.0;
      const Real erg=left ? erl : err;
      const Real lambdag=left ? laml : lamr;
      const Real ag=left ? arl : arr;
      const Real fer=ag*erg*am;
      if (active_rad_face) {
        radflux(k,j,i)=pfld->mixed_frame_transport ? fer : 0.0;
        pfld->rad_face_g[dir](k,j,i)=erg;
      }
      if (face_coupled && !radiation_pressure_in_flux) flxi[IVX]-=lambdag*erg;
      if (face_coupled) flxi[IEN]-=fer;
      flx(IDN,k,j,i)=flxi[IDN]; flx(ivx,k,j,i)=flxi[IVX];
      flx(ivy,k,j,i)=flxi[IVY]; flx(ivz,k,j,i)=flxi[IVZ];
      flx(IEN,k,j,i)=flxi[IEN];
      ey(k,j,i)=0.0; ez(k,j,i)=0.0;
      pmy_block->phydro->vf[dir](k,j,i)=am;
      wct(k,j,i)=GetWeightForCT(flxi[IDN],wli[IDN],wri[IDN],dxw(i),dt);
      continue;
    }
#endif

    // Compute L/R states for selected conserved variables
    Real bxsq = bxi*bxi;
    // (KGF): group transverse vector components for floating-point associativity symmetry
    Real pbl = 0.5*(bxsq + (SQR(wli[IBY]) + SQR(wli[IBZ])));  // magnetic pressure (l/r)
    Real pbr = 0.5*(bxsq + (SQR(wri[IBY]) + SQR(wri[IBZ])));
    Real kel = 0.5*wli[IDN]*(SQR(wli[IVX]) + (SQR(wli[IVY]) + SQR(wli[IVZ])));
    Real ker = 0.5*wri[IDN]*(SQR(wri[IVX]) + (SQR(wri[IVY]) + SQR(wri[IVZ])));

    ul.d  = wli[IDN];
    ul.mx = wli[IVX]*ul.d;
    ul.my = wli[IVY]*ul.d;
    ul.mz = wli[IVZ]*ul.d;
    if (GENERAL_EOS) {
      ul.e  = peos->EgasFromRhoP(ul.d, wli[IPR]) + kel + pbl;
    } else {
      ul.e  = wli[IPR]*igm1 + kel + pbl;
    }
    ul.by = wli[IBY];
    ul.bz = wli[IBZ];

    ur.d  = wri[IDN];
    ur.mx = wri[IVX]*ur.d;
    ur.my = wri[IVY]*ur.d;
    ur.mz = wri[IVZ]*ur.d;
    if (GENERAL_EOS) {
      ur.e  = peos->EgasFromRhoP(ur.d, wri[IPR]) + ker + pbr;
    } else {
      ur.e  = wri[IPR]*igm1 + ker + pbr;
    }
    ur.by = wri[IBY];
    ur.bz = wri[IBZ];

#if NRMGFLD_ENABLED
    if (face_coupled) {
      ul.e += erl;
      ur.e += err;
    }
#endif

    //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

    Real cfl, cfr;
#if NRMGFLD_ENABLED
    // Replace the gas acoustic modulus gamma*P by the gas+radiation modulus
    // rho*c_s^2 + lambda*(E_r+P_r) in the standard fast-wave formula.
    const Real agl=GENERAL_EOS ? peos->AsqFromRhoP(wli[IDN],wli[IPR])*wli[IDN]
                               : peos->GetGamma()*wli[IPR];
    const Real agr=GENERAL_EOS ? peos->AsqFromRhoP(wri[IDN],wri[IPR])*wri[IDN]
                               : peos->GetGamma()*wri[IPR];
    const Real asl=agl+(face_coupled ? laml*(erl+prl) : 0.0);
    const Real asr=agr+(face_coupled ? lamr*(err+prr) : 0.0);
    const Real btl=SQR(wli[IBY])+SQR(wli[IBZ]);
    const Real btr=SQR(wri[IBY])+SQR(wri[IBZ]);
    const Real ql=SQR(bxi)+btl+asl, qr=SQR(bxi)+btr+asr;
    const Real tl=SQR(bxi)+btl-asl, tr=SQR(bxi)+btr-asr;
    cfl=std::sqrt(0.5*(ql+std::sqrt(SQR(tl)+4.0*asl*btl))/wli[IDN]);
    cfr=std::sqrt(0.5*(qr+std::sqrt(SQR(tr)+4.0*asr*btr))/wri[IDN]);
#else
    cfl = pmy_block->peos->FastMagnetosonicSpeed(wli,bxi);
    cfr = pmy_block->peos->FastMagnetosonicSpeed(wri,bxi);
#endif

    spd[0] = std::min( wli[IVX]-cfl, wri[IVX]-cfr );
    spd[4] = std::max( wli[IVX]+cfl, wri[IVX]+cfr );

    // Real cfmax = std::max(cfl,cfr);
    // if (wli[IVX] <= wri[IVX]) {
    //   spd[0] = wli[IVX] - cfmax;
    //   spd[4] = wri[IVX] + cfmax;
    // } else {
    //   spd[0] = wri[IVX] - cfmax;
    //   spd[4] = wli[IVX] + cfmax;
    // }

    //--- Step 3.  Compute L/R fluxes

    Real ptl = wli[IPR] + pbl + prl; // gas + magnetic + radiation pressure
    Real ptr = wri[IPR] + pbr + prr;

    fl.d  = ul.mx;
    fl.mx = ul.mx*wli[IVX] + ptl - bxsq;
    fl.my = ul.my*wli[IVX] - bxi*ul.by;
    fl.mz = ul.mz*wli[IVX] - bxi*ul.bz;
    fl.e  = wli[IVX]*(ul.e + ptl - bxsq) - bxi*(wli[IVY]*ul.by + wli[IVZ]*ul.bz);
    fl.by = ul.by*wli[IVX] - bxi*wli[IVY];
    fl.bz = ul.bz*wli[IVX] - bxi*wli[IVZ];

    fr.d  = ur.mx;
    fr.mx = ur.mx*wri[IVX] + ptr - bxsq;
    fr.my = ur.my*wri[IVX] - bxi*ur.by;
    fr.mz = ur.mz*wri[IVX] - bxi*ur.bz;
    fr.e  = wri[IVX]*(ur.e + ptr - bxsq) - bxi*(wri[IVY]*ur.by + wri[IVZ]*ur.bz);
    fr.by = ur.by*wri[IVX] - bxi*wri[IVY];
    fr.bz = ur.bz*wri[IVX] - bxi*wri[IVZ];

    //--- Step 4.  Compute middle and Alfven wave speeds

    Real sdl = spd[0] - wli[IVX];  // S_i-u_i (i=L or R)
    Real sdr = spd[4] - wri[IVX];

    // S_M: eqn (38) of Miyoshi & Kusano
    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
    spd[2] = (sdr*ur.mx - sdl*ul.mx + (ptl - ptr))/(sdr*ur.d - sdl*ul.d);

    Real sdml   = spd[0] - spd[2];  // S_i-S_M (i=L or R)
    Real sdmr   = spd[4] - spd[2];
    Real sdml_inv = 1.0/sdml;
    Real sdmr_inv = 1.0/sdmr;
    // eqn (43) of Miyoshi & Kusano
    ulst.d = ul.d * sdl * sdml_inv;
    urst.d = ur.d * sdr * sdmr_inv;
    Real ulst_d_inv = 1.0/ulst.d;
    Real urst_d_inv = 1.0/urst.d;
    Real sqrtdl = std::sqrt(ulst.d);
    Real sqrtdr = std::sqrt(urst.d);

    // eqn (51) of Miyoshi & Kusano
    spd[1] = spd[2] - std::abs(bxi)/sqrtdl;
    spd[3] = spd[2] + std::abs(bxi)/sqrtdr;

    //--- Step 5.  Compute intermediate states
    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
    // TODO(felker): place an assertion that ptstl==ptstr
    Real ptstl = ptl + ul.d*sdl*(spd[2]-wli[IVX]);
    Real ptstr = ptr + ur.d*sdr*(spd[2]-wri[IVX]);
    // Real ptstl = ptl + ul.d*sdl*(sdl-sdml); // these equations had issues when averaged
    // Real ptstr = ptr + ur.d*sdr*(sdr-sdmr);
    Real ptst = 0.5*(ptstr + ptstl);  // total pressure (star state)

    // ul* - eqn (39) of M&K
    ulst.mx = ulst.d * spd[2];
    if (std::abs(ul.d*sdl*sdml-bxsq) < (SMALL_NUMBER)*ptst) {
      // Degenerate case
      ulst.my = ulst.d * wli[IVY];
      ulst.mz = ulst.d * wli[IVZ];

      ulst.by = ul.by;
      ulst.bz = ul.bz;
    } else {
      // eqns (44) and (46) of M&K
      Real tmp = bxi*(sdl - sdml)/(ul.d*sdl*sdml - bxsq);
      ulst.my = ulst.d * (wli[IVY] - ul.by*tmp);
      ulst.mz = ulst.d * (wli[IVZ] - ul.bz*tmp);

      // eqns (45) and (47) of M&K
      tmp = (ul.d*SQR(sdl) - bxsq)/(ul.d*sdl*sdml - bxsq);
      ulst.by = ul.by * tmp;
      ulst.bz = ul.bz * tmp;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstl = (ulst.mx*bxi+(ulst.my*ulst.by+ulst.mz*ulst.bz))*ulst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    ulst.e = (sdl*ul.e - ptl*wli[IVX] + ptst*spd[2] +
              bxi*(wli[IVX]*bxi + (wli[IVY]*ul.by + wli[IVZ]*ul.bz) - vbstl))*sdml_inv;

    // ur* - eqn (39) of M&K
    urst.mx = urst.d * spd[2];
    if (std::abs(ur.d*sdr*sdmr - bxsq) < (SMALL_NUMBER)*ptst) {
      // Degenerate case
      urst.my = urst.d * wri[IVY];
      urst.mz = urst.d * wri[IVZ];

      urst.by = ur.by;
      urst.bz = ur.bz;
    } else {
      // eqns (44) and (46) of M&K
      Real tmp = bxi*(sdr - sdmr)/(ur.d*sdr*sdmr - bxsq);
      urst.my = urst.d * (wri[IVY] - ur.by*tmp);
      urst.mz = urst.d * (wri[IVZ] - ur.bz*tmp);

      // eqns (45) and (47) of M&K
      tmp = (ur.d*SQR(sdr) - bxsq)/(ur.d*sdr*sdmr - bxsq);
      urst.by = ur.by * tmp;
      urst.bz = ur.bz * tmp;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstr = (urst.mx*bxi+(urst.my*urst.by+urst.mz*urst.bz))*urst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    urst.e = (sdr*ur.e - ptr*wri[IVX] + ptst*spd[2] +
              bxi*(wri[IVX]*bxi + (wri[IVY]*ur.by + wri[IVZ]*ur.bz) - vbstr))*sdmr_inv;
    // ul** and ur** - if Bx is near zero, same as *-states
    if (0.5*bxsq < (SMALL_NUMBER)*ptst) {
      uldst = ulst;
      urdst = urst;
    } else {
      Real invsumd = 1.0/(sqrtdl + sqrtdr);
      Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

      uldst.d = ulst.d;
      urdst.d = urst.d;

      uldst.mx = ulst.mx;
      urdst.mx = urst.mx;

      // eqn (59) of M&K
      Real tmp = invsumd*(sqrtdl*(ulst.my*ulst_d_inv) + sqrtdr*(urst.my*urst_d_inv) +
                          bxsig*(urst.by - ulst.by));
      uldst.my = uldst.d * tmp;
      urdst.my = urdst.d * tmp;

      // eqn (60) of M&K
      tmp = invsumd*(sqrtdl*(ulst.mz*ulst_d_inv) + sqrtdr*(urst.mz*urst_d_inv) +
                     bxsig*(urst.bz - ulst.bz));
      uldst.mz = uldst.d * tmp;
      urdst.mz = urdst.d * tmp;

      // eqn (61) of M&K
      tmp = invsumd*(sqrtdl*urst.by + sqrtdr*ulst.by +
                     bxsig*sqrtdl*sqrtdr*((urst.my*urst_d_inv) - (ulst.my*ulst_d_inv)));
      uldst.by = urdst.by = tmp;

      // eqn (62) of M&K
      tmp = invsumd*(sqrtdl*urst.bz + sqrtdr*ulst.bz +
                     bxsig*sqrtdl*sqrtdr*((urst.mz*urst_d_inv) - (ulst.mz*ulst_d_inv)));
      uldst.bz = urdst.bz = tmp;

      // eqn (63) of M&K
      tmp = spd[2]*bxi + (uldst.my*uldst.by + uldst.mz*uldst.bz)/uldst.d;
      uldst.e = ulst.e - sqrtdl*bxsig*(vbstl - tmp);
      urdst.e = urst.e + sqrtdr*bxsig*(vbstr - tmp);
    }

    //--- Step 6.  Compute flux
    uldst.d = spd[1] * (uldst.d - ulst.d);
    uldst.mx = spd[1] * (uldst.mx - ulst.mx);
    uldst.my = spd[1] * (uldst.my - ulst.my);
    uldst.mz = spd[1] * (uldst.mz - ulst.mz);
    uldst.e = spd[1] * (uldst.e - ulst.e);
    uldst.by = spd[1] * (uldst.by - ulst.by);
    uldst.bz = spd[1] * (uldst.bz - ulst.bz);

    ulst.d = spd[0] * (ulst.d - ul.d);
    ulst.mx = spd[0] * (ulst.mx - ul.mx);
    ulst.my = spd[0] * (ulst.my - ul.my);
    ulst.mz = spd[0] * (ulst.mz - ul.mz);
    ulst.e = spd[0] * (ulst.e - ul.e);
    ulst.by = spd[0] * (ulst.by - ul.by);
    ulst.bz = spd[0] * (ulst.bz - ul.bz);

    urdst.d = spd[3] * (urdst.d - urst.d);
    urdst.mx = spd[3] * (urdst.mx - urst.mx);
    urdst.my = spd[3] * (urdst.my - urst.my);
    urdst.mz = spd[3] * (urdst.mz - urst.mz);
    urdst.e = spd[3] * (urdst.e - urst.e);
    urdst.by = spd[3] * (urdst.by - urst.by);
    urdst.bz = spd[3] * (urdst.bz - urst.bz);

    urst.d = spd[4] * (urst.d  - ur.d);
    urst.mx = spd[4] * (urst.mx - ur.mx);
    urst.my = spd[4] * (urst.my - ur.my);
    urst.mz = spd[4] * (urst.mz - ur.mz);
    urst.e = spd[4] * (urst.e - ur.e);
    urst.by = spd[4] * (urst.by - ur.by);
    urst.bz = spd[4] * (urst.bz - ur.bz);

    if (spd[0] >= 0.0) {
      // return Fl if flow is supersonic
      flxi[IDN] = fl.d;
      flxi[IVX] = fl.mx;
      flxi[IVY] = fl.my;
      flxi[IVZ] = fl.mz;
      flxi[IEN] = fl.e;
      flxi[IBY] = fl.by;
      flxi[IBZ] = fl.bz;
      vf = wli[IVX];
    } else if (spd[4] <= 0.0) {
      // return Fr if flow is supersonic
      flxi[IDN] = fr.d;
      flxi[IVX] = fr.mx;
      flxi[IVY] = fr.my;
      flxi[IVZ] = fr.mz;
      flxi[IEN] = fr.e;
      flxi[IBY] = fr.by;
      flxi[IBZ] = fr.bz;
      vf = wri[IVX];
    } else if (spd[1] >= 0.0) {
      // return Fl*
      flxi[IDN] = fl.d  + ulst.d;
      flxi[IVX] = fl.mx + ulst.mx;
      flxi[IVY] = fl.my + ulst.my;
      flxi[IVZ] = fl.mz + ulst.mz;
      flxi[IEN] = fl.e  + ulst.e;
      flxi[IBY] = fl.by + ulst.by;
      flxi[IBZ] = fl.bz + ulst.bz;
      vf = spd[2];
    } else if (spd[2] >= 0.0) {
      // return Fl**
      flxi[IDN] = fl.d  + ulst.d + uldst.d;
      flxi[IVX] = fl.mx + ulst.mx + uldst.mx;
      flxi[IVY] = fl.my + ulst.my + uldst.my;
      flxi[IVZ] = fl.mz + ulst.mz + uldst.mz;
      flxi[IEN] = fl.e  + ulst.e + uldst.e;
      flxi[IBY] = fl.by + ulst.by + uldst.by;
      flxi[IBZ] = fl.bz + ulst.bz + uldst.bz;
      vf = spd[2];
    } else if (spd[3] > 0.0) {
      // return Fr**
      flxi[IDN] = fr.d + urst.d + urdst.d;
      flxi[IVX] = fr.mx + urst.mx + urdst.mx;
      flxi[IVY] = fr.my + urst.my + urdst.my;
      flxi[IVZ] = fr.mz + urst.mz + urdst.mz;
      flxi[IEN] = fr.e + urst.e + urdst.e;
      flxi[IBY] = fr.by + urst.by + urdst.by;
      flxi[IBZ] = fr.bz + urst.bz + urdst.bz;
      vf = spd[2];
    } else {
      // return Fr*
      flxi[IDN] = fr.d  + urst.d;
      flxi[IVX] = fr.mx + urst.mx;
      flxi[IVY] = fr.my + urst.my;
      flxi[IVZ] = fr.mz + urst.mz;
      flxi[IEN] = fr.e  + urst.e;
      flxi[IBY] = fr.by + urst.by;
      flxi[IBZ] = fr.bz + urst.bz;
      vf = spd[2];
    }

#if NRMGFLD_ENABLED
    const bool left=vf >= 0.0;
    const Real erg=left ? erl : err;
    const Real lambdag=left ? laml : lamr;
    const Real ag=left ? arl : arr;
    const Real fer=ag*erg*vf;
    if (active_rad_face
        && (!std::isfinite(vf) || !std::isfinite(erg)
            || !std::isfinite(ag) || !std::isfinite(fer))) bad_rad_flux |= 1;
    if (active_rad_face) {
      radflux(k,j,i)=pfld->mixed_frame_transport ? fer : 0.0;
      pfld->rad_face_g[dir](k,j,i)=erg;
    }
    if (face_coupled && !radiation_pressure_in_flux) flxi[IVX]-=lambdag*erg;
    if (face_coupled) flxi[IEN]-=fer;
#endif

    flx(IDN,k,j,i) = flxi[IDN];
    flx(ivx,k,j,i) = flxi[IVX];
    flx(ivy,k,j,i) = flxi[IVY];
    flx(ivz,k,j,i) = flxi[IVZ];
    flx(IEN,k,j,i) = flxi[IEN];
    ey(k,j,i) = -flxi[IBY];
    ez(k,j,i) =  flxi[IBZ];

#if NRMGFLD_ENABLED
    // vf is allocated only for NR-FLD mixed-frame transport.
    pmy_block->phydro->vf[ivx-IVX](k,j,i) = vf;
#endif

    wct(k,j,i) = GetWeightForCT(flxi[IDN], wli[IDN], wri[IDN], dxw(i), dt);
  }
#if NRMGFLD_ENABLED
  // Throwing an exception is forbidden inside an OpenMP SIMD region by Clang/ICX.
  // The reduction records a failure in the vector loop; rescan only on that exceptional
  // path to construct a useful diagnostic without penalizing the normal HLLD-FLD path.
  for (int i=il; bad_rad_flux != 0 && i<=iu; ++i) {
    const bool active_rad_face=(dir == X1DIR)
        ? (j >= pmy_block->js && j <= pmy_block->je
           && k >= pmy_block->ks && k <= pmy_block->ke)
        : ((dir == X2DIR)
           ? (i >= pmy_block->is && i <= pmy_block->ie
              && k >= pmy_block->ks && k <= pmy_block->ke)
           : (i >= pmy_block->is && i <= pmy_block->ie
              && j >= pmy_block->js && j <= pmy_block->je));
    if (!active_rad_face) continue;
    const Real vf_face=pmy_block->phydro->vf[dir](k,j,i);
    const bool left=vf_face >= 0.0;
    const Real er_face=left ? radl(RadFLD::ERAD,k,j,i) : radr(RadFLD::ERAD,k,j,i);
    const Real a_face=left ? radl(RadFLD::ARAD,k,j,i) : radr(RadFLD::ARAD,k,j,i);
    const Real fer=a_face*er_face*vf_face;
    if (!std::isfinite(vf_face) || !std::isfinite(er_face)
        || !std::isfinite(a_face) || !std::isfinite(fer)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in HLLD-FLD radiation flux at (k,j,i)=("
          << k << "," << j << "," << i << ") dir=" << dir
          << " vf=" << vf_face << " er=" << er_face
          << " arad=" << a_face << " fer=" << fer
          << " rhoL=" << wl(IDN,i) << " rhoR=" << wr(IDN,i)
          << " pL=" << wl(IPR,i) << " pR=" << wr(IPR,i)
          << " vxL=" << wl(ivx,i) << " vxR=" << wr(ivx,i)
          << " byL=" << wl(IBY,i) << " byR=" << wr(IBY,i)
          << " bzL=" << wl(IBZ,i) << " bzR=" << wr(IBZ,i)
          << " bx=" << bx(k,j,i);
      ATHENA_ERROR(msg);
    }
  }
  if (bad_rad_flux != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in HLLD-FLD radiation flux: non-finite SIMD result "
        << "could not be localized, dir=" << dir << " k=" << k << " j=" << j;
    ATHENA_ERROR(msg);
  }
#endif
  return;
}
