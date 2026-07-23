//========================================================================================
//! \file hllc_fld.cpp
//! \brief Gas HLLC solver using an FLD-modified signal speed.

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
  const int dir = ivx-IVX;
  const int ivy = IVX+(dir+1)%3;
  const int ivz = IVX+(dir+2)%3;
  FLD2 *pfld = pmy_block->prfld2;
  const bool coupled = pfld->is_couple && !pfld->only_rad
                       && pfld->include_radiation_force;
  AthenaArray<Real> &radl = pfld->rad_face_l[dir];
  AthenaArray<Real> &radr = pfld->rad_face_r[dir];
  AthenaArray<Real> &radflux = pfld->u_rad_flux[dir];

  Real wli[NHYDRO], wri[NHYDRO];
  Real fi[NHYDRO], fl[NHYDRO], fr[NHYDRO];
  const Real gamma = GENERAL_EOS ? std::nan("") : pmy_block->peos->GetGamma();
  const Real igm1 = GENERAL_EOS ? 0.0 : 1.0/(gamma-1.0);

#pragma omp simd private(wli,wri,fi,fl,fr)
  for (int i=il; i<=iu; ++i) {
    wli[IDN]=wl(IDN,i); wli[IVX]=wl(ivx,i); wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i); wli[IPR]=wl(IPR,i);
    wri[IDN]=wr(IDN,i); wri[IVX]=wr(ivx,i); wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i); wri[IPR]=wr(IPR,i);

    const Real erl = std::max(radl(RadFLD2::ERAD,k,j,i),TINY_NUMBER);
    const Real err = std::max(radr(RadFLD2::ERAD,k,j,i),TINY_NUMBER);
    const Real laml = std::max(radl(RadFLD2::LAMBDA,k,j,i),0.0);
    const Real lamr = std::max(radr(RadFLD2::LAMBDA,k,j,i),0.0);
    const Real prl = coupled ? laml*erl : 0.0;
    const Real prr = coupled ? lamr*err : 0.0;

    const Real eil = GENERAL_EOS
        ? pmy_block->peos->EgasFromRhoP(wli[IDN],wli[IPR]) : wli[IPR]*igm1;
    const Real eir = GENERAL_EOS
        ? pmy_block->peos->EgasFromRhoP(wri[IDN],wri[IPR]) : wri[IPR]*igm1;
    const Real kel = 0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ]));
    const Real ker = 0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ]));
    const Real etl = eil+kel;
    const Real etr = eir+ker;

    const Real cgl = pmy_block->peos->SoundSpeed(wli);
    const Real cgr = pmy_block->peos->SoundSpeed(wri);
    const Real cl = std::sqrt(SQR(cgl)+(coupled ? laml*(erl+prl)/wli[IDN] : 0.0));
    const Real cr = std::sqrt(SQR(cgr)+(coupled ? lamr*(err+prr)/wri[IDN] : 0.0));
    const Real rhoa=0.5*(wli[IDN]+wri[IDN]);
    const Real ca=0.5*(cl+cr);
    // Radiation changes only the signal-speed estimate.  The HLLC pressure,
    // star state, and returned hydro flux remain the ordinary gas quantities.
    const Real pmid=0.5*(wli[IPR]+wri[IPR]
                         +(wli[IVX]-wri[IVX])*rhoa*ca);
    const Real umid=0.5*(wli[IVX]+wri[IVX]
                         +(wli[IPR]-wri[IPR])/(rhoa*ca));
    const Real rhol=wli[IDN]+(wli[IVX]-umid)*rhoa/ca;
    const Real rhor=wri[IDN]+(umid-wri[IVX])*rhoa/ca;

    Real ql,qr;
    if (GENERAL_EOS) {
      const Real gl=pmy_block->peos->AsqFromRhoP(rhol,wli[IPR])*rhol
                    /std::max(wli[IPR],TINY_NUMBER);
      const Real gr=pmy_block->peos->AsqFromRhoP(rhor,wri[IPR])*rhor
                    /std::max(wri[IPR],TINY_NUMBER);
      ql=(pmid<=wli[IPR])?1.0:std::sqrt(1.0+(gl+1.0)/(2.0*gl)
                                      *(pmid/wli[IPR]-1.0));
      qr=(pmid<=wri[IPR])?1.0:std::sqrt(1.0+(gr+1.0)/(2.0*gr)
                                      *(pmid/wri[IPR]-1.0));
    } else {
      ql=(pmid<=wli[IPR])?1.0:std::sqrt(1.0+(gamma+1.0)/(2.0*gamma)
                                      *(pmid/wli[IPR]-1.0));
      qr=(pmid<=wri[IPR])?1.0:std::sqrt(1.0+(gamma+1.0)/(2.0*gamma)
                                      *(pmid/wri[IPR]-1.0));
    }
    const Real al=wli[IVX]-cl*ql, ar=wri[IVX]+cr*qr;
    const Real bp=ar>0.0?ar:TINY_NUMBER, bm=al<0.0?al:-TINY_NUMBER;
    const Real vlbm=wli[IVX]-bm, vrbp=wri[IVX]-bp;
    const Real ml=wli[IDN]*(wli[IVX]-al), mr=-wri[IDN]*(wri[IVX]-ar);
    const Real tl=wli[IPR]+(wli[IVX]-al)*wli[IDN]*wli[IVX];
    const Real tr=wri[IPR]+(wri[IVX]-ar)*wri[IDN]*wri[IVX];
    const Real am=(tl-tr)/(ml+mr);
    const Real cp=std::max((ml*tr+mr*tl)/(ml+mr),0.0);

    fl[IDN]=wli[IDN]*vlbm; fr[IDN]=wri[IDN]*vrbp;
    fl[IVX]=wli[IDN]*wli[IVX]*vlbm+wli[IPR];
    fr[IVX]=wri[IDN]*wri[IVX]*vrbp+wri[IPR];
    fl[IVY]=wli[IDN]*wli[IVY]*vlbm; fr[IVY]=wri[IDN]*wri[IVY]*vrbp;
    fl[IVZ]=wli[IDN]*wli[IVZ]*vlbm; fr[IVZ]=wri[IDN]*wri[IVZ]*vrbp;
    fl[IEN]=etl*vlbm+wli[IPR]*wli[IVX];
    fr[IEN]=etr*vrbp+wri[IPR]*wri[IVX];

    Real sl,sr,sm;
    if (am>=0.0) {sl=am/(am-bm); sr=0.0; sm=-bm/(am-bm);}
    else {sl=0.0; sr=-am/(bp-am); sm=bp/(bp-am);}
    fi[IDN]=sl*fl[IDN]+sr*fr[IDN];
    fi[IVX]=sl*fl[IVX]+sr*fr[IVX]+sm*cp;
    fi[IVY]=sl*fl[IVY]+sr*fr[IVY];
    fi[IVZ]=sl*fl[IVZ]+sr*fr[IVZ];
    fi[IEN]=sl*fl[IEN]+sr*fr[IEN]+sm*cp*am;

    // Use the upwind reconstructed radiation state as the interface Godunov
    // state.  No radiation star state is constructed in this simplified
    // solver.
    const Real erg = am>=0.0 ? erl : err;
    pfld->rad_face_g[dir](k,j,i)=erg;
    const Real fer = coupled
        ? (am>=0.0 ? radl(RadFLD2::ARAD,k,j,i)
                   : radr(RadFLD2::ARAD,k,j,i))*erg*am
        : sl*erl*vlbm+sr*err*vrbp;
    radflux(k,j,i)=fer;
    flx(IDN,k,j,i)=fi[IDN]; flx(ivx,k,j,i)=fi[IVX];
    flx(ivy,k,j,i)=fi[IVY]; flx(ivz,k,j,i)=fi[IVZ];
    flx(IEN,k,j,i)=fi[IEN];
    pmy_block->phydro->vf[dir](k,j,i)=am;
  }
}
