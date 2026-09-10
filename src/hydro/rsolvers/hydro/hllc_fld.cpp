//========================================================================================
//! \file hllc_fld.cpp
//! \brief HLLC solver whose wave construction includes scalar radiation pressure.

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
  const int dir=ivx-IVX;
  const int ivy=IVX+(dir+1)%3, ivz=IVX+(dir+2)%3;
  FLD *pfld=pmy_block->prfld;
  const bool coupled = pfld->pressure_coupling_mode
      != RadFLD::PressureCouplingMode::kOff;
  const bool radiation_pressure_in_flux = pfld->pressure_coupling_mode
      == RadFLD::PressureCouplingMode::kFlux;
  AthenaArray<Real> &radl=pfld->rad_face_l[dir];
  AthenaArray<Real> &radr=pfld->rad_face_r[dir];
  AthenaArray<Real> &radflux=pfld->u_rad_flux[dir];

  Real wli[NHYDRO],wri[NHYDRO],fi[NHYDRO],fl[NHYDRO],fr[NHYDRO];
  const Real gamma=GENERAL_EOS?std::nan(""):pmy_block->peos->GetGamma();
  const Real igm1=GENERAL_EOS?0.0:1.0/(gamma-1.0);

#pragma omp simd private(wli,wri,fi,fl,fr)
  for (int i=il; i<=iu; ++i) {
    wli[IDN]=wl(IDN,i); wli[IVX]=wl(ivx,i); wli[IVY]=wl(ivy,i);
    wli[IVZ]=wl(ivz,i); wli[IPR]=wl(IPR,i);
    wri[IDN]=wr(IDN,i); wri[IVX]=wr(ivx,i); wri[IVY]=wr(ivy,i);
    wri[IVZ]=wr(ivz,i); wri[IPR]=wr(IPR,i);

    const Real erl=std::max(radl(RadFLD::ERAD,k,j,i),TINY_NUMBER);
    const Real err=std::max(radr(RadFLD::ERAD,k,j,i),TINY_NUMBER);
    const Real laml=std::max(radl(RadFLD::LAMBDA,k,j,i),0.0);
    const Real lamr=std::max(radr(RadFLD::LAMBDA,k,j,i),0.0);
    const Real arl=std::max(radl(RadFLD::ARAD,k,j,i),0.0);
    const Real arr=std::max(radr(RadFLD::ARAD,k,j,i),0.0);
    const Real prl=coupled?laml*erl:0.0;
    const Real prr=coupled?lamr*err:0.0;
    const Real ptl=wli[IPR]+prl, ptr=wri[IPR]+prr;

    const Real eil=GENERAL_EOS?pmy_block->peos->EgasFromRhoP(wli[IDN],wli[IPR])
                              :wli[IPR]*igm1;
    const Real eir=GENERAL_EOS?pmy_block->peos->EgasFromRhoP(wri[IDN],wri[IPR])
                              :wri[IPR]*igm1;
    const Real kel=0.5*wli[IDN]*(SQR(wli[IVX])+SQR(wli[IVY])+SQR(wli[IVZ]));
    const Real ker=0.5*wri[IDN]*(SQR(wri[IVX])+SQR(wri[IVY])+SQR(wri[IVZ]));
    const Real egl=eil+kel, egr=eir+ker;
    // E_r is included only to split the resulting total-energy flux.  No
    // independent radiation star state is constructed.
    const Real etl=egl+(coupled?erl:0.0), etr=egr+(coupled?err:0.0);

    const Real cgl=pmy_block->peos->SoundSpeed(wli);
    const Real cgr=pmy_block->peos->SoundSpeed(wri);
    const Real cl=std::sqrt(SQR(cgl)+(coupled?laml*(erl+prl)/wli[IDN]:0.0));
    const Real cr=std::sqrt(SQR(cgr)+(coupled?lamr*(err+prr)/wri[IDN]:0.0));
    const Real rhoa=0.5*(wli[IDN]+wri[IDN]), ca=0.5*(cl+cr);
    const Real pmid=0.5*(ptl+ptr+(wli[IVX]-wri[IVX])*rhoa*ca);
    const Real umid=0.5*(wli[IVX]+wri[IVX]+(ptl-ptr)/(rhoa*ca));
    const Real rhol=wli[IDN]+(wli[IVX]-umid)*rhoa/ca;
    const Real rhor=wri[IDN]+(umid-wri[IVX])*rhoa/ca;

    Real ql,qr;
    if (GENERAL_EOS) {
      const Real gl=pmy_block->peos->AsqFromRhoP(rhol,wli[IPR])*rhol
                    /std::max(wli[IPR],TINY_NUMBER);
      const Real gr=pmy_block->peos->AsqFromRhoP(rhor,wri[IPR])*rhor
                    /std::max(wri[IPR],TINY_NUMBER);
      ql=(pmid<=ptl)?1.0:std::sqrt(1.0+(gl+1.0)/(2.0*gl)*(pmid/ptl-1.0));
      qr=(pmid<=ptr)?1.0:std::sqrt(1.0+(gr+1.0)/(2.0*gr)*(pmid/ptr-1.0));
    } else {
      ql=(pmid<=ptl)?1.0:std::sqrt(1.0+(gamma+1.0)/(2.0*gamma)
                                      *(pmid/ptl-1.0));
      qr=(pmid<=ptr)?1.0:std::sqrt(1.0+(gamma+1.0)/(2.0*gamma)
                                      *(pmid/ptr-1.0));
    }
    const Real al=wli[IVX]-cl*ql, ar=wri[IVX]+cr*qr;
    const Real bp=ar>0.0?ar:TINY_NUMBER, bm=al<0.0?al:-TINY_NUMBER;
    const Real vlbm=wli[IVX]-bm, vrbp=wri[IVX]-bp;
    const Real ml=wli[IDN]*(wli[IVX]-al), mr=-wri[IDN]*(wri[IVX]-ar);
    const Real tl=ptl+(wli[IVX]-al)*wli[IDN]*wli[IVX];
    const Real tr=ptr+(wri[IVX]-ar)*wri[IDN]*wri[IVX];
    const Real am=(tl-tr)/(ml+mr);
    const Real cp=std::max((ml*tr+mr*tl)/(ml+mr),0.0);

    fl[IDN]=wli[IDN]*vlbm; fr[IDN]=wri[IDN]*vrbp;
    fl[IVX]=wli[IDN]*wli[IVX]*vlbm+ptl;
    fr[IVX]=wri[IDN]*wri[IVX]*vrbp+ptr;
    fl[IVY]=wli[IDN]*wli[IVY]*vlbm; fr[IVY]=wri[IDN]*wri[IVY]*vrbp;
    fl[IVZ]=wli[IDN]*wli[IVZ]*vlbm; fr[IVZ]=wri[IDN]*wri[IVZ]*vrbp;
    fl[IEN]=etl*vlbm+ptl*wli[IVX];
    fr[IEN]=etr*vrbp+ptr*wri[IVX];
    Real a,b,c;
    if (am>=0.0) {a=am/(am-bm); b=0.0; c=-bm/(am-bm);}
    else {a=0.0; b=-am/(bp-am); c=bp/(bp-am);}
    fi[IDN]=a*fl[IDN]+b*fr[IDN];
    fi[IVX]=a*fl[IVX]+b*fr[IVX]+c*cp;
    fi[IVY]=a*fl[IVY]+b*fr[IVY];
    fi[IVZ]=a*fl[IVZ]+b*fr[IVZ];
    fi[IEN]=a*fl[IEN]+b*fr[IEN]+c*cp*am;

    const bool left=am>=0.0;
    const Real erg=left?erl:err;
    const Real lambdag=left?laml:lamr;
    const Real ag=left?arl:arr;
    const Real fer=ag*erg*am;
    radflux(k,j,i)=pfld->mixed_frame_transport ? fer : 0.0;
    pfld->rad_face_g[dir](k,j,i)=erg;
    if (coupled && !radiation_pressure_in_flux) {
      // Radiation pressure participated in the HLLC state construction, but
      // the returned gas momentum flux excludes it because the matching
      // -lambda grad(E_r) force is applied explicitly from Godunov faces.
      fi[IVX]-=lambdag*erg;
    }
    if (coupled) {
      fi[IEN]-=fer;
    }
    flx(IDN,k,j,i)=fi[IDN]; flx(ivx,k,j,i)=fi[IVX];
    flx(ivy,k,j,i)=fi[IVY]; flx(ivz,k,j,i)=fi[IVZ];
    flx(IEN,k,j,i)=fi[IEN];
    pmy_block->phydro->vf[dir](k,j,i)=am;
  }
}
