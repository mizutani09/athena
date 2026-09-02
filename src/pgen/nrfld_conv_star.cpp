//========================================================================================
//! \file nrfld_conv_star.cpp
//! \brief A three-dimensional composite modified-Lane-Emden RSG with NR-FLD.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fld/fld.hpp"
#include "../fld/opacity_table.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !NRMGFLD_ENABLED
#error "nrfld_conv_star requires -nrmgfld"
#endif

namespace {
constexpr Real kG = 6.67430e-8;
constexpr Real kKb = 1.380649e-16;
// Atomic mass unit: consistent with the R_gas convention used by the FLD/EOS code.
constexpr Real kMh = 1.66053906660e-24;
constexpr Real kAr = 7.5657e-15;

Real rho_unit, egas_unit, leng_unit, time_unit, vel_unit;
Real mu, gamma_gas, kappa_p, kappa_r;
Real rstar, mstar, luminosity, heat_radius, perturb_frac;
Real relax_end, relax_tau, heating_start, heating_ramp;
Real core_anchor_radius, core_anchor_width, core_anchor_tau;
Real atmosphere_anchor_radius, atmosphere_anchor_width, atmosphere_anchor_tau;
Real rho_floor, temp_floor, pgas_floor, erad_floor;
bool use_opacity_table;
bool discrete_hse_gravity;
bool use_profile_opacity;
bool core_anchor_thermodynamics;
constexpr int kHseAccel = 0;
constexpr int kHseInitialized = 0;
UserOpacityTable *opacity_table = nullptr;
std::vector<Real> vr, vrho, vptot, vtemp, vpgas, verad, vmenv, vgrav, vnpoly, vkappa;

Real Interp(const std::vector<Real> &v, Real r) {
  if (r <= vr.front()) return v.front();
  if (r >= vr.back()) return v.back();
  auto it = std::lower_bound(vr.begin(), vr.end(), r);
  int q = static_cast<int>(it-vr.begin());
  Real w = (r-vr[q-1])/(vr[q]-vr[q-1]);
  return v[q-1] + w*(v[q]-v[q-1]);
}

void ReadProfile(const std::string &name) {
  std::ifstream in(name.c_str());
  if (!in) {
    std::stringstream msg; msg << "Cannot open stellar profile " << name; ATHENA_ERROR(msg);
  }
  Real r, rho, pt, t, pg, er, me, grav, np, kap;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    if (ss >> r >> rho >> pt >> t >> pg >> er >> me >> grav >> np) {
      if (!(ss >> kap)) kap=-1.0;
      vr.push_back(r); vrho.push_back(rho); vptot.push_back(pt); vtemp.push_back(t);
      vpgas.push_back(pg); verad.push_back(er); vmenv.push_back(me);
      vgrav.push_back(grav); vnpoly.push_back(np);
      vkappa.push_back(kap);
    }
  }
  if (vr.size() < 16) {
    std::stringstream msg; msg << "Stellar profile has fewer than 16 rows"; ATHENA_ERROR(msg);
  }
}

Real ProfileDerivative(const std::vector<Real> &v, std::size_t i) {
  if (i==0) return (v[1]-v[0])/(vr[1]-vr[0]);
  if (i+1==v.size()) return (v[i]-v[i-1])/(vr[i]-vr[i-1]);
  const Real hm=vr[i]-vr[i-1], hp=vr[i+1]-vr[i];
  return -hp/(hm*(hm+hp))*v[i-1]
         +(hp-hm)/(hm*hp)*v[i]
         +hm/(hp*(hm+hp))*v[i+1];
}

void WriteHSEProfile(const std::string &name) {
  std::ofstream out(name.c_str());
  if (!out) {
    std::stringstream msg; msg << "Cannot write stellar HSE profile " << name;
    ATHENA_ERROR(msg);
  }
  out << "# Initial one-dimensional hydrostatic-equilibrium diagnostic (cgs)\n"
      << "# Positive force magnitudes satisfy rho_g = -dP/dr.\n"
      << "# rel_total=(rho_g+dPtot_dr)/(|rho_g|+|dPtot_dr|); "
         "rel_fld=(rho_g+dPfld_dr)/(|rho_g|+|dPfld_dr|).\n"
      << "# [1]=r_cm [2]=r_over_Rstar [3]=rho_g_cm3 [4]=Tgas_K [5]=Trad_K "
         "[6]=Pgas [7]=Prad [8]=Ptot [9]=g_cm_s2 [10]=rho_g "
         "[11]=minus_dPtot_dr [12]=minus_dPfld_dr [13]=rel_total "
         "[14]=rel_fld [15]=lambda [16]=kappa_R_cm2_g\n";
  out << std::scientific << std::setprecision(16);
  for (std::size_t i=0;i<vr.size();++i) {
    const Real dpg=ProfileDerivative(vpgas,i);
    const Real der=ProfileDerivative(verad,i);
    const Real dpt=ProfileDerivative(vptot,i);
    Real kr=kappa_r;
    if (use_profile_opacity && vkappa[i]>0.0) kr=vkappa[i];
    if (use_opacity_table) kr=opacity_table->GetOpacity(RadFLD::SIGMA_R,vrho[i],vtemp[i]);
    const Real ratio=std::abs(der)/(std::max(kr*vrho[i],TINY_NUMBER)
                                      *std::max(verad[i],TINY_NUMBER));
    const Real lambda=RadFLD::FluxLimiter(ratio,false);
    const Real dpfld=dpg+lambda*der;
    const Real rhog=vrho[i]*vgrav[i];
    const Real rel_total=(rhog+dpt)/(std::abs(rhog)+std::abs(dpt)+TINY_NUMBER);
    const Real rel_fld=(rhog+dpfld)/(std::abs(rhog)+std::abs(dpfld)+TINY_NUMBER);
    const Real trad=std::pow(std::max(verad[i]/kAr,TINY_NUMBER),0.25);
    out << vr[i] << ' ' << vr[i]/rstar << ' ' << vrho[i] << ' ' << vtemp[i] << ' '
        << trad << ' ' << vpgas[i] << ' ' << verad[i]/3.0 << ' ' << vptot[i] << ' '
        << vgrav[i] << ' ' << rhog << ' ' << -dpt << ' ' << -dpfld << ' '
        << rel_total << ' ' << rel_fld << ' ' << lambda << ' ' << kr << '\n';
  }
}

void StateAtRadius(Real r, Real &rho, Real &pgas, Real &erad, Real &temp,
                   Real &grav, Real &npoly) {
  if (r <= rstar) {
    rho=Interp(vrho,r); pgas=Interp(vpgas,r); erad=Interp(verad,r);
    temp=Interp(vtemp,r); grav=Interp(vgrav,r); npoly=Interp(vnpoly,r);
  } else {
    temp=temp_floor;
    Real expo=kG*mstar*mu*kMh/kKb/temp*(1.0/r-1.0/rstar);
    rho=std::max(rho_floor*std::exp(expo), rho_floor*1.0e-4);
    pgas=std::max(rho*kKb*temp/(mu*kMh), pgas_floor*1.0e-4);
    erad=erad_floor*SQR(rstar/r);
    grav=kG*mstar/(r*r);
    npoly=vnpoly.back();
  }
}

void EquilibriumAcceleration(MeshBlock *pmb, int k, int j, int i,
                             Real &ax, Real &ay, Real &az) {
  Real x=pmb->pcoord->x1v(i), y=pmb->pcoord->x2v(j), z=pmb->pcoord->x3v(k);
  Real dx=pmb->pcoord->dx1f(i), dy=pmb->pcoord->dx2f(j), dz=pmb->pcoord->dx3f(k);
  Real rho,pg,er,t,g,np;
  StateAtRadius(std::sqrt(x*x+y*y+z*z)*leng_unit,rho,pg,er,t,g,np);

  auto pressure_pair = [](Real xa, Real ya, Real za, Real xb, Real yb, Real zb,
                          Real &dpg, Real &der) {
    Real ra=std::sqrt(xa*xa+ya*ya+za*za)*leng_unit;
    Real rb=std::sqrt(xb*xb+yb*yb+zb*zb)*leng_unit;
    Real rhoa,pga,era,ta,ga,npa, rhob,pgb,erb,tb,gb,npb;
    StateAtRadius(ra,rhoa,pga,era,ta,ga,npa);
    StateAtRadius(rb,rhob,pgb,erb,tb,gb,npb);
    dpg=pgb-pga; der=erb-era;
  };

  Real dpgx,derx,dpgy,dery,dpgz,derz;
  pressure_pair(x-0.5*dx,y,z,x+0.5*dx,y,z,dpgx,derx);
  pressure_pair(x,y-0.5*dy,z,x,y+0.5*dy,z,dpgy,dery);
  pressure_pair(x,y,z-0.5*dz,x,y,z+0.5*dz,dpgz,derz);
  Real grad_er=std::sqrt(SQR(derx/(dx*leng_unit))+SQR(dery/(dy*leng_unit))
                         +SQR(derz/(dz*leng_unit)));
  Real opacity_density=std::max(rho,rho_floor*1.0e-2);
  Real kr=kappa_r;
  if (use_opacity_table) kr=opacity_table->GetOpacity(RadFLD::SIGMA_R,rho,t);
  Real flux_ratio=grad_er/(std::max(kr*opacity_density,TINY_NUMBER)
                           *std::max(er,TINY_NUMBER));
  Real lambda=RadFLD::FluxLimiter(flux_ratio,false);
  ax=(dpgx+lambda*derx)/(dx*leng_unit*rho)*time_unit*time_unit/leng_unit;
  ay=(dpgy+lambda*dery)/(dy*leng_unit*rho)*time_unit*time_unit/leng_unit;
  az=(dpgz+lambda*derz)/(dz*leng_unit*rho)*time_unit*time_unit/leng_unit;
}

void InitializeGodunovBalancedGravity(MeshBlock *pmb) {
  FLD *fld=pmb->prfld;
  Hydro *hyd=pmb->phydro;
  for (int k=pmb->ks;k<=pmb->ke;++k) for (int j=pmb->js;j<=pmb->je;++j)
    for (int i=pmb->is;i<=pmb->ie;++i) {
      Real idx=1.0/pmb->pcoord->dx1f(i);
      Real idy=1.0/pmb->pcoord->dx2f(j);
      Real idz=1.0/pmb->pcoord->dx3f(k);
      Real gx=0.5*(fld->u_rad(k,j,i+1)-fld->u_rad(k,j,i-1))*idx;
      Real gy=0.5*(fld->u_rad(k,j+1,i)-fld->u_rad(k,j-1,i))*idy;
      Real gz=0.5*(fld->u_rad(k+1,j,i)-fld->u_rad(k-1,j,i))*idz;
      Real ratio=std::sqrt(gx*gx+gy*gy+gz*gz)
                 /(std::max(fld->sigma_r(k,j,i),TINY_NUMBER)
                   *std::max(fld->u_rad(k,j,i),TINY_NUMBER));
      Real lambda=RadFLD::FluxLimiter(ratio,fld->fixed_flux_limiter);
      Real rho=std::max(hyd->w(IDN,k,j,i),TINY_NUMBER);
      Real divx=(hyd->flux[X1DIR](IM1,k,j,i+1)-hyd->flux[X1DIR](IM1,k,j,i))*idx;
      Real divy=(hyd->flux[X2DIR](IM2,k,j+1,i)-hyd->flux[X2DIR](IM2,k,j,i))*idy;
      Real divz=(hyd->flux[X3DIR](IM3,k+1,j,i)-hyd->flux[X3DIR](IM3,k,j,i))*idz;
      divx+=lambda*(fld->rad_face_g[X1DIR](k,j,i+1)-fld->rad_face_g[X1DIR](k,j,i))*idx;
      divy+=lambda*(fld->rad_face_g[X2DIR](k,j+1,i)-fld->rad_face_g[X2DIR](k,j,i))*idy;
      divz+=lambda*(fld->rad_face_g[X3DIR](k+1,j,i)-fld->rad_face_g[X3DIR](k,j,i))*idz;
      pmb->ruser_meshblock_data[kHseAccel](0,k,j,i)=divx/rho;
      pmb->ruser_meshblock_data[kHseAccel](1,k,j,i)=divy/rho;
      pmb->ruser_meshblock_data[kHseAccel](2,k,j,i)=divz/rho;
    }
  pmb->iuser_meshblock_data[kHseInitialized](0)=1;
}

void Opacity(MeshBlock *pmb, AthenaArray<Real> &u_fld, AthenaArray<Real> &prim) {
  (void)u_fld;
  int kl=pmb->ks-NGHOST, ku=pmb->ke+NGHOST;
  int jl=pmb->js-NGHOST, ju=pmb->je+NGHOST;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  for (int k=kl;k<=ku;++k) for (int j=jl;j<=ju;++j) for (int i=il;i<=iu;++i) {
    // Prevent the optically thin atmosphere from producing an extreme jump
    // in the implicit diffusion coefficient on the commissioning grid.
    Real rho=std::max(prim(IDN,k,j,i)*rho_unit, rho_floor*1.0e-2);
    Real p=std::max(prim(IPR,k,j,i)*egas_unit, pgas_floor*1.0e-4);
    Real temp=p*mu*kMh/(rho*kKb);
    Real kp=kappa_p, kr=kappa_r;
    if (use_profile_opacity && !vkappa.empty() && vkappa.front()>0.0) {
      Real radius=std::sqrt(SQR(pmb->pcoord->x1v(i))+SQR(pmb->pcoord->x2v(j))
                            +SQR(pmb->pcoord->x3v(k)))*leng_unit;
      kp=kr=Interp(vkappa,std::min(radius,rstar));
    }
    if (use_opacity_table) {
      kp=opacity_table->GetOpacity(RadFLD::SIGMA_P,rho,temp);
      kr=opacity_table->GetOpacity(RadFLD::SIGMA_R,rho,temp);
    }
    pmb->prfld->sigma_p(k,j,i)=std::max(kp*rho*leng_unit,TINY_NUMBER);
    pmb->prfld->sigma_r(k,j,i)=std::max(kr*rho*leng_unit,TINY_NUMBER);
  }
}

void GravityAndHeating(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
    AthenaArray<Real> &cons_scalar) {
  (void)time; (void)prim_scalar; (void)bcc; (void)cons_scalar;
  if (discrete_hse_gravity
      && pmb->iuser_meshblock_data[kHseInitialized](0)==0)
    InitializeGodunovBalancedGravity(pmb);
  // Radiation diffusion changes the pressure profile during the damped
  // commissioning phase. Re-freeze the gravity once, from that relaxed
  // finite-volume state, immediately before the damping is released.
  if (discrete_hse_gravity && relax_end > 0.0 && time >= relax_end
      && pmb->iuser_meshblock_data[kHseInitialized](0)==1) {
    InitializeGodunovBalancedGravity(pmb);
    pmb->iuser_meshblock_data[kHseInitialized](0)=2;
  }
  Real heat_weight=(heating_ramp > 0.0)
      ? std::max(0.0,std::min(1.0,(time-heating_start)/heating_ramp))
      : (time >= heating_start ? 1.0 : 0.0);
  Real qphys=(luminosity > 0.0)
      ? luminosity/(4.0*M_PI*heat_radius*heat_radius*heat_radius/3.0) : 0.0;
  Real qcode=heat_weight*qphys*time_unit/egas_unit;
  for (int k=pmb->ks;k<=pmb->ke;++k) for (int j=pmb->js;j<=pmb->je;++j)
    for (int i=pmb->is;i<=pmb->ie;++i) {
      Real x=pmb->pcoord->x1v(i), y=pmb->pcoord->x2v(j), z=pmb->pcoord->x3v(k);
      Real rc=std::sqrt(x*x+y*y+z*z), rp=rc*leng_unit;
      Real rho,pg,er,t,g,np; StateAtRadius(rp,rho,pg,er,t,g,np);
      Real acode=g*time_unit*time_unit/leng_unit;
      Real invr=1.0/std::max(rc,TINY_NUMBER);
      Real ax=-acode*x*invr, ay=-acode*y*invr, az=-acode*z*invr;
      if (discrete_hse_gravity) {
        ax=pmb->ruser_meshblock_data[kHseAccel](0,k,j,i);
        ay=pmb->ruser_meshblock_data[kHseAccel](1,k,j,i);
        az=pmb->ruser_meshblock_data[kHseAccel](2,k,j,i);
      }
      if (time < relax_end && relax_tau > 0.0) {
        ax-=prim(IVX,k,j,i)/relax_tau;
        ay-=prim(IVY,k,j,i)/relax_tau;
        az-=prim(IVZ,k,j,i)/relax_tau;
      }
      Real den=prim(IDN,k,j,i);
      cons(IM1,k,j,i)+=dt*den*ax; cons(IM2,k,j,i)+=dt*den*ay;
      cons(IM3,k,j,i)+=dt*den*az;
      if (NON_BAROTROPIC_EOS) {
        cons(IEN,k,j,i)+=dt*den*(ax*prim(IVX,k,j,i)+ay*prim(IVY,k,j,i)
                                                +az*prim(IVZ,k,j,i));
        if (rp < heat_radius) cons(IEN,k,j,i)+=dt*qcode;
      }
    }
}

Real HistoryMaxMach(MeshBlock *pmb, int iout) {
  (void)iout; Real ans=0.0; Real gamma=pmb->peos->GetGamma();
  for (int k=pmb->ks;k<=pmb->ke;++k) for (int j=pmb->js;j<=pmb->je;++j)
    for (int i=pmb->is;i<=pmb->ie;++i) {
      Real x=pmb->pcoord->x1v(i), y=pmb->pcoord->x2v(j), z=pmb->pcoord->x3v(k);
      Real r=std::sqrt(x*x+y*y+z*z)*leng_unit;
      Real rho=pmb->phydro->w(IDN,k,j,i), p=pmb->phydro->w(IPR,k,j,i);
      if (r >= 0.9*rstar || rho < 1.0e-4) continue;
      Real v2=SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVY,k,j,i))
             +SQR(pmb->phydro->w(IVZ,k,j,i));
      ans=std::max(ans,std::sqrt(v2/std::max(gamma*p/rho,TINY_NUMBER)));
    }
  return ans;
}
} // namespace

namespace {
void RadiationEscapeBoundary(Coordinates *pco, AthenaArray<Real> &erad,
                             int axis, bool inner, int is, int ie, int js, int je,
                             int ks, int ke, int ngh) {
  if (axis==1) {
    for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int n=1;n<=ngh;++n) {
      const int i=inner?is-n:ie+n;
      const Real rg=SQR(pco->x1v(i))+SQR(pco->x2v(j))+SQR(pco->x3v(k));
      erad(k,j,i)=erad_floor/egas_unit*SQR(rstar/leng_unit)/std::max(rg,TINY_NUMBER);
    }
  } else if (axis==2) {
    for (int k=ks;k<=ke;++k) for (int n=1;n<=ngh;++n) for (int i=is;i<=ie;++i) {
      const int j=inner?js-n:je+n;
      const Real rg=SQR(pco->x1v(i))+SQR(pco->x2v(j))+SQR(pco->x3v(k));
      erad(k,j,i)=erad_floor/egas_unit*SQR(rstar/leng_unit)/std::max(rg,TINY_NUMBER);
    }
  } else {
    for (int n=1;n<=ngh;++n) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
      const int k=inner?ks-n:ke+n;
      const Real rg=SQR(pco->x1v(i))+SQR(pco->x2v(j))+SQR(pco->x3v(k));
      erad(k,j,i)=erad_floor/egas_unit*SQR(rstar/leng_unit)/std::max(rg,TINY_NUMBER);
    }
  }
}

void GasCopyBoundary(AthenaArray<Real> &ugas, int axis, bool inner,
                     int is, int ie, int js, int je, int ks, int ke, int ngh) {
  if (axis==1) {
    const int ir=inner?is:ie;
    for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int n=1;n<=ngh;++n)
      ugas(k,j,inner?is-n:ie+n)=ugas(k,j,ir);
  } else if (axis==2) {
    const int jr=inner?js:je;
    for (int k=ks;k<=ke;++k) for (int n=1;n<=ngh;++n) for (int i=is;i<=ie;++i)
      ugas(k,inner?js-n:je+n,i)=ugas(k,jr,i);
  } else {
    const int kr=inner?ks:ke;
    for (int n=1;n<=ngh;++n) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i)
      ugas(inner?ks-n:ke+n,j,i)=ugas(kr,j,i);
  }
}
} // namespace

#define DEFINE_FLD_ESCAPE(NAME,AXIS,INNER) \
void NAME(MeshBlock*, Coordinates *pco, FLD*, const AthenaArray<Real>&, \
          AthenaArray<Real> &erad, Real, Real, int is,int ie,int js,int je,int ks,int ke,int ngh) { \
  RadiationEscapeBoundary(pco,erad,AXIS,INNER,is,ie,js,je,ks,ke,ngh); \
}
DEFINE_FLD_ESCAPE(StarFLDInnerX1,1,true)
DEFINE_FLD_ESCAPE(StarFLDOuterX1,1,false)
DEFINE_FLD_ESCAPE(StarFLDInnerX2,2,true)
DEFINE_FLD_ESCAPE(StarFLDOuterX2,2,false)
DEFINE_FLD_ESCAPE(StarFLDInnerX3,3,true)
DEFINE_FLD_ESCAPE(StarFLDOuterX3,3,false)
#undef DEFINE_FLD_ESCAPE

#define DEFINE_NR_ESCAPE(NAME,AXIS,INNER) \
void NAME(MeshBlock*, AthenaArray<Real> &erad, AthenaArray<Real> &ugas, Coordinates *pco, \
          const AthenaArray<Real>&, Real, Real, int is,int ie,int js,int je,int ks,int ke,int ngh) { \
  RadiationEscapeBoundary(pco,erad,AXIS,INNER,is,ie,js,je,ks,ke,ngh); \
  GasCopyBoundary(ugas,AXIS,INNER,is,ie,js,je,ks,ke,ngh); \
}
DEFINE_NR_ESCAPE(StarNRInnerX1,1,true)
DEFINE_NR_ESCAPE(StarNROuterX1,1,false)
DEFINE_NR_ESCAPE(StarNRInnerX2,2,true)
DEFINE_NR_ESCAPE(StarNROuterX2,2,false)
DEFINE_NR_ESCAPE(StarNRInnerX3,3,true)
DEFINE_NR_ESCAPE(StarNROuterX3,3,false)
#undef DEFINE_NR_ESCAPE

namespace {
void HydroOutflowBoundary(AthenaArray<Real> &w, int axis, bool inner,
                          int is,int ie,int js,int je,int ks,int ke,int ngh) {
  if (axis==1) {
    const int ir=inner?is:ie;
    for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int n=1;n<=ngh;++n) {
      const int i=inner?is-n:ie+n;
      for (int q=0;q<NHYDRO;++q) w(q,k,j,i)=w(q,k,j,ir);
      if ((inner && w(IVX,k,j,i)>0.0)||(!inner && w(IVX,k,j,i)<0.0)) w(IVX,k,j,i)=0.0;
    }
  } else if (axis==2) {
    const int jr=inner?js:je;
    for (int k=ks;k<=ke;++k) for (int n=1;n<=ngh;++n) for (int i=is;i<=ie;++i) {
      const int j=inner?js-n:je+n;
      for (int q=0;q<NHYDRO;++q) w(q,k,j,i)=w(q,k,jr,i);
      if ((inner && w(IVY,k,j,i)>0.0)||(!inner && w(IVY,k,j,i)<0.0)) w(IVY,k,j,i)=0.0;
    }
  } else {
    const int kr=inner?ks:ke;
    for (int n=1;n<=ngh;++n) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
      const int k=inner?ks-n:ke+n;
      for (int q=0;q<NHYDRO;++q) w(q,k,j,i)=w(q,kr,j,i);
      if ((inner && w(IVZ,k,j,i)>0.0)||(!inner && w(IVZ,k,j,i)<0.0)) w(IVZ,k,j,i)=0.0;
    }
  }
}
} // namespace

#define DEFINE_HYDRO_OUTFLOW(NAME,AXIS,INNER) \
void NAME(MeshBlock*, Coordinates*, AthenaArray<Real> &w, FaceField&, Real, Real, \
          int is,int ie,int js,int je,int ks,int ke,int ngh) { \
  HydroOutflowBoundary(w,AXIS,INNER,is,ie,js,je,ks,ke,ngh); \
}
DEFINE_HYDRO_OUTFLOW(StarHydroInnerX1,1,true)
DEFINE_HYDRO_OUTFLOW(StarHydroOuterX1,1,false)
DEFINE_HYDRO_OUTFLOW(StarHydroInnerX2,2,true)
DEFINE_HYDRO_OUTFLOW(StarHydroOuterX2,2,false)
DEFINE_HYDRO_OUTFLOW(StarHydroInnerX3,3,true)
DEFINE_HYDRO_OUTFLOW(StarHydroOuterX3,3,false)
#undef DEFINE_HYDRO_OUTFLOW

void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho_unit=pin->GetReal("hydro","rho_unit"); egas_unit=pin->GetReal("hydro","egas_unit");
  leng_unit=pin->GetReal("hydro","leng_unit"); mu=pin->GetOrAddReal("hydro","mu",0.62);
  gamma_gas=pin->GetReal("hydro","gamma"); vel_unit=std::sqrt(egas_unit/rho_unit);
  time_unit=leng_unit/vel_unit;
  ReadProfile(pin->GetString("problem","stellar_profile_file"));
  rstar=pin->GetOrAddReal("problem","rstar_cgs",vr.back());
  mstar=pin->GetReal("problem","mstar_cgs");
  luminosity=pin->GetOrAddReal("problem","luminosity_cgs",0.0);
  heat_radius=pin->GetOrAddReal("problem","heat_radius_fraction",0.2)*rstar;
  perturb_frac=pin->GetOrAddReal("problem","velocity_perturb_fraction",1.0e-3);
  relax_end=pin->GetOrAddReal("problem","relaxation_end",0.1);
  relax_tau=pin->GetOrAddReal("problem","relaxation_timescale",0.01);
  heating_start=pin->GetOrAddReal("problem","heating_start",relax_end);
  heating_ramp=pin->GetOrAddReal("problem","heating_ramp",0.1);
  core_anchor_radius=pin->GetOrAddReal("problem","core_anchor_radius_fraction",0.15)*rstar;
  core_anchor_width=pin->GetOrAddReal("problem","core_anchor_width_fraction",0.05)*rstar;
  core_anchor_tau=pin->GetOrAddReal("problem","core_anchor_timescale",0.01);
  core_anchor_thermodynamics=
      pin->GetOrAddBoolean("problem","core_anchor_thermodynamics",false);
  atmosphere_anchor_radius=
      pin->GetOrAddReal("problem","atmosphere_anchor_radius_fraction",1.05)*rstar;
  atmosphere_anchor_width=
      pin->GetOrAddReal("problem","atmosphere_anchor_width_fraction",0.10)*rstar;
  atmosphere_anchor_tau=
      pin->GetOrAddReal("problem","atmosphere_anchor_timescale",0.01);
  kappa_p=pin->GetOrAddReal("fld","const_opacity_P",0.01);
  kappa_r=pin->GetOrAddReal("fld","const_opacity_R",0.01);
  use_opacity_table=pin->GetOrAddBoolean("fld","use_opacity_table",false);
  use_profile_opacity=pin->GetOrAddBoolean("fld","use_profile_opacity",true);
  if (use_opacity_table) opacity_table=new UserOpacityTable(pin);
  rho_floor=vrho.back(); temp_floor=vtemp.back(); pgas_floor=vpgas.back(); erad_floor=verad.back();
  if (Globals::my_rank==0) {
    WriteHSEProfile(pin->GetOrAddString("problem","stellar_hse_profile_file",
                                       "stellar_hse_profile.dat"));
  }
  discrete_hse_gravity=
      pin->GetOrAddBoolean("problem","discrete_hse_gravity",true);
  EnrollUserExplicitSourceFunction(GravityAndHeating);
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0,HistoryMaxMach,"Mach_max",UserHistoryOperation::max);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  (void)pin; prfld->EnrollOpacityFunction(Opacity);
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[kHseAccel].NewAthenaArray(3,ncells3,ncells2,ncells1);
  ruser_meshblock_data[kHseAccel].ZeroClear();
  AllocateIntUserMeshBlockDataField(1);
  iuser_meshblock_data[kHseInitialized].NewAthenaArray(1);
  iuser_meshblock_data[kHseInitialized](0)=0;
  AllocateUserOutputVariables(6);
  SetUserOutputVariableName(0,"radius"); SetUserOutputVariableName(1,"Tgas");
  SetUserOutputVariableName(2,"Trad"); SetUserOutputVariableName(3,"Ptot");
  SetUserOutputVariableName(4,"grav"); SetUserOutputVariableName(5,"npoly");
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  (void)pin; Real igm1=1.0/(gamma_gas-1.0);
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    Real x=pcoord->x1v(i), y=pcoord->x2v(j), z=pcoord->x3v(k);
    Real r=std::sqrt(x*x+y*y+z*z), rho,pg,er,t,g,np;
    StateAtRadius(r*leng_unit,rho,pg,er,t,g,np);
    Real rhoc=rho/rho_unit, pgc=pg/egas_unit, erc=er/egas_unit;
    Real weight=0.0;
    if (r*leng_unit > 0.7*rstar && r*leng_unit < rstar)
      weight=std::sin(M_PI*(r*leng_unit/rstar-0.7)/0.3);
    Real phase=std::sin(12.9898*x+78.233*y+37.719*z+0.123*gid);
    Real cs=std::sqrt(gamma_gas*pgc/rhoc), vrad=perturb_frac*weight*phase*cs;
    Real invr=1.0/std::max(r,TINY_NUMBER);
    Real vx=vrad*x*invr, vy=vrad*y*invr, vz=vrad*z*invr;
    phydro->u(IDN,k,j,i)=rhoc; phydro->u(IM1,k,j,i)=rhoc*vx;
    phydro->u(IM2,k,j,i)=rhoc*vy; phydro->u(IM3,k,j,i)=rhoc*vz;
    phydro->u(IEN,k,j,i)=pgc*igm1+0.5*rhoc*(vx*vx+vy*vy+vz*vz);
    prfld->u_gas(k,j,i)=pgc*igm1; prfld->u_rad(k,j,i)=erc;
  }
}

void MeshBlock::UserWorkInLoop() {
  const Real gm1=peos->GetGamma()-1.0;
  const Real dt=pmy_mesh->dt;
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    const Real x=pcoord->x1v(i), y=pcoord->x2v(j), z=pcoord->x3v(k);
    const Real rp=std::sqrt(x*x+y*y+z*z)*leng_unit;
    Real weight=0.0;
    if (core_anchor_radius > 0.0 && core_anchor_tau > 0.0 && rp <= core_anchor_radius) {
      weight=1.0;
    } else if (core_anchor_radius > 0.0 && core_anchor_tau > 0.0
               && core_anchor_width > 0.0 && rp < core_anchor_radius+core_anchor_width) {
      const Real s=(rp-core_anchor_radius)/core_anchor_width;
      weight=0.5*(1.0+std::cos(M_PI*s));
    }
    if (weight > 0.0) {
      Real rho0,pg0,er0,t0,g0,np0;
      StateAtRadius(rp,rho0,pg0,er0,t0,g0,np0);
      rho0/=rho_unit; pg0/=egas_unit; er0/=egas_unit;
      const Real f=1.0-std::exp(-weight*dt/core_anchor_tau);
      const Real rho=phydro->u(IDN,k,j,i);
      const Real mx=phydro->u(IM1,k,j,i), my=phydro->u(IM2,k,j,i);
      const Real mz=phydro->u(IM3,k,j,i);
      const Real ek=0.5*(mx*mx+my*my+mz*mz)/std::max(rho,TINY_NUMBER);
      const Real eint=std::max(phydro->u(IEN,k,j,i)-ek,pg0/gm1*1.0e-8);
      phydro->u(IM1,k,j,i)=(1.0-f)*mx;
      phydro->u(IM2,k,j,i)=(1.0-f)*my;
      phydro->u(IM3,k,j,i)=(1.0-f)*mz;
      const Real eint_new=core_anchor_thermodynamics
          ? (1.0-f)*eint+f*pg0/gm1 : eint;
      if (core_anchor_thermodynamics) phydro->u(IDN,k,j,i)=(1.0-f)*rho+f*rho0;
      const Real rho_new=phydro->u(IDN,k,j,i);
      const Real ek_new=0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                           +SQR(phydro->u(IM3,k,j,i)))/std::max(rho_new,TINY_NUMBER);
      phydro->u(IEN,k,j,i)=eint_new+ek_new;
      if (core_anchor_thermodynamics) {
        prfld->u_gas(k,j,i)=eint_new;
        prfld->u_rad(k,j,i)=(1.0-f)*prfld->u_rad(k,j,i)+f*er0;
      }
    }
    if (atmosphere_anchor_radius > 0.0 && atmosphere_anchor_tau > 0.0
        && rp > atmosphere_anchor_radius) {
      Real w=1.0;
      if (atmosphere_anchor_width > 0.0
          && rp < atmosphere_anchor_radius+atmosphere_anchor_width) {
        const Real s=(rp-atmosphere_anchor_radius)/atmosphere_anchor_width;
        w=0.5*(1.0-std::cos(M_PI*s));
      }
      const Real f=1.0-std::exp(-w*dt/atmosphere_anchor_tau);
      const Real target=erad_floor/egas_unit*SQR(rstar/rp);
      prfld->u_rad(k,j,i)=(1.0-f)*prfld->u_rad(k,j,i)+f*target;
    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  (void)pin;
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    Real x=pcoord->x1v(i),y=pcoord->x2v(j),z=pcoord->x3v(k),r=std::sqrt(x*x+y*y+z*z);
    Real rho=phydro->w(IDN,k,j,i), pg=phydro->w(IPR,k,j,i);
    Real tg=pg*egas_unit*mu*kMh/(std::max(rho*rho_unit,TINY_NUMBER)*kKb);
    Real tr=std::pow(std::max(prfld->u_rad(k,j,i)*egas_unit/kAr,TINY_NUMBER),0.25);
    Real drho,dpg,der,dt,g,np; StateAtRadius(r*leng_unit,drho,dpg,der,dt,g,np);
    user_out_var(0,k,j,i)=r; user_out_var(1,k,j,i)=tg; user_out_var(2,k,j,i)=tr;
    user_out_var(3,k,j,i)=pg+prfld->u_rad(k,j,i)/3.0;
    user_out_var(4,k,j,i)=g*time_unit*time_unit/leng_unit; user_out_var(5,k,j,i)=np;
  }
}
