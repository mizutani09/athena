//========================================================================================
//! \file nrfld_star_in_box.cpp
//! \brief Freytag-style fixed-potential RSG commissioning problem using existing NR-FLD.
//! This ideal-gas/constant-opacity baseline is NOT a CO5BOLD reproduction.
//! See doc/freytag_star_in_box.md for references, assumptions and validation gates.
//========================================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
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
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if !NRMGFLD_ENABLED || MAGNETIC_FIELDS_ENABLED || !NON_BAROTROPIC_EOS
#error "nrfld_star_in_box requires -nrmgfld, an energy EOS, and no MHD"
#endif
#if defined(RSG_PRODUCTION) && !GENERAL_EOS
#error "rsg_star_in_box requires --eos=general/eos_table"
#endif

namespace {
constexpr Real kG=6.67430e-8, kMsun=1.98847e33, kRsun=6.957e10;
constexpr Real kLsun=3.828e33, kAr=7.5657e-15, kRgas=8.3144621e7;
Real rho_unit, energy_unit, length_unit, time_unit, velocity_unit;
Real gm, radius, rcore, router, rheat, drag_tau, gamma_gas, mu, tsurface;
Real index_poly, perturb, rho_surface, heat_norm, lum_code, boundary_er, bc_scale;
Real dfloor, pfloor, kp_const, kr_const;
Real drag_taper_start, drag_taper_end, grid_mass_tolerance;
Real marshak_alpha;
std::uint64_t perturb_seed;
bool profile_initialization=false, marshak_boundary=false;
std::vector<Real> profile_r, profile_rho, profile_egas, profile_erad, profile_temp;
UserOpacityTable *opacity_table=nullptr;

void Require(bool valid, const std::string &message) {
  if (!valid) {
    std::stringstream msg; msg << "nrfld_star_in_box: " << message;
    ATHENA_ERROR(msg);
  }
}
Real Positive(ParameterInput *pin, const char *block, const char *name) {
  const Real v=pin->GetReal(block,name);
  Require(std::isfinite(v) && v>0.0, std::string(block)+"/"+name+" must be positive");
  return v;
}
// Freytag et al. (2012), Eq. 41. All arguments and results are in code units.
Real Potential(Real r) {
  const Real s=std::pow(r/router,8);
  const Real a=std::pow(rcore,4)+std::pow(r,4)/std::sqrt(1.0+s);
  return -gm/std::pow(a,0.25);
}
Real GravityMagnitude(Real r) {
  const Real s=std::pow(r/router,8);
  const Real a=std::pow(rcore,4)+std::pow(r,4)/std::sqrt(1.0+s);
  return gm*r*r*r/(std::pow(1.0+s,1.5)*std::pow(a,1.25));
}
Real Temperature(Real r) {
  const Real gas_constant=kRgas/mu/(velocity_unit*velocity_unit);
  return tsurface+std::max(Potential(radius)-Potential(r),0.0)
                         /((index_poly+1.0)*gas_constant);
}
Real DensityShape(Real r) {
  if (r<=radius) return std::pow(Temperature(r)/tsurface,index_poly);
  const Real ciso2=kRgas*tsurface/mu/(velocity_unit*velocity_unit);
  return std::exp(-(Potential(r)-Potential(radius))/ciso2);
}
Real CoreWeight(Real r) {
  if (r>=rheat) return 0.0;
  return SQR(1.0-SQR(r/rheat));
}
Real RadialPosition(Coordinates *c, int k, int j, int i) {
  return std::sqrt(SQR(c->x1v(i))+SQR(c->x2v(j))+SQR(c->x3v(k)));
}
void LoadHseProfile(const std::string &filename) {
  std::ifstream stream(filename);
  Require(stream.good(), "cannot open problem/hse_model_file="+filename);
  profile_r.clear(); profile_rho.clear(); profile_egas.clear();
  profile_erad.clear(); profile_temp.clear();
  std::string line;
  while (std::getline(stream,line)) {
    if (line.empty() || line[0]=='#') continue;
    std::stringstream row(line);
    Real r,d,e,er,t;
    if (!(row>>r>>d>>e>>er>>t)) continue;
    Require(r>=0.0 && d>0.0 && e>0.0 && er>0.0 && t>0.0,
            "HSE profile values must be finite and positive");
    Require(profile_r.empty() || r>profile_r.back(),
            "HSE profile radii must be strictly increasing");
    profile_r.push_back(r/length_unit);
    profile_rho.push_back(d/rho_unit);
    profile_egas.push_back(e/energy_unit);
    profile_erad.push_back(er/energy_unit);
    profile_temp.push_back(t);
  }
  Require(profile_r.size()>=4,"HSE profile needs at least four rows");
}
Real InterpolateProfile(const std::vector<Real> &values, Real r) {
  if (r<=profile_r.front()) return values.front();
  if (r>=profile_r.back()) return values.back();
  auto upper=std::upper_bound(profile_r.begin(),profile_r.end(),r);
  const std::size_t q=static_cast<std::size_t>(upper-profile_r.begin()-1);
  const Real f=(r-profile_r[q])/(profile_r[q+1]-profile_r[q]);
  return std::exp((1.0-f)*std::log(values[q])+f*std::log(values[q+1]));
}
Real InitialDensity(Real r) {
  return profile_initialization ? InterpolateProfile(profile_rho,r)
                                : rho_surface*DensityShape(r);
}
Real Kinetic(const AthenaArray<Real> &u,int k,int j,int i) {
  return 0.5*(SQR(u(IM1,k,j,i))+SQR(u(IM2,k,j,i))+SQR(u(IM3,k,j,i)))
             /u(IDN,k,j,i);
}
void Opacity(MeshBlock *mb,AthenaArray<Real>&,AthenaArray<Real> &w) {
  for (int k=mb->ks-NGHOST;k<=mb->ke+NGHOST;++k)
    for (int j=mb->js-NGHOST;j<=mb->je+NGHOST;++j)
      for (int i=mb->is-NGHOST;i<=mb->ie+NGHOST;++i) {
        const Real d=w(IDN,k,j,i)*rho_unit;
#if GENERAL_EOS
        const Real eg=mb->peos->EgasFromRhoP(w(IDN,k,j,i),w(IPR,k,j,i));
        const Real t=mb->peos->TempFromRhoEg(w(IDN,k,j,i),eg)
                    *mb->peos->ptable->tUnit;
#else
        const Real t=w(IPR,k,j,i)*energy_unit/d*mu/kRgas;
#endif
        Real kp=kp_const,kr=kr_const;
        if (opacity_table) {
          try {
            kp=opacity_table->GetOpacity(RadFLD::SIGMA_P,d,t);
            kr=opacity_table->GetOpacity(RadFLD::SIGMA_R,d,t);
          } catch (const std::runtime_error &error) {
            std::stringstream msg;
            const int ia=std::max(mb->is,std::min(mb->ie,i));
            const int ja=std::max(mb->js,std::min(mb->je,j));
            const int ka=std::max(mb->ks,std::min(mb->ke,k));
            msg << error.what() << "\nRSG opacity state: gid=" << mb->gid
                << " (k,j,i)=(" << k << "," << j << "," << i << ")"
                << " x=(" << mb->pcoord->x1v(i) << ","
                << mb->pcoord->x2v(j) << "," << mb->pcoord->x3v(k) << ")"
                << " rho_code=" << w(IDN,k,j,i)
                << " pressure_code=" << w(IPR,k,j,i)
                << " temperature_K=" << t
                << " velocity=(" << w(IVX,k,j,i) << ","
                << w(IVY,k,j,i) << "," << w(IVZ,k,j,i) << ")"
                << " nearest_active=(" << ka << "," << ja << "," << ia
                << ") active_rho_code=" << w(IDN,ka,ja,ia)
                << " active_pressure_code=" << w(IPR,ka,ja,ia)
                << " active_velocity=(" << w(IVX,ka,ja,ia) << ","
                << w(IVY,ka,ja,ia) << "," << w(IVZ,ka,ja,ia) << ").";
            ATHENA_ERROR(msg);
          }
        }
        Require(std::isfinite(kp) && kp>=0.0 && std::isfinite(kr) && kr>0.0,
                "opacity must have finite kappa_P>=0 and kappa_R>0");
        mb->prfld->sigma_p(k,j,i)=kp*d*length_unit;
        mb->prfld->sigma_r(k,j,i)=kr*d*length_unit;
      }
}
void GravityHeating(MeshBlock *mb,Real,Real dt,const AthenaArray<Real> &w,
                    const AthenaArray<Real>&,const AthenaArray<Real>&,
                    AthenaArray<Real> &u,AthenaArray<Real>&) {
  for (int k=mb->ks;k<=mb->ke;++k) for (int j=mb->js;j<=mb->je;++j)
    for (int i=mb->is;i<=mb->ie;++i) {
      const Real r=RadialPosition(mb->pcoord,k,j,i);
      const Real factor=(r>0.0) ? -GravityMagnitude(r)/r : 0.0;
      const Real ax=factor*mb->pcoord->x1v(i),ay=factor*mb->pcoord->x2v(j);
      const Real az=factor*mb->pcoord->x3v(k),d=w(IDN,k,j,i);
      u(IM1,k,j,i)+=dt*d*ax; u(IM2,k,j,i)+=dt*d*ay; u(IM3,k,j,i)+=dt*d*az;
      u(IEN,k,j,i)+=dt*(d*(ax*w(IVX,k,j,i)+ay*w(IVY,k,j,i)+az*w(IVZ,k,j,i))
                                  +lum_code*CoreWeight(r)/heat_norm);
    }
}
// Inward/outward velocities are both allowed, as in the CO5BOLD open boundary.
// Fixed specific internal energy; exponential density in the external potential.
void HydroBoundary(MeshBlock *mb,Coordinates *c,AthenaArray<Real> &w,int axis,bool inner,
                   int is,int ie,int js,int je,int ks,int ke,int ngh) {
  int lo[3]={is,js,ks},hi[3]={ie,je,ke};
  const int ref=inner?lo[axis]:hi[axis];
  lo[axis]=inner?ref-ngh:ref+1; hi[axis]=inner?ref-1:ref+ngh;
  for (int k=lo[2];k<=hi[2];++k) for (int j=lo[1];j<=hi[1];++j)
    for (int i=lo[0];i<=hi[0];++i) {
      int q[3]={i,j,k}; q[axis]=ref;
      const Real d=w(IDN,q[2],q[1],q[0]),p=w(IPR,q[2],q[1],q[0]);
      const Real dp=Potential(RadialPosition(c,k,j,i))
                   -Potential(RadialPosition(c,q[2],q[1],q[0]));
      const Real f=std::exp(std::max(-50.0,std::min(50.0,-dp*d/(bc_scale*p))));
      for (int n=0;n<NHYDRO;++n) w(n,k,j,i)=w(n,q[2],q[1],q[0]);
      w(IDN,k,j,i)=std::max(dfloor,d*f);
#if GENERAL_EOS
      const Real espec=mb->peos->EgasFromRhoP(d,p)/d;
      w(IPR,k,j,i)=std::max(pfloor,
          mb->peos->PresFromRhoEg(w(IDN,k,j,i),w(IDN,k,j,i)*espec));
#else
      w(IPR,k,j,i)=std::max(pfloor,p*w(IDN,k,j,i)/d);
#endif
    }
}
// The production Marshak condition is imposed directly as a face flux in
// NRFLD.  Its ghost state is zero-gradient and is used only to evaluate the
// lagged face diffusion coefficient.  The old fixed bath remains available
// explicitly for solver-regression comparisons.
void RadiationBoundary(AthenaArray<Real> &er,AthenaArray<Real> *eg,int axis,
                       bool inner,int is,int ie,int js,int je,int ks,int ke,int ngh) {
  int lo[3]={is,js,ks},hi[3]={ie,je,ke};
  const int ref=inner?lo[axis]:hi[axis];
  lo[axis]=inner?ref-ngh:ref+1; hi[axis]=inner?ref-1:ref+ngh;
  for (int k=lo[2];k<=hi[2];++k) for (int j=lo[1];j<=hi[1];++j)
    for (int i=lo[0];i<=hi[0];++i) {
      int q[3]={i,j,k};q[axis]=ref;
      er(k,j,i)=marshak_boundary ? er(q[2],q[1],q[0]) : boundary_er;
      if (eg) (*eg)(k,j,i)=(*eg)(q[2],q[1],q[0]);
    }
}
// The Newton correction must be ZERO in ghost centres. The built-in MG
// "zerofixed" is antisymmetric about the face, so it is NOT this derivative.
void CorrectionBoundary(AthenaArray<Real> &u,int nv,int axis,bool inner,
                        int is,int ie,int js,int je,int ks,int ke,int ngh) {
  int lo[3]={is,js,ks},hi[3]={ie,je,ke};
  const int ref=inner?lo[axis]:hi[axis];
  lo[axis]=inner?ref-ngh:ref+1;hi[axis]=inner?ref-1:ref+ngh;
  for (int n=0;n<nv;++n) for (int k=lo[2];k<=hi[2];++k)
    for (int j=lo[1];j<=hi[1];++j) for (int i=lo[0];i<=hi[0];++i) u(n,k,j,i)=0.0;
}
Real History(MeshBlock *mb,int out) {
  if (out>=7) return mb->ruser_meshblock_data[0](out-7);
  Real sum=0.0;
  for (int k=mb->ks;k<=mb->ke;++k) for (int j=mb->js;j<=mb->je;++j)
    for (int i=mb->is;i<=mb->ie;++i) {
      const auto &u=mb->phydro->u;
      const Real r=RadialPosition(mb->pcoord,k,j,i),d=u(IDN,k,j,i);
      const Real ek=Kinetic(u,k,j,i),eg=u(IEN,k,j,i)-ek;
      Real val=0.0;
      if (out==0) val=mb->prfld->u_rad(k,j,i);
      if (out==1) val=eg;
      if (out==2) val=d*Potential(r);
      if (out==3) val=(r>0.0) ? (u(IM1,k,j,i)*mb->pcoord->x1v(i)
              +u(IM2,k,j,i)*mb->pcoord->x2v(j)+u(IM3,k,j,i)*mb->pcoord->x3v(k))/r : 0.0;
      if (out==4) val=d*r*r;
      if (out==5) val=lum_code*CoreWeight(r)/heat_norm;
      if (out==6) {
        const Real p=mb->phydro->w(IPR,k,j,i);
        const Real asq=mb->peos->AsqFromRhoP(d,p);
        sum=std::max(sum,std::sqrt(2.0*ek/d/std::max(asq,TINY_NUMBER)));
        continue;
      }
      sum+=val*mb->pcoord->GetCellVolume(k,j,i);
    }
  return sum;
}
}  // namespace

#define FACE_CALLBACKS(TAG,AXIS,INNER) \
void Hydro##TAG(MeshBlock *mb,Coordinates *c,AthenaArray<Real> &w,FaceField&,Real,Real, \
    int is,int ie,int js,int je,int ks,int ke,int ngh) { \
  HydroBoundary(mb,c,w,AXIS,INNER,is,ie,js,je,ks,ke,ngh); } \
void FLD##TAG(MeshBlock*,Coordinates*,FLD*,const AthenaArray<Real>&, \
    AthenaArray<Real> &er,Real,Real,int is,int ie,int js,int je,int ks,int ke,int ngh) { \
  RadiationBoundary(er,nullptr,AXIS,INNER,is,ie,js,je,ks,ke,ngh); } \
void NR##TAG(MeshBlock*,AthenaArray<Real> &er,AthenaArray<Real> &eg,Coordinates*, \
    const AthenaArray<Real>&,Real,Real,int is,int ie,int js,int je,int ks,int ke,int ngh) { \
  RadiationBoundary(er,&eg,AXIS,INNER,is,ie,js,je,ks,ke,ngh); } \
void MG##TAG(AthenaArray<Real> &u,Real,int nv,int is,int ie,int js,int je,int ks,int ke, \
    int ngh,const MGCoordinates&) { \
  CorrectionBoundary(u,nv,AXIS,INNER,is,ie,js,je,ks,ke,ngh); }
FACE_CALLBACKS(IX1,0,true)
FACE_CALLBACKS(OX1,0,false)
FACE_CALLBACKS(IX2,1,true)
FACE_CALLBACKS(OX2,1,false)
FACE_CALLBACKS(IX3,2,true)
FACE_CALLBACKS(OX3,2,false)
#undef FACE_CALLBACKS

void Mesh::InitUserMeshData(ParameterInput *pin) {
  Require(std::string(COORDINATE_SYSTEM)=="cartesian" && f2 && f3 && !multilevel,
          "use a uniform 3D Cartesian mesh; AMR/SMR normalization is not implemented");
  rho_unit=Positive(pin,"hydro","rho_unit");
  energy_unit=Positive(pin,"hydro","egas_unit");
  length_unit=Positive(pin,"hydro","leng_unit");
  velocity_unit=std::sqrt(energy_unit/rho_unit);time_unit=length_unit/velocity_unit;
  gamma_gas=Positive(pin,"hydro","gamma");mu=Positive(pin,"hydro","mu");
  Require(gamma_gas>1.0,"gamma must exceed one");
  dfloor=Positive(pin,"hydro","dfloor");pfloor=Positive(pin,"hydro","pfloor");
  radius=Positive(pin,"problem","radius_rsun")*kRsun/length_unit;
  gm=kG*Positive(pin,"problem","potential_mass_msun")*kMsun
     /(length_unit*SQR(velocity_unit));
  const Real gas_mass=Positive(pin,"problem","gas_mass_msun")*kMsun
                     /(rho_unit*std::pow(length_unit,3));
  rcore=Positive(pin,"problem","core_radius_fraction")*radius;
  router=Positive(pin,"problem","outer_flattening_fraction")*radius;
  rheat=Positive(pin,"problem","heating_radius_fraction")*radius;
  Require(rheat<=rcore && rcore<radius && router>radius,"require rheat<=rcore<R<router");
  tsurface=Positive(pin,"problem","surface_temperature");
  index_poly=Positive(pin,"problem","polytropic_index");
  bc_scale=Positive(pin,"problem","boundary_scale_height_factor");
  const std::string radiation_boundary = pin->GetOrAddString(
      "problem","radiation_boundary","fixed_bath");
  Require(radiation_boundary=="fixed_bath" || radiation_boundary=="marshak_vacuum",
          "problem/radiation_boundary must be fixed_bath or marshak_vacuum");
  marshak_boundary=(radiation_boundary=="marshak_vacuum");
  const Real external_temperature=pin->GetOrAddReal(
      "problem","radiation_external_temperature",
      pin->GetOrAddReal("problem","radiation_bath_temperature",10.0));
  Require(std::isfinite(external_temperature) && external_temperature>=0.0,
          "problem/radiation_external_temperature must be finite and nonnegative");
  boundary_er=kAr*std::pow(external_temperature,4)/energy_unit;
  marshak_alpha=pin->GetOrAddReal("problem","marshak_alpha",0.5);
  Require(std::isfinite(marshak_alpha) && marshak_alpha>0.0,
          "problem/marshak_alpha must be finite and positive");
  drag_tau=pin->GetOrAddReal("problem","core_drag_timescale",-1.0);
  const Real drag_tau_tdyn=pin->GetOrAddReal("problem","drag_tau_tdyn",-1.0);
  const Real tdyn=std::sqrt(radius*radius*radius/gm);
  if (drag_tau<0.0 && drag_tau_tdyn>=0.0) {
    drag_tau=drag_tau_tdyn*tdyn;
  }
  drag_taper_start=pin->GetOrAddReal("problem","drag_taper_start",-1.0);
  drag_taper_end=pin->GetOrAddReal("problem","drag_taper_end",-1.0);
  const Real taper_start_tdyn=pin->GetOrAddReal("problem","drag_taper_start_tdyn",-1.0);
  const Real taper_end_tdyn=pin->GetOrAddReal("problem","drag_taper_end_tdyn",-1.0);
  if (taper_start_tdyn>=0.0 || taper_end_tdyn>=0.0) {
    Require(drag_taper_start<0.0 && drag_taper_end<0.0,
            "choose drag taper in code time or dynamical-time units, not both");
    drag_taper_start=taper_start_tdyn*tdyn;
    drag_taper_end=taper_end_tdyn*tdyn;
  }
  Require((drag_taper_start<0.0 && drag_taper_end<0.0)
          || (drag_taper_start>=0.0 && drag_taper_end>drag_taper_start),
          "drag taper must be disabled or satisfy 0<=start<end");
  perturb=pin->GetReal("problem","velocity_perturb_fraction");
  const int seed=pin->GetOrAddInteger("problem","perturb_seed",350307);
  Require(seed>=0,"perturb_seed must be nonnegative");
  perturb_seed=static_cast<std::uint64_t>(seed);
  grid_mass_tolerance=pin->GetOrAddReal("problem","grid_mass_tolerance",5.0e-3);
  Require(grid_mass_tolerance>0.0,"grid_mass_tolerance must be positive");
  const Real lum=pin->GetReal("problem","luminosity_lsun");
  kp_const=pin->GetReal("fld","const_opacity_P");
  kr_const=Positive(pin,"fld","const_opacity_R");
  Require(std::isfinite(drag_tau) && drag_tau>=0.0 && std::isfinite(perturb) && perturb>=0.0
          && std::isfinite(lum) && lum>=0.0 && std::isfinite(kp_const) && kp_const>=0.0,
          "drag timescale, perturbation, luminosity and Planck opacity must be nonnegative");
  lum_code=lum*kLsun*time_unit/(energy_unit*std::pow(length_unit,3));
  const bool radiation_force=pin->GetOrAddBoolean("fld","include_radiation_force",false);
  if (pin->GetOrAddBoolean("fld","use_opacity_table",false))
    opacity_table=new UserOpacityTable(pin);

  const std::string profile_file=pin->GetOrAddString("problem","hse_model_file","");
  profile_initialization=!profile_file.empty();
  if (profile_initialization) {
    LoadHseProfile(profile_file);
    rho_surface=InterpolateProfile(profile_rho,radius);
  }
  Require(!radiation_force || profile_initialization,
          "radiation force requires an imported profile including radiation support");
#if defined(RSG_PRODUCTION)
  Require(profile_initialization,"production RSG requires problem/hse_model_file");
  Require(opacity_table!=nullptr,"production RSG requires fld/use_opacity_table=true");
  Require(!pin->GetOrAddBoolean("hydro","eos_table_clamp",false),
          "production RSG forbids hydro/eos_table_clamp");
  Require(radiation_force,
          "production material-only RSG requires fld/include_radiation_force=true");
  Require(marshak_boundary,
          "production RSG requires problem/radiation_boundary=marshak_vacuum");
#endif
#if GENERAL_EOS
  Require(profile_initialization,"general EOS initialization requires an HSE profile");
#endif

  const char *names[6]={"ix1_bc","ox1_bc","ix2_bc","ox2_bc","ix3_bc","ox3_bc"};
  BValFunc hydro[6]={HydroIX1,HydroOX1,HydroIX2,HydroOX2,HydroIX3,HydroOX3};
  FLDBoundaryFunc fld[6]={FLDIX1,FLDOX1,FLDIX2,FLDOX2,FLDIX3,FLDOX3};
  NRBoundaryFunc nr[6]={NRIX1,NROX1,NRIX2,NROX2,NRIX3,NROX3};
  MGBoundaryFunc mg[6]={MGIX1,MGOX1,MGIX2,MGOX2,MGIX3,MGOX3};
  for (int q=0;q<6;++q) {
    Require(pin->GetString("mesh",names[q])=="user"
            && pin->GetString("nrfld",names[q])=="user"
            && pin->GetString("nrfld",std::string("linearMG_")+names[q])=="user",
            "all hydro, NR and linearMG boundaries must be user");
    auto face=static_cast<BoundaryFace>(q);
    EnrollUserBoundaryFunction(face,hydro[q]);EnrollUserFLDBoundaryFunction(face,fld[q]);
    EnrollUserNRBoundaryFunction(face,nr[q]);EnrollUserLinearMGBoundaryFunction(face,mg[q]);
  }
  // Root-grid quadrature, once on rank zero. Includes all atmosphere and corners.
  // Independent of block decomposition and reproducible on restart (same mesh).
  const Real lo[3]={mesh_size.x1min,mesh_size.x2min,mesh_size.x3min};
  const Real hi[3]={mesh_size.x1max,mesh_size.x2max,mesh_size.x3max};
  const int nx[3]={mesh_size.nx1,mesh_size.nx2,mesh_size.nx3};
  Real dx[3];
  for (int a=0;a<3;++a) {
    dx[a]=(hi[a]-lo[a])/nx[a];
    Require(lo[a]<-radius && hi[a]>radius,"box must contain the whole star");
    Require(pin->GetOrAddReal("mesh","x"+std::to_string(a+1)+"rat",1.0)==1.0,
            "stretched mesh is unsupported");
    Require(std::abs(dx[a]/dx[0]-1.0)<1e-12,"NR-FLD requires cubic cells");
  }
  Real norm[2]={0.0,0.0};
  if (Globals::my_rank==0) {
    for (int k=0;k<nx[2];++k) for (int j=0;j<nx[1];++j) for (int i=0;i<nx[0];++i) {
      const Real r=std::sqrt(SQR(lo[0]+(i+0.5)*dx[0])+SQR(lo[1]+(j+0.5)*dx[1])
                             +SQR(lo[2]+(k+0.5)*dx[2]));
      norm[0]+=(profile_initialization ? InterpolateProfile(profile_rho,r)
                                      : DensityShape(r))*dx[0]*dx[1]*dx[2];
      norm[1]+=CoreWeight(r)*dx[0]*dx[1]*dx[2];
    }
  }
#ifdef MPI_PARALLEL
  MPI_Bcast(norm,2,MPI_ATHENA_REAL,0,MPI_COMM_WORLD);
#endif
  Require(norm[0]>0.0 && norm[1]>0.0,"empty mass or heating quadrature; resolve the core");
  if (profile_initialization) {
    const Real rel=std::abs(norm[0]/gas_mass-1.0);
    Require(rel<=grid_mass_tolerance,
            "mapped profile mass misses gas_mass_msun; regenerate the 1D profile for this grid");
  } else {
    rho_surface=gas_mass/norm[0];
  }
  heat_norm=norm[1];
  EnrollUserExplicitSourceFunction(GravityHeating);
  AllocateUserHistoryOutput(12);
  const char *hist[12]={"Erad","Egas","Epotential","radial_momentum","mass_r2",
      "Lheat","Mach_max","Edrag_removed","Pdrag_x_removed","Pdrag_y_removed",
      "Pdrag_z_removed","drag_power_removed"};
  for (int q=0;q<12;++q) EnrollUserHistoryOutput(q,History,hist[q],
      q==6 ? UserHistoryOperation::max : UserHistoryOperation::sum);
  if (Globals::my_rank==0) std::cout << std::setprecision(16)
      << "STAR_IN_BOX mode=" << (profile_initialization ? "tabular-profile" : "commissioning-polytrope")
      << ", fixed Freytag potential\n"
      << "STAR_IN_BOX rho_surface=" << rho_surface << " heat_norm=" << heat_norm
      << " gas_mass=" << gas_mass << " Lheat=" << lum_code
      << " time_unit_s=" << time_unit << " core_cells=" << 2.0*rcore/dx[0] << '\n';
}

void MeshBlock::InitUserMeshBlockData(ParameterInput*) {
  prfld->EnrollOpacityFunction(Opacity);
  if (marshak_boundary) {
    for (int face=0; face<6; ++face) prfld->marshak_face_boundary[face]=true;
    prfld->marshak_top_alpha=marshak_alpha;
    prfld->marshak_top_erad_ext=boundary_er;
  }
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(5);
  for (int q=0;q<5;++q) ruser_meshblock_data[0](q)=0.0;
  AllocateUserOutputVariables(7);
  const char *names[7]={"Tgas","Trad","Er","phi","g","vr","heat"};
  for (int q=0;q<7;++q) SetUserOutputVariableName(q,names[q]);
}
void MeshBlock::ProblemGenerator(ParameterInput*) {
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    const Real r=RadialPosition(pcoord,k,j,i);
    const Real d_profile=InitialDensity(r);
    const Real floor_tolerance=64.0*std::numeric_limits<Real>::epsilon();
    if (!std::isfinite(d_profile)
        || d_profile < dfloor*(1.0-floor_tolerance)) {
      std::stringstream message;
      message << "initial density is materially below hydro/dfloor at gid=" << gid
              << " (k,j,i)=(" << k << "," << j << "," << i << ")"
              << ", r=" << r << ", rho=" << d_profile
              << ", dfloor=" << dfloor;
      Require(false,message.str());
    }
    // A log interpolation of a profile plateau that is exactly equal to the
    // configured floor can land one ULP below it.  Accept only that roundoff-
    // sized discrepancy and preserve specific internal energy when snapping
    // density back to the exact numerical bound.
    const Real d=std::max(d_profile,dfloor);
    const Real eg_profile=profile_initialization
        ? InterpolateProfile(profile_egas,r) : -1.0;
    const Real eg=profile_initialization ? eg_profile*d/d_profile : -1.0;
    const Real t=profile_initialization ? InterpolateProfile(profile_temp,r) : Temperature(r);
#if GENERAL_EOS
    const Real p_profile=peos->PresFromRhoEg(d,eg);
#else
    const Real p_profile=d*kRgas*t/mu/SQR(velocity_unit);
#endif
    if (!std::isfinite(p_profile)
        || p_profile < pfloor*(1.0-floor_tolerance)) {
      std::stringstream message;
      message << "initial pressure is materially below hydro/pfloor at gid=" << gid
              << " (k,j,i)=(" << k << "," << j << "," << i << ")"
              << ", r=" << r << ", pressure=" << p_profile
              << ", pfloor=" << pfloor << ", rho=" << d;
      Require(false,message.str());
    }
    const Real p=std::max(p_profile,pfloor);
    const Real w=(r>rcore && r<radius) ? std::sin(M_PI*(r-rcore)/(radius-rcore)) : 0.0;
    const Real cs=std::sqrt(peos->AsqFromRhoP(d,p));
    const int m1=5+static_cast<int>(perturb_seed%7);
    const int m2=7+static_cast<int>((perturb_seed/7)%7);
    const int m3=9+static_cast<int>((perturb_seed/49)%7);
    // Coordinates only: changing MPI ranks or MeshBlocks must not change the seed.
    // Every component is odd under x->-x,y->-y,z->-z, so a centered symmetric
    // Cartesian mesh has zero net seeded momentum up to roundoff.
    const Real vx=perturb*w*cs*std::sin((m1*pcoord->x2v(j)+m2*pcoord->x3v(k))/radius);
    const Real vy=perturb*w*cs*std::sin((m2*pcoord->x3v(k)+m3*pcoord->x1v(i))/radius);
    const Real vz=perturb*w*cs*std::sin((m3*pcoord->x1v(i)+m1*pcoord->x2v(j))/radius);
    phydro->u(IDN,k,j,i)=d;phydro->u(IM1,k,j,i)=d*vx;
    phydro->u(IM2,k,j,i)=d*vy;phydro->u(IM3,k,j,i)=d*vz;
#if GENERAL_EOS
    const Real eint=(profile_initialization && p_profile>=pfloor)
        ? eg : peos->EgasFromRhoP(d,p);
#else
    const Real eint=profile_initialization ? eg : p/(gamma_gas-1.0);
#endif
    phydro->u(IEN,k,j,i)=eint+0.5*d*(vx*vx+vy*vy+vz*vz);
    prfld->u_gas(k,j,i)=eint;
    prfld->u_rad(k,j,i)=profile_initialization ? InterpolateProfile(profile_erad,r)
        : kAr*std::pow(t,4)/energy_unit*((r>radius) ? SQR(radius/r) : 1.0);
  }
}
void MeshBlock::UserWorkInLoop() {
  ruser_meshblock_data[0](4)=0.0;
  if (drag_tau<=0.0) return;
  Real taper=1.0;
  if (drag_taper_start>=0.0) {
    taper=std::max(0.0,std::min(1.0,
        (drag_taper_end-pmy_mesh->time)/(drag_taper_end-drag_taper_start)));
  }
  if (taper<=0.0) return;
  // Exact kinetic damping, only in the artificial core. No gas/radiation anchor.
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    const Real r=RadialPosition(pcoord,k,j,i);
    if (r>=rcore) continue;
    const Real f=std::exp(-taper*pmy_mesh->dt*SQR(1.0-SQR(r/rcore))/drag_tau);
    auto &u=phydro->u;
    const Real oldk=Kinetic(u,k,j,i);
    const Real oldm[3]={u(IM1,k,j,i),u(IM2,k,j,i),u(IM3,k,j,i)};
    u(IM1,k,j,i)*=f;u(IM2,k,j,i)*=f;u(IM3,k,j,i)*=f;
    const Real loss=oldk*(1.0-f*f);
    const Real vol=pcoord->GetCellVolume(k,j,i);
    u(IEN,k,j,i)-=loss;
    ruser_meshblock_data[0](0)+=loss*vol;
    ruser_meshblock_data[0](1)+=(oldm[0]-u(IM1,k,j,i))*vol;
    ruser_meshblock_data[0](2)+=(oldm[1]-u(IM2,k,j,i))*vol;
    ruser_meshblock_data[0](3)+=(oldm[2]-u(IM3,k,j,i))*vol;
    ruser_meshblock_data[0](4)+=loss*vol/std::max(pmy_mesh->dt,TINY_NUMBER);
    phydro->w(IVX,k,j,i)=u(IM1,k,j,i)/u(IDN,k,j,i);
    phydro->w(IVY,k,j,i)=u(IM2,k,j,i)/u(IDN,k,j,i);
    phydro->w(IVZ,k,j,i)=u(IM3,k,j,i)/u(IDN,k,j,i);
  }
}
void MeshBlock::UserWorkBeforeOutput(ParameterInput*) {
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    const Real r=RadialPosition(pcoord,k,j,i),d=phydro->u(IDN,k,j,i);
    const Real eg=phydro->u(IEN,k,j,i)-Kinetic(phydro->u,k,j,i);
#if GENERAL_EOS
    user_out_var(0,k,j,i)=peos->TempFromRhoEg(d,eg)*peos->ptable->tUnit;
#else
    const Real p=(gamma_gas-1.0)*eg;
    user_out_var(0,k,j,i)=p/d*SQR(velocity_unit)*mu/kRgas;
#endif
    user_out_var(1,k,j,i)=std::pow(prfld->u_rad(k,j,i)*energy_unit/kAr,0.25);
    user_out_var(2,k,j,i)=prfld->u_rad(k,j,i);
    user_out_var(3,k,j,i)=Potential(r);user_out_var(4,k,j,i)=GravityMagnitude(r);
    user_out_var(5,k,j,i)=(r>0.0) ? (phydro->u(IM1,k,j,i)*pcoord->x1v(i)
        +phydro->u(IM2,k,j,i)*pcoord->x2v(j)+phydro->u(IM3,k,j,i)*pcoord->x3v(k))/(d*r) : 0.0;
    user_out_var(6,k,j,i)=lum_code*CoreWeight(r)/heat_norm;
  }
}
void Mesh::UserWorkAfterLoop(ParameterInput*) {
  if (opacity_table) {opacity_table->ReportDiagnostics(std::cout);delete opacity_table;
    opacity_table=nullptr;}
}
