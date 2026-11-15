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
Real gravity_quad_full_radius, gravity_quad_taper_radius;
Real marshak_alpha;
Real short_test_max_dt=-1.0;
Real prep_thermal_tau=-1.0, prep_thermal_rmin, prep_thermal_rfull;
Real prep_thermal_rflat_end, prep_thermal_rmax, prep_thermal_dr;
Real prep_thermal_taper_start=-1.0, prep_thermal_taper_end=-1.0;
std::uint64_t perturb_seed;
bool profile_initialization=false, profile_cell_average_trial=false;
bool marshak_boundary=false;
bool core_quadrature_gravity=false;
std::vector<Real> profile_r, profile_rho, profile_egas, profile_erad, profile_temp;
UserOpacityTable *opacity_table=nullptr;

void Require(bool valid, const std::string &message) {
  if (!valid) {
    std::stringstream msg; msg << "nrfld_star_in_box: " << message;
    ATHENA_ERROR(msg);
  }
}
Real ShortTestTimeStep(MeshBlock*) { return short_test_max_dt; }
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
// Fourth-order cell average of rho*g on a smooth uniform mesh. A linear
// reconstruction of cell-average rho suffices because the omitted quadratic
// part has zero cell mean and its correlation with g first enters at O(h^4).
bool CellAverageGravityForce(MeshBlock *mb,const AthenaArray<Real> &w,
                             int k,int j,int i,Real force[3]) {
  const Real hx=mb->pcoord->dx1f(i),hy=mb->pcoord->dx2f(j);
  const Real hz=mb->pcoord->dx3f(k);
  const Real rho=w(IDN,k,j,i);
  const Real grad[3]={
    (w(IDN,k,j,i+1)-w(IDN,k,j,i-1))/(2.0*hx),
    (w(IDN,k,j+1,i)-w(IDN,k,j-1,i))/(2.0*hy),
    (w(IDN,k+1,j,i)-w(IDN,k-1,j,i))/(2.0*hz)};
  if (!(rho>0.0) || !std::isfinite(rho)
      || !std::isfinite(grad[0]) || !std::isfinite(grad[1])
      || !std::isfinite(grad[2])) return false;
  force[0]=force[1]=force[2]=0.0;
  const Real gauss=1.0/(2.0*std::sqrt(3.0));
  for (int iz=-1;iz<=1;iz+=2) for (int iy=-1;iy<=1;iy+=2)
    for (int ix=-1;ix<=1;ix+=2) {
      const Real offset[3]={ix*gauss*hx,iy*gauss*hy,iz*gauss*hz};
      const Real node_rho=rho+grad[0]*offset[0]
          +grad[1]*offset[1]+grad[2]*offset[2];
      if (!(node_rho>0.0) || !std::isfinite(node_rho)) return false;
      const Real node[3]={mb->pcoord->x1v(i)+offset[0],
          mb->pcoord->x2v(j)+offset[1],mb->pcoord->x3v(k)+offset[2]};
      const Real node_r=std::sqrt(SQR(node[0])+SQR(node[1])+SQR(node[2]));
      const Real factor=(node_r>0.0) ? -GravityMagnitude(node_r)/node_r : 0.0;
      for (int dir=0;dir<3;++dir) force[dir]+=0.125*node_rho*factor*node[dir];
    }
  return std::isfinite(force[0]) && std::isfinite(force[1])
      && std::isfinite(force[2]);
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
Real PrepThermalWindow(Real r) {
  if (r<=prep_thermal_rmin || r>=prep_thermal_rmax) return 0.0;
  if (r<prep_thermal_rfull) {
    const Real t=(r-prep_thermal_rmin)/(prep_thermal_rfull-prep_thermal_rmin);
    return t*t*(3.0-2.0*t);
  }
  if (r<=prep_thermal_rflat_end) return 1.0;
  const Real t=(prep_thermal_rmax-r)/(prep_thermal_rmax-prep_thermal_rflat_end);
  return t*t*(3.0-2.0*t);
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
struct ProfileCellAverage { Real rho, egas, erad; };
ProfileCellAverage AverageProfileInCell(Coordinates *c, int k, int j, int i) {
  // Positive tensor-product Gauss quadrature of the imported *conservative*
  // fields.  The imported profile is much steeper than the Cartesian cells
  // near the radiative transition; point sampling is not a cell average.
  constexpr Real nodes[4]={-0.8611363115940526,-0.3399810435848563,
                            0.3399810435848563, 0.8611363115940526};
  constexpr Real weights[4]={0.1739274225687269,0.3260725774312731,
                             0.3260725774312731,0.1739274225687269};
  const Real half_width[3]={0.5*c->dx1f(i),0.5*c->dx2f(j),0.5*c->dx3f(k)};
  const Real center[3]={c->x1v(i),c->x2v(j),c->x3v(k)};
  ProfileCellAverage average{0.0,0.0,0.0};
  for (int iz=0;iz<4;++iz) for (int iy=0;iy<4;++iy) for (int ix=0;ix<4;++ix) {
    const Real x=center[0]+nodes[ix]*half_width[0];
    const Real y=center[1]+nodes[iy]*half_width[1];
    const Real z=center[2]+nodes[iz]*half_width[2];
    const Real r=std::sqrt(x*x+y*y+z*z);
    const Real w=weights[ix]*weights[iy]*weights[iz];
    average.rho+=w*InterpolateProfile(profile_rho,r);
    average.egas+=w*InterpolateProfile(profile_egas,r);
    average.erad+=w*InterpolateProfile(profile_erad,r);
  }
  return average;
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
      const Real x=mb->pcoord->x1v(i),y=mb->pcoord->x2v(j);
      const Real z=mb->pcoord->x3v(k),d=w(IDN,k,j,i);
      Real force[3]={d*factor*x,d*factor*y,d*factor*z};
      if (core_quadrature_gravity && r<gravity_quad_taper_radius) {
        Real averaged[3];
        if (CellAverageGravityForce(mb,w,k,j,i,averaged)) {
          const Real t=(r-gravity_quad_full_radius)
              /(gravity_quad_taper_radius-gravity_quad_full_radius);
          const Real blend=(r<=gravity_quad_full_radius)
              ? 1.0 : 1.0-3.0*SQR(t)+2.0*t*t*t;
          force[0]+=blend*(averaged[0]-d*factor*x);
          force[1]+=blend*(averaged[1]-d*factor*y);
          force[2]+=blend*(averaged[2]-d*factor*z);
        }
      }
      u(IM1,k,j,i)+=dt*force[0]; u(IM2,k,j,i)+=dt*force[1];
      u(IM3,k,j,i)+=dt*force[2];
      u(IEN,k,j,i)+=dt*(force[0]*w(IVX,k,j,i)+force[1]*w(IVY,k,j,i)
                                  +force[2]*w(IVZ,k,j,i)
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
  if (out>=18) return mb->ruser_meshblock_data[0](out-13);
  if (out>=12) {
    // Audit the actual Robin flux used by NRFLD on each Cartesian box face.
    // The diffusion coefficient is lagged during the most recent Newton solve.
    // The implicit operator removes both interior diffusion and its Marshak
    // boundary flux when cut_diff is set.  Do not report a formal Robin flux
    // that was never applied to the radiation energy equation.
    if (!marshak_boundary || mb->prfld->cut_diff
        || mb->pmy_mesh->time<=0.0) return 0.0;
    const int face=out-12,axis=face/2;
    const bool inner=(face%2==0);
    const RegionSize &block=mb->block_size;
    const RegionSize &mesh=mb->pmy_mesh->mesh_size;
    const bool physical=(axis==0 ? (inner ? block.x1min==mesh.x1min
                                             : block.x1max==mesh.x1max)
                         : axis==1 ? (inner ? block.x2min==mesh.x2min
                                             : block.x2max==mesh.x2max)
                                   : (inner ? block.x3min==mesh.x3min
                                            : block.x3max==mesh.x3max));
    if (!physical) return 0.0;
    Real flux=0.0;
    for (int k=mb->ks;k<=mb->ke;++k) for (int j=mb->js;j<=mb->je;++j)
      for (int i=mb->is;i<=mb->ie;++i) {
        if ((axis==0 && i!=(inner ? mb->is : mb->ie))
            || (axis==1 && j!=(inner ? mb->js : mb->je))
            || (axis==2 && k!=(inner ? mb->ks : mb->ke))) continue;
        const Real dx=mb->pcoord->dx1f(i);
        const Real dface=std::max(mb->prfld->marshak_dface_all(face,k,j,i),
                                  TINY_NUMBER);
        const Real acoef=marshak_alpha*mb->prfld->c_ph*0.5*dx;
        const Real erad=mb->prfld->u_rad(k,j,i);
        const Real eb=(dface*erad+acoef*boundary_er)/(dface+acoef);
        flux+=marshak_alpha*mb->prfld->c_ph*(eb-boundary_er)*dx*dx;
      }
    return flux;
  }
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
  const bool static_refinement_trial = pin->GetOrAddBoolean(
      "problem", "static_refinement_trial", false);
  short_test_max_dt=pin->GetOrAddReal("problem", "short_test_max_dt", -1.0);
  Require(short_test_max_dt<0.0 || (std::isfinite(short_test_max_dt)
          && short_test_max_dt>0.0),
          "problem/short_test_max_dt must be positive when specified");
  Require(std::string(COORDINATE_SYSTEM)=="cartesian" && f2 && f3,
          "use a 3D Cartesian mesh");
  Require(!multilevel || (static_refinement_trial && !adaptive
          && pin->GetString("mesh", "refinement") == "static"
          && pin->GetInteger("time", "xorder") == 2),
          "multilevel RSG requires the opt-in static, xorder=2 trial");
  Require(!multilevel || (!pin->GetOrAddBoolean("time", "correct_ic", false)
          && !pin->GetOrAddBoolean("time", "correct_ic_fld_radiation", false)),
          "second-order static refinement trial requires both initial cell-average "
          "corrections off");
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
  core_quadrature_gravity=pin->GetOrAddBoolean(
      "problem","core_quadrature_gravity",false);
  Require(!core_quadrature_gravity ||
          (pin->GetInteger("time","xorder")==4 && NGHOST>=4),
          "core_quadrature_gravity requires xorder=4 and at least four ghosts");
  gravity_quad_full_radius=rcore*pin->GetOrAddReal(
      "problem","gravity_quadrature_full_radius_rcore",1.0);
  gravity_quad_taper_radius=rcore*pin->GetOrAddReal(
      "problem","gravity_quadrature_taper_radius_rcore",1.5);
  Require(!core_quadrature_gravity ||
          (gravity_quad_full_radius>=rcore
           && gravity_quad_taper_radius>gravity_quad_full_radius
           && gravity_quad_taper_radius<radius),
          "gravity quadrature requires Rcore<=full<taper<Rstar");
  router=Positive(pin,"problem","outer_flattening_fraction")*radius;
  rheat=Positive(pin,"problem","heating_radius_fraction")*radius;
  Require(rheat<=rcore && rcore<radius && router>radius,"require rheat<=rcore<R<router");
  if (multilevel) {
    Require(!core_quadrature_gravity,
            "static refinement trial requires problem/core_quadrature_gravity=false");
    int regions=0;
    for (InputBlock *block=pin->pfirst_block; block!=nullptr; block=block->pnext) {
      if (block->block_name.compare(0,10,"refinement") != 0) continue;
      ++regions;
      Real d2=0.0;
      for (int axis=1;axis<=3;++axis) {
        const std::string key="x"+std::to_string(axis);
        const Real lo=pin->GetReal(block->block_name,key+"min");
        const Real hi=pin->GetReal(block->block_name,key+"max");
        const Real min_abs=(lo>0.0 ? lo : hi<0.0 ? -hi : 0.0);
        d2+=SQR(min_abs);
      }
      Require(pin->GetInteger(block->block_name,"level")==1
              && std::sqrt(d2)>rheat,
              "trial refinement patches must be one level and outside heating sphere");
    }
    Require(regions>0,"static refinement trial needs a refinement region");
  }
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
  prep_thermal_tau=pin->GetOrAddReal("problem","prep_thermal_tau",-1.0);
  prep_thermal_rmin=pin->GetOrAddReal("problem","prep_thermal_rmin_rsun",790.0)
      *kRsun/length_unit;
  prep_thermal_rfull=pin->GetOrAddReal("problem","prep_thermal_rfull_rsun",805.0)
      *kRsun/length_unit;
  prep_thermal_rflat_end=pin->GetOrAddReal(
      "problem","prep_thermal_rflat_end_rsun",830.0)*kRsun/length_unit;
  prep_thermal_rmax=pin->GetOrAddReal("problem","prep_thermal_rmax_rsun",850.0)
      *kRsun/length_unit;
  prep_thermal_dr=pin->GetOrAddReal("problem","prep_thermal_bin_width_rsun",1.0)
      *kRsun/length_unit;
  prep_thermal_taper_start=pin->GetOrAddReal(
      "problem","prep_thermal_taper_start",-1.0);
  prep_thermal_taper_end=pin->GetOrAddReal(
      "problem","prep_thermal_taper_end",-1.0);
  Require(prep_thermal_tau<0.0 || (std::isfinite(prep_thermal_tau)
          && prep_thermal_tau>0.0
          && 0.0<=prep_thermal_rmin && prep_thermal_rmin<prep_thermal_rfull
          && prep_thermal_rfull<=prep_thermal_rflat_end
          && prep_thermal_rflat_end<prep_thermal_rmax
          && std::isfinite(prep_thermal_dr) && prep_thermal_dr>0.0
          && (prep_thermal_rmax-prep_thermal_rmin)/prep_thermal_dr<=1000.0
          && ((prep_thermal_taper_start<0.0 && prep_thermal_taper_end<0.0)
              || (prep_thermal_taper_start>=0.0
                  && prep_thermal_taper_end>prep_thermal_taper_start))),
          "prep thermal support requires imported profile, positive tau, "
          "ordered radii, <=1000 bins, and valid optional taper");
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
  Require(prep_thermal_tau<0.0 || profile_initialization,
          "prep thermal support requires an imported profile");
  // Reproducible WOR-122 control only.  It did not pass the surface-force or
  // luminosity gates and must stay disabled in production inputs.
  profile_cell_average_trial=pin->GetOrAddBoolean(
      "problem", "profile_cell_average_trial", false);
  Require(!profile_cell_average_trial || (profile_initialization && !multilevel
          && !pin->GetOrAddBoolean("time", "correct_ic", false)
          && !pin->GetOrAddBoolean("time", "correct_ic_fld_radiation", false)),
          "profile cell-average trial requires a uniform imported profile "
          "and both generic initial corrections off");
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
  if (short_test_max_dt>0.0) EnrollUserTimeStepFunction(ShortTestTimeStep);
  AllocateUserHistoryOutput(20);
  const char *hist[20]={"Erad","Egas","Epotential","radial_momentum","mass_r2",
      "Lheat","Mach_max","Edrag_removed","Pdrag_x_removed","Pdrag_y_removed",
      "Pdrag_z_removed","drag_power_removed","Lrad_xm","Lrad_xp",
      "Lrad_ym","Lrad_yp","Lrad_zm","Lrad_zp","Eprep_thermal","Pprep_thermal"};
  for (int q=0;q<20;++q) EnrollUserHistoryOutput(q,History,hist[q],
      q==6 ? UserHistoryOperation::max : UserHistoryOperation::sum);
  if (Globals::my_rank==0) std::cout << std::setprecision(16)
      << "STAR_IN_BOX mode=" << (profile_initialization ? "tabular-profile" : "commissioning-polytrope")
      << ", fixed Freytag potential\n"
      << "STAR_IN_BOX rho_surface=" << rho_surface << " heat_norm=" << heat_norm
      << " gas_mass=" << gas_mass << " Lheat=" << lum_code
      << " time_unit_s=" << time_unit << " core_cells=" << 2.0*rcore/dx[0]
      << " static_refinement_trial=" << static_refinement_trial
      << " profile_cell_average_trial=" << profile_cell_average_trial
      << " prep_thermal_tau=" << prep_thermal_tau << '\n';
}

void MeshBlock::InitUserMeshBlockData(ParameterInput*) {
  prfld->EnrollOpacityFunction(Opacity);
  if (marshak_boundary) {
    for (int face=0; face<6; ++face) prfld->marshak_face_boundary[face]=true;
    prfld->marshak_top_alpha=marshak_alpha;
    prfld->marshak_top_erad_ext=boundary_er;
  }
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(7);
  for (int q=0;q<7;++q) ruser_meshblock_data[0](q)=0.0;
  AllocateUserOutputVariables(7);
  const char *names[7]={"Tgas","Trad","Er","phi","g","vr","heat"};
  for (int q=0;q<7;++q) SetUserOutputVariableName(q,names[q]);
}
void MeshBlock::ProblemGenerator(ParameterInput*) {
  for (int k=ks;k<=ke;++k) for (int j=js;j<=je;++j) for (int i=is;i<=ie;++i) {
    const Real r=RadialPosition(pcoord,k,j,i);
    const ProfileCellAverage averaged=profile_cell_average_trial
        ? AverageProfileInCell(pcoord,k,j,i) : ProfileCellAverage{0.0,0.0,0.0};
    const Real d_profile=profile_cell_average_trial ? averaged.rho : InitialDensity(r);
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
        ? (profile_cell_average_trial ? averaged.egas
                                      : InterpolateProfile(profile_egas,r)) : -1.0;
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
    prfld->u_rad(k,j,i)=profile_initialization
        ? (profile_cell_average_trial ? averaged.erad
                                      : InterpolateProfile(profile_erad,r))
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
void Mesh::UserWorkInLoop() {
  // This deliberately nonphysical preparation source acts after the implicit
  // FLD solve. A common correction to each radial shell's mean gas specific
  // energy retains angular thermal fluctuations and leaves all velocities
  // free to develop. It is disabled in ordinary RSG runs.
  for (int b=0;b<nblocal;++b) my_blocks(b)->ruser_meshblock_data[0](6)=0.0;
  if (prep_thermal_tau<=0.0) return;
  Real taper=1.0;
  if (prep_thermal_taper_start>=0.0)
    taper=std::max(0.0,std::min(1.0,
        (prep_thermal_taper_end-(time+dt))
        /(prep_thermal_taper_end-prep_thermal_taper_start)));
  if (taper<=0.0) return;
  const Real factor=-std::expm1(-taper*dt/prep_thermal_tau);
  const int nbins=static_cast<int>(std::ceil(
      (prep_thermal_rmax-prep_thermal_rmin)/prep_thermal_dr));
  std::vector<Real> local(2*nbins,0.0),global(2*nbins,0.0);
  for (int b=0;b<nblocal;++b) {
    MeshBlock *mb=my_blocks(b);
    for (int k=mb->ks;k<=mb->ke;++k) for (int j=mb->js;j<=mb->je;++j)
      for (int i=mb->is;i<=mb->ie;++i) {
        const Real r=RadialPosition(mb->pcoord,k,j,i);
        const Real window=PrepThermalWindow(r);
        if (window<=0.0) continue;
        const int q=static_cast<int>((r-prep_thermal_rmin)/prep_thermal_dr);
        const auto &u=mb->phydro->u;
        const Real rho=u(IDN,k,j,i);
        const Real egas=u(IEN,k,j,i)-Kinetic(u,k,j,i);
        const Real target_specific=InterpolateProfile(profile_egas,r)
            /InterpolateProfile(profile_rho,r);
        const Real volume=mb->pcoord->GetCellVolume(k,j,i);
        local[2*q]+=window*rho*volume;
        local[2*q+1]+=window*(rho*target_specific-egas)*volume;
      }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(local.data(),global.data(),2*nbins,MPI_ATHENA_REAL,
                MPI_SUM,MPI_COMM_WORLD);
#else
  global=local;
#endif
  for (int b=0;b<nblocal;++b) {
    MeshBlock *mb=my_blocks(b);
    Real injected=0.0;
    for (int k=mb->ks;k<=mb->ke;++k) for (int j=mb->js;j<=mb->je;++j)
      for (int i=mb->is;i<=mb->ie;++i) {
        const Real r=RadialPosition(mb->pcoord,k,j,i);
        const Real window=PrepThermalWindow(r);
        if (window<=0.0) continue;
        const int q=static_cast<int>((r-prep_thermal_rmin)/prep_thermal_dr);
        if (global[2*q]<=0.0) continue;
        auto &u=mb->phydro->u;
        const Real rho=u(IDN,k,j,i);
        const Real old_egas=u(IEN,k,j,i)-Kinetic(u,k,j,i);
        const Real delta=factor*window*rho*global[2*q+1]/global[2*q];
        const Real new_egas=old_egas+delta;
        Require(std::isfinite(new_egas) && new_egas>0.0,
                "prep thermal support produced nonpositive gas energy");
        const Real pressure=mb->peos->PresFromRhoEg(rho,new_egas);
        Require(std::isfinite(pressure) && pressure>=pfloor,
                "prep thermal support produced invalid gas pressure");
        u(IEN,k,j,i)+=delta;
        mb->phydro->w(IPR,k,j,i)=pressure;
        mb->prfld->u_gas(k,j,i)=new_egas;
        injected+=delta*mb->pcoord->GetCellVolume(k,j,i);
      }
    mb->ruser_meshblock_data[0](5)+=injected;
    mb->ruser_meshblock_data[0](6)=injected/dt;
    // The next explicit radiation force must see the supported gas state.
    mb->prfld->UpdateOpacity(mb,mb->prfld->u_rad,mb->phydro->w);
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
