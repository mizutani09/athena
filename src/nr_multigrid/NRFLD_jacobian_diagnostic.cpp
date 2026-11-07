// Optional Test C operator audit. No diagnostic arrays exist when disabled.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "NRFLD.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../linear_solver/linear_solver.hpp"
#include "../task_list/nr_task_list.hpp"
#include "../utils/nrfld_prototype_flags.hpp"

namespace {
constexpr int di[6]={-1,1,0,0,0,0};
constexpr int dj[6]={0,0,-1,1,0,0};
constexpr int dk[6]={0,0,0,0,-1,1};
const int cf[6]={linearSolver::DXMF,linearSolver::DXPF,linearSolver::DYMF,
                linearSolver::DYPF,linearSolver::DZMF,linearSolver::DZPF};

void Snapshot(AthenaArray<Real> &dest, const AthenaArray<Real> &source) {
  // Copy assignment does not allocate. Empty optional arrays must not be
  // passed to AthenaArray's allocating copy constructor either.
  if (source.data() != nullptr) dest=AthenaArray<Real>(source);
}

// Use the difference form in extended precision to distinguish interpolation
// errors from cancellation of the large diagonal/offdiagonal products.
long double Residual(const AthenaArray<Real> &u, const AthenaArray<Real> &old,
                     const AthenaArray<Real> &coeff, Real fac, int k,int j,int i) {
  const long double center=u(k,j,i);
  long double r=center-old(k,j,i);
  for (int f=0; f<6; ++f)
    r+=static_cast<long double>(fac)*coeff(cf[f],k,j,i)
       *(static_cast<long double>(u(k+dk[f],j+dj[f],i+di[f]))-center);
  return r;
}

int Region(MeshBlock *mb,int k,int j,int i) {
  int region=0;
  const bool face[6]={i==mb->is,i==mb->ie,j==mb->js,j==mb->je,
                      k==mb->ks,k==mb->ke};
  const RegionSize &block=mb->block_size;
  const RegionSize &mesh=mb->pmy_mesh->mesh_size;
  const bool physical[6]={block.x1min==mesh.x1min,block.x1max==mesh.x1max,
                          block.x2min==mesh.x2min,block.x2max==mesh.x2max,
                          block.x3min==mesh.x3min,block.x3max==mesh.x3max};
  for (int f=0;f<6;++f) if (face[f]) {
    if (physical[f] && mb->pmy_mesh->mesh_bcs[f]!=BoundaryFlag::periodic) {
      region=3;
      continue;
    }
    const int level=mb->pbval->nblevel[1+dk[f]][1+dj[f]][1+di[f]];
    region=std::max(region,level==mb->loc.level ? 1 : 2);
  }
  return region;
}

struct Sums {
  long double error{0}, fd{0}, av{0}, tangent_error{0}, ghost_error{0}, ghost_ref{0};
  Real max_error{0}, x{0}, y{0}, z{0};
  int gid{-1}, level{-1};
  std::int64_t cells{0};
};
}  // namespace

void NRFLDDriver::RunJacobianDiagnostics() {
  NewtonRaphsonDriver::RunJacobianDiagnostics(); // preserve legacy local check
  global_jv_solve_=0;
  if (global_jv_ && dt_>0.0) CheckGlobalJacobian(false);
}

void NRFLDDriver::RunDirectionDiagnostic() {
  if (global_jv_ && dt_>0.0 && global_jv_solve_==global_jv_iteration_)
    CheckGlobalJacobian(true);
  ++global_jv_solve_;
}

// Frozen-coefficient, full-domain two-equation Jv and eliminated Schur audit.
// The radiation direction is exchanged through the same correction boundary
// path as FGMRES.  The smooth probe uses independent gas/radiation directions;
// the Newton probe eliminates the local gas row as linearMG does.
void NRFLDDriver::CheckCoupledGlobalJacobian(bool actual_direction) {
  struct Saved {
    NRFLD *nr;
    AthenaArray<Real> u, gas, delta, coeff, deriv, baseline_deriv, src,
                      baseline_src, defcoeff,
                      coarse_u, coarse_delta, moments, coarse_moments, marshak;
    AthenaArray<Real> baseline, gas_baseline, vr, vg;
    bool initialized;
  };
  std::vector<Saved> saved(vnr_.size());
  for (std::size_t b=0; b<saved.size(); ++b) {
    Saved &s=saved[b]; s.nr=static_cast<NRFLD*>(vnr_[b]);
    NRFLD *nr=s.nr;
    Snapshot(s.u,nr->u_); Snapshot(s.gas,nr->u_gas_);
    Snapshot(s.delta,nr->delta_u_); Snapshot(s.coeff,nr->coeff_);
    Snapshot(s.deriv,nr->derivetive_); Snapshot(s.src,nr->src_);
    Snapshot(s.defcoeff,nr->def_coeff_);
    Snapshot(s.coarse_u,nr->NewtonRaphson::coarse_u_);
    Snapshot(s.coarse_delta,nr->coarse_delta_u_);
    if (NRFLDPrototype::SharedCoeff()) {
      Snapshot(s.moments,nr->interface_moments_);
      Snapshot(s.coarse_moments,nr->coarse_interface_moments_);
    }
    Snapshot(s.marshak,nr->pmy_block_->prfld->marshak_dface);
    s.initialized=nr->fixed_linear_coefficients_initialized_;
  }
  auto restore=[&]() {
    for (Saved &s:saved) {
      NRFLD *nr=s.nr;
      nr->u_=s.u; nr->u_gas_=s.gas; nr->delta_u_=s.delta;
      nr->coeff_=s.coeff; nr->derivetive_=s.deriv; nr->src_=s.src;
      nr->def_coeff_=s.defcoeff;
      nr->NewtonRaphson::coarse_u_=s.coarse_u;
      nr->coarse_delta_u_=s.coarse_delta;
      if (NRFLDPrototype::SharedCoeff()) {
        nr->interface_moments_=s.moments;
        nr->coarse_interface_moments_=s.coarse_moments;
      }
      nr->pmy_block_->prfld->marshak_dface=s.marshak;
      nr->fixed_linear_coefficients_initialized_=s.initialized;
    }
  };
  NewtonRaphsonTaskList boundary(this,NewtonRaphsonTaskList::Mode::boundary);
  boundary.DoTaskListOneStage(stage_);
  for (Saved &s:saved) {
    NRFLD *nr=s.nr;
    nr->coeff_=s.coeff; nr->derivetive_=s.deriv;
    nr->fixed_linear_coefficients_initialized_=true;
    nr->CalculateCoefficients(nr->uold_,nr->u_,nr->def_coeff_,
                              nr->coeff_,nr->derivetive_,nr->src_,dt_);
    Snapshot(s.baseline_deriv,nr->derivetive_);
    Snapshot(s.baseline_src,nr->src_);
  }
  Real amplitude=0.0;
  for (Saved &s:saved) {
    NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
    Snapshot(s.baseline,nr->u_); Snapshot(s.gas_baseline,nr->u_gas_);
    Snapshot(s.vr,nr->u_); s.vr.ZeroClear();
    Snapshot(s.vg,nr->u_gas_); s.vg.ZeroClear();
    const Real length=mb->pmy_mesh->mesh_size.x1max-mb->pmy_mesh->mesh_size.x1min;
    for (int k=mb->ks;k<=mb->ke;++k)
      for (int j=mb->js;j<=mb->je;++j)
        for (int i=mb->is;i<=mb->ie;++i) {
          const Real x=(mb->pcoord->x1v(i)-mb->pmy_mesh->mesh_size.x1min)/length;
          const Real shape=0.55*std::sin(2.0*3.141592653589793*x)
                          +0.25*std::cos(4.0*3.141592653589793*x);
          const Real vr=actual_direction ? s.delta(k,j,i)
                                          : s.baseline(k,j,i)*shape;
          const Real denom=s.baseline_deriv(NewtonRaphsonFLD::dFg_deg,k,j,i);
          const Real vg=actual_direction
              ? -s.baseline_deriv(NewtonRaphsonFLD::dFg_dEr,k,j,i)*vr/denom
              : 0.35*s.gas_baseline(k,j,i)*shape;
          s.vr(k,j,i)=vr; s.vg(k,j,i)=vg;
          amplitude=std::max(amplitude,std::abs(vr)/
                  std::max(std::abs(s.baseline(k,j,i)),TINY_NUMBER));
          amplitude=std::max(amplitude,std::abs(vg)/
                  std::max(std::abs(s.gas_baseline(k,j,i)),TINY_NUMBER));
        }
    nr->delta_u_=s.vr;
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&amplitude,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (amplitude==0.0 || !std::isfinite(amplitude)) {
    restore();
    if (Globals::my_rank==0)
      std::cout << "[NRFLD coupled Jv skipped] reason=zero_or_invalid_direction"
                << std::endl;
    return;
  }
  for (Saved &s:saved) {
    MeshBlock *mb=s.nr->pmy_block_;
    for (int k=mb->ks;k<=mb->ke;++k)
      for (int j=mb->js;j<=mb->je;++j)
        for (int i=mb->is;i<=mb->ie;++i) {
          s.vr(k,j,i)/=amplitude; s.vg(k,j,i)/=amplitude;
          s.nr->delta_u_(k,j,i)=s.vr(k,j,i);
        }
  }
  plmgd_->SynchronizeCorrectionBoundaries(stage_);
  for (Saved &s:saved) s.vr=s.nr->delta_u_;
  const auto flags=std::cout.flags(); const auto precision=std::cout.precision();
  std::cout << std::scientific << std::setprecision(16);
  for (int pass=0;pass<4;++pass) {
    const Real h=global_jv_epsilon_*std::pow(0.1,pass);
    for (Saved &s:saved) {
      NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
      nr->u_=s.baseline; nr->u_gas_=s.gas_baseline;
      for (int k=mb->ks;k<=mb->ke;++k)
        for (int j=mb->js;j<=mb->je;++j)
          for (int i=mb->is;i<=mb->ie;++i) {
            nr->u_(k,j,i)+=h*s.vr(k,j,i);
            nr->u_gas_(k,j,i)+=h*s.vg(k,j,i);
          }
    }
    boundary.DoTaskListOneStage(stage_);
    // Preserve the diffusion closure and face coefficients of the production
    // Newton solve. CalculateCoefficients still reevaluates its two residuals.
    for (Saved &s:saved) {
      NRFLD *nr=s.nr;
      nr->derivetive_=s.baseline_deriv; nr->coeff_=s.coeff;
      nr->fixed_linear_coefficients_initialized_=true;
      nr->CalculateCoefficients(nr->uold_,nr->u_,nr->def_coeff_,
                                nr->coeff_,nr->derivetive_,nr->src_,dt_);
    }
    // error^2, reference^2 for gas, radiation, Schur and the analytic
    // full-system-vs-assembled-Schur check, split by interior/same/AMR face.
    long double sums[4][8] = {};
    long long counts[4] = {};
    for (Saved &s:saved) {
      NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
      const Real fac=dt_/SQR(mb->pcoord->dx1f(mb->is));
      for (int k=mb->ks;k<=mb->ke;++k)
        for (int j=mb->js;j<=mb->je;++j)
          for (int i=mb->is;i<=mb->ie;++i) {
            const auto &d=s.baseline_deriv;
            const long double vr=s.vr(k,j,i),vg=s.vg(k,j,i);
            const long double jg=d(NewtonRaphsonFLD::dFg_deg,k,j,i)*vg
                +d(NewtonRaphsonFLD::dFg_dEr,k,j,i)*vr;
            long double jr=d(NewtonRaphsonFLD::dFr_deg,k,j,i)*vg
                +d(NewtonRaphsonFLD::dFr_dEr,k,j,i)*vr;
            long double schur=(fac*s.coeff(linearSolver::DCCF,k,j,i)
                +s.coeff(linearSolver::DCCS,k,j,i))*vr;
            for (int f=0;f<6;++f) {
              const long double vn=s.vr(k+dk[f],j+dj[f],i+di[f]);
              jr+=fac*s.coeff(cf[f],k,j,i)*vn;
              schur+=(fac*s.coeff(cf[f],k,j,i)
                  +s.coeff(linearSolver::DXMS+f,k,j,i))*vn;
            }
            const long double fdg=(nr->derivetive_(NewtonRaphsonFLD::Fg,k,j,i)
                         -d(NewtonRaphsonFLD::Fg,k,j,i))/h;
            const long double fdr=(nr->derivetive_(NewtonRaphsonFLD::Fr,k,j,i)
                         -d(NewtonRaphsonFLD::Fr,k,j,i))/h;
            const long double ratio=d(NewtonRaphsonFLD::dFr_deg,k,j,i)
                              /d(NewtonRaphsonFLD::dFg_deg,k,j,i);
            // Production forms Ftotal directly and src=-Ftotal+Fg/dFg_deg.
            // Freeze the eliminated gas denominator at the baseline when
            // differencing; separately rounded Fr and Fg lose the small
            // Schur signal in stiff matter/radiation exchange.
            const long double inv_base=1.0L
                /d(NewtonRaphsonFLD::dFg_deg,k,j,i);
            const long double inv_pert=1.0L
                /nr->derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i);
            const long double fg_pert=nr->derivetive_(NewtonRaphsonFLD::Fg,k,j,i);
            const long double stable_pert=-nr->src_(k,j,i)
                +fg_pert*(inv_pert-inv_base);
            const long double fdschur=(stable_pert+s.baseline_src(k,j,i))/h;
            const long double jacschur=jr-ratio*jg;
            const long double e[4]={fdg-jg,fdr-jr,fdschur-schur,
                                    jacschur-schur};
            const long double ref[4]={jg,jr,schur,schur};
            const int region=Region(mb,k,j,i);
            for (int r : {0,region}) {
              for (int n=0;n<4;++n) {
                sums[r][2*n]+=e[n]*e[n];
                sums[r][2*n+1]+=ref[n]*ref[n];
              }
              ++counts[r];
              if (region==0) break;
            }
          }
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,sums,32,MPI_LONG_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,counts,4,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
#endif
    if (Globals::my_rank==0)
      for (int r=0;r<4;++r) if (counts[r]) {
        std::cout << "[NRFLD coupled Jv] direction="
                  << (actual_direction ? "newton" : "smooth")
                  << " epsilon=" << h << " region=" << r
                  << " cells=" << counts[r];
        if (actual_direction)
          std::cout << " gas_null_fd_l2=" << std::sqrt(sums[r][0])
                    << " gas_null_jac_l2=" << std::sqrt(sums[r][1]);
        else
          std::cout << " rel0=" << std::sqrt(sums[r][0]
                        /std::max(sums[r][1],1.e-300L));
        for (int n=1;n<4;++n)
          std::cout << " rel" << n << "="
                    << std::sqrt(sums[r][2*n]
                        /std::max(sums[r][2*n+1],1.e-300L));
        std::cout << std::endl;
      }
  }
  restore();
  std::cout.flags(flags); std::cout.precision(precision);
}

void NRFLDDriver::CheckGlobalJacobian(bool actual_direction) {
  bool coupled=false;
  for (auto base:vnr_) {
    NRFLD *nr=static_cast<NRFLD*>(base);
    FLD *fld=nr->pmy_block_->prfld;
    coupled |= fld->is_couple && !fld->only_rad && !fld->fixed_u_rad;
  }
  if (coupled) {
    bool supported=true;
    for (auto base:vnr_) {
      NRFLD *nr=static_cast<NRFLD*>(base);
      FLD *fld=nr->pmy_block_->prfld;
      supported &= fld->is_couple && !fld->only_rad && !fld->fixed_u_rad
                   && !fld->MixedFrameCoupled() && !fld->marshak_top_boundary;
      const bool periodic=std::all_of(nr_mesh_bcs_,nr_mesh_bcs_+6,
          [](BoundaryFlag flag) { return flag==BoundaryFlag::periodic; });
      const bool mach5_user_x1=nr_mesh_bcs_[0]==BoundaryFlag::user
          && nr_mesh_bcs_[1]==BoundaryFlag::user
          && std::all_of(nr_mesh_bcs_+2,nr_mesh_bcs_+6,
              [](BoundaryFlag flag) { return flag==BoundaryFlag::periodic; });
      supported &= periodic || mach5_user_x1;
    }
    if (!supported) {
      std::stringstream msg;
      msg << "coupled global_jacobian_check requires periodic Test C or "
             "user-x1 Mach 5 without mixed drift.";
      ATHENA_ERROR(msg);
    }
    CheckCoupledGlobalJacobian(actual_direction);
    return;
  }
  for (auto base:vnr_) {
    NRFLD *nr=static_cast<NRFLD*>(base);
    FLD *fld=nr->pmy_block_->prfld;
    bool supported=fld->only_rad && !fld->fixed_u_rad && !fld->is_couple
                   && !fld->MixedFrameCoupled() && !fld->marshak_top_boundary;
    for (int f=0;f<6;++f) supported &= nr_mesh_bcs_[f]==BoundaryFlag::periodic;
    if (!supported) {
      std::stringstream msg;
      msg << "global_jacobian_check requires periodic, uncoupled only_rad Test C.";
      ATHENA_ERROR(msg);
    }
  }

  struct Saved {
    NRFLD *nr;
    AthenaArray<Real> u, delta, coarse_u, coarse_delta, moments,
                      coarse_moments, coeff, deriv, defcoeff,
                      src, mixed, marshak, baseline, direction, tangent;
    bool initialized;
  };
  std::vector<Saved> saved(vnr_.size());
  for (std::size_t b=0;b<vnr_.size();++b) {
    Saved &s=saved[b];
    s.nr=static_cast<NRFLD*>(vnr_[b]);
    NRFLD *nr=s.nr;
    Snapshot(s.u,nr->u_); Snapshot(s.delta,nr->delta_u_); Snapshot(s.coeff,nr->coeff_);
    Snapshot(s.coarse_u,nr->NewtonRaphson::coarse_u_);
    Snapshot(s.coarse_delta,nr->coarse_delta_u_);
    if (NRFLDPrototype::SharedCoeff()) {
      Snapshot(s.moments,nr->interface_moments_);
      Snapshot(s.coarse_moments,nr->coarse_interface_moments_);
    }
    Snapshot(s.deriv,nr->derivetive_); Snapshot(s.defcoeff,nr->def_coeff_);
    Snapshot(s.src,nr->src_); Snapshot(s.mixed,nr->mixed_drift_);
    Snapshot(s.marshak,nr->pmy_block_->prfld->marshak_dface);
    s.initialized=nr->fixed_linear_coefficients_initialized_;
  }
  auto restore=[&]() {
    for (Saved &s:saved) {
      NRFLD *nr=s.nr;
      nr->u_=s.u; nr->delta_u_=s.delta; nr->coeff_=s.coeff;
      nr->NewtonRaphson::coarse_u_=s.coarse_u;
      nr->coarse_delta_u_=s.coarse_delta;
      if (NRFLDPrototype::SharedCoeff()) {
        nr->interface_moments_=s.moments;
        nr->coarse_interface_moments_=s.coarse_moments;
      }
      nr->derivetive_=s.deriv; nr->def_coeff_=s.defcoeff; nr->src_=s.src;
      nr->mixed_drift_=s.mixed; nr->pmy_block_->prfld->marshak_dface=s.marshak;
      nr->fixed_linear_coefficients_initialized_=s.initialized;
    }
  };
  NewtonRaphsonTaskList boundary(this,NewtonRaphsonTaskList::Mode::boundary);
  boundary.DoTaskListOneStage(stage_);
  if (NRFLDPrototype::SharedCoeff()) SyncInterfaceFaceMoments();
  Real baseline_ghost_change=0.0, amplitude=0.0;
  for (Saved &s:saved) {
    NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
    Snapshot(s.baseline,nr->u_);
    Snapshot(s.direction,nr->delta_u_); s.direction.ZeroClear();
    Snapshot(s.tangent,nr->u_);
    for (int k=0;k<nr->u_.GetDim3();++k)
      for (int j=0;j<nr->u_.GetDim2();++j)
        for (int i=0;i<nr->u_.GetDim1();++i)
          baseline_ghost_change=std::max(baseline_ghost_change,
                                        std::abs(s.u(k,j,i)-nr->u_(k,j,i)));
    const Real length=mb->pmy_mesh->mesh_size.x1max-mb->pmy_mesh->mesh_size.x1min;
    for (int k=mb->ks;k<=mb->ke;++k)
      for (int j=mb->js;j<=mb->je;++j)
        for (int i=mb->is;i<=mb->ie;++i) {
          const Real x=(mb->pcoord->x1v(i)-mb->pmy_mesh->mesh_size.x1min)/length;
          const Real v=actual_direction ? s.delta(k,j,i)
              : s.baseline(k,j,i)*(0.55*std::sin(2.0*3.141592653589793*x)
                                     +0.25*std::cos(4.0*3.141592653589793*x));
          s.direction(k,j,i)=v;
          amplitude=std::max(amplitude,std::abs(v)/
                             std::max(std::abs(s.baseline(k,j,i)),TINY_NUMBER));
        }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&baseline_ghost_change,1,MPI_ATHENA_REAL,MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&amplitude,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  // Normalize the actual MG direction to avoid positivity clipping. Preserve
  // its shape and sign; epsilon is relative to the energy, not to raw delta.
  if (actual_direction && amplitude==0.0) {
    restore();
    if (Globals::my_rank==0)
      std::cout << "[NRFLD Jv skipped] direction=newton iteration=" << global_jv_solve_
                << " reason=zero_direction" << std::endl;
    return;
  }
  amplitude=std::max(amplitude,static_cast<Real>(1.e-30));
  for (Saved &s:saved) {
    MeshBlock *mb=s.nr->pmy_block_;
    for (int k=mb->ks;k<=mb->ke;++k)
      for (int j=mb->js;j<=mb->je;++j)
        for (int i=mb->is;i<=mb->ie;++i) s.direction(k,j,i)/=amplitude;
    s.nr->delta_u_=s.direction;
  }
  plmgd_->SynchronizeCorrectionBoundaries(stage_);
  for (Saved &s:saved) s.direction=s.nr->delta_u_;

  const auto flags=std::cout.flags(); const auto precision=std::cout.precision();
  std::cout << std::scientific << std::setprecision(16);
  if (Globals::my_rank==0) std::cout << "[NRFLD Jv setup] direction=" << (actual_direction ? "newton" : "smooth")
            << " iteration=" << (actual_direction ? global_jv_solve_ : -1)
            << " baseline_ghost_change=" << baseline_ghost_change
            << " direction_scale=" << amplitude << " coefficient_policy=lagged_per_step"
            << std::endl;

  // frozen is the production residual; refreshed is a hypothetical closure
  // derivative and must never be interpreted as the production Newton Jv.
  for (int mode=0;mode<2;++mode) {
    std::vector<AthenaArray<Real>> base_coeff(saved.size());
    for (std::size_t b=0;b<saved.size();++b) {
      Saved &s=saved[b]; NRFLD *nr=s.nr;
      nr->u_=s.baseline;
      nr->derivetive_=s.deriv;
      if (mode) {
        nr->CalculateCoefficientsOnce(nr->u_,nr->pmy_block_->phydro->w,
                                       nr->def_coeff_,nr->derivetive_);
      }
    }
    if (NRFLDPrototype::SharedCoeff()) SyncInterfaceFaceMoments();
    for (std::size_t b=0;b<saved.size();++b) {
      Saved &s=saved[b]; NRFLD *nr=s.nr;
      if (mode) {
        nr->fixed_linear_coefficients_initialized_=false;
        nr->CalculateCoefficientsTask(dt_);
        Snapshot(base_coeff[b],nr->coeff_);
      } else Snapshot(base_coeff[b],s.coeff);
    }
    for (int pass=0;pass<4;++pass) {
      const Real h=global_jv_epsilon_*std::pow(0.1,pass);
      for (Saved &s:saved) {
        NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
        nr->u_=s.baseline;
        for (int k=mb->ks;k<=mb->ke;++k)
          for (int j=mb->js;j<=mb->je;++j)
            for (int i=mb->is;i<=mb->ie;++i)
              nr->u_(k,j,i)+=h*s.direction(k,j,i);
      }
      boundary.DoTaskListOneStage(stage_);
      for (Saved &s:saved) {
        NRFLD *nr=s.nr;
        if (!mode) nr->derivetive_=s.deriv;
        if (mode)
          nr->CalculateCoefficientsOnce(nr->u_,nr->pmy_block_->phydro->w,
                                         nr->def_coeff_,nr->derivetive_);
      }
      if (NRFLDPrototype::SharedCoeff()) SyncInterfaceFaceMoments();
      if (mode)
        for (Saved &s:saved) {
          s.nr->fixed_linear_coefficients_initialized_=false;
          s.nr->CalculateCoefficientsTask(dt_);
        }
      std::array<Sums,4> sums;
      for (std::size_t b=0;b<saved.size();++b) {
        Saved &s=saved[b]; NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
        if (!mode) nr->coeff_=base_coeff[b];
        for (int k=0;k<nr->u_.GetDim3();++k)
          for (int j=0;j<nr->u_.GetDim2();++j)
            for (int i=0;i<nr->u_.GetDim1();++i)
              s.tangent(k,j,i)=(nr->u_(k,j,i)-s.baseline(k,j,i))/h;
        const Real fac=dt_/SQR(mb->pcoord->dx1f(mb->is));
        const Real volume=mb->pcoord->GetCellVolume(mb->ks,mb->js,mb->is);
        for (int k=mb->ks;k<=mb->ke;++k)
          for (int j=mb->js;j<=mb->je;++j)
            for (int i=mb->is;i<=mb->ie;++i) {
              const long double fd=(Residual(nr->u_,nr->uold_,nr->coeff_,fac,k,j,i)
                      -Residual(s.baseline,nr->uold_,base_coeff[b],fac,k,j,i))/h;
              // A(v) = v + dt sum c_face*(v_nb-v); no exchange in Test C.
              long double av=s.direction(k,j,i), tangent=s.tangent(k,j,i);
              for (int f=0;f<6;++f) {
                const long double c=static_cast<long double>(fac)*base_coeff[b](cf[f],k,j,i);
                av+=c*(static_cast<long double>(s.direction(k+dk[f],j+dj[f],i+di[f]))
                       -s.direction(k,j,i));
                tangent+=c*(static_cast<long double>(s.tangent(k+dk[f],j+dj[f],i+di[f]))
                            -s.tangent(k,j,i));
              }
              const long double error=fd-av;
              for (int region : {Region(mb,k,j,i),3}) {
                Sums &a=sums[region];
                a.error+=volume*error*error; a.fd+=volume*fd*fd; a.av+=volume*av*av;
                a.tangent_error+=volume*(fd-tangent)*(fd-tangent); ++a.cells;
                for (int f=0;f<6;++f) {
                  const Real v=s.direction(k+dk[f],j+dj[f],i+di[f]);
                  const Real t=s.tangent(k+dk[f],j+dj[f],i+di[f]);
                  a.ghost_error+=volume*SQR(t-v); a.ghost_ref+=volume*SQR(v);
                }
                if (std::abs(error)>a.max_error) {
                  a.max_error=std::abs(error); a.gid=mb->gid; a.level=mb->loc.level;
                  a.x=mb->pcoord->x1v(i); a.y=mb->pcoord->x2v(j); a.z=mb->pcoord->x3v(k);
                }
              }
            }
      }
      const char *regions[4]={"interior","same_level_face","coarse_fine_face","all"};
      for (int region=0;region<4;++region) {
        Sums &a=sums[region];
#ifdef MPI_PARALLEL
        long double totals[6]={a.error,a.fd,a.av,a.tangent_error,
                               a.ghost_error,a.ghost_ref};
        MPI_Allreduce(MPI_IN_PLACE,totals,6,MPI_LONG_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        a.error=totals[0]; a.fd=totals[1]; a.av=totals[2];
        a.tangent_error=totals[3]; a.ghost_error=totals[4]; a.ghost_ref=totals[5];
        const std::int64_t local_cells=a.cells;
        MPI_Allreduce(MPI_IN_PLACE,&a.cells,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
        const Real local_max=a.max_error;
        MPI_Allreduce(MPI_IN_PLACE,&a.max_error,1,MPI_ATHENA_REAL,MPI_MAX,
                      MPI_COMM_WORLD);
        int owner=(local_max==a.max_error && local_cells>0) ? Globals::my_rank : nranks_;
        MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        if (a.cells>0) {
          Real xyz[3]={a.x,a.y,a.z}; int ids[2]={a.gid,a.level};
          MPI_Bcast(xyz,3,MPI_ATHENA_REAL,owner,MPI_COMM_WORLD);
          MPI_Bcast(ids,2,MPI_INT,owner,MPI_COMM_WORLD);
          a.x=xyz[0]; a.y=xyz[1]; a.z=xyz[2]; a.gid=ids[0]; a.level=ids[1];
        }
#endif
        if (!a.cells || Globals::my_rank!=0) continue;
        const long double ref=std::max({a.fd,a.av,1.e-300L});
        std::cout << "[NRFLD global Jv] direction=" << (actual_direction ? "newton" : "smooth")
                  << " iteration=" << (actual_direction ? global_jv_solve_ : -1)
                  << " closure=" << (mode ? "refreshed" : "frozen")
                  << " epsilon=" << h << " region=" << regions[region]
                  << " cells=" << a.cells << " l2_rel=" << std::sqrt(a.error/ref)
                  << " tangent_l2_rel=" << std::sqrt(a.tangent_error/ref)
                  << " ghost_l2_rel=" << std::sqrt(a.ghost_error/std::max(a.ghost_ref,1.e-300L))
                  << " max_abs=" << a.max_error << " gid=" << a.gid << " level=" << a.level
                  << " x=" << a.x << " y=" << a.y << " z=" << a.z << std::endl;
      }
    }
  }
  restore();
  std::cout.flags(flags); std::cout.precision(precision);
}
