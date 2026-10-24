// Optional Test C operator audit. No diagnostic arrays exist when disabled.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#include "NRFLD.hpp"
#include "../coordinates/coordinates.hpp"
#include "../linear_solver/linear_solver.hpp"
#include "../task_list/nr_task_list.hpp"

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
  for (int f=0;f<6;++f) if (face[f]) {
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

void NRFLDDriver::CheckGlobalJacobian(bool actual_direction) {
  if (nranks_!=1) {
    std::stringstream msg;
    msg << "global_jacobian_check currently requires one MPI rank (Test C).";
    ATHENA_ERROR(msg);
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
    AthenaArray<Real> u, delta, coarse_u, coarse_delta, coeff, deriv, defcoeff,
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
      nr->derivetive_=s.deriv; nr->def_coeff_=s.defcoeff; nr->src_=s.src;
      nr->mixed_drift_=s.mixed; nr->pmy_block_->prfld->marshak_dface=s.marshak;
      nr->fixed_linear_coefficients_initialized_=s.initialized;
    }
  };
  NewtonRaphsonTaskList boundary(this,NewtonRaphsonTaskList::Mode::boundary);
  boundary.DoTaskListOneStage(stage_);
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
  // Normalize the actual MG direction to avoid positivity clipping. Preserve
  // its shape and sign; epsilon is relative to the energy, not to raw delta.
  if (actual_direction && amplitude==0.0) {
    restore();
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
  std::cout << "[NRFLD Jv setup] direction=" << (actual_direction ? "newton" : "smooth")
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
      if (mode) {
        nr->CalculateCoefficientsOnce(nr->u_,nr->pmy_block_->phydro->w,
                                       nr->def_coeff_,nr->derivetive_);
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
      std::array<Sums,4> sums;
      for (std::size_t b=0;b<saved.size();++b) {
        Saved &s=saved[b]; NRFLD *nr=s.nr; MeshBlock *mb=nr->pmy_block_;
        if (mode) {
          nr->CalculateCoefficientsOnce(nr->u_,mb->phydro->w,nr->def_coeff_,nr->derivetive_);
          nr->fixed_linear_coefficients_initialized_=false;
          nr->CalculateCoefficientsTask(dt_);
        } else nr->coeff_=base_coeff[b];
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
        const Sums &a=sums[region]; if (!a.cells) continue;
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
