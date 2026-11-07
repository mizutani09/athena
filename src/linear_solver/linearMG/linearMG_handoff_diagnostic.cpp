// Single-rank Test C audit of the MG-to-Newton correction handoff.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include "../../globals.hpp"
#include "linearMG.hpp"
#include "../linear_solver.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../newton_raphson/Newton_Raphson.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

void linearMGDriver::PrintHandoffDiagnostic(const char *stage, bool native) {
  const int di[6]={-1,1,0,0,0,0}, dj[6]={0,0,-1,1,0,0}, dk[6]={0,0,0,0,-1,1};
  const int cf[6]={linearSolver::DXMF,linearSolver::DXPF,linearSolver::DYMF,
                   linearSolver::DYPF,linearSolver::DZMF,linearSolver::DZPF};
  const int cs[6]={linearSolver::DXMS,linearSolver::DXPS,linearSolver::DYMS,
                   linearSolver::DYPS,linearSolver::DZMS,linearSolver::DZPS};
  const int mm[6]={linearSolver::CCM,linearSolver::CCP,linearSolver::CMC,
                   linearSolver::CPC,linearSolver::MCC,linearSolver::PCC};
  struct Sum {
    long double r2{0}, b2{0}, double_r2{0}, native_r2{0}, roundoff2{0};
    long double delta_integral{0}, delta2_integral{0}, rhs_integral{0};
    long double r_integral{0}, volume{0};
    Real delta_min{std::numeric_limits<Real>::max()};
    Real delta_max{-std::numeric_limits<Real>::max()};
    Real updated_min{std::numeric_limits<Real>::max()};
    std::int64_t nonpositive{0};
    Real maximum{0}, x{0}, y{0}, z{0};
    int gid{-1},level{-1}; std::int64_t cells{0};
  };
  std::array<Sum,4> sums;
  for (auto base:vmg_) {
    linearMG *mg=static_cast<linearMG*>(base);
    NewtonRaphson *nr=mg->pnr_; MeshBlock *mb=mg->pmy_block_;
    const int l=mg->nlevel_-1;
    auto u=[&](int k,int j,int i) {
      return native ? mg->u_[l](k-mb->ks+mg->ngh_,j-mb->js+mg->ngh_,i-mb->is+mg->ngh_)
                    : nr->delta_u_(k,j,i);
    };
    const auto &c=nr->coeff_;
    const Real fac=dt_/SQR(mb->pcoord->dx1f(mb->is));
    const Real volume=mb->pcoord->GetCellVolume(mb->ks,mb->js,mb->is);
    for (int k=mb->ks;k<=mb->ke;++k)
      for (int j=mb->js;j<=mb->je;++j)
        for (int i=mb->is;i<=mb->ie;++i) {
          const Real center=u(k,j,i), rhs=nr->src_(k,j,i);
          const Real diag=fac*c(linearSolver::DCCF,k,j,i)+c(linearSolver::DCCS,k,j,i);
          Real direct=diag*center;
          long double row_f=c(linearSolver::DCCF,k,j,i);
          long double stable=static_cast<long double>(c(linearSolver::DCCS,k,j,i))*center;
          const int mk=k-mb->ks+mg->ngh_,mj=j-mb->js+mg->ngh_,mi=i-mb->is+mg->ngh_;
          long double native_av=static_cast<long double>(mg->matrix_[l](linearSolver::CCC,mk,mj,mi))*center;
          int region=0;
          const bool on[6]={i==mb->is,i==mb->ie,j==mb->js,j==mb->je,k==mb->ks,k==mb->ke};
          for (int f=0;f<6;++f) {
            const Real nb=u(k+dk[f],j+dj[f],i+di[f]);
            row_f+=c(cf[f],k,j,i);
            stable+=static_cast<long double>(fac)*c(cf[f],k,j,i)
                      *(static_cast<long double>(nb)-center)
                    +static_cast<long double>(c(cs[f],k,j,i))*nb;
            direct+=(fac*c(cf[f],k,j,i)+c(cs[f],k,j,i))*nb;
            native_av+=static_cast<long double>(mg->matrix_[l](mm[f],mk,mj,mi))*nb;
            if (on[f]) region=std::max(region,
                mb->pbval->nblevel[1+dk[f]][1+dj[f]][1+di[f]]==mb->loc.level ? 1:2);
          }
          stable+=static_cast<long double>(fac)*row_f*center;
          const long double r=rhs-stable, rd=rhs-direct;
          const long double rn=mg->src_[l](mk,mj,mi)-native_av;
          for (int category:{region,3}) {
            Sum &s=sums[category]; ++s.cells;
            s.r2+=volume*r*r; s.b2+=volume*static_cast<long double>(rhs)*rhs;
            s.double_r2+=volume*rd*rd; s.native_r2+=volume*rn*rn;
            s.roundoff2+=volume*(r-rd)*(r-rd);
            s.volume+=volume; s.delta_integral+=volume*static_cast<long double>(center);
            s.delta2_integral+=volume*static_cast<long double>(center)*center;
            s.rhs_integral+=volume*static_cast<long double>(rhs);
            s.r_integral+=volume*r;
            s.delta_min=std::min(s.delta_min,center); s.delta_max=std::max(s.delta_max,center);
            const Real updated=nr->u_(k,j,i)+center;
            s.updated_min=std::min(s.updated_min,updated);
            if (updated<=0.0) ++s.nonpositive;
            if (std::abs(r)>s.maximum) {
              s.maximum=std::abs(r); s.gid=mb->gid; s.level=mb->loc.level;
              s.x=mb->pcoord->x1v(i); s.y=mb->pcoord->x2v(j); s.z=mb->pcoord->x3v(k);
            }
          }
        }
  }
  const auto flags=std::cout.flags(); const auto precision=std::cout.precision();
  std::cout << std::scientific << std::setprecision(16);
  const char *regions[4]={"interior","same_level_face","coarse_fine_face","all"};
  for (int region=0;region<4;++region) {
    Sum &s=sums[region];
#ifdef MPI_PARALLEL
    long double totals[10]={s.r2,s.b2,s.double_r2,s.native_r2,s.roundoff2,
                            s.delta_integral,s.delta2_integral,s.rhs_integral,
                            s.r_integral,s.volume};
    MPI_Allreduce(MPI_IN_PLACE,totals,10,MPI_LONG_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    s.r2=totals[0]; s.b2=totals[1]; s.double_r2=totals[2]; s.native_r2=totals[3];
    s.roundoff2=totals[4]; s.delta_integral=totals[5];
    s.delta2_integral=totals[6]; s.rhs_integral=totals[7];
    s.r_integral=totals[8]; s.volume=totals[9];
    const std::int64_t local_cells=s.cells;
    std::int64_t counts[2]={s.cells,s.nonpositive};
    MPI_Allreduce(MPI_IN_PLACE,counts,2,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
    s.cells=counts[0]; s.nonpositive=counts[1];
    const Real local_max=s.maximum;
    Real minima[2]={s.delta_min,s.updated_min}, maxima[2]={s.delta_max,s.maximum};
    MPI_Allreduce(MPI_IN_PLACE,minima,2,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,maxima,2,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
    s.delta_min=minima[0]; s.updated_min=minima[1];
    s.delta_max=maxima[0]; s.maximum=maxima[1];
    int owner=(local_cells>0 && local_max==s.maximum) ? Globals::my_rank
                                                     : Globals::nranks;
    MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if (s.cells>0) {
      Real xyz[3]={s.x,s.y,s.z}; int ids[2]={s.gid,s.level};
      MPI_Bcast(xyz,3,MPI_ATHENA_REAL,owner,MPI_COMM_WORLD);
      MPI_Bcast(ids,2,MPI_INT,owner,MPI_COMM_WORLD);
      s.x=xyz[0]; s.y=xyz[1]; s.z=xyz[2]; s.gid=ids[0]; s.level=ids[1];
    }
#endif
    if (!s.cells || Globals::my_rank!=0) continue;
    const auto denom=std::max(s.b2,1.e-300L);
    const long double delta_mean=s.delta_integral/s.volume;
    const long double fluctuation=std::max(0.0L,
        s.delta2_integral/s.volume-delta_mean*delta_mean);
    std::cout << "[NRFLD handoff] solve=" << handoff_diagnostic_count_
              << " stage=" << stage << " region=" << regions[region] << " cells=" << s.cells
              << " common_rel=" << std::sqrt(s.r2/denom)
              << " double_rel=" << std::sqrt(s.double_r2/denom)
              << " native_row_rel=" << std::sqrt(s.native_r2/denom)
              << " roundoff_rel=" << std::sqrt(s.roundoff2/denom)
              << " delta_min=" << s.delta_min << " delta_max=" << s.delta_max
              << " delta_mean=" << delta_mean
              << " delta_fluct_rms=" << std::sqrt(fluctuation)
              << " rhs_mean=" << s.rhs_integral/s.volume
              << " residual_mean=" << s.r_integral/s.volume
              << " rhs_rms=" << std::sqrt(s.b2/s.volume)
              << " residual_rms=" << std::sqrt(s.r2/s.volume)
              << " full_step_min=" << s.updated_min << " nonpositive=" << s.nonpositive
              << " smoothing_only=" << smoothing_only_
              << " max_abs=" << s.maximum << " gid=" << s.gid << " level=" << s.level
              << " x=" << s.x << " y=" << s.y << " z=" << s.z << std::endl;
  }
  std::cout.flags(flags); std::cout.precision(precision);
}
