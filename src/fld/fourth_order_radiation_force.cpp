//========================================================================================
//! \file fourth_order_radiation_force.cpp
//! \brief Opt-in cell-average FLD force for smooth, uniform 3D meshes.

#include <cmath>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "fld.hpp"

namespace {
// Reconstruct a quartic from the five cell averages centered at offsets
// -2..2.  Values and dimensionless derivatives are evaluated at the two
// Gauss points s=+/-1/(2 sqrt(3)) of the central cell.  The coefficients are
// the inverse of the exact polynomial cell-average moment matrix.
constexpr Real value_weight[2][5] = {
  {-0.028297119567088226,  0.20139476939454554, 0.99861111111111167,
   -0.19954291754269499,  0.02783415660412537},
  { 0.027834156604125374, -0.19954291754269454, 0.99861111111111167,
    0.20139476939454509, -0.028297119567088226}
};
constexpr Real gradient_weight[2][5] = {
  {0.11540834828831258, -1.0836418610813972,  0.76980035891950227,
   0.24969147225193614, -0.051258318378354005},
  {0.051258318378354005, -0.24969147225193614, -0.76980035891950227,
   1.0836418610813972, -0.11540834828831258}
};
}  // namespace

bool FLD::FourthOrderRadiationForce(int k, int j, int i,
                                    const AthenaArray<Real> &prim,
                                    Real force[3]) const {
  const MeshBlock *pmb = pmy_block;
  const Real hx = pmb->pcoord->dx1f(i);
  const Real hy = pmb->pcoord->dx2f(j);
  const Real hz = pmb->pcoord->dx3f(k);
  Real erad[5][5][5], rho[5][5][5], kappa[5][5][5];
  for (int z = 0; z < 5; ++z) {
    for (int y = 0; y < 5; ++y) {
      for (int x = 0; x < 5; ++x) {
        const int kk = k + z - 2, jj = j + y - 2, ii = i + x - 2;
        const Real e = u_rad(kk,jj,ii);
        const Real r = prim(IDN,kk,jj,ii);
        const Real sigma = sigma_r(kk,jj,ii);
        if (!(e > 0.0) || !(r > 0.0) || !(sigma >= 0.0)
            || !std::isfinite(e) || !std::isfinite(r)
            || !std::isfinite(sigma)) return false;
        erad[z][y][x] = e;
        rho[z][y][x] = r;
        kappa[z][y][x] = sigma/r;
      }
    }
  }

  force[0] = force[1] = force[2] = 0.0;
  Real averaged_gradient[3] = {0.0, 0.0, 0.0};
  for (int qz = 0; qz < 2; ++qz) {
    for (int qy = 0; qy < 2; ++qy) {
      for (int qx = 0; qx < 2; ++qx) {
        Real e = 0.0, r = 0.0, kap = 0.0;
        Real gx = 0.0, gy = 0.0, gz = 0.0;
        for (int z = 0; z < 5; ++z) {
          const Real vz = value_weight[qz][z];
          const Real dz = gradient_weight[qz][z];
          for (int y = 0; y < 5; ++y) {
            const Real vy = value_weight[qy][y];
            const Real dy = gradient_weight[qy][y];
            for (int x = 0; x < 5; ++x) {
              const Real vx = value_weight[qx][x];
              const Real dx = gradient_weight[qx][x];
              const Real sample = erad[z][y][x];
              const Real weight = vz*vy*vx;
              e += weight*sample;
              r += weight*rho[z][y][x];
              kap += weight*kappa[z][y][x];
              gx += vz*vy*dx*sample;
              gy += vz*dy*vx*sample;
              gz += dz*vy*vx*sample;
            }
          }
        }
        gx /= hx;
        gy /= hy;
        gz /= hz;
        const Real sigma = r*kap;
        if (!(e > 0.0) || !(r > 0.0) || !(kap >= 0.0)
            || !std::isfinite(e) || !std::isfinite(r)
            || !std::isfinite(sigma) || !std::isfinite(gx)
            || !std::isfinite(gy) || !std::isfinite(gz)) return false;
        const RadFLD::ClosureValues closure = RadFLD::EvaluateClosure(
            std::hypot(gx, std::hypot(gy, gz)), sigma, e,
            fixed_flux_limiter);
        if (!closure.valid) return false;
        const Real factor = -0.125*closure.lambda;
        force[0] += factor*gx;
        force[1] += factor*gy;
        force[2] += factor*gz;
        averaged_gradient[0] += 0.125*gx;
        averaged_gradient[1] += 0.125*gy;
        averaged_gradient[2] += 0.125*gz;
      }
    }
  }
  // Anchor the constant-limiter component to the same Godunov face Er
  // difference as the gas pressure flux.  This exactly preserves its
  // discrete pressure balance.  The difference between this face gradient
  // and the quadrature gradient is O(h^4) on smooth xorder=4 meshes, so the
  // variable-limiter product retains fourth-order accuracy.
  const Real lambda_ref = rad_state_cc_(RadFLD::LAMBDA,k,j,i);
  const Real face_gradient[3] = {
    (rad_face_g[X1DIR](k,j,i+1)-rad_face_g[X1DIR](k,j,i))/hx,
    (rad_face_g[X2DIR](k,j+1,i)-rad_face_g[X2DIR](k,j,i))/hy,
    (rad_face_g[X3DIR](k+1,j,i)-rad_face_g[X3DIR](k,j,i))/hz
  };
  for (int direction = 0; direction < 3; ++direction)
    force[direction] += lambda_ref*(averaged_gradient[direction]
                                    - face_gradient[direction]);
  return std::isfinite(force[0]) && std::isfinite(force[1])
      && std::isfinite(force[2]);
}
