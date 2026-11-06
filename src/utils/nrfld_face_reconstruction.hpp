#ifndef UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_
#define UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_

#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "nrfld_prototype_flags.hpp"

namespace NRFLDFaceReconstruction {
// The two fine cells in each tangential direction share one coarse cell.
// Their difference estimates the coarse-center to fine-subface offset for a
// linear field. The same linear expression applies to an iterate and a Jv.
inline Real TangentialOffset(const AthenaArray<Real> &u, int n,
                             const MeshBlock *mb, int face,
                             int k, int j, int i) {
  const int pi=mb->is+((i-mb->is)^1);
  const int pj=mb->js+((j-mb->js)^1);
  const int pk=mb->ks+((k-mb->ks)^1);
  const Real center=u(n,k,j,i);
  if (face < 2)
    return 0.5*(center-u(n,k,pj,i))+0.5*(center-u(n,pk,j,i));
  if (face < 4)
    return 0.5*(center-u(n,k,j,pi))+0.5*(center-u(n,pk,j,i));
  return 0.5*(center-u(n,k,j,pi))+0.5*(center-u(n,k,pj,i));
}

inline Real NextFine(const AthenaArray<Real> &u, int n, int face,
                     int k, int j, int i) {
  const int inward=(face%2==0 ? 1 : -1);
  if (face < 2) return u(n,k,j,i+inward);
  if (face < 4) return u(n,k,j+inward,i);
  return u(n,k+inward,j,i);
}

inline Real CoarseScale() {
  return 4.0/3.0-(4.0/15.0)*NRFLDPrototype::NormalWeight();
}

inline Real EffectiveFine(const AthenaArray<Real> &u, int n, int face,
                          int k, int j, int i, Real tangential_offset) {
  const Real theta=NRFLDPrototype::NormalWeight();
  const Real a=2.0/3.0-(2.0/15.0)*theta;
  const Real next_weight=(theta/5.0)/a;
  const Real center=u(n,k,j,i);
  const Real next=theta>0.0 ? NextFine(u,n,face,k,j,i) : center;
  return (1.0-next_weight)*center+next_weight*next-tangential_offset;
}

inline Real FineGhost(Real coarse_face, Real fine_center, Real fine_next) {
  const Real old_ghost=(fine_center+2.0*coarse_face)/3.0;
  const Real quadratic_ghost=(8.0/15.0)*coarse_face
      +(2.0/3.0)*fine_center-(1.0/5.0)*fine_next;
  return old_ghost+NRFLDPrototype::NormalWeight()*(quadratic_ghost-old_ghost);
}
}  // namespace NRFLDFaceReconstruction

#endif  // UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_
