#ifndef UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_
#define UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_

#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"

namespace NRFLDFaceReconstruction {
inline Real CoarseReference(const AthenaArray<Real> &coarse,
                           const MeshBlock *mb, int face,
                           int k, int j, int i) {
  const int ck=mb->cks+(k-mb->ks)/2;
  const int cj=mb->cjs+(j-mb->js)/2;
  const int ci=mb->cis+(i-mb->is)/2;
  if (face == 0) return coarse(0,ck,cj,mb->cis-1);
  if (face == 1) return coarse(0,ck,cj,mb->cie+1);
  if (face == 2) return coarse(0,ck,mb->cjs-1,ci);
  if (face == 3) return coarse(0,ck,mb->cje+1,ci);
  if (face == 4) return coarse(0,mb->cks-1,cj,ci);
  return coarse(0,mb->cke+1,cj,ci);
}
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
}  // namespace NRFLDFaceReconstruction

#endif  // UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_
