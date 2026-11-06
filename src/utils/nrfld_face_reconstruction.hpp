#ifndef UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_
#define UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_

#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"

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
}  // namespace NRFLDFaceReconstruction

#endif  // UTILS_NRFLD_FACE_RECONSTRUCTION_HPP_
