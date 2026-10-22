// Smooth periodic 3-D performance problem; supports HD/MHD, FLD and MG gravity.
#include <cmath>
#include <sstream>
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../fld/fld.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {
Real benchmark_dt;
Real FixedStep(MeshBlock*) { return benchmark_dt; }
Real RadiationEnergy(MeshBlock *pmb, int) {
  Real sum = 0.0;
#if NRMGFLD_ENABLED
  for (int k = pmb->ks; k <= pmb->ke; ++k)
    for (int j = pmb->js; j <= pmb->je; ++j)
      for (int i = pmb->is; i <= pmb->ie; ++i)
        sum += pmb->prfld->u_rad(k,j,i) * pmb->pcoord->GetCellVolume(k,j,i);
#endif
  return sum;
}
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  benchmark_dt = pin->GetReal("problem", "dt");
  if (!(benchmark_dt > 0.0)) {
    std::stringstream msg;
    msg << "benchmark dt must be positive";
    ATHENA_ERROR(msg);
  }
  EnrollUserTimeStepFunction(FixedStep);
  if (SELF_GRAVITY_ENABLED) SetGravitationalConstant(0.01);
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, RadiationEnergy, "Er_volume");
}

void MeshBlock::InitUserMeshBlockData(ParameterInput*) {
  AllocateUserOutputVariables(1);
  SetUserOutputVariableName(0, "Er");
}

void MeshBlock::ProblemGenerator(ParameterInput*) {
  const Real b0 = MAGNETIC_FIELDS_ENABLED ? 0.1 : 0.0;
  // Uniform face-centred B is exactly divergence-free, also across block boundaries.
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke+1; ++k)
      for (int j = js; j <= je+1; ++j)
        for (int i = is; i <= ie+1; ++i) {
          pfield->b.x1f(k,j,i) = b0;
          pfield->b.x2f(k,j,i) = b0;
          pfield->b.x3f(k,j,i) = b0;
        }
  }
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        const Real x = TWO_PI*pcoord->x1v(i);
        const Real y = TWO_PI*pcoord->x2v(j);
        const Real z = TWO_PI*pcoord->x3v(k);
        const Real rho = 1.0 + 0.1*std::sin(x)*std::sin(y)*std::sin(z);
        const Real egas = 1.0/(peos->GetGamma()-1.0);
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.1*rho;
        phydro->u(IM2,k,j,i) = 0.05*rho;
        phydro->u(IM3,k,j,i) = -0.02*rho;
        phydro->u(IEN,k,j,i) = egas + 0.5*rho*(0.01+0.0025+0.0004)
                                      + 1.5*b0*b0;
#if NRMGFLD_ENABLED
        prfld->u_gas(k,j,i) = egas;
        prfld->u_rad(k,j,i) = prfld->a_r*(1.0+0.1*std::cos(x)*std::cos(y)*std::cos(z));
#endif
      }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput*) {
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
#if NRMGFLD_ENABLED
        user_out_var(0,k,j,i) = prfld->u_rad(k,j,i);
#else
        user_out_var(0,k,j,i) = 0.0;
#endif
      }
}
