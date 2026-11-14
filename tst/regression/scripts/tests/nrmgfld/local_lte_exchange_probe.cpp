// Exercise the production NRFLD::AddDifference local gas solve on one cell.
// The Python regression supplies independent scalar reference roots.
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "fld/fld.hpp"
#include "nr_multigrid/NRFLD.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

struct GasAccess : NRFLD {
  static AthenaArray<Real> NRFLD::* Gas() { return &GasAccess::u_gas_; }
  static NewtonRaphsonDriver *NRFLD::* Driver() { return &GasAccess::pmy_driver_; }
};

struct DriverAccess : NewtonRaphsonDriver {
  static Real NewtonRaphsonDriver::* Dt() { return &DriverAccess::dt_; }
  static Real NewtonRaphsonDriver::* Step() { return &DriverAccess::step_scale_; }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &Globals::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks);
  {
    ParameterInput pin;
    std::ifstream input(argv[1]);
    pin.LoadFromStream(input);
    pin.SetString("hydro", "eos_file_name", argv[2]);
    pin.SetString("fld", "opacity_table_file", argv[3]);
    pin.SetString("fld", "include_mixed_frame_terms", "false");
    pin.SetString("nrfld", "exact_local_gas_solve", "true");
    const bool material = argc > 7 && std::string(argv[7]) == "material";
    if (material) {
      pin.SetString("hydro", "eos_rho_unit", "1e-8");
      pin.SetString("hydro", "eos_egas_unit", "1e5");
      pin.SetString("hydro", "eos_temp_unit", "1");
      pin.SetString("hydro", "eos_table_clamp", "false");
      pin.SetString("hydro", "dfloor", "1e-13");
      pin.SetString("hydro", "pfloor", "1.4e-15");
    }
    Mesh mesh(&pin);
    MeshBlock *mb = mesh.my_blocks(0);
    auto *nr = static_cast<NRFLD*>(mb->pnr);
    auto *fld = mb->prfld;
    auto *driver = nr->*GasAccess::Driver();
    driver->*DriverAccess::Dt() = 1.0;
    driver->*DriverAccess::Step() = 1.0;
    fld->a_r = argc > 6 ? std::atof(argv[6]) : 1.0;
    fld->c_ph = 1.0;
    fld->is_couple = true;
    fld->only_rad = false;
    const int k = mb->ks+1, j = mb->js+1, i = mb->is+1;
    const Real rho = argc > 5 ? std::atof(argv[5]) : 1.0;
    AthenaArray<Real> delta;
    delta.NewAthenaArray(mb->ncells3, mb->ncells2, mb->ncells1);
    std::ifstream cases(argv[4]);
    int id, fixed;
    Real old_gas, old_rad, new_rad, coupling;
    std::cout << std::setprecision(17);
    while (cases >> id >> fixed >> old_gas >> old_rad >> new_rad >> coupling) {
      fld->fixed_u_rad = fixed != 0;
      fld->cut_diff = material && !fixed;
      fld->sigma_p(k,j,i) = coupling;
      for (int z=0; z<mb->ncells3; ++z)
        for (int y=0; y<mb->ncells2; ++y)
          for (int x=0; x<mb->ncells1; ++x) {
            mb->phydro->w(IDN,z,y,x) = rho;
            mb->phydro->w(IPR,z,y,x) = mb->peos->PresFromRhoEg(rho, old_gas);
            mb->phydro->u(IDN,z,y,x) = rho;
            mb->phydro->u(IM1,z,y,x) = 0.0;
            mb->phydro->u(IM2,z,y,x) = 0.0;
            mb->phydro->u(IM3,z,y,x) = 0.0;
            mb->phydro->u(IEN,z,y,x) = old_gas;
            fld->u_rad(z,y,x) = old_rad;
          }
      nr->LoadVariables();
      (nr->*GasAccess::Gas())(k,j,i) = old_gas;
      nr->u_(k,j,i) = old_rad;
      nr->uold_(k,j,i) = old_rad;
      nr->derivetive_(NewtonRaphsonFLD::Fg,k,j,i) = 0.0;
      nr->derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i) = 1.0;
      nr->derivetive_(NewtonRaphsonFLD::dFg_dEr,k,j,i) = 0.0;
      delta.ZeroClear();
      delta(k,j,i) = fixed ? 0.0 : new_rad-old_rad;
      nr->AddDifference(nr->u_, delta, nr->derivetive_);
      const Real gas = (nr->*GasAccess::Gas())(k,j,i);
      const Real rad = nr->u_(k,j,i);
      std::cout << id << ' ' << gas << ' ' << rad << ' '
                << mb->peos->TempFromRhoEg(rho, gas) << '\n';
    }
    // A cut-diffusion run must also suppress the physical Marshak face.
    // Otherwise a nominally local exchange test silently retains a sink.
    if (!material) {
      fld->cut_diff = true;
      fld->fixed_u_rad = false;
      fld->marshak_face_boundary[0] = true;
      fld->marshak_top_alpha = 0.5;
      fld->marshak_top_erad_ext = 0.0;
      nr->coeff_.ZeroClear();
      for (int z=0; z<mb->ncells3; ++z)
        for (int y=0; y<mb->ncells2; ++y)
          for (int x=0; x<mb->ncells1; ++x) {
            fld->u_gas(z,y,x) = (nr->*GasAccess::Gas())(z,y,x) = 1.0;
            nr->u_(z,y,x) = nr->uold_(z,y,x) = 1.0;
            nr->def_coeff_(NewtonRaphsonFLD::DRHO,z,y,x) = 1.0;
            nr->derivetive_(NewtonRaphsonFLD::dFg_deg,z,y,x) = 1.0;
            fld->marshak_dface_all(0,z,y,x) = 1.0;
          }
      nr->CalculateDefect(nr->def_, nr->u_, nr->uold_, nr->coeff_,
                          nr->def_coeff_, false);
      std::cout << "marshak_cut_diff " << nr->def_(mb->ks,mb->js,mb->is) << '\n';
      // Re-enable the face and balance its actual Marshak flux against the
      // radiation transient. The external gas input is applied before the FLD
      // solve, as in the production hydro source stage.
      fld->cut_diff = false;
      const Real dx = mb->pcoord->dx1f(mb->is);
      const Real dface = fld->marshak_dface_all(0,mb->ks,mb->js,mb->is);
      const Real acoef = fld->marshak_top_alpha*fld->c_ph*0.5*dx;
      const Real boundary_erad = dface/(dface+acoef);
      const Real outward_flux = fld->marshak_top_alpha*fld->c_ph*boundary_erad;
      const Real boundary_loss_density = outward_flux/dx;
      nr->uold_(mb->ks,mb->js,mb->is) = 1.0+boundary_loss_density;
      nr->CalculateDefect(nr->def_, nr->u_, nr->uold_, nr->coeff_,
                          nr->def_coeff_, false);
      const Real external_input = 0.1;
      const Real energy_balance = (1.0+1.0)
          - ((1.0-external_input)+(1.0+boundary_loss_density))
          - external_input + boundary_loss_density;
      std::cout << "marshak_flux_balance "
                << nr->def_(mb->ks,mb->js,mb->is) << ' '
                << energy_balance << ' ' << boundary_loss_density << '\n';
    }
  }
  MPI_Finalize();
}
