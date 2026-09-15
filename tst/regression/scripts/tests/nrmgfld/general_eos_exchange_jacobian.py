"""Differentiate the production gas residual after changing the Newton state."""

import glob
import os
import subprocess

import scripts.utils.athena as athena
from scripts.tests.nrmgfld.adiabatic_gradient import (
    _write_nonideal_table, _write_opacity_table)


_PROBE = r'''
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "nr_multigrid/NRFLD.hpp"
#include <fstream>
#include <cmath>
#include <iostream>
#include <iomanip>

// A pointer to a protected member avoids casting an existing base object to
// a fictitious derived object. No test-only accessors enter production code.
struct Access : NRFLD {
  static AthenaArray<Real> NRFLD::* Gas() { return &Access::u_gas_; }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &Globals::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks);
  bool ok = true;
  {
    ParameterInput pin;
    std::ifstream input(argv[1]);
    pin.LoadFromStream(input);
    pin.SetString("hydro", "eos_file_name", argv[2]);
    pin.SetString("fld", "opacity_table_file", argv[3]);
    Mesh mesh(&pin);
    MeshBlock *mb = mesh.my_blocks(0);
    auto *nr = static_cast<NRFLD*>(mb->pnr);
    auto &gas = nr->*Access::Gas();
    auto *fld = mb->prfld;
    fld->c_ph = 10.0;
    fld->a_r = 1.0;
    for (int k=0; k<mb->ncells3; ++k)
      for (int j=0; j<mb->ncells2; ++j)
        for (int i=0; i<mb->ncells1; ++i) {
          mb->phydro->w(IDN,k,j,i) = 1.0;
          mb->phydro->w(IPR,k,j,i) = mb->peos->PresFromRhoEg(1.0,1.0);
          fld->u_rad(k,j,i) = 0.5;
          fld->sigma_p(k,j,i) = fld->sigma_r(k,j,i) = 1.0;
        }
    nr->LoadVariables();
    nr->CalculateCoefficientsOnce(nr->u_, mb->phydro->w,
                                  nr->def_coeff_, nr->derivetive_);
    const int k=mb->ks+1, j=mb->js+1, i=mb->is+1;
    const Real dt = 0.2;
    auto assemble = [&]() {
      nr->CalculateCoefficients(nr->uold_, nr->u_, nr->def_coeff_,
                                 nr->coeff_, nr->derivetive_, nr->src_, dt);
    };
    for (Real energy : {0.2, 0.7, 1.8}) {
      gas(k,j,i) = energy;
      assemble();
      const Real analytic = nr->derivetive_(NewtonRaphsonFLD::dFg_deg,k,j,i);
      const Real h = 1.e-6*energy;
      gas(k,j,i) = energy+h;
      assemble();
      const Real plus = nr->derivetive_(NewtonRaphsonFLD::Fg,k,j,i);
      gas(k,j,i) = energy-h;
      assemble();
      const Real minus = nr->derivetive_(NewtonRaphsonFLD::Fg,k,j,i);
      const Real numerical = (plus-minus)/(2*h);
      const Real error = std::abs(analytic-numerical)/std::abs(numerical);
      std::cout << std::setprecision(16) << "energy=" << energy
                << " relative_error=" << error << '\n';
      ok &= std::isfinite(error) && error < 1.e-7;
    }
  }  // Destroy mesh and its communicators before MPI_Finalize.
  MPI_Finalize();
  if (ok) std::cout << "PASS\n";
  return ok ? 0 : 1;
}
'''


def prepare(**kwargs):
    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='simple_convection_lhllc_fld', coord='cartesian',
                     flux='lhllc', eos='general/eos_table', cxx='g++', **kwargs)
    athena.make()
    root = os.path.abspath(athena.athena_rel_path)
    # A header-only API update must rebuild callers, including main's
    # by-value NewtonSolveResult. Query make without changing source mtimes.
    database = subprocess.run([
        'make', '-qp', 'OBJ_DIR:='+os.path.abspath('obj')+'/',
        'EXE_DIR:='+os.path.abspath('bin')+'/',
    ], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if database.returncode not in (0, 1):
        raise RuntimeError(database.stderr)
    target = os.path.abspath('obj/main.o')+':'
    prerequisites = next(line for line in database.stdout.splitlines()
                         if line.startswith(target))
    required = ('src/newton_raphson/Newton_Raphson.hpp', 'src/defs.hpp', 'Makefile')
    if any(header not in prerequisites.split() for header in required) \
            or '.DEFAULT_GOAL := all' not in database.stdout:
        raise RuntimeError('make does not track header/configuration changes')
    objects = sorted(glob.glob(os.path.abspath(os.path.join('obj', '*.o'))))
    objects = [p for p in objects if os.path.basename(p) != 'main.o']
    command = ['mpicxx', '-std=c++11', '-O2', '-I'+os.path.join(root, 'src'),
               '-x', 'c++', '-', '-x', 'none'] + objects
    command += ['-lhdf5', '-o', 'bin/general_eos_exchange_jacobian']
    subprocess.run(command, input=_PROBE.encode(), check=True)
    _write_nonideal_table('bin/exchange_jacobian_eos.tab')
    _write_opacity_table('bin/exchange_jacobian_opacity.h5')


def run(**kwargs):
    command = [kwargs['mpirun_cmd']] + kwargs['mpirun_opts']
    if os.geteuid() == 0 and kwargs['mpirun_cmd'] == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command += [
        '-n', '1', './general_eos_exchange_jacobian',
        '../'+athena.athena_rel_path+'inputs/radiation/athinput.adiabatic_gradient',
        'exchange_jacobian_eos.tab', 'exchange_jacobian_opacity.h5']
    with open('bin/general_eos_exchange_jacobian.log', 'w') as log:
        subprocess.check_call(command, cwd='bin', stdout=log,
                              stderr=subprocess.STDOUT)


def analyze():
    with open('bin/general_eos_exchange_jacobian.log') as log:
        return 'PASS' in log.read().splitlines()
