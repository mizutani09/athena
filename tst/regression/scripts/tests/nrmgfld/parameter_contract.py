"""Regression tests for the NR-FLD parameter and opacity contracts."""

import os
import subprocess

import h5py
import numpy as np

import scripts.utils.athena as athena


_LOGS = []


def _write_opacity_table(path):
    axis = np.array([-3.0, 0.0, 3.0])
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('axes/log10_density', data=axis)
        handle.create_dataset('axes/log10_temperature', data=axis)
        handle.create_dataset('kappa/log10_planck', data=np.full((3, 3), 0.0))
        handle.create_dataset('kappa/log10_rosseland', data=np.full((3, 3), 0.0))


def prepare(**kwargs):
    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='nrfld_sound', coord='cartesian', flux='lhllc',
                     eos='adiabatic', **kwargs)
    athena.make()
    os.rename('bin/athena', 'bin/contract_constant')

    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='simple_convection_lhllc_fld', coord='cartesian',
                     flux='lhllc', eos='adiabatic', **kwargs)
    athena.make()
    os.rename('bin/athena', 'bin/contract_table')
    _write_opacity_table('bin/contract_opacity.h5')


def _mpirun(executable, input_file, mpirun_cmd, mpirun_opts, extra):
    command = [mpirun_cmd]
    command.extend(mpirun_opts)
    if os.geteuid() == 0 and mpirun_cmd == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command.extend([
        '-n', '1', './' + executable, '-i',
        '../' + athena.athena_rel_path + 'inputs/radiation/' + input_file,
    ])
    command.extend(extra)
    return command


def _run(executable, input_file, name, mpirun_cmd, mpirun_opts, extra,
         expect_success=True):
    filename = 'bin/parameter_contract_{}.log'.format(name)
    environment = os.environ.copy()
    environment['ATHENA_FLD_OPACITY_DIAGNOSTICS'] = '1'
    with open(filename, 'w') as log_file:
        result = subprocess.run(
            _mpirun(executable, input_file, mpirun_cmd, mpirun_opts, extra),
            cwd='bin', stdout=log_file, stderr=subprocess.STDOUT,
            env=environment)
    if expect_success and result.returncode != 0:
        raise RuntimeError('case {} returned {}'.format(name, result.returncode))
    # Athena's fatal-error path may terminate an MPI run with status 0; the
    # analyze() stage checks the required diagnostic text in the log.
    _LOGS.append((filename, name))


def run(**kwargs):
    mpirun_cmd = kwargs['mpirun_cmd']
    mpirun_opts = kwargs['mpirun_opts']
    small_sound = [
        'time/nlim=1', 'mesh/nx1=8', 'mesh/nx2=8', 'mesh/nx3=8',
        'meshblock/nx1=8', 'meshblock/nx2=8', 'meshblock/nx3=8',
    ]

    _run('contract_constant', 'athinput.nrfld_sound', 'constant_off',
         mpirun_cmd, mpirun_opts,
         small_sound + ['fld/is_couple=false', 'fld/only_rad=true'])
    _run('contract_table', 'athinput.simple_convection_lhllc_fld', 'table_off',
         mpirun_cmd, mpirun_opts, [
             'time/nlim=0', 'time/tlim=0.0',
             'mesh/nx1=4', 'mesh/nx2=4', 'mesh/nx3=4',
             'meshblock/nx1=4', 'meshblock/nx2=4', 'meshblock/nx3=4',
             'fld/is_couple=false', 'fld/only_rad=true',
             'fld/include_radiation_force=false',
             'fld/opacity_table_file=contract_opacity.h5',
         ])

    for name, value in (('reduced_c_factor', '1.0e-2'),
                        ('cut_Pnablav', 'false')):
        _run('contract_constant', 'athinput.nrfld_sound', 'reject_' + name,
             mpirun_cmd, mpirun_opts,
             ['time/nlim=0', 'time/tlim=0.0', 'fld/{}={}'.format(name, value)],
             expect_success=False)

    for name, extra in (
            ('normal', []),
            ('only_rad', ['fld/only_rad=true']),
            ('fixed_u_rad', ['fld/fixed_u_rad=true']),
            ('cut_diff', ['fld/cut_diff=true'])):
        _run('contract_constant', 'athinput.nrfld_sound', name,
             mpirun_cmd, mpirun_opts, small_sound + extra)


def analyze():
    expected_rejections = {
        'reject_reduced_c_factor': "Parameter 'reduced_c_factor' in block 'fld'",
        'reject_cut_Pnablav': "Parameter 'cut_Pnablav' in block 'fld'",
    }
    for filename, name in _LOGS:
        with open(filename, 'r') as log_file:
            output = log_file.read()
        if 'FATAL ERROR' not in output and name.startswith('reject_'):
            return False
        if name in expected_rejections and expected_rejections[name] not in output:
            return False
        if name in ('constant_off', 'table_off'):
            if 'FLD_OPACITY_CONTRACT is_couple=0 sigma_p_max=0' not in output:
                return False
            if 'sigma_r_max=0' in output:
                return False
        if name in ('normal', 'only_rad', 'fixed_u_rad', 'cut_diff'):
            if 'cycle=1' not in output:
                return False
    return True
