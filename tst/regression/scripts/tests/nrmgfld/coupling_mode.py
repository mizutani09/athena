"""Regression test for the shared HLLC/LHLLC FLD pressure-coupling mode."""

import os
import re
import subprocess

import scripts.utils.athena as athena


_LOGS = []


def prepare(**kwargs):
    # Each solver is configured in run() so that this one test covers both
    # HLLC-FLD and LHLLC-FLD.  The test cases are deliberately 8^3 and one
    # cycle; they exercise mode ownership without starting a long convection run.
    pass


def _build(flux):
    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='nrfld_sound', coord='cartesian', flux=flux,
                     eos='adiabatic')
    athena.make()


def _run_case(solver, mode, mpirun_cmd, mpirun_opts, extra):
    command = [mpirun_cmd]
    command.extend(mpirun_opts)
    if os.geteuid() == 0 and mpirun_cmd == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command.extend([
        '-n', '1', './athena', '-i',
        '../' + athena.athena_rel_path + 'inputs/radiation/athinput.nrfld_sound',
        'time/nlim=1',
        'mesh/nx1=8', 'mesh/nx2=8', 'mesh/nx3=8',
        'meshblock/nx1=8', 'meshblock/nx2=8', 'meshblock/nx3=8',
    ])
    command.extend(extra)
    filename = 'bin/coupling_mode_{}_{}.log'.format(solver, mode)
    with open(filename, 'w') as log_file:
        subprocess.check_call(command, cwd='bin', stdout=log_file,
                              stderr=subprocess.STDOUT)
    _LOGS.append((filename, mode))


def run(**kwargs):
    for solver in ('hllc', 'lhllc'):
        _build(solver)
        _run_case(solver, 'source', kwargs['mpirun_cmd'], kwargs['mpirun_opts'], [])
        _run_case(solver, 'flux', kwargs['mpirun_cmd'], kwargs['mpirun_opts'],
                  ['fld/pressure_coupling=flux'])
        _run_case(solver, 'off', kwargs['mpirun_cmd'], kwargs['mpirun_opts'],
                  ['fld/include_radiation_force=false'])


def analyze():
    mode_pattern = re.compile(
        r'FLD_PRESSURE_COUPLING mode=(\S+) .*active=(\d+) '
        r'source_force=(\d+) pressure_flux=(\d+)')
    for filename, expected in _LOGS:
        with open(filename, 'r') as log_file:
            output = log_file.read()
        if 'FATAL ERROR' in output or 'status=failed' in output:
            return False
        match = mode_pattern.search(output)
        if match is None:
            return False
        mode, active, source_force, pressure_flux = match.groups()
        if expected == 'source':
            expected_values = ('source', '1', '1', '0')
        elif expected == 'flux':
            expected_values = ('flux', '1', '0', '1')
        else:
            expected_values = ('off', '0', '0', '0')
        if (mode, active, source_force, pressure_flux) != expected_values:
            return False
        if 'cycle=1' not in output:
            return False
    return True
