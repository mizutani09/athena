"""Regression tests for FLD opacity domain, value, and I/O error policies."""

import os
import re
import subprocess

import h5py
import numpy as np

import scripts.utils.athena as athena


_CASES = []


def _fields():
    planck = np.empty((3, 3))
    rosseland = np.empty((3, 3))
    for j in range(3):
        for i in range(3):
            planck[j, i] = 10.0 + 2.0*j + 3.0*i
            rosseland[j, i] = 20.0 + 4.0*j + 5.0*i
    return planck, rosseland


def _write_table(path, planck=None, rosseland=None, omit=None):
    default_planck, default_rosseland = _fields()
    if planck is None:
        planck = default_planck
    if rosseland is None:
        rosseland = default_rosseland
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('axes/log10_temperature', data=[3.0, 4.0, 5.0])
        handle.create_dataset('axes/log10_density', data=[-12.0, -10.0, -8.0])
        if omit != 'planck':
            handle.create_dataset('kappa/planck', data=planck)
        if omit != 'rosseland':
            handle.create_dataset('kappa/rosseland', data=rosseland)


def _write_log_table(path):
    planck = np.tile(np.array([0.0, 2.0, 4.0]), (3, 1))
    rosseland = np.zeros((3, 3))
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('axes/log10_temperature', data=[3.0, 4.0, 5.0])
        handle.create_dataset('axes/log10_density', data=[-12.0, -10.0, -8.0])
        handle.create_dataset('kappa/log10_planck', data=planck)
        handle.create_dataset('kappa/log10_rosseland', data=rosseland)


def _write_tp_table(path):
    planck, rosseland = _fields()
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('axes/log10_temperature', data=[3.0, 4.0, 5.0])
        handle.create_dataset('axes/log10_pressure', data=[2.0, 3.0, 4.0])
        handle.create_dataset('kappa/planck', data=planck)
        handle.create_dataset('kappa/rosseland', data=rosseland)


def prepare(**kwargs):
    _CASES.clear()
    athena.configure('nrmgfld', 'hdf5', 'mpi', 'omp', prob='test_opacity_table',
                     coord='cartesian', eos='adiabatic', **kwargs)
    athena.make()

    _write_table(os.path.join('bin', 'opacity_domain_valid.h5'))
    _write_table(os.path.join('bin', 'opacity_domain_missing.h5'), omit='rosseland')

    planck, rosseland = _fields()
    invalid_fields = {
        'negative_planck': (planck.copy(), rosseland.copy()),
        'nan_rosseland': (planck.copy(), rosseland.copy()),
        'inf_planck': (planck.copy(), rosseland.copy()),
    }
    invalid_fields['negative_planck'][0][1, 1] = -1.0
    invalid_fields['nan_rosseland'][1][1, 1] = np.nan
    invalid_fields['inf_planck'][0][1, 1] = np.inf
    for name, fields in invalid_fields.items():
        _write_table(os.path.join('bin', 'opacity_domain_' + name + '.h5'),
                     planck=fields[0], rosseland=fields[1])

    negative_slope = np.tile(np.array([3.0, 2.0, 1.0]), (3, 1))
    _write_table(os.path.join('bin', 'opacity_domain_negative_extrap.h5'),
                 planck=negative_slope)
    _write_log_table(os.path.join('bin', 'opacity_domain_overflow_extrap.h5'))
    _write_tp_table(os.path.join('bin', 'opacity_domain_tp.h5'))
    with open(os.path.join('bin', 'opacity_domain_corrupt.h5'), 'wb') as stream:
        stream.write(b'not an HDF5 file')


def _run_case(name, mpirun_cmd, mpirun_opts, filename='opacity_domain_valid.h5',
              rho='1e-10', temperature='1e4', policy=None, repetitions=1,
              expected_values=None, expected_counts=None, expected_error=None,
              extra=None, ranks=1, axis='trho'):
    input_path = os.path.join('..', athena.athena_rel_path,
                              'inputs/radiation/athinput.test_opacity_table')
    command = [mpirun_cmd]
    command.extend(mpirun_opts)
    if os.geteuid() == 0 and mpirun_cmd == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command.extend([
        '-n', str(ranks), './athena', '-i', input_path,
        'fld/opacity_table_file=' + filename,
        'fld/opacity_table_file_type=hdf5',
        'fld/opacity_table_axis=' + axis,
        'problem/rho=' + rho,
        'problem/temp=' + temperature,
        'problem/opacity_probe_repetitions=' + str(repetitions),
    ])
    if policy is not None:
        command.append('fld/opacity_table_domain_policy=' + policy)
    if extra:
        command.extend(extra)
    log_path = os.path.join('bin', 'opacity_domain_' + name + '.log')
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = '2'
    with open(log_path, 'w') as log:
        subprocess.run(command, cwd='bin', env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=False)
    _CASES.append((name, log_path, expected_values, expected_counts, expected_error))


def run(**kwargs):
    launcher = kwargs['mpirun_cmd']
    options = kwargs['mpirun_opts']

    _run_case('lower_boundary', launcher, options, rho='1e-12', temperature='1e3',
              policy='error', expected_values=(10.0, 20.0), expected_counts=(0, 0))
    _run_case('upper_boundary', launcher, options, rho='1e-8', temperature='1e5',
              policy='error', expected_values=(20.0, 38.0), expected_counts=(0, 0))
    _run_case('legacy_extrapolate', launcher, options, temperature='1e6',
              expected_values=(21.0, 39.0), expected_counts=(2, 0))
    pressure_coordinate = np.log10(1.0e-10*8.314462618e7*1.0e4/0.6)
    _run_case('fixed_mu_tp', launcher, options, filename='opacity_domain_tp.h5',
              axis='tp', policy='error',
              expected_values=(13.0 + 2.0*(pressure_coordinate - 2.0),
                               25.0 + 4.0*(pressure_coordinate - 2.0)),
              expected_counts=(0, 0))
    _run_case('clamp_parallel', launcher, options, temperature='1e6', policy='clamp',
              repetitions=2000, expected_values=(18.0, 34.0),
              expected_counts=(4000, 4000))
    _run_case('clamp_mpi_parallel', launcher, options, temperature='1e6', policy='clamp',
              repetitions=1000, ranks=2, extra=['mesh/nx1=8', 'mesh/x1max=0.3'],
              expected_values=(18.0, 34.0), expected_counts=(4000, 4000))
    _run_case('error_outside', launcher, options, temperature='1e6', policy='error',
              expected_error='opacity lookup is outside the table domain')
    _run_case('negative_extrapolation', launcher, options,
              filename='opacity_domain_negative_extrap.h5', temperature='1e7',
              policy='extrapolate', expected_error='opacity extrapolation returned a negative')
    _run_case('zero_extrapolation', launcher, options,
              filename='opacity_domain_negative_extrap.h5', temperature='1e6',
              policy='extrapolate', expected_error='opacity is zero')
    _run_case('nonfinite_result', launcher, options,
              filename='opacity_domain_overflow_extrap.h5', temperature='1e308',
              policy='extrapolate', expected_error='opacity is non-finite after decoding')

    for name, rho, temperature in [
            ('nan_density', 'nan', '1e4'), ('inf_density', 'inf', '1e4'),
            ('negative_density', '-1', '1e4'), ('nan_temperature', '1e-10', 'nan'),
            ('inf_temperature', '1e-10', 'inf')]:
        _run_case(name, launcher, options, rho=rho, temperature=temperature,
                  expected_error='must be finite and positive')

    for name in ['negative_planck', 'nan_rosseland', 'inf_planck']:
        _run_case(name, launcher, options,
                  filename='opacity_domain_' + name + '.h5',
                  expected_error='Every stored value must be finite and every decoded opacity')
    _run_case('missing_dataset', launcher, options,
              filename='opacity_domain_missing.h5',
              expected_error='Could not auto-detect dataset for Rosseland opacity')
    _run_case('corrupt_file', launcher, options,
              filename='opacity_domain_corrupt.h5',
              expected_error="Could not open HDF5 file 'opacity_domain_corrupt.h5'")
    _run_case('invalid_policy', launcher, options, policy='nearest',
              expected_error="must be 'error', 'clamp', or 'extrapolate'")
    _run_case('invalid_mu', launcher, options, extra=['hydro/mu=0'],
              expected_error='hydro/mu must be finite and positive')


def analyze():
    value_pattern = re.compile(
        r'Rosseland opacity: ([^ ]+) cm\^2/g.*Planck opacity: ([^ ]+) cm\^2/g',
        re.DOTALL)
    count_pattern = re.compile(
        r'FLD_OPACITY_DIAGNOSTICS out_of_domain=(\d+) clamped=(\d+)')
    for name, log_path, expected_values, expected_counts, expected_error in _CASES:
        with open(log_path, 'r') as log:
            output = log.read()
        if expected_error is not None:
            if expected_error not in output:
                return False
            continue
        match = value_pattern.search(output)
        counts = count_pattern.findall(output)
        if match is None or len(counts) != 1:
            return False
        rosseland = float(match.group(1))
        planck = float(match.group(2))
        if not np.isclose(planck, expected_values[0], rtol=1.0e-12):
            return False
        if not np.isclose(rosseland, expected_values[1], rtol=1.0e-12):
            return False
        if tuple(int(value) for value in counts[0]) != expected_counts:
            return False
        # The intentional test-pgen exception adds a fixed MPI stack trace in
        # this environment. Thousands of clamp events must still stay O(1).
        if name == 'clamp_parallel' and len(output.splitlines()) > 60:
            return False
    return True
