"""Regression tests for FLD opacity-table axes, shapes, and value encodings."""

import os
import re
import subprocess

import h5py
import numpy as np

import scripts.utils.athena as athena


_CASES = []


def _write_hdf5(path, temperature=None, density=None, planck=None,
                rosseland=None, names=None):
    if temperature is None:
        temperature = np.array([3.0, 4.0, 5.0])
    if density is None:
        density = np.array([-12.0, -10.0, -8.0])
    if planck is None:
        planck = np.full((density.size, temperature.size), np.log10(2.0))
    if rosseland is None:
        rosseland = np.full((density.size, temperature.size), np.log10(3.0))
    if names is None:
        names = ('axes/log10_temperature', 'axes/log10_density',
                 'kappa/log10_planck', 'kappa/log10_rosseland')
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(names[0], data=temperature)
        handle.create_dataset(names[1], data=density)
        handle.create_dataset(names[2], data=planck)
        handle.create_dataset(names[3], data=rosseland)


def _write_ascii(path, nvar=2, x2_limits=(-12.0, -8.0),
                 x1_limits=(3.0, 5.0), planck=np.log10(2.0), rosseland=3.0):
    with open(path, 'w') as table:
        table.write('{} 3 3\n'.format(nvar))
        table.write('{} {}\n'.format(*x2_limits))
        table.write('{} {}\n'.format(*x1_limits))
        table.write(' '.join(['1'] * nvar) + '\n')
        values = [planck, rosseland]
        for ivar in range(nvar):
            for _ in range(3):
                table.write('{0} {0} {0}\n'.format(values[ivar]))


def prepare(**kwargs):
    _CASES.clear()
    athena.configure('nrmgfld', 'hdf5', 'mpi', prob='test_opacity_table',
                     coord='cartesian', eos='adiabatic', **kwargs)
    athena.make()

    standard = os.path.join('bin', 'opacity_schema_standard.h5')
    _write_hdf5(standard)
    _write_hdf5(os.path.join('bin', 'opacity_schema_float_axis.h5'),
                temperature=np.linspace(3.0, 5.0, 128, dtype=np.float32))

    _write_hdf5(os.path.join('bin', 'opacity_schema_mixed.h5'),
                rosseland=np.full((3, 3), 3.0),
                names=('axes/log10_temperature', 'axes/log10_density',
                       'kappa/log10_planck', 'kappa/rosseland'))

    custom_names = ('custom/t', 'custom/rho', 'custom/p', 'custom/r')
    _write_hdf5(os.path.join('bin', 'opacity_schema_custom.h5'),
                rosseland=np.full((3, 3), 3.0), names=custom_names)

    invalid_axes = {
        'nonuniform': np.array([0.0, 1.0, 4.0]),
        'empty': np.array([]),
        'singleton': np.array([0.0]),
        'reverse': np.array([2.0, 1.0, 0.0]),
        'duplicate': np.array([0.0, 1.0, 1.0]),
        'nan': np.array([0.0, np.nan, 2.0]),
    }
    for name, axis in invalid_axes.items():
        _write_hdf5(os.path.join('bin', 'opacity_schema_' + name + '.h5'),
                    temperature=axis)

    _write_hdf5(os.path.join('bin', 'opacity_schema_shape.h5'),
                planck=np.zeros((2, 3)))
    _write_ascii(os.path.join('bin', 'opacity_schema_linear.ascii'), planck=2.0)
    _write_ascii(os.path.join('bin', 'opacity_schema_mixed.ascii'))
    _write_ascii(os.path.join('bin', 'opacity_schema_nan.ascii'),
                 x1_limits=(np.nan, 5.0))
    _write_ascii(os.path.join('bin', 'opacity_schema_fields.ascii'), nvar=1)


def _run_case(name, filename, mpirun_cmd, mpirun_opts, extra=None,
              expected_error=None, expected_values=None):
    input_path = os.path.join('..', athena.athena_rel_path,
                              'inputs/radiation/athinput.test_opacity_table')
    file_type = 'ascii' if filename.endswith('.ascii') else 'hdf5'
    command = [mpirun_cmd]
    command.extend(mpirun_opts)
    if os.geteuid() == 0 and mpirun_cmd == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command.extend([
        '-n', '1', './athena', '-i', input_path,
        'fld/opacity_table_file=' + filename,
        'fld/opacity_table_file_type=' + file_type,
        'fld/opacity_table_axis=trho',
    ])
    if extra:
        command.extend(extra)
    log_path = os.path.join('bin', 'opacity_schema_' + name + '.log')
    with open(log_path, 'w') as log:
        subprocess.run(command, cwd='bin', stdout=log, stderr=subprocess.STDOUT)
    _CASES.append((name, log_path, expected_error, expected_values))


def run(**kwargs):
    mpirun_cmd = kwargs['mpirun_cmd']
    mpirun_opts = kwargs['mpirun_opts']
    _run_case('standard', 'opacity_schema_standard.h5', mpirun_cmd, mpirun_opts,
              expected_values=(2.0, 3.0))
    _run_case('float_axis', 'opacity_schema_float_axis.h5',
              mpirun_cmd, mpirun_opts, expected_values=(2.0, 3.0))
    _run_case('mixed_auto', 'opacity_schema_mixed.h5', mpirun_cmd, mpirun_opts,
              expected_values=(2.0, 3.0))
    _run_case('custom_explicit', 'opacity_schema_custom.h5', mpirun_cmd, mpirun_opts,
              extra=[
        'fld/opacity_table_temperature_dataset=custom/t',
        'fld/opacity_table_x2_dataset=custom/rho',
        'fld/opacity_table_planck_dataset=custom/p',
        'fld/opacity_table_rosseland_dataset=custom/r',
        'fld/opacity_table_planck_format=log10',
        'fld/opacity_table_rosseland_format=linear',
    ], expected_values=(2.0, 3.0))
    _run_case('custom_ambiguous', 'opacity_schema_custom.h5', mpirun_cmd, mpirun_opts,
              extra=[
        'fld/opacity_table_temperature_dataset=custom/t',
        'fld/opacity_table_x2_dataset=custom/rho',
        'fld/opacity_table_planck_dataset=custom/p',
        'fld/opacity_table_rosseland_dataset=custom/r',
    ], expected_error='Cannot infer whether planck dataset')
    _run_case('invalid_format', 'opacity_schema_standard.h5', mpirun_cmd, mpirun_opts,
              extra=['fld/opacity_table_planck_format=natural_log'],
              expected_error="must be 'auto', 'linear', or 'log10'")

    expected_axis_errors = {
        'nonuniform': 'is not uniformly spaced',
        'empty': 'must each contain at least two points',
        'singleton': 'must each contain at least two points',
        'reverse': 'must be strictly increasing',
        'duplicate': 'must be strictly increasing',
        'nan': 'contains a non-finite value',
    }
    for name, message in expected_axis_errors.items():
        _run_case(name, 'opacity_schema_' + name + '.h5', mpirun_cmd, mpirun_opts,
                  expected_error=message)
    _run_case('shape', 'opacity_schema_shape.h5', mpirun_cmd, mpirun_opts,
              expected_error='must match coordinate arrays')
    _run_case('ascii_linear', 'opacity_schema_linear.ascii', mpirun_cmd, mpirun_opts,
              expected_values=(2.0, 3.0))
    _run_case('ascii_mixed', 'opacity_schema_mixed.ascii', mpirun_cmd, mpirun_opts,
              extra=[
        'fld/opacity_table_planck_format=log10',
        'fld/opacity_table_rosseland_format=linear',
    ], expected_values=(2.0, 3.0))
    _run_case('ascii_nan', 'opacity_schema_nan.ascii', mpirun_cmd, mpirun_opts,
              expected_error='Failed to parse x1 limits')
    _run_case('ascii_fields', 'opacity_schema_fields.ascii', mpirun_cmd, mpirun_opts,
              expected_error='must contain exactly 2 fields')


def analyze():
    opacity_pattern = re.compile(
        r'Rosseland opacity: ([^ ]+) cm\^2/g.*Planck opacity: ([^ ]+) cm\^2/g',
        re.DOTALL)
    for name, log_path, expected_error, expected_values in _CASES:
        with open(log_path, 'r') as log:
            output = log.read()
        if expected_error is not None:
            if expected_error not in output:
                return False
            continue
        if 'FATAL ERROR in Read' in output or 'FATAL ERROR in UserOpacityTable' in output:
            return False
        match = opacity_pattern.search(output)
        if match is None:
            return False
        rosseland = float(match.group(1))
        planck = float(match.group(2))
        if not np.isclose(planck, expected_values[0], rtol=1.0e-12):
            return False
        if not np.isclose(rosseland, expected_values[1], rtol=1.0e-12):
            return False
    return True
