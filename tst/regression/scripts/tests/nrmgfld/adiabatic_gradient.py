"""Regression test for EOS-consistent adiabatic temperature gradients."""

import os
import importlib.util
from pathlib import Path
from shutil import copy2, move

import h5py
import numpy as np
from scipy.optimize import brentq

import scripts.utils.athena as athena
from scripts.utils.EquationOfState.writeEOS import write_varlist


# Synthetic closure in x=log10(rho), y=log10(u):
# log10(P)=C*x+D*y, log10(T)=A*x+B*y, log10(S)=M*x+N*y.
# It is deliberately non-ideal, and its Gamma1 field is deliberately not a
# substitute for the exact isentropic temperature-pressure slope.
A, B = 0.1, 0.7
C, D = 1.2, 0.8
M, N = -0.6, 1.1
EXPECTED_NONIDEAL = (A*N - B*M)/(C*N - D*M)
DIRECT_NABLA = 0.2375


def _write_opacity_table(path):
    axis = np.array([-3.0, 0.0, 3.0])
    opacity = np.zeros((axis.size, axis.size))
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('axes/log10_density', data=axis)
        handle.create_dataset('axes/log10_temperature', data=axis)
        handle.create_dataset('kappa/log10_planck', data=opacity)
        handle.create_dataset('kappa/log10_rosseland', data=opacity)


def _synthetic_fields(log_rho, log_u):
    y, x = np.meshgrid(log_u, log_rho, indexing='ij')
    log_p = C*x + D*y
    log_t = A*x + B*y
    log_s = 3.0 + M*x + N*y
    log_q = y  # placeholder; inverse fields are filled separately below
    del log_q
    return x, y, log_p, log_t, log_s


def _write_nonideal_table(path, include_entropy=True, include_temperature=True,
                          direct_nabla=None):
    limits = np.array([-2.0, 2.0])
    log_rho = np.linspace(*limits, num=33)
    log_u = np.linspace(*limits, num=41)
    x, y, log_p, log_t, log_s = _synthetic_fields(log_rho, log_u)
    forward_p_over_e = 10.0**(log_p - x - y)

    # Inverse fields use q=log10(P/rho) as their x2 coordinate.
    q, xi = np.meshgrid(log_u, log_rho, indexing='ij')
    yi = (q - (C - 1.0)*xi)/D
    log_pi = C*xi + D*yi
    inverse_e_over_p = 10.0**(xi + yi - log_pi)
    # This is dlogP/dlogrho along the synthetic constant-entropy direction.
    gamma1 = np.full_like(q, C - D*M/N)
    fields = [forward_p_over_e, inverse_e_over_p, gamma1]
    names = ['p/e(e/rho,rho)', 'e/p(p/rho,rho)',
             'asq*rho/p(p/rho,rho)']
    if include_temperature:
        fields.append(10.0**log_t)
        names.append('temperature(e/rho,rho)')
    if include_entropy:
        entropy_forward = 10.0**log_s
        entropy_inverse = 10.0**(3.0 + M*xi + N*yi)
        fields += [entropy_forward, entropy_inverse]
        names += ['entropy(e/rho,rho)', 'entropy(p/rho,rho)']
    if direct_nabla is not None:
        if not include_entropy:
            raise ValueError('the direct nabla_ad field occupies schema index 6')
        fields.append(np.full_like(log_p, direct_nabla))
        names.append('nabla_ad(e/rho,rho)')
    write_varlist(
        limits, limits, fields, fn=path, out_type='ascii', eOp=1.0,
        ratios=np.ones(len(fields)), var_names=names,
        opt={'sep': ' ', 'format': '%.16e'})


def prepare(**kwargs):
    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='simple_convection_lhllc_fld', coord='cartesian',
                     flux='lhllc', eos='general/eos_table', **kwargs)
    athena.make()
    move(os.path.join('bin', 'athena'), os.path.join('bin', 'athena_nabla_table'))

    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='simple_convection_lhllc_fld', coord='cartesian',
                     flux='lhllc', eos='general/ideal', **kwargs)
    athena.make()
    move(os.path.join('bin', 'athena'), os.path.join('bin', 'athena_nabla_ideal'))

    _write_nonideal_table(os.path.join('bin', 'adiabatic_gradient_nonideal.tab'))
    _write_nonideal_table(
        os.path.join('bin', 'adiabatic_gradient_no_entropy.tab'),
        include_entropy=False)
    _write_nonideal_table(
        os.path.join('bin', 'adiabatic_gradient_no_temperature.tab'),
        include_entropy=False, include_temperature=False)
    _write_nonideal_table(
        os.path.join('bin', 'adiabatic_gradient_direct.tab'),
        direct_nabla=DIRECT_NABLA)
    _write_opacity_table(os.path.join('bin', 'adiabatic_gradient_opacity.h5'))


def run(**kwargs):
    input_file = 'radiation/athinput.adiabatic_gradient'
    for eos_name in ('table', 'ideal'):
        copy2(os.path.join('bin', 'athena_nabla_' + eos_name),
              os.path.join('bin', 'athena'))
        profile = 'adiabatic_gradient_{0}_profile.txt'.format(eos_name)
        path = os.path.join('bin', profile)
        if os.path.exists(path):
            os.remove(path)
        athena.run(input_file, [
            'job/problem_id=adiabatic_gradient_' + eos_name,
            'problem/profile_output=' + profile,
        ])
    copy2(os.path.join('bin', 'athena_nabla_table'), os.path.join('bin', 'athena'))
    athena.run(input_file, [
        'job/problem_id=adiabatic_gradient_direct',
        'hydro/eos_file_name=adiabatic_gradient_direct.tab',
        'problem/profile_output=adiabatic_gradient_direct_profile.txt',
    ])


def _load_profile(name):
    return np.loadtxt(os.path.join('bin', name))


def _independent_isentrope_slope(eos, x0, y0, width):
    """Finite chord along S(x,y)=S(x0,y0), independent of API partials."""
    s0 = eos._field(4, eos._coordinate(4, y0), x0)
    points = []
    for sign in (-1.0, 1.0):
        x = x0 + sign*width
        if not eos.logrho_min <= x <= eos.logrho_max:
            continue

        def entropy_error(y):
            return eos._field(4, eos._coordinate(4, y), x) - s0

        try:
            y = brentq(entropy_error, eos.logx2_min, eos.logx2_max)
        except ValueError:
            continue
        log_p = eos._field(0, eos._coordinate(0, y), x) + x + y
        log_t = eos._field(3, eos._coordinate(3, y), x)
        points.append((log_p, log_t))
    if len(points) == 1:  # one-sided boundary chord includes the base point
        log_p0 = eos._field(0, eos._coordinate(0, y0), x0) + x0 + y0
        log_t0 = eos._field(3, eos._coordinate(3, y0), x0)
        points.append((log_p0, log_t0))
    return (points[1][1] - points[0][1])/(points[1][0] - points[0][0])


def analyze():
    table_profile = _load_profile('adiabatic_gradient_table_profile.txt')
    ideal_profile = _load_profile('adiabatic_gradient_ideal_profile.txt')
    direct_profile = _load_profile('adiabatic_gradient_direct_profile.txt')
    if not np.allclose(table_profile[:, 6], EXPECTED_NONIDEAL,
                       rtol=2.0e-10, atol=2.0e-12):
        return False
    if not np.allclose(ideal_profile[:, 6], 0.4,
                       rtol=2.0e-13, atol=2.0e-15):
        return False
    if not np.allclose(direct_profile[:, 6], DIRECT_NABLA,
                       rtol=2.0e-13, atol=2.0e-15):
        return False

    # Exercise the Python profile implementation at the interior and all four
    # rho/u corners.  The one-sided edge derivative must retain the same slope.
    profile_script = Path(athena.athena_rel_path) / 'scripts' / \
        'build_solar_initial_profile.py'
    spec = importlib.util.spec_from_file_location('solar_profile_task02', profile_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    AthenaEosTable = module.AthenaEosTable
    eos = AthenaEosTable(Path('bin/adiabatic_gradient_nonideal.tab'))
    for x, y in [(0.0, 0.0), (-2.0, -2.0), (-2.0, 2.0),
                 (2.0, -2.0), (2.0, 2.0)]:
        if not np.isclose(eos.nabla_ad_from_log_state(x, y),
                          EXPECTED_NONIDEAL, rtol=2.0e-10, atol=2.0e-12):
            return False

    # Use edge midpoints for finite isentropic chords: at two diagonal
    # corners this particular isentrope immediately leaves the rectangular
    # table in both coordinate directions.
    for x, y in [(0.0, 0.0), (-2.0, 0.0), (2.0, 0.0),
                 (0.0, -2.0), (0.0, 2.0)]:
        for width in (1.0e-2, 1.0e-3, 1.0e-4):
            if not np.isclose(_independent_isentrope_slope(eos, x, y, width),
                              EXPECTED_NONIDEAL, rtol=2.0e-10, atol=2.0e-12):
                return False

    # The four-field legacy schema is supported by Gamma1 plus the P/T
    # Jacobian.  A three-field table has no temperature derivative and fails.
    legacy = AthenaEosTable(Path('bin/adiabatic_gradient_no_entropy.tab'))
    if not np.isclose(legacy.nabla_ad_from_log_state(0.0, 0.0),
                      EXPECTED_NONIDEAL, rtol=2.0e-10, atol=2.0e-12):
        return False
    try:
        AthenaEosTable(Path('bin/adiabatic_gradient_no_temperature.tab'))
    except ValueError as error:
        return 'standard P, Gamma1, and T fields' in str(error)
    return False
