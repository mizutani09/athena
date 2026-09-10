"""Regression test for EOS-consistent opacity temperatures in simple convection."""

import os
from shutil import copy2, move

import h5py
import numpy as np

import scripts.utils.athena as athena
from scripts.utils.EquationOfState.eos import SimpleHydrogen
from scripts.utils.EquationOfState.writeEOS import write_varlist


def _write_opacity_table(path, temperature_dependent):
    log_density = np.array([-3.0, 0.0, 3.0])
    log_temperature = np.array([-3.0, 0.0, 3.0])
    if temperature_dependent:
        planck = np.tile(log_temperature, (log_density.size, 1))
        rosseland = 2.0*planck
    else:
        planck = np.full((log_density.size, log_temperature.size), np.log10(2.0))
        rosseland = np.full((log_density.size, log_temperature.size), np.log10(3.0))
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('axes/log10_density', data=log_density)
        handle.create_dataset('axes/log10_temperature', data=log_temperature)
        handle.create_dataset('kappa/log10_planck', data=planck)
        handle.create_dataset('kappa/log10_rosseland', data=rosseland)


def _write_hydrogen_eos_table(path):
    """Write the four-field synthetic non-ideal table required by this pgen."""
    log_rho_limits = np.array([-3.0, 3.0])
    log_espec_limits = np.array([-3.0, 3.0])
    log_rho = np.linspace(*log_rho_limits, num=64)
    log_espec = np.linspace(*log_espec_limits, num=128)
    rho = 10.0**log_rho
    espec = 10.0**log_espec
    hydrogen = SimpleHydrogen()
    shape = (log_espec.size, log_rho.size)
    pres_over_egas = np.empty(shape)
    egas_over_pres = np.empty(shape)
    asq_rho_over_pres = np.empty(shape)
    temperature = np.empty(shape)
    pressure_ratio = 1.5
    for ie, es in enumerate(espec):
        for ir, density in enumerate(rho):
            egas = density*es
            temp_e = hydrogen.T_of_rho_ei(density, egas)
            pressure = hydrogen.p_of_rho_T(density, temp_e)
            pres_over_egas[ie, ir] = pressure/egas
            temperature[ie, ir] = temp_e

            pressure_inverse = density*es/pressure_ratio
            temp_p = hydrogen.T_of_rho_p(density, pressure_inverse)
            egas_inverse = hydrogen.ei_of_rho_T(density, temp_p)
            egas_over_pres[ie, ir] = egas_inverse/pressure_inverse
            asq_rho_over_pres[ie, ir] = hydrogen.gamma1(density, temp_p)

    write_varlist(
        log_rho_limits, log_espec_limits,
        [pres_over_egas, egas_over_pres, asq_rho_over_pres, temperature],
        fn=path, out_type='ascii', eOp=pressure_ratio,
        ratios=np.array([1.0, pressure_ratio, pressure_ratio, 1.0]),
        var_names=['p/e(e/rho,rho)', 'e/p(p/rho,rho)',
                   'asq*rho/p(p/rho,rho)', 'temperature(e/rho,rho)'])


def prepare(**kwargs):
    athena.configure('nrmgfld', 'hdf5', prob='simple_convection_lhllc_fld',
                     coord='cartesian', flux='lhllc', eos='adiabatic', **kwargs)
    athena.make()
    move(os.path.join('bin', 'athena'), os.path.join('bin', 'athena_adiabatic'))

    athena.configure('nrmgfld', 'hdf5', prob='simple_convection_lhllc_fld',
                     coord='cartesian', flux='lhllc', eos='general/eos_table',
                     **kwargs)
    athena.make()
    move(os.path.join('bin', 'athena'), os.path.join('bin', 'athena_tabulated'))

    _write_hydrogen_eos_table(
        os.path.join('bin', 'opacity_temperature_hydrogen.tab'))
    _write_opacity_table(os.path.join('bin', 'opacity_temperature_dependent.h5'),
                         temperature_dependent=True)
    _write_opacity_table(os.path.join('bin', 'opacity_temperature_constant.h5'),
                         temperature_dependent=False)


def run(**kwargs):
    input_file = 'radiation/athinput.opacity_temperature'
    for eos_name in ('adiabatic', 'tabulated'):
        copy2(os.path.join('bin', 'athena_' + eos_name), os.path.join('bin', 'athena'))
        for table_name in ('dependent', 'constant'):
            problem_id = 'opacity_temperature_{0}_{1}'.format(
                eos_name, table_name)
            history_path = os.path.join('bin', problem_id + '.hst')
            if os.path.exists(history_path):
                os.remove(history_path)
            arguments = [
                'job/problem_id=' + problem_id,
                'fld/opacity_table_file=opacity_temperature_{0}.h5'.format(table_name),
            ]
            athena.run(input_file, arguments)


def analyze():
    # The pgen's opt-in guard checks the EOS-derived opacity in every active
    # and ghost cell.  Athena's fatal-error path can return
    # a zero process status, so require the initial history output from every
    # case rather than relying only on athena.run reaching this point.
    return all(os.path.isfile(os.path.join(
        'bin', 'opacity_temperature_{0}_{1}.hst'.format(eos_name, table_name)))
        for eos_name in ('adiabatic', 'tabulated')
        for table_name in ('dependent', 'constant'))
