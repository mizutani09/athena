"""Compare the production one-cell LTE update with an independent EOS root.

The coupled cases prescribe the radiation correction from a scalar total-energy
root with known boundary loss and external input. AddDifference must recover
the gas state; the prescribed radiation correction tests the exchange ledger.
"""

import importlib.util
import math
import os
from pathlib import Path
import subprocess

import scripts.utils.athena as athena
from scripts.tests.nrmgfld.adiabatic_gradient import (
    _write_nonideal_table, _write_opacity_table)


ROOT = Path(__file__).resolve().parents[5]


def _temperature_table(path='bin/local_lte_eos.tab', rho_unit=1., energy_unit=1.):
    source = ROOT / 'tools/diagnose_rsg_lte_exchange.py'
    spec = importlib.util.spec_from_file_location('lte_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TableTemperature(path, rho_unit, energy_unit, 1.)


def _root(func, lower, upper):
    lo, hi = func(lower), func(upper)
    if not lo <= 0. <= hi:
        raise AssertionError(f'root not bracketed: {lo}, {hi}')
    for _ in range(180):
        middle = lower + (upper-lower)*0.5
        if middle == lower or middle == upper:
            break
        if func(middle) < 0.:
            lower = middle
        else:
            upper = middle
    return lower + (upper-lower)*0.5


def prepare(**kwargs):
    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='simple_convection_lhllc_fld', coord='cartesian',
                     flux='lhllc', eos='general/eos_table', cxx='g++', **kwargs)
    athena.make()
    objects = sorted(str(p) for p in Path('obj').glob('*.o')
                     if p.name != 'main.o')
    source = Path('scripts/tests/nrmgfld/local_lte_exchange_probe.cpp')
    command = ['mpicxx', '-std=c++11', '-O2', '-I'+str(ROOT / 'src'),
               str(source), *objects, '-lhdf5', '-o', 'bin/local_lte_exchange_probe']
    subprocess.run(command, check=True)
    _write_nonideal_table('bin/local_lte_eos.tab', temperature_kink=True)
    _write_opacity_table('bin/local_lte_opacity.h5')


def run(**kwargs):
    table = _temperature_table()
    records = []
    for old_gas in (0.2, 1., 1.8):
        temperature = table.temperature_and_slope(1., old_gas)[0]
        equilibrium = temperature**4
        old_rad = 1.15*equilibrium
        for coupling in (0.1, 1.e4, 1.e12):
            fixed_root = _root(
                lambda gas: gas-old_gas + coupling*(
                    table.temperature_and_slope(1., gas)[0]**4-old_rad),
                0.01, 100.)
            records.append((len(records), 1, old_gas, old_rad,
                            old_rad, coupling, fixed_root, 0., 0.))
            source = 0.03*old_gas
            boundary_loss = 0.01*old_rad
            total = old_gas + old_rad + source-boundary_loss
            coupled_root = _root(
                lambda gas: gas-old_gas + coupling*(
                    table.temperature_and_slope(1., gas)[0]**4-(total-gas)),
                0.01, total)
            records.append((len(records), 0, old_gas, old_rad,
                            total-coupled_root, coupling, coupled_root,
                            source, boundary_loss))
    with open('bin/local_lte_cases.txt', 'w') as stream:
        for row in records:
            stream.write(' '.join(f'{value:.17g}' for value in row[:6])+'\n')
    with open('bin/local_lte_reference.txt', 'w') as stream:
        for row in records:
            stream.write(' '.join(f'{value:.17g}' for value in row)+'\n')
    radiation_constant = 7.5657e-20
    density = 125.
    material = _temperature_table(
        ROOT / 'data/eos/rsg_tomida2013_material_eos.tab', 1.e-8, 1.e5)
    material_records = []
    x2_at_100 = material.coordinates(density, 100.)[0]
    knot_index = round((x2_at_100-material.x2min)*material.x2norm)
    knot_x2 = material.x2min+knot_index/material.x2norm
    knot_energy = 10.**(knot_x2+math.log10(density*1.e-8)
                        -math.log10(material.ratios[3]*1.e5))
    states = [(10., None, (1.e4, 1.e12)),
              (100., None, (1.e4, 1.e12)),
              (knot_energy, None, (1.e4, 1.e12)),
              (453.487553374, None, (1.e4, 1.e12)),
              (453.487553374, 7.14792218147, (6.91260968506e12,)),
              (1000., None, (1.e4, 1.e12))]
    for old_gas, prescribed_rad, couplings in states:
        equilibrium = radiation_constant*material.temperature_and_slope(
            density, old_gas)[0]**4
        old_rad = 1.02*equilibrium if prescribed_rad is None else prescribed_rad
        for coupling in couplings:
            for fixed in (1, 0):
                total = old_gas+old_rad
                def residual(gas):
                    erad = old_rad if fixed else old_rad+(old_gas-gas)
                    return gas-old_gas + coupling*(
                        radiation_constant*material.temperature_and_slope(
                            density, gas)[0]**4-erad)
                root = _root(residual, max(2., old_gas*0.1),
                             old_gas*2. if fixed else total)
                material_records.append((len(material_records), fixed,
                                         old_gas, old_rad,
                                         old_rad if fixed else old_rad+(old_gas-root),
                                         coupling, root, 0., 0.))
    with open('bin/local_lte_material_cases.txt', 'w') as stream:
        for row in material_records:
            stream.write(' '.join(f'{value:.17g}' for value in row[:6])+'\n')
    with open('bin/local_lte_material_reference.txt', 'w') as stream:
        for row in material_records:
            stream.write(' '.join(f'{value:.17g}' for value in row)+'\n')
    command = [kwargs['mpirun_cmd'], *kwargs['mpirun_opts']]
    if os.geteuid() == 0 and kwargs['mpirun_cmd'] == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command += ['-n', '1', './local_lte_exchange_probe',
                '../'+athena.athena_rel_path+'inputs/radiation/athinput.adiabatic_gradient',
                'local_lte_eos.tab', 'local_lte_opacity.h5', 'local_lte_cases.txt']
    with open('bin/local_lte_exchange.log', 'w') as log:
        subprocess.run(command, cwd='bin', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    material_command = command[:]
    material_command[material_command.index('local_lte_eos.tab')] = str(
        ROOT / 'data/eos/rsg_tomida2013_material_eos.tab')
    material_command[material_command.index('local_lte_cases.txt')] = \
        'local_lte_material_cases.txt'
    material_command.extend((str(density), str(radiation_constant), 'material'))
    with open('bin/local_lte_material_exchange.log', 'w') as log:
        subprocess.run(material_command, cwd='bin', stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def _analyze_dataset(reference_path, log_path, require_marshak):
    table = (_temperature_table() if require_marshak else _temperature_table(
        ROOT / 'data/eos/rsg_tomida2013_material_eos.tab', 1.e-8, 1.e5))
    density = 1. if require_marshak else 125.
    expected = {int(parts[0]): list(map(float, parts[1:]))
                for line in Path(reference_path).read_text().splitlines()
                if (parts := line.split())}
    observed = {}
    marshak_cut_diff = None
    marshak_flux_balance = None
    for line in Path(log_path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            observed[int(parts[0])] = tuple(map(float, parts[1:]))
        if len(parts) == 2 and parts[0] == 'marshak_cut_diff':
            marshak_cut_diff = float(parts[1])
        if len(parts) == 4 and parts[0] == 'marshak_flux_balance':
            marshak_flux_balance = tuple(map(float, parts[1:]))
    if observed.keys() != expected.keys():
        return None
    if require_marshak and (marshak_cut_diff is None
                            or not math.isfinite(marshak_cut_diff)
                            or abs(marshak_cut_diff) > 1.e-14
                            or marshak_flux_balance is None
                            or not all(map(math.isfinite, marshak_flux_balance))
                            or abs(marshak_flux_balance[0]) > 1.e-14
                            or abs(marshak_flux_balance[1]) > 1.e-14
                            or marshak_flux_balance[2] <= 0.):
        return None
    max_root_error = 0.
    max_balance_error = 0.
    for index, (gas, rad, temperature) in observed.items():
        (fixed, old_gas, old_rad, target_rad, coupling, reference,
         source, boundary_loss) = expected[index]
        if not all(map(math.isfinite, (gas, rad, temperature))):
            return None
        if gas <= 0. or rad <= 0. or temperature <= 0.:
            return None
        x2, x1 = table.coordinates(density, gas)
        if not (table.x2min-1.e-12 <= x2 <= table.x2max+1.e-12
                and table.x1min-1.e-12 <= x1 <= table.x1max+1.e-12):
            return None
        root_error = abs(gas-reference)/max(1., abs(reference))
        max_root_error = max(max_root_error, root_error)
        if root_error > 2.e-12:
            return None
        if abs(rad-target_rad) > 2.e-14*max(1., abs(target_rad)):
            return None
        if not fixed:
            balance_error = abs((gas+rad)-(
                old_gas+old_rad+source-boundary_loss))/max(1., old_gas+old_rad)
            max_balance_error = max(max_balance_error, balance_error)
            if balance_error > 2.e-12:
                return None
    return len(observed), max_root_error, max_balance_error


def analyze():
    synthetic = _analyze_dataset('bin/local_lte_reference.txt',
                                 'bin/local_lte_exchange.log', True)
    material = _analyze_dataset('bin/local_lte_material_reference.txt',
                                'bin/local_lte_material_exchange.log', False)
    if synthetic is None or material is None:
        return False
    print(f'local_lte_exchange: synthetic {synthetic[0]} cases, material '
          f'{material[0]} cases; max root error '
          f'{max(synthetic[1], material[1]):.3e}, '
          f'max source/flux balance error '
          f'{max(synthetic[2], material[2]):.3e}')
    return True
