"""Regression test for coupled gas/radiation Newton residual norms."""

import os
import re
import subprocess

import scripts.utils.athena as athena


def prepare(**kwargs):
    athena.configure('nrmgfld', 'hdf5', 'mpi',
                     prob='nrfld_sound', coord='cartesian', flux='lhllc',
                     eos='adiabatic', **kwargs)
    athena.make()


def run(**kwargs):
    command = [kwargs['mpirun_cmd']]
    command.extend(kwargs['mpirun_opts'])
    if os.geteuid() == 0 and kwargs['mpirun_cmd'] == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command.extend([
        '-n', '2', './athena', '-i',
        '../' + athena.athena_rel_path + 'inputs/radiation/athinput.nrfld_sound',
        'time/nlim=2',
        'meshblock/nx1=16', 'meshblock/nx2=16', 'meshblock/nx3=16',
        # Make both independent residuals measurable in this short case.
        'problem/kappa_P=1e8', 'problem/kappa_R=1e8',
    ])
    with open('bin/coupled_residual_norms.log', 'w') as log_file:
        subprocess.check_call(command, cwd='bin', stdout=log_file,
                              stderr=subprocess.STDOUT)


def _summaries():
    summaries = []
    pattern = re.compile(r'^\[NR\] status=(\S+) (.*)$')
    with open('bin/coupled_residual_norms.log', 'r') as log_file:
        for line in log_file:
            match = pattern.match(line.strip())
            if not match:
                continue
            values = {'status': match.group(1)}
            for item in match.group(2).split():
                key, value = item.split('=', 1)
                values[key] = float(value)
            summaries.append(values)
    return summaries


def analyze():
    summaries = _summaries()
    if not any(item['status'] == 'dt_zero_initialization'
               for item in summaries):
        return False
    converged = [item for item in summaries if item['status'] == 'converged']
    if not converged:
        return False
    final = converged[-1]
    required = ('initial_gas_l2', 'initial_gas_max',
                'initial_radiation_l2', 'initial_radiation_max',
                'final_gas_l2', 'final_gas_max',
                'final_radiation_l2', 'final_radiation_max')
    if any(key not in final for key in required):
        return False
    if not (final['initial_gas_l2'] > 0.0
            and final['initial_radiation_l2'] > 0.0):
        return False
    return final['final_l2'] <= 1.0e-10 and final['final_max'] <= 1.0e-10
