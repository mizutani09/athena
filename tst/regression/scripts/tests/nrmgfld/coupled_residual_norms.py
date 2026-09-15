"""Regression test for gas/total-energy and single-equation Newton norms."""

import math
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
    _run_case('coupled', kwargs, [
        'meshblock/nx1=16', 'meshblock/nx2=16', 'meshblock/nx3=16',
        # Make both independent residuals measurable with stiff exchange.
        'problem/kappa_P=1e8', 'problem/kappa_R=1e8',
    ], 2)
    small = ['mesh/nx1=8', 'mesh/nx2=8', 'mesh/nx3=8',
             'meshblock/nx1=8', 'meshblock/nx2=8', 'meshblock/nx3=8']
    _run_case('only_rad', kwargs, small + ['fld/only_rad=true'], 1)
    _run_case('fixed_u_rad', kwargs,
              small + ['fld/fixed_u_rad=true', 'problem/kappa_P=1e4'], 1)


def _run_case(name, kwargs, extra, ranks):
    command = [kwargs['mpirun_cmd']]
    command.extend(kwargs['mpirun_opts'])
    if os.geteuid() == 0 and kwargs['mpirun_cmd'] == 'mpirun' \
            and '--allow-run-as-root' not in command:
        command.append('--allow-run-as-root')
    command.extend([
        '-n', str(ranks), './athena', '-i',
        '../' + athena.athena_rel_path + 'inputs/radiation/athinput.nrfld_sound',
        'time/nlim=2', 'job/problem_id=residual_' + name,
    ])
    command.extend(extra)
    with open('bin/coupled_residual_norms_' + name + '.log', 'w') as log_file:
        subprocess.check_call(command, cwd='bin', stdout=log_file,
                              stderr=subprocess.STDOUT)


def _summaries(name):
    summaries = []
    pattern = re.compile(r'^\[NR\] status=(\S+) (.*)$')
    with open('bin/coupled_residual_norms_' + name + '.log', 'r') as log_file:
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
    active = {'coupled': ('gas', 'total_energy'),
              'only_rad': ('radiation',), 'fixed_u_rad': ('gas',)}
    channels = ('gas', 'radiation', 'total_energy')
    for name, equations in active.items():
        summaries = _summaries(name)
        if not any(item['status'] == 'dt_zero_initialization'
                   for item in summaries):
            return False
        converged = [item for item in summaries if item['status'] == 'converged']
        if len(converged) != 2:
            return False
        for item in summaries:
            if item['status'] not in ('converged', 'dt_zero_initialization'):
                return False
            for prefix in ('initial_', 'final_'):
                required = [prefix + channel + '_' + norm
                            for channel in channels for norm in ('l2', 'max')]
                if any(key not in item or not math.isfinite(item[key])
                       for key in required):
                    return False
                if any(item[prefix + channel + '_' + norm] != 0.0
                       for channel in channels if channel not in equations
                       for norm in ('l2', 'max')):
                    return False
                combined_l2 = math.sqrt(sum(item[prefix + channel + '_l2']**2
                                            for channel in equations))
                combined_max = max(item[prefix + channel + '_max']
                                   for channel in equations)
                if not math.isclose(item[prefix + 'l2'], combined_l2,
                                    rel_tol=1e-12, abs_tol=1e-30):
                    return False
                if item[prefix + 'max'] != combined_max:
                    return False
        if any(not any(item['initial_' + channel + '_l2'] > 0.0
                       for item in converged) for channel in equations):
            return False
        if any(item['final_l2'] > 1e-10 or item['final_max'] > 1e-10
               for item in converged):
            return False
    return True
