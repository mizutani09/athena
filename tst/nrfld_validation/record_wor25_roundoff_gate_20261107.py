#!/usr/bin/env python3
"""Record the isolated, opt-in FGMRES roundoff-gate trials."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_roundoff_gate_trials')
OUT = ROOT / 'tst/nrfld_validation/wor25_roundoff_gate_20261107.json'
BINARY = Path('/tmp/wor25_roundoff_gate_build/bin/athena')
SOURCE_GIT_HASH = '2e2017d2e14286725895b1c531c9e70910256c2c'
RUNS = {
    'strict_gate_valid_r1': (1, 'testc', 2, 'variable', True),
    'strict_gate_valid_r2': (2, 'testc', 2, 'variable', True),
    'strict_control_r1': (1, 'testc', 2, 'variable', False),
    'five_gate_r1': (1, 'testc', 5, 'variable', True),
    'five_gate_r2': (2, 'testc', 5, 'variable', True),
    'fixed_gate_r1': (1, 'testc', 1, 'fixed', True),
    'fixed_gate_r2': (2, 'testc', 1, 'fixed', True),
    'fixed_two_gate_r2': (2, 'testc', 2, 'fixed', True),
    'uniform_gate_r1': (1, 'uniform', 1, 'fixed', True),
    'uniform_gate_r2': (2, 'uniform', 1, 'fixed', True),
    'uniform_control_r1': (1, 'uniform', 1, 'fixed', False),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(line, name, cast=float, default=None):
    match = re.search(r'(?:^|\s)' + name + r'=([\w.eE+-]+)', line)
    return cast(match.group(1)) if match else default


def parse(name, ranks, problem, steps, limiter, gate):
    directory = TRIALS / name
    if not (directory / 'time.log').exists() or not (directory / 'run.log').exists():
        return None
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    exit_match = re.search(r'Exit status:\s*(\d+)', timing)
    if exit_match is None:
        return None
    input_name = ('wor25_roundoff_gate_' + problem + '_20261107.in' if gate else
                  'wor25_k1_uniform_20261107.in' if problem == 'uniform' else
                  'testc_threelevel_shifted_20261107.in')
    input_file = ROOT / 'tst/nrfld_validation' / input_name
    fg = []
    nr = []
    audits = []
    for line in log.splitlines():
        if line.startswith('[LINEAR_FGMRES]'):
            fg.append({key: field(line, key, cast)
                       for key, cast in [('iterations', int),
                                         ('relative_residual', float),
                                         ('roundoff_accepted', int),
                                         ('roundoff_predicted_rel', float),
                                         ('roundoff_gap_rel', float),
                                         ('roundoff_closure_rel', float)]})
        elif line.startswith('[NR] status='):
            if field(line, 'status', str) == 'dt_zero_initialization':
                continue
            nr.append({'status': field(line, 'status', str),
                       'iterations': field(line, 'iterations', int),
                       'final_l2': field(line, 'final_l2'),
                       'final_max': field(line, 'final_max')})
        elif line.startswith('[LINEAR_FGMRES_RESTART_AUDIT]'):
            audits.append({'iterations': field(line, 'total_iterations', int),
                           'projected': field(line, 'projected_relative_residual'),
                           'true': field(line, 'true_relative_residual')})
    faces = compare([row for file in sorted((directory / 'mpi').rglob('stdout'))
                     for row in read_faces(file)])
    energy = [float(v) for v in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    outputs = sorted((directory / 'output').glob('*.athdf'))
    final = None
    if len(outputs) > 1:
        with h5py.File(outputs[-1]) as data:
            digest = hashlib.sha256()
            for key in sorted(data):
                item = data[key]
                if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
                    digest.update(key.encode())
                    digest.update(item.dtype.str.encode())
                    digest.update(str(item.shape).encode())
                    digest.update(np.ascontiguousarray(item[...]).tobytes())
            final = {'path': str(outputs[-1]), 'numeric_sha256': digest.hexdigest()}
    face_max = max((x['relative_mismatch'] for x in faces), default=None)
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing)
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)
    command = re.search(r'Command being timed: "([^"]+)"', timing)
    exit_code = int(exit_match.group(1))
    return {'name': name, 'ranks': ranks, 'problem': problem, 'steps': steps,
            'limiter': limiter, 'roundoff_gate': gate,
            'source_git_hash': SOURCE_GIT_HASH,
            'input': str(input_file), 'input_sha256': sha(input_file),
            'binary_sha256': sha(BINARY), 'command': command.group(1) if command else None,
            'environment': {'OMP_NUM_THREADS': 1,
                            'ATHENA_NRFLD_CENTERED_FACE_TRIAL': problem == 'testc',
                            'ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE': problem == 'uniform',
                            'ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF': problem == 'uniform',
                            'ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV': 0 if problem == 'uniform' else None,
                            'ATHENA_NRFLD_INTERFACE_LEDGER': problem == 'testc',
                            'ATHENA_NRFLD_ENERGY_LEDGER': True},
            'exit_status': exit_code, 'elapsed': wall.group(1) if wall else None,
            'peak_rss_kb': int(rss.group(1)) if rss else None,
            'newton_history': nr, 'fgmres_history': fg,
            'fgmres_total_iterations': sum(x['iterations'] for x in fg),
            'roundoff_accepted_solves': sum(x['roundoff_accepted'] == 1 for x in fg),
            'max_true_relative_residual': max((x['relative_residual'] for x in fg), default=None),
            'restart_audit': audits, 'face_pairs': faces,
            'max_face_relative_mismatch': face_max,
            'face_pass_1e-6': face_max <= 1e-6 if face_max is not None else None,
            'energy_absolute_relative_balance_history': energy,
            'max_energy_absolute_relative_balance': max(energy, default=None),
            'final_output': final,
            'solver_pass': exit_code == 0 and len(nr) == steps and
                           all(x['status'] == 'converged' for x in nr),
            'log_path': str(directory / 'run.log')}


def compare_final(a, b):
    x = a['final_output']
    y = b['final_output']
    if x is None or y is None:
        return None
    with h5py.File(x['path']) as left, h5py.File(y['path']) as right:
        exact = all(np.array_equal(left[key][...], right[key][...])
                    for key in left if isinstance(left[key], h5py.Dataset)
                    and np.issubdtype(left[key].dtype, np.number))
        aa = left['user_out_var'][...]
        bb = right['user_out_var'][...]
        abs_diff = float(np.max(np.abs(aa-bb)))
        scale = max(float(np.max(np.abs(aa))), float(np.max(np.abs(bb))))
    return {'left': a['name'], 'right': b['name'],
            'all_numeric_arrays_exact': exact,
            'max_abs_user_out_var': abs_diff,
            'max_rel_user_out_var': abs_diff/scale if scale else 0.0}


def main():
    runs = [parse(name, *config) for name, config in RUNS.items()]
    runs = [run for run in runs if run is not None]
    by_name = {run['name']: run for run in runs}
    pairs = [('strict_gate_valid_r1', 'strict_gate_valid_r2'),
             ('five_gate_r1', 'five_gate_r2'),
             ('fixed_gate_r1', 'fixed_gate_r2'),
             ('uniform_gate_r1', 'uniform_gate_r2'),
             ('uniform_gate_r1', 'uniform_control_r1')]
    comparisons = [compare_final(by_name[a], by_name[b]) for a, b in pairs
                   if a in by_name and b in by_name]
    guard_directory = TRIALS / 'invalid_cap_r1'
    guard_log = (guard_directory / 'run.log').read_text(errors='replace')
    guard_timing = (guard_directory / 'time.log').read_text(errors='replace')
    guard = {'name': 'invalid_cap_r1', 'source_git_hash': SOURCE_GIT_HASH,
             'binary_sha256': sha(BINARY),
             'input': str(ROOT / 'tst/nrfld_validation/wor25_roundoff_gate_uniform_20261107.in'),
             'input_sha256': sha(ROOT / 'tst/nrfld_validation/wor25_roundoff_gate_uniform_20261107.in'),
             'override': 'nrfld/fgmres_roundoff_cap=1e-9',
             'expected_fatal': 'optional FGMRES roundoff cap must be within [target, 10*target]' in guard_log,
             'exit_status': int(re.search(r'Exit status:\s*(\d+)', guard_timing).group(1)),
             'log_path': str(guard_directory / 'run.log')}
    OUT.write_text(json.dumps({'runs': runs, 'output_comparisons': comparisons,
                               'invalid_configuration_guard': guard}, indent=2) + '\n')
    for run in runs:
        print(run['name'], run['solver_pass'], run['fgmres_total_iterations'],
              run['roundoff_accepted_solves'], run['max_face_relative_mismatch'])


if __name__ == '__main__':
    main()
