#!/usr/bin/env python3
"""Summarize isolated default Test C operator and restart trials."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor24_wor25_default_testc_trials')
OUT = ROOT / 'tst/nrfld_validation/wor24_wor25_default_testc_20261107.json'
SOURCE_PARENT = '2e2017d2e14286725895b1c531c9e70910256c2c'
SOURCE_COMMIT = 'b0849726084aec3fc055adf3da31f36dc72404e2'
NEW_BINARY = '62c3a7f2ebc15af56062c9d90669deae7b5e60890ca33d3a51d413fd35c4d121'
BASE_BINARY = '3ebe0aff6addf0b51956d3eedf4483601b9c59cc75f3ed040d930673eb864bf8'
INVALID_PILOT_BINARY = '5a7b840bf1b2c4c3938a6c5345f861a76ed733b89a80c52eaf95779006da62e9'
RUNS = {
    'default_fixedcap_testc_r1': (1, 'variable', 2, 'testc', 'new'),
    'default_fixedcap_testc_r2': (2, 'variable', 2, 'testc', 'new'),
    'default_fixed_limiter_r2': (2, 'fixed', 1, 'testc', 'new'),
    'default_restart_seed_r2': (2, 'variable', 1, 'restart', 'new'),
    'default_restart_continue_r2': (2, 'variable', 1, 'restart', 'new'),
    'optout_old_face_r1': (1, 'variable', 1, 'testc', 'new'),
    'default_uniform_r1': (1, 'fixed', 1, 'uniform', 'new'),
    'base_uniform_r1': (1, 'fixed', 1, 'uniform', 'base'),
    'default_testc_r1': (1, 'variable', 2, 'testc', 'invalid_pilot'),
    'default_testc_r2': (2, 'variable', 2, 'testc', 'invalid_pilot'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(line, name, cast=float):
    match = re.search(r'(?:^|\s)' + name + r'=([\w.eE+-]+)', line)
    return cast(match.group(1)) if match else None


def parse(name, ranks, limiter, steps, problem, binary_kind):
    directory = TRIALS / name
    log_path = directory / 'run.log'
    time_path = directory / 'time.log'
    if not log_path.exists() or not time_path.exists():
        return None
    log = log_path.read_text(errors='replace')
    timing = time_path.read_text(errors='replace')
    exit_match = re.search(r'Exit status:\s*(\d+)', timing)
    if not exit_match:
        return None
    input_name = ('wor25_k1_uniform_20261107.in' if problem == 'uniform' else
                  'wor24_wor25_default_restart_testc_20261107.in'
                  if problem == 'restart' else 'testc_threelevel_shifted_20261107.in')
    input_file = ROOT / 'tst/nrfld_validation' / input_name
    fg, nr = [], []
    for line in log.splitlines():
        if line.startswith('[LINEAR_FGMRES]'):
            fg.append({'iterations': field(line, 'iterations', int),
                       'true_relative_residual': field(line, 'relative_residual'),
                       'roundoff_accepted': field(line, 'roundoff_accepted', int),
                       'roundoff_cap': field(line, 'roundoff_cap')})
        elif line.startswith('[NR] status='):
            if field(line, 'status', str) == 'dt_zero_initialization' or (
                    field(line, 'iterations', int) == 0 and field(line, 'dt') == 0.0):
                continue
            nr.append({'status': field(line, 'status', str),
                       'iterations': field(line, 'iterations', int),
                       'final_l2': field(line, 'final_l2'),
                       'final_max': field(line, 'final_max')})
    faces = compare([row for path in sorted((directory / 'mpi').rglob('stdout'))
                     for row in read_faces(path)])
    energy = [float(value) for value in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    outputs = sorted((directory / 'output').glob('*.athdf'))
    final = None
    if outputs:
        final_file = outputs[-1]
        with h5py.File(final_file) as data:
            digest = hashlib.sha256()
            for key in sorted(data):
                item = data[key]
                if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
                    digest.update(key.encode())
                    digest.update(item.dtype.str.encode())
                    digest.update(str(item.shape).encode())
                    digest.update(np.ascontiguousarray(item[...]).tobytes())
            final = {'path': str(final_file), 'file_sha256': sha(final_file),
                     'numeric_arrays_sha256': digest.hexdigest()}
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing)
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)
    command = re.search(r'Command being timed: "([^"]+)"', timing)
    binary = (NEW_BINARY if binary_kind == 'new' else BASE_BINARY
              if binary_kind == 'base' else INVALID_PILOT_BINARY)
    face_max = max((row['relative_mismatch'] for row in faces), default=None)
    return {'name': name, 'source_parent_git_hash': SOURCE_PARENT,
            'source_git_hash': SOURCE_COMMIT if binary_kind == 'new' else SOURCE_PARENT,
            'source_variant': binary_kind, 'binary_sha256': binary,
            'ranks': ranks, 'limiter': limiter, 'requested_steps': steps,
            'input': str(input_file), 'input_sha256': sha(input_file),
            'command': command.group(1) if command else None,
            'environment': {'OMP_NUM_THREADS': 1,
                            'ATHENA_NRFLD_CENTERED_FACE_TRIAL': 0 if name == 'optout_old_face_r1' else None,
                            'ATHENA_NRFLD_INTERFACE_LEDGER': problem != 'uniform',
                            'ATHENA_NRFLD_ENERGY_LEDGER': True},
            'exit_status': int(exit_match.group(1)),
            'elapsed': wall.group(1) if wall else None,
            'peak_rss_kb': int(rss.group(1)) if rss else None,
            'effective_operator_path': field(log, 'path', str),
            'newton_history': nr, 'fgmres_history': fg,
            'fgmres_total_iterations': sum(x['iterations'] for x in fg),
            'roundoff_accepted_solves': sum(x['roundoff_accepted'] == 1 for x in fg),
            'max_true_relative_residual': max((x['true_relative_residual'] for x in fg), default=None),
            'face_pairs': faces, 'max_face_relative_mismatch': face_max,
            'face_pass_1e-6': face_max <= 1e-6 if face_max is not None else None,
            'energy_absolute_relative_balance_history': energy,
            'max_energy_absolute_relative_balance': max(energy, default=None),
            'final_output': final,
            'solver_pass': int(exit_match.group(1)) == 0 and len(nr) == steps and
                           all(x['status'] == 'converged' for x in nr),
            'trial_validity': 'invalid_initialization_pilot' if binary_kind == 'invalid_pilot' else 'valid',
            'log_path': str(log_path)}


def compare_hdf(left_path, right_path, left_name, right_name):
    with h5py.File(left_path) as left, h5py.File(right_path) as right:
        keys = [key for key in left if isinstance(left[key], h5py.Dataset)
                and np.issubdtype(left[key].dtype, np.number)]
        exact = all(np.array_equal(left[key][...], right[key][...]) for key in keys)
        aa, bb = left['user_out_var'][...], right['user_out_var'][...]
        abs_diff = float(np.max(np.abs(aa-bb)))
        scale = max(float(np.max(np.abs(aa))), float(np.max(np.abs(bb))))
    return {'left': left_name, 'right': right_name,
            'all_final_numeric_hdf_arrays_exact': exact,
            'max_abs_user_out_var': abs_diff,
            'max_rel_user_out_var': abs_diff/scale if scale else 0.0}


def main():
    runs = [parse(name, *config) for name, config in RUNS.items()]
    runs = [run for run in runs if run is not None]
    by_name = {run['name']: run for run in runs}
    pairs = [('default_fixedcap_testc_r1', 'default_fixedcap_testc_r2'),
             ('default_fixedcap_testc_r2', 'default_restart_continue_r2'),
             ('default_uniform_r1', 'base_uniform_r1')]
    comparisons = []
    for a, b in pairs:
        if a in by_name and b in by_name and by_name[a]['final_output'] and by_name[b]['final_output']:
            comparisons.append(compare_hdf(by_name[a]['final_output']['path'],
                                            by_name[b]['final_output']['path'], a, b))
    direct_first = TRIALS / 'default_fixedcap_testc_r2/output/nrfld_diff_amr_interface.fields.00001.athdf'
    if direct_first.exists() and 'default_restart_seed_r2' in by_name and by_name['default_restart_seed_r2']['final_output']:
        comparisons.append(compare_hdf(direct_first,
                                        by_name['default_restart_seed_r2']['final_output']['path'],
                                        'direct_r2_cycle1', 'restart_seed_r2_cycle1'))
    explicit = Path('/tmp/wor25_roundoff_gate_trials/strict_gate_valid_r2/output')
    if 'default_fixedcap_testc_r2' in by_name and by_name['default_fixedcap_testc_r2']['final_output']:
        exp_hdf = sorted(explicit.glob('*.athdf'))[-1]
        comparisons.append(compare_hdf(by_name['default_fixedcap_testc_r2']['final_output']['path'],
                                        exp_hdf, 'default_fixedcap_testc_r2', 'explicit_gate_r2'))
    checkpoint = TRIALS / 'default_restart_seed_r2/output/nrfld_diff_amr_interface.final.rst'
    checkpoint_info = None
    if checkpoint.exists():
        raw = checkpoint.read_bytes()
        checkpoint_info = {'path': str(checkpoint), 'sha256': sha(checkpoint),
                           'serialized_cap_1e-7': b'fgmres_roundoff_cap' in raw and b'1e-07' in raw}
    OUT.write_text(json.dumps({'runs': runs, 'output_comparisons': comparisons,
                               'restart_checkpoint': checkpoint_info}, indent=2) + '\n')
    for run in runs:
        print(run['name'], run['solver_pass'], run['fgmres_total_iterations'],
              run['roundoff_accepted_solves'], run['max_face_relative_mismatch'])


if __name__ == '__main__':
    main()
