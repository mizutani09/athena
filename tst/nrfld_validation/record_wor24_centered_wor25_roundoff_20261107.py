#!/usr/bin/env python3
"""Summarize isolated centered-face and FGMRES roundoff solver runs."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor24_centered_wor25_roundoff_trials')
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
OUT = ROOT / 'tst/nrfld_validation/wor24_centered_wor25_roundoff_20261107.json'
SOURCE_BASE = '2d6821233229ed68bc172ed94d85b0d81243736d'
FINAL_SOURCE = 'e786abc429684cf413d23f423b6d48df5838437d'
FINAL_BINARY = '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'

# These are the binaries used at execution time, before later rebuilds.
RUNS = {
    'centered_coupled_var_r1': ('on', 1, '61b88c156f76e297898148126d9ec469d61dcc5a6e9c13d59c6ea6425a705f36'),
    'baseline_coupled_var_fast_r1': ('off', 1, '61b88c156f76e297898148126d9ec469d61dcc5a6e9c13d59c6ea6425a705f36'),
    'centered_coupled_var_fast_r1': ('on', 1, '61b88c156f76e297898148126d9ec469d61dcc5a6e9c13d59c6ea6425a705f36'),
    'centered_coupled_var_fast_r2': ('on', 2, '61b88c156f76e297898148126d9ec469d61dcc5a6e9c13d59c6ea6425a705f36'),
    'centered_jv_r1': ('on', 1, 'cdb5297e3b2718cd41041a8f9d5f6edf8035d0703026b53325d82e77ad57752d'),
    'centered_2step_audit_r1': ('on', 1, '2a45a893a194ea93c4dcf9e7ae1329216195bbb653e03cd38bee4d97041ba6a4'),
    'centered_2step_long_update_r1': ('on+long_double_update', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'centered_jv_r2': ('on', 2, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'uniform_default_r1': ('off', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'uniform_centered_r1': ('on', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'centered_fixed_outflow_r1': ('on, unsupported outflow guard', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'centered_fixed_periodic_r1': ('on', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'centered_2step_truecap5_r1': ('on, true residual cap 5e-8 input trial', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'centered_fixed_cap5_r1': ('on, true residual cap 5e-8 input trial', 1, 'bbc33e37518295cee6a93d81300d1f8b4237ef72c182a3cb7fbab13b8c6a87fd'),
    'baseline_2step_audit_r1': ('off, update audit', 1, '2e9e544908197d0dc915ff24f42e4dc01047a81c708dee914734dad9a9378b8c'),
    'centered_weakdiff_2step_r1': ('on, opacity x1000', 1, '2e9e544908197d0dc915ff24f42e4dc01047a81c708dee914734dad9a9378b8c'),
    'centered_2step_truecap10_r1': ('on, true residual cap 1e-7 input trial', 1, '2e9e544908197d0dc915ff24f42e4dc01047a81c708dee914734dad9a9378b8c'),
    'centered_difference_jv_guard': ('invalid legacy-path guard attempt', 1, '42f84993ea6468767459617cccbccee1e03024471c0b83d9ed1261c608b09176'),
    'centered_difference_jv_guard_supported': ('on, expected assembled-Jv guard', 1, '42f84993ea6468767459617cccbccee1e03024471c0b83d9ed1261c608b09176'),
    'uniform_final_guard_r1': ('on, final-source smoke', 1, '42f84993ea6468767459617cccbccee1e03024471c0b83d9ed1261c608b09176'),
    'centered_2step_truecap10_r2': ('on, true residual cap 1e-7 input trial', 2, '42f84993ea6468767459617cccbccee1e03024471c0b83d9ed1261c608b09176'),
    'centered_weakdiff_2step_r1_retry': ('on, opacity x1000, retry after timeout', 1, '42f84993ea6468767459617cccbccee1e03024471c0b83d9ed1261c608b09176'),
    'uniform_forced_default_r1': ('off, forced conservative uniform', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'uniform_forced_centered_r1': ('on, forced conservative uniform', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'uniform_forced_centered_r2': ('on, forced conservative uniform', 2, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_twolevel_var_r1': ('on, two-level coupled variable limiter', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_twolevel_var_r2_correct': ('on, two-level coupled variable limiter', 2, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_weakdiff_2step_r2': ('on, opacity x1000', 2, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_fixed_weakdiff_r1': ('on, fixed limiter, opacity x1000', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_fixed_weakdiff_r2': ('on, fixed limiter, opacity x1000', 2, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_final_jv_r1': ('on, final-source coupled Jv', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_final_jv_r2': ('on, final-source coupled Jv', 2, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_final_nojv_r1': ('on, final-source Jv-off control', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_5step_truecap10_r1': ('on, five-step true residual cap 1e-7 pilot', 1, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
    'centered_5step_truecap10_r2': ('on, five-step true residual cap 1e-7 pilot', 2, '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grab(pattern, text, default=None):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def parse(name, mode, ranks, binary):
    directory = TRIALS / name
    input_file = (ROOT / 'tst/nrfld_validation/wor25_k1_uniform_20261107.in'
                  if name.startswith('uniform_') or
                  name == 'centered_difference_jv_guard' else
                  ROOT / 'tst/nrfld_validation/testc_twolevel_shifted_20261107.in'
                  if name.startswith('centered_twolevel_') else INPUT)
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    stdout = sorted((directory / 'mpi').rglob('stdout'))
    faces = compare([row for file in stdout for row in read_faces(file)])
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    nr = []
    for line in log.splitlines():
        if line.startswith('[NR] status='):
            fields = dict(re.findall(r'(\w+)=([\w.eE+-]+)', line))
            nr.append({'status': fields['status'],
                       'iterations': int(fields['iterations']),
                       'final_l2': float(fields['final_l2']),
                       'final_max': float(fields['final_max'])})
    energy = [float(v) for v in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    jv = []
    for line in log.splitlines():
        if line.startswith('[NRFLD coupled Jv]'):
            jv.append(dict(re.findall(r'(\w+)=([\w.+-]+)', line)))
    updates = []
    for line in log.splitlines():
        if line.startswith('[LINEAR_FGMRES_UPDATE_AUDIT]'):
            updates.append({key: float(value) for key, value in re.findall(
                r'(\w+)=([\d.eE+-]+)', line)})
    exit_status = int(grab(r'Exit status:\s*(\d+)', timing, '-1'))
    face_worst = max((p['relative_mismatch'] for p in faces), default=None)
    outputs = sorted((directory / 'output').glob('*.athdf'))
    final_output = None
    if len(outputs) > 1:
        with h5py.File(outputs[-1]) as data:
            values = data['user_out_var'][0]
            numeric_digest = hashlib.sha256()
            for key in sorted(data):
                item = data[key]
                if not isinstance(item, h5py.Dataset) or not np.issubdtype(item.dtype, np.number):
                    continue
                numeric_digest.update(key.encode())
                numeric_digest.update(item.dtype.str.encode())
                numeric_digest.update(str(item.shape).encode())
                numeric_digest.update(np.ascontiguousarray(item[...]).tobytes())
            final_output = {'file': str(outputs[-1]),
                            'file_sha256': sha(outputs[-1]),
                            'numeric_arrays_sha256': numeric_digest.hexdigest(),
                            'radiation_min': float(np.min(values)),
                            'radiation_max': float(np.max(values))}
    target = 1e-7 if 'cap10' in name else 5e-8 if 'cap5' in name else 1e-8
    solver_pass = (exit_status == 0 and any(x['status'] == 'converged' for x in nr)
                   and fg and max(x['true_relative_residual'] for x in fg) < target)
    return {
        'name': name, 'centered_face_trial': mode, 'ranks': ranks,
        'source_base_git_hash': SOURCE_BASE,
        'source_git_hash': FINAL_SOURCE if binary == FINAL_BINARY else None,
        'binary_sha256': binary,
        'input': str(input_file), 'input_sha256': sha(input_file),
        'command': grab(r'Command being timed: "([^"]+)"', timing),
        'opacity_override': 8.1863327197866663e-5 if 'weakdiff' in name else None,
        'effective_operator_path': grab(r'\[NRFLD_INTERFACE_MODE\] path=(\w+)', log),
        'environment': {'OMP_NUM_THREADS': 1,
                        'ATHENA_NRFLD_CENTERED_FACE_TRIAL':
                            mode.startswith('on') or 'invalid legacy' in mode,
                        'ATHENA_NRFLD_FGMRES_LONG_DOUBLE_UPDATE_TRIAL':
                            'long_double_update' in mode,
                        'ATHENA_NRFLD_FGMRES_UPDATE_AUDIT': 'audit' in name,
                        'ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE':
                            name.startswith('uniform_forced_'),
                        'ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF':
                            name.startswith('uniform_forced_'),
                        'ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV':
                            0 if name.startswith('uniform_forced_') else None,
                        'ATHENA_NRFLD_INTERFACE_LEDGER': not name.startswith('uniform_'),
                        'ATHENA_NRFLD_ENERGY_LEDGER':
                            not name.startswith('uniform_') or
                            name.startswith('uniform_forced_')},
        'elapsed': grab(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': (None if exit_status == 124 else int(grab(
            r'Maximum resident set size \(kbytes\):\s*(\d+)', timing, '0'))),
        'exit_status': exit_status, 'newton_history': nr,
        'fgmres_history': fg, 'fgmres_total_iterations': sum(x['iterations'] for x in fg),
        'max_true_relative_residual': max((x['true_relative_residual'] for x in fg), default=None),
        'jv': jv, 'update_audit': updates,
        'face_pairs': faces, 'max_face_relative_mismatch': face_worst,
        'energy_absolute_relative_balance_history': energy,
        'max_implicit_energy_absolute_relative_balance': max(energy, default=None),
        'final_output': final_output,
        'fgmres_fatal': grab(r'(FGMRES did not reach its requested relative residual:[^\n]*)', log),
        'fgmres_target': target, 'solver_pass': solver_pass,
        'strict_target_met_by_history': (solver_pass and
                                         max(x['true_relative_residual'] for x in fg) < 1e-8),
        'expected_support_guard': 'Conservative NR-FLD requires either periodic Test C' in log,
        'trial_validity': ('invalid_legacy_path' if name == 'centered_difference_jv_guard'
                           else 'valid'),
        'face_pass': face_worst <= 1e-6 if face_worst is not None else None,
        'log_path': str(directory / 'run.log'),
    }


def main():
    runs = [parse(name, *config) for name, config in RUNS.items()
            if (TRIALS / name / 'time.log').exists()]
    by_name = {run['name']: run for run in runs}
    comparisons = []
    for left, right in (
            ('centered_coupled_var_fast_r1', 'centered_coupled_var_fast_r2'),
            ('centered_jv_r1', 'centered_jv_r2'),
            ('centered_weakdiff_2step_r1_retry', 'centered_weakdiff_2step_r2'),
            ('centered_twolevel_var_r1', 'centered_twolevel_var_r2_correct'),
            ('centered_fixed_weakdiff_r1', 'centered_fixed_weakdiff_r2'),
            ('centered_final_jv_r1', 'centered_final_jv_r2'),
            ('centered_final_jv_r1', 'centered_final_nojv_r1'),
            ('centered_2step_truecap10_r1', 'centered_2step_truecap10_r2'),
            ('centered_5step_truecap10_r1', 'centered_5step_truecap10_r2'),
            ('centered_fixed_cap5_r1', 'centered_fixed_cap5_r2'),
            ('uniform_forced_default_r1', 'uniform_forced_centered_r1'),
            ('uniform_forced_centered_r1', 'uniform_forced_centered_r2')):
        if left not in by_name or right not in by_name:
            continue
        a = by_name[left]['final_output']
        b = by_name[right]['final_output']
        if not a or not b:
            continue
        with h5py.File(a['file']) as x, h5py.File(b['file']) as y:
            exact = all(np.array_equal(x[key][...], y[key][...])
                        for key in x if isinstance(x[key], h5py.Dataset)
                        and np.issubdtype(x[key].dtype, np.number))
        comparisons.append({'left': left, 'right': right,
                            'all_final_numeric_hdf_arrays_exact': exact})
    OUT.write_text(json.dumps({'runs': runs, 'output_comparisons': comparisons},
                              indent=2) + '\n')
    for run in runs:
        print(run['name'], run['solver_pass'], run['face_pass'],
              run['fgmres_total_iterations'], run['max_face_relative_mismatch'])


if __name__ == '__main__':
    main()
