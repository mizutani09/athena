#!/usr/bin/env python3
"""Record coupled two/three-level Test C controls and rejected FGMRES trials."""

import glob
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
BINARY = Path('/tmp/wor25_23_24_resolution_roundoff_production_testc_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(pattern, text, default=None):
    found = re.search(pattern, text)
    return found.group(1) if found else default


def run(name, level, mode, ranks, opacity='q1e7', trial='base'):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    command = get(r'Command being timed: "([^"]+)"', timing)
    input_path = ROOT / 'tst/nrfld_validation' / (
        'testc_threelevel_shifted_20261107.in' if level == 3 else
        'testc_twolevel_shifted_20261107.in')
    files = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    rows = [row for file in files for row in read_faces(Path(file))]
    pairs = compare(rows)
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)} for n, r in
          re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    newton = [{'status': status, 'iterations': int(n)} for status, n in
              re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log)]
    energy = [float(x) for x in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    jv = []
    for line in log.splitlines():
        if not line.startswith('[NRFLD coupled Jv]') or 'region=2' not in line:
            continue
        values = dict(re.findall(r'(\w+)=([\w.+-]+)', line))
        if values.get('epsilon') != '1.0000000000000000e-04':
            continue
        jv.append({'direction': values['direction'], 'epsilon': float(values['epsilon']),
                   'coarse_fine_cells': int(values['cells']),
                   'relative_l2': float(values['rel2'])})
    hdf = sorted((directory / 'output').glob('*.fields.*.athdf'))
    final_output = None
    if len(hdf) > 1:
        with h5py.File(hdf[-1]) as final:
            final_output = {'file': str(hdf[-1]),
                            'minimum_radiation_energy': float(np.min(final['user_out_var'][0])),
                            'maximum_radiation_energy': float(np.max(final['user_out_var'][0]))}
    exit_status = int(get(r'Exit status:\s*(\d+)', timing))
    solver_pass = bool(newton) and any(row['status'] == 'converged' for row in newton) \
        and all(row['status'] in ('converged', 'dt_zero_initialization') for row in newton) \
        and bool(fg) and all(row['true_relative_residual'] < 1.2e-8 if trial == 'tol12'
                             else row['true_relative_residual'] < 1e-8 for row in fg) \
        and exit_status == 0
    face_pass = bool(pairs) and all(row['relative_mismatch'] <= 1e-6 for row in pairs)
    worst = max(pairs, key=lambda row: row['relative_mismatch']) if pairs else None
    return {
        'name': name, 'level_count': level, 'limiter': mode, 'ranks': ranks,
        'threads_per_rank': 1, 'opacity_case': opacity, 'trial': trial,
        'source_commit_before_run': '9a8567ed', 'binary_sha256': sha(BINARY),
        'input_path': str(input_path), 'input_sha256': sha(input_path),
        'command': command,
        'overrides': command.split(' -d ', 1)[-1].split(' ', 1)[-1] if command else None,
        'environment': {'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'run_log': str(directory / 'run.log'), 'face_logs': files,
        'exit_status': exit_status,
        'wall_time': get(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(get(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'fgmres_history': fg, 'fgmres_total_iterations': sum(row['iterations'] for row in fg),
        'maximum_true_relative_residual': max((row['true_relative_residual'] for row in fg),
                                              default=None),
        'newton_history': newton, 'fgmres_fatal': get(
            r'(FGMRES did not reach its requested relative residual:[^\n]*)', log),
        'coupled_jv_coarse_fine_h1e4': jv,
        'max_implicit_energy_absolute_relative_balance': max(energy) if energy else None,
        'coarse_fine_face_pair_count': len(pairs),
        'worst_face_pair': worst,
        'max_face_absolute_mismatch': max((abs(row['mismatch']) for row in pairs), default=None),
        'max_face_relative_mismatch': worst['relative_mismatch'] if worst else None,
        'final_output': final_output, 'solver_pass': solver_pass, 'face_pass': face_pass,
        'evaluation': ('FAIL_SOLVER' if not solver_pass else
                       'FAIL_FACE_RELATIVE_ONLY' if not face_pass else 'PASS'),
    }


def hdf_equal(name1, name2):
    file1 = sorted((TRIALS / name1 / 'output').glob('*.fields.*.athdf'))[-1]
    file2 = sorted((TRIALS / name2 / 'output').glob('*.fields.*.athdf'))[-1]
    with h5py.File(file1) as a, h5py.File(file2) as b:
        return all(np.array_equal(a[key][:], b[key][:]) for key in a
                   if isinstance(a[key], h5py.Dataset))


base = []
for level in (2, 3):
    for mode in ('fixed', 'variable'):
        for ranks in (1, 2):
            prefix = 'twolevel' if level == 2 else 'threelevel'
            base.append(run(f'{prefix}_coupled_{mode}_r{ranks}', level, mode, ranks))
for mode in ('fixed', 'variable'):
    for ranks in (1, 2):
        base.append(run(f'threelevel_coupled_q1e6_{mode}_r{ranks}', 3, mode,
                        ranks, opacity='q1e6'))
trials = []
for trial, ranks in [('max160', 2), ('v2', 1), ('v2', 2),
                     ('restart40', 2), ('unbatched', 1), ('unbatched', 2),
                     ('tol12', 1), ('tol12', 2)]:
    trials.append(run(f'threelevel_coupled_fixed_{trial}_r{ranks}', 3,
                      'fixed', ranks, trial=trial))
rank_pairs = {}
for level in (2, 3):
    for mode in ('fixed', 'variable'):
        for opacity in ('q1e7', 'q1e6') if level == 3 else ('q1e7',):
            prefix = 'twolevel' if level == 2 else 'threelevel'
            suffix = '' if opacity == 'q1e7' else '_q1e6'
            names = [f'{prefix}_coupled{suffix}_{mode}_r{rank}' for rank in (1, 2)]
            if all((TRIALS / name / 'output').exists() and len(list(
                    (TRIALS / name / 'output').glob('*.fields.*.athdf'))) > 1
                   for name in names):
                rank_pairs[f'{prefix}_{opacity}_{mode}'] = {
                    'rank1': names[0], 'rank2': names[1],
                    'final_hdf_numeric_arrays_exact': hdf_equal(*names)}
report = json.loads(REPORT.read_text())
report['threelevel_coupled_screen'] = {
    'scope': 'Coupled Newton/FGMRES, coarse/fine Jv and face ledger, 2/3 levels, fixed/variable limiter, 1/2 MPI ranks; q1e7 and q1e6 3-level opacity; FGMRES options in fixed q1e7 case.',
    'base_cases': base, 'fgmres_option_trials': trials,
    'rank_comparisons': rank_pairs,
    'decision': 'Two-level coupled controls pass face tolerance on both ranks. Three-level coupled cases converge except q1e7 fixed 2-rank strict FGMRES target, but all completed three-level cases fail the 1e-6 relative face criterion on very small face flux. Smooth coarse/fine Schur Jv and global energy balance remain small. Max iterations, 2 MG cycles, restart width and unbatched reductions do not give a robust strict-target fix. Relaxed 1.2e-8 target is a diagnostic only; it does not address face mismatch. Retain opt-in roundoff-scale diagnostic and leave conservation discrepancy unresolved.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'base': [(r['name'], r['evaluation'], r['max_face_relative_mismatch'])
                          for r in base],
                  'trials': [(r['name'], r['evaluation'],
                              r['maximum_true_relative_residual']) for r in trials],
                  'rank_comparisons': rank_pairs}, indent=2))
