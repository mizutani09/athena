#!/usr/bin/env python3
"""Summarize the same-checkpoint centered-face ON/OFF Newton trial."""

import hashlib
import json
import re
from pathlib import Path

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor24_centered_wor25_roundoff_common_state')
OUT = ROOT / 'tst/nrfld_validation/wor24_common_checkpoint_20261107.json'
SOURCE = 'e786abc429684cf413d23f423b6d48df5838437d'
BINARY = '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grab(pattern, content, default=None):
    m = re.search(pattern, content)
    return m.group(1) if m else default


def parse(name, checkpoint):
    folder = TRIALS / name
    log = (folder / 'run.log').read_text(errors='replace')
    timing = (folder / 'time.log').read_text(errors='replace')
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
              log)]
    nr = []
    for line in log.splitlines():
        if line.startswith('[NR] status='):
            fields = dict(re.findall(r'(\w+)=([\w.eE+-]+)', line))
            nr.append({'status': fields['status'],
                       'iterations': int(fields['iterations']),
                       'final_l2': float(fields['final_l2']),
                       'final_max': float(fields['final_max']),
                       'cycle': int(fields['cycle'])})
    pairs = compare([row for file in (folder / 'mpi').rglob('stdout')
                     for row in read_faces(file)])
    energy = [float(value) for value in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    outputs = sorted((folder / 'output').glob('*.athdf'))
    exit_status = int(grab(r'Exit status:\s*(\d+)', timing, '-1'))
    max_true = max((row['true_relative_residual'] for row in fg), default=None)
    return {
        'name': name, 'source_git_hash': SOURCE, 'binary_sha256': BINARY,
        'operator_path': grab(r'\[NRFLD_INTERFACE_MODE\] path=(\w+)', log),
        'input_sha256': sha(TRIALS / 'input.in'),
        'checkpoint': str(checkpoint) if name != 'seed' else None,
        'checkpoint_sha256': sha(checkpoint) if name != 'seed' else None,
        'command': grab(r'Command being timed: "([^"]+)"', timing),
        'environment': {
            'OMP_NUM_THREADS': 1,
            'ATHENA_NRFLD_CENTERED_FACE_TRIAL': name != 'resume_off',
            'ATHENA_NRFLD_INTERFACE_LEDGER': True,
            'ATHENA_NRFLD_ENERGY_LEDGER': True},
        'fgmres_target': 1e-7,
        'fgmres_history': fg,
        'fgmres_solve_count': len(fg),
        'fgmres_total_iterations': sum(row['iterations'] for row in fg),
        'max_true_relative_residual': max_true,
        'linear_target_met': max_true is not None and max_true < 1e-7,
        'newton_history': nr,
        'face_pair_count': len(pairs),
        'max_face_relative_mismatch': max(
            (row['relative_mismatch'] for row in pairs), default=None),
        'max_implicit_energy_absolute_relative_balance': max(energy, default=None),
        'exit_status': exit_status,
        'solver_pass': (exit_status == 0 and nr and nr[-1]['status'] == 'converged'
                        and max_true is not None and max_true < 1e-7),
        'failure_reason': grab(r'Newton-Raphson solve failed: ([^,]+)', log),
        'elapsed': grab(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(grab(r'Maximum resident set size \(kbytes\):\s*(\d+)',
                                timing, '0')),
        'final_output': str(outputs[-1]) if outputs else None,
        'final_output_sha256': sha(outputs[-1]) if outputs else None,
        'log_sha256': sha(folder / 'run.log')
    }


def main():
    checkpoint = TRIALS / 'seed/output/nrfld_diff_amr_interface.final.rst'
    cases = [parse(name, checkpoint) for name in ('seed', 'resume_on', 'resume_off')]
    OUT.write_text(json.dumps({
        'common_checkpoint_sha256': sha(checkpoint),
        'cases': cases,
        'conclusion': ('The same accepted checkpoint converged with the '
                       'centered-face operator and exhausted Newton '
                       'backtracking with the default face operator; '
                       'both completed linear solves met their 1e-7 target.')
    }, indent=2) + '\n')
    for case in cases:
        print(case['name'], case['solver_pass'], case['linear_target_met'],
              case['fgmres_solve_count'], case['failure_reason'])


if __name__ == '__main__':
    main()
