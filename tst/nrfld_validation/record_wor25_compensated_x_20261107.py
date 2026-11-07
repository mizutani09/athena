#!/usr/bin/env python3
"""Record same-binary FGMRES restart compensation ON/OFF experiments."""

import hashlib
import json
import re
from pathlib import Path

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_compensated_x_trials')
OUT = ROOT / 'tst/nrfld_validation/wor25_compensated_x_20261107.json'
SOURCE = '2ecbea05f13e29fcfa4905e1cbedd2342059be14'
BINARY = '90489fbefee919580859cdc19e92321ec078bbb3d4ce306c559460415e0f6dfb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grab(pattern, content, default=None):
    m = re.search(pattern, content)
    return m.group(1) if m else default


def parse(name):
    folder = TRIALS / name
    log = (folder / 'run.log').read_text(errors='replace')
    timing = (folder / 'time.log').read_text(errors='replace')
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
              log)]
    restarts = [{'iterations': int(n), 'projected': float(p), 'true': float(t),
                 'rhs_norm': float(b)}
                for n, p, t, b in re.findall(
                    r'\[LINEAR_FGMRES_RESTART_AUDIT\] total_iterations=(\d+) '
                    r'used=\d+ projected_relative_residual=([\deE+.-]+) '
                    r'true_relative_residual=([\deE+.-]+) rhs_norm=([\deE+.-]+)',
                    log)]
    nr = []
    for line in log.splitlines():
        if line.startswith('[NR] status='):
            fields = dict(re.findall(r'(\w+)=([\w.eE+-]+)', line))
            nr.append({'status': fields['status'],
                       'iterations': int(fields['iterations']),
                       'final_l2': float(fields['final_l2']),
                       'cycle': int(fields['cycle'])})
    pairs = compare([row for path in (folder / 'mpi').rglob('stdout')
                     for row in read_faces(path)])
    energy = [float(v) for v in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    return {
        'name': name, 'source_base_git_hash': SOURCE,
        'trial_patch_sha256': sha(ROOT / 'tst/nrfld_validation/wor25_compensated_x_test_only.patch'),
        'binary_sha256': BINARY,
        'input': str(ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'),
        'input_sha256': sha(ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'),
        'command': grab(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1,
                        'ATHENA_NRFLD_CENTERED_FACE_TRIAL': True,
                        'ATHENA_NRFLD_FGMRES_COMPENSATED_X_TRIAL':
                            name == 'strict_centered_r1',
                        'ATHENA_NRFLD_FGMRES_RESTART_AUDIT': True,
                        'ATHENA_NRFLD_INTERFACE_LEDGER': True,
                        'ATHENA_NRFLD_ENERGY_LEDGER': True},
        'newton_history': nr, 'fgmres_history': fg,
        'fgmres_solve_count': len(fg),
        'fgmres_total_iterations': sum(row['iterations'] for row in fg),
        'restart_audit': restarts,
        'last_restart_audit': restarts[-1],
        'face_pair_count': len(pairs),
        'max_face_relative_mismatch': max(
            (pair['relative_mismatch'] for pair in pairs), default=None),
        'max_implicit_energy_absolute_relative_balance': max(energy, default=None),
        'elapsed': grab(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(grab(r'Maximum resident set size \(kbytes\):\s*(\d+)',
                                timing, '0')),
        'exit_status': int(grab(r'Exit status:\s*(\d+)', timing, '-1')),
        'strict_pass': int(grab(r'Exit status:\s*(\d+)', timing, '-1')) == 0
                       and nr[-1]['status'] == 'converged'
                       and nr[-1]['cycle'] == 1,
        'failure': grab(r'(FGMRES did not reach its requested relative residual:[^\n]*)',
                        log),
        'log_sha256': sha(folder / 'run.log')
    }


def main():
    runs = [parse(name) for name in ('strict_control_r1', 'strict_centered_r1')]
    OUT.write_text(json.dumps({'runs': runs}, indent=2) + '\n')
    for run in runs:
        print(run['name'], run['strict_pass'], run['last_restart_audit'])


if __name__ == '__main__':
    main()
