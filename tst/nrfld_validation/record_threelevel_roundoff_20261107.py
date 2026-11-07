#!/usr/bin/env python3
"""Record pulse-amplitude and test-only face roundoff-scale diagnostics."""

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
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
BASE_BINARY = Path('/tmp/wor25_23_24_resolution_mode_k1_audit_build/bin/athena')
PROBE_BINARY = Path('/tmp/wor25_23_24_resolution_threelevel_roundoff_build/bin/athena')
PATCH = ROOT / 'tst/nrfld_validation/threelevel_roundoff_scale_test_only.patch'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(pattern, text):
    found = re.search(pattern, text)
    if found is None:
        raise ValueError(pattern)
    return found.group(1)


def hdf_equal(name1, name2):
    files1 = sorted((TRIALS / name1 / 'output').glob('*.athdf'))
    files2 = sorted((TRIALS / name2 / 'output').glob('*.athdf'))
    if len(files1) != len(files2):
        return False
    for one, two in zip(files1, files2):
        with h5py.File(one) as a, h5py.File(two) as b:
            for key in a:
                if isinstance(a[key], h5py.Dataset) and not np.array_equal(a[key][:], b[key][:]):
                    return False
    return True


def record(name, amplitude, binary, fixed_limiter=True):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace') if (directory / 'time.log').exists() else log
    rows = []
    logs = sorted((directory / 'mpi').glob('**/rank.*/stdout'))
    for path in logs:
        rows.extend(read_faces(path))
    pairs = compare(rows)
    worst = max(pairs, key=lambda pair: pair['relative_mismatch'])
    coarse = next(row for row in rows if row['gid'] == worst['coarse_gid']
                  and row['cycle'] == worst['cycle'] and row['stage'] == worst['stage'])
    axis = worst['axis']
    side = worst['side']
    opposite = 'max' if side == 'min' else 'min'
    fine = [row for row in rows if row['level'] == coarse['level'] + 1
            and row['cycle'] == coarse['cycle'] and row['stage'] == coarse['stage']
            and row[f'x{axis}{opposite}'] == coarse[f'x{axis}{side}']
            and all(row[f'x{other}min'] >= coarse[f'x{other}min']
                    and row[f'x{other}max'] <= coarse[f'x{other}max']
                    for other in range(1, 4) if other != axis)]
    key = f"{'xyz'[axis-1]}{'minus' if side == 'min' else 'plus'}"
    opposite_key = f"{'xyz'[axis-1]}{'plus' if side == 'min' else 'minus'}"
    scale = None
    if f'{key}_roundoff_scale' in coarse:
        scale = coarse[f'{key}_roundoff_scale'] + sum(
            row[f'{opposite_key}_roundoff_scale'] for row in fine)
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    nr = re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log)
    energy = [float(x) for x in re.findall(r'absolute_relative_balance=([\deE+.-]+)', log)]
    final = sorted((directory / 'output').glob('*.fields.*.athdf'))[-1]
    with h5py.File(final) as hdf:
        er = hdf['user_out_var'][0]
        er_min = float(er.min())
    return {
        'name': name, 'base_binary_source_commit': '9ee735fa',
        'binary_sha256': sha(binary),
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'overrides': {'problem/erad_amplitude': amplitude,
                      'fld/fixed_flux_limiter': fixed_limiter,
                      'nrfld/linear_solver': 'fgmres',
                      'nrfld/fgmres_linearity_check': False},
        'command': value(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'run_log': str(directory / 'run.log'), 'face_logs': list(map(str, logs)),
        'exit_status': int(value(r'Exit status:\s*(\d+)', timing)),
        'wall_time': value(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(value(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_statuses': nr, 'fgmres_history': fg,
        'max_true_relative_residual': max(f['true_relative_residual'] for f in fg),
        'face_pair_count': len(pairs), 'worst_pair': worst,
        'worst_pair_coarse_level': int(coarse['level']),
        'worst_pair_fine_gids': [int(row['gid']) for row in fine],
        'worst_pair_roundoff_scale': scale,
        'worst_pair_abs_mismatch_over_roundoff_scale': abs(worst['mismatch']) / scale if scale else None,
        'max_implicit_energy_absolute_relative_balance': max(energy),
        'final_er_min': er_min,
        'solver_pass': nr[-1][0] == 'converged' and all(f['true_relative_residual'] < 1e-8 for f in fg),
        'face_pass': worst['relative_mismatch'] <= 1e-6,
        'evaluation': ('PASS' if worst['relative_mismatch'] <= 1e-6
                       else 'FAIL_FACE_RELATIVE_ONLY'),
    }


cases = [
    record('threelevel_amp001_r1', 0.01, BASE_BINARY),
    record('testc_threelevel_shifted_r1', 0.1, BASE_BINARY),
    record('threelevel_amp1_r1', 1.0, BASE_BINARY),
    record('threelevel_roundoff_scale_r1', 0.1, PROBE_BINARY),
    record('threelevel_variable_amp001_r1', 0.01, BASE_BINARY, False),
    record('testc_threelevel_shifted_variable_r1', 0.1, BASE_BINARY, False),
    record('threelevel_variable_amp1_r1', 1.0, BASE_BINARY, False),
]
report = json.loads(REPORT.read_text())
report['threelevel_roundoff_scale_probe'] = {
    'scope': 'Fixed/variable limiter q=1e7 three-level shifted Test C, 1 MPI rank, one step; amplitude screen and fixed-limiter test-only face diagnostic',
    'probe_patch_sha256': sha(PATCH), 'cases': cases,
    'probe_hdf_exact_vs_base': hdf_equal('threelevel_roundoff_scale_r1', 'testc_threelevel_shifted_r1'),
    'decision': 'All amplitude cases converge. Fixed limiter: all three fail the 1e-6 per-face criterion, with worst-face absolute mismatch O(1e-14) and lower relative mismatch at larger amplitude. Variable limiter: amplitude 0.01/0.1 fail, amplitude 1.0 passes at 2.86e-9; limiter nonlinearity means flux does not scale linearly. At fixed amplitude 0.1, mismatch 2.40e-14 is 5.3% of the local epsilon-weighted face scale 4.56e-13. Scale is heuristic, not a rigorous error bound. Test-only diagnostic does not change fields. No production operator or tolerance change; root cause of arithmetic discrepancy not proven.',
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'cases': [
    {'name': r['name'], 'flux': r['worst_pair']['coarse_flux'],
     'absolute_mismatch': abs(r['worst_pair']['mismatch']),
     'relative_mismatch': r['worst_pair']['relative_mismatch'],
     'roundoff_scale': r['worst_pair_roundoff_scale']} for r in cases],
    'probe_hdf_exact_vs_base': report['threelevel_roundoff_scale_probe']['probe_hdf_exact_vs_base']},
    indent=2))
