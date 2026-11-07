#!/usr/bin/env python3
"""Record an opt-in FGMRES linearity audit at every coupled solve."""

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
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
BINARY = Path('/tmp/wor25_23_24_resolution_fgmres_norm_audit_build/bin/athena')
OLD_BINARY = Path('/tmp/wor25_23_24_resolution_roundoff_production_testc_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find(pattern, source):
    match = re.search(pattern, source)
    return match.group(1) if match else None


def trial(name, ranks, enabled, binary):
    folder = TRIALS / name
    log = (folder / 'run.log').read_text(errors='replace')
    timing = (folder / 'time.log').read_text(errors='replace')
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
              log)]
    lin = [{'additivity_rel': float(a), 'homogeneity_rel': float(h),
            'solve_index': int(i)} for a, h, i in re.findall(
                r'\[LINEAR_FGMRES_LINEARITY\] additivity_rel=([\deE+.-]+) '
                r'homogeneity_rel=([\deE+.-]+)[^\n]*solve_index=(\d+)', log)]
    nr = [{'status': s, 'iterations': int(n), 'cycle': int(c)}
          for s, n, c in re.findall(
              r'\[NR\] status=(\w+) iterations=(\d+)[^\n]*cycle=(\d+)', log)]
    stdout = sorted(glob.glob(str(folder / 'mpi/**/stdout'), recursive=True))
    faces = compare([row for file in stdout for row in read_faces(Path(file))])
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1,
        'audit_enabled': enabled, 'binary_sha256': sha(binary),
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'command': find(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1,
                        'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1,
                        'ATHENA_NRFLD_FGMRES_LINEARITY_EACH_SOLVE': int(enabled),
                        'ATHENA_NRFLD_FGMRES_NORM_AUDIT': int(enabled)},
        'run_log': str(folder / 'run.log'), 'face_logs': stdout,
        'exit_status': int(find(r'Exit status:\s*(\d+)', timing)),
        'wall_time': find(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(find(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_history': nr, 'fgmres_history': fg,
        'fgmres_linearity_history': lin,
        'fgmres_fatal': find(r'(FGMRES did not reach[^\n]*)', log),
        'first_cycle_face_count': len(faces),
        'first_cycle_max_face_relative_mismatch': max(
            (x['relative_mismatch'] for x in faces), default=None),
        'evaluation': 'FAIL_STRICT_SOLVER',
    }


def hdf_equal(a, b):
    files_a = sorted((TRIALS / a / 'output').glob('*.fields.*.athdf'))
    files_b = sorted((TRIALS / b / 'output').glob('*.fields.*.athdf'))
    if len(files_a) != len(files_b):
        return False
    for path_a, path_b in zip(files_a, files_b):
        with h5py.File(path_a) as x, h5py.File(path_b) as y:
            for key in x:
                if isinstance(x[key], h5py.Dataset):
                    if key not in y or not np.array_equal(x[key][:], y[key][:]):
                        return False
    return True


cases = []
for rank in (1, 2):
    for suffix, enabled in [('linearity_after_off', False),
                            ('linearity_after', True)]:
        cases.append(trial(f'threelevel_coupled_{suffix}_r{rank}', rank,
                           enabled, BINARY))
cases.append(trial('threelevel_coupled_variable_restart80_r1', 1,
                   False, OLD_BINARY))
comparisons = []
for rank in (1, 2):
    off, on = cases[(rank - 1)*2:(rank - 1)*2 + 2]
    comparisons.append({
        'ranks': rank,
        'same_binary': off['binary_sha256'] == on['binary_sha256'],
        'fgmres_history_exact': off['fgmres_history'] == on['fgmres_history'],
        'newton_history_exact': off['newton_history'] == on['newton_history'],
        'all_available_hdf_numeric_arrays_exact': hdf_equal(off['name'], on['name']),
        'max_audited_additivity_rel': max(
            x['additivity_rel'] for x in on['fgmres_linearity_history']),
        'max_audited_homogeneity_rel': max(
            x['homogeneity_rel'] for x in on['fgmres_linearity_history']),
        'failed_solve_linearity': on['fgmres_linearity_history'][-1],
    })
report = json.loads(REPORT.read_text())
report['wor25_linearity_each_solve_audit'] = {
    'scope': 'Coupled variable-limiter 3-level shifted Test C, two steps, '
             '1/2 ranks; post-solve opt-in linearity audit ON/OFF on the same '
             'binary. One-rank restart-80 control on older isolated binary.',
    'cases': cases, 'on_off_comparisons': comparisons,
    'decision': 'The failing second-step FGMRES solve is linear to ~3e-16 on '
                'the probe vectors. Audit ON/OFF histories and HDF arrays '
                'match exactly within each rank and binary; the strict '
                '1e-8 solve still fails at 80 iterations. Restart 80 also '
                'fails on the older binary. No tolerance or operator change.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
for row in comparisons:
    print(row)
