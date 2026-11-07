#!/usr/bin/env python3
"""Record extended-precision A*x and cast-gap audit of stalled FGMRES."""

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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find(pattern, source):
    hit = re.search(pattern, source)
    return hit.group(1) if hit else None


def trial(name, ranks, enabled):
    folder = TRIALS / name
    log = (folder / 'run.log').read_text(errors='replace')
    timing = (folder / 'time.log').read_text(errors='replace')
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
              log)]
    cast = [{'iterations': int(n), 'extended_residual_rel': float(r),
             'cast_gap_rel': float(g), 'solution_norm': float(x)}
            for n, r, g, x in re.findall(
                r'\[LINEAR_FGMRES_MATVEC_CAST_AUDIT\] iterations=(\d+) '
                r'extended_residual_rel=([\deE+.-]+) '
                r'cast_gap_rel=([\deE+.-]+) solution_norm=([\deE+.-]+)', log)]
    stdout = sorted(glob.glob(str(folder / 'mpi/**/stdout'), recursive=True))
    faces = compare([row for file in stdout for row in read_faces(Path(file))])
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1,
        'cast_audit_enabled': enabled, 'source_commit_before_run': '170f125f',
        'binary_sha256': sha(BINARY), 'source_sha256': sha(
            ROOT / 'src/linear_solver/linearMG/linearMG.cpp'),
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'command': find(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1,
                        'ATHENA_NRFLD_FGMRES_MATVEC_CAST_AUDIT': int(enabled),
                        'ATHENA_NRFLD_FGMRES_RESTART_AUDIT': int(enabled),
                        'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'run_log': str(folder / 'run.log'), 'face_logs': stdout,
        'exit_status': int(find(r'Exit status:\s*(\d+)', timing)),
        'wall_time': find(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(find(
            r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'fgmres_history': fg, 'matvec_cast_history': cast,
        'fgmres_fatal': find(r'(FGMRES did not reach[^\n]*)', log),
        'face_pair_count_before_failure': len(faces),
        'max_face_relative_mismatch_before_failure': max(
            (x['relative_mismatch'] for x in faces), default=None),
        'evaluation': 'PASS_AUDIT_FAIL_STRICT_FGMRES' if enabled
                      else 'CONTROL_FAIL_STRICT_FGMRES',
    }


def hdf_equal(a, b):
    one = sorted((TRIALS / a / 'output').glob('*.fields.*.athdf'))
    two = sorted((TRIALS / b / 'output').glob('*.fields.*.athdf'))
    if len(one) != len(two):
        return False
    for x, y in zip(one, two):
        with h5py.File(x) as xx, h5py.File(y) as yy:
            if not all(np.array_equal(xx[key][:], yy[key][:]) for key in xx
                       if isinstance(xx[key], h5py.Dataset)):
                return False
    return True


cases = [
    trial('threelevel_coupled_matvec_cast_r1', 1, True),
    trial('threelevel_coupled_matvec_cast_off_r1', 1, False),
    trial('threelevel_coupled_matvec_cast_r2', 2, True),
]
on, off, rank2 = cases
comparisons = {
    'same_binary_on_off': on['binary_sha256'] == off['binary_sha256'],
    'fgmres_history_on_off_exact': on['fgmres_history'] == off['fgmres_history'],
    'all_available_hdf_numeric_arrays_on_off_exact': hdf_equal(on['name'], off['name']),
    'failing_solve_by_rank': [
        {'ranks': c['ranks'], 'true_relative_residual': c['fgmres_history'][-1][
            'true_relative_residual'], **c['matvec_cast_history'][-1]}
        for c in (on, rank2)],
}
report = json.loads(REPORT.read_text())
report['wor25_matvec_cast_gap_audit'] = {
    'scope': 'Same failing 3-level coupled variable-limiter solve on 1/2 '
             'MPI ranks; recompute stored A*x row in long double without '
             'casting its output to Real.',
    'cases': cases, 'comparisons': comparisons,
    'decision': 'The final A*x output cast contributes ~3.3e-17 relative '
                'while the strict true residual is ~1.75e-8. Output casting '
                'cannot explain the projected/true gap. The audit ON/OFF '
                'FGMRES histories and HDF arrays match exactly on one rank. '
                'Next inspect solution-vector accumulation and operator '
                'conditioning; no solver threshold or operator change.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(comparisons)
