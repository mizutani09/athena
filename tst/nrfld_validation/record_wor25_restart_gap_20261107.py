#!/usr/bin/env python3
"""Record FGMRES projected/true residual gap and a rejected stop-factor trial."""

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
PATCH = ROOT / 'tst/nrfld_validation/fgmres_projected_stop_test_only.patch'
AUDIT_BINARY_SHA = 'd69bac32b578c064c694e6ca99497e7377754df079e44cddce0c831ee4f00c85'
CANDIDATE_BINARY_SHA = '9429247b044d9c5829d559a6884c9da91a831902d46acaab7f7edf20144e164a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find(pattern, source):
    hit = re.search(pattern, source)
    return hit.group(1) if hit else None


def trial(name, ranks, mode):
    folder = TRIALS / name
    log = (folder / 'run.log').read_text(errors='replace')
    timing = (folder / 'time.log').read_text(errors='replace')
    pending = []
    solves = []
    for line in log.splitlines():
        hit = re.search(
            r'\[LINEAR_FGMRES_RESTART_AUDIT\] total_iterations=(\d+) '
            r'used=(\d+) projected_relative_residual=([\deE+.-]+) '
            r'true_relative_residual=([\deE+.-]+) rhs_norm=([\deE+.-]+)', line)
        if hit:
            n, used, projected, true, rhs = hit.groups()
            pending.append({'total_iterations': int(n), 'used': int(used),
                            'projected_relative_residual': float(projected),
                            'true_relative_residual': float(true),
                            'rhs_norm': float(rhs)})
            continue
        hit = re.search(
            r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
            line)
        if hit:
            solves.append({'iterations': int(hit.group(1)),
                           'true_relative_residual': float(hit.group(2)),
                           'restart_history': pending})
            pending = []
    assert not pending
    stdout = sorted(glob.glob(str(folder / 'mpi/**/stdout'), recursive=True))
    faces = compare([row for file in stdout for row in read_faces(Path(file))])
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1, 'mode': mode,
        'source_commit_before_run': '170f125f',
        'binary_sha256': CANDIDATE_BINARY_SHA if mode == 'factor_0.001'
                         else AUDIT_BINARY_SHA,
        'test_only_patch_sha256': sha(PATCH) if mode == 'factor_0.001' else None,
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'command': find(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1,
                        'ATHENA_NRFLD_FGMRES_RESTART_AUDIT':
                            0 if mode == 'audit_off' else 1,
                        'ATHENA_NRFLD_FGMRES_PROJECTED_STOP_FACTOR':
                            0.001 if mode == 'factor_0.001' else None,
                        'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'run_log': str(folder / 'run.log'), 'face_logs': stdout,
        'exit_status': int(find(r'Exit status:\s*(\d+)', timing)),
        'wall_time': find(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(find(
            r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'fgmres_solves': solves,
        'fgmres_fatal': find(r'(FGMRES did not reach[^\n]*)', log),
        'face_pair_count_before_failure': len(faces),
        'max_face_relative_mismatch_before_failure': max(
            (x['relative_mismatch'] for x in faces), default=None),
        'evaluation': 'FAIL_STRICT_FGMRES',
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
    trial('threelevel_coupled_restart_audit_r1', 1, 'audit_on'),
    trial('threelevel_coupled_restart_audit_off_r1', 1, 'audit_off'),
    trial('threelevel_coupled_restart_audit_r2', 2, 'audit_on'),
    trial('threelevel_coupled_projected_factor001_r1', 1, 'factor_0.001'),
]
on, off, rank2, candidate = cases
assert sha(PATCH) == '56542ad0e357bff146c192e23cc431fd174650868b3063d9547a6f69c4ae473a'
comparisons = {
    'same_binary_on_off': on['binary_sha256'] == off['binary_sha256'],
    'fgmres_history_on_off_exact': [
        (s['iterations'], s['true_relative_residual']) for s in on['fgmres_solves']
    ] == [
        (s['iterations'], s['true_relative_residual']) for s in off['fgmres_solves']],
    'all_available_hdf_numeric_arrays_on_off_exact': hdf_equal(on['name'], off['name']),
    'failed_solve_final_projected_true_by_rank': [
        {'ranks': case['ranks'],
         'projected': case['fgmres_solves'][-1]['restart_history'][-1][
             'projected_relative_residual'],
         'true': case['fgmres_solves'][-1]['true_relative_residual']}
        for case in (on, rank2)],
    'factor_trial_final_true': candidate['fgmres_solves'][-1][
        'true_relative_residual'],
    'factor_trial_min_true_during_failed_solve': min(
        x['true_relative_residual']
        for x in candidate['fgmres_solves'][-1]['restart_history']),
}
report = json.loads(REPORT.read_text())
report['wor25_fgmres_restart_gap_audit'] = {
    'scope': '3-level shifted coupled variable-limiter Test C, two steps, '
             '1/2 MPI ranks, strict true residual target 1e-8. One-rank '
             'test-only projected stop factor 0.001.',
    'cases': cases, 'comparisons': comparisons,
    'decision': 'At the failing solve, the projected residual falls far below '
                '1e-8 while independently recomputed true residual stalls '
                'around 1.75e-8 on both ranks. Forcing projected stop to '
                '1e-11-scale does not lower true residual below 1e-8. '
                'Candidate stop-factor code is reverted; no tolerance change. '
                'Next isolate solution accumulation and independent A*x '
                'rounding before changing Krylov arithmetic.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(comparisons)
