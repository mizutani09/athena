#!/usr/bin/env python3
"""Record opt-in FGMRES absolute norm diagnostics near a strict floor."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
BINARY = Path('/tmp/wor25_23_24_resolution_fgmres_norm_audit_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match(pattern, source, default=None):
    found = re.search(pattern, source)
    return found.group(1) if found else default


def record(name, ranks, audit):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    norm_rows = [{'iterations': int(n), 'rhs_norm': float(b),
                  'true_residual_norm': float(r), 'target_norm': float(t)}
                 for n, b, r, t in re.findall(
                     r'\[LINEAR_FGMRES_NORM_AUDIT\] iterations=(\d+) '
                     r'rhs_norm=([\deE+.-]+) true_residual_norm=([\deE+.-]+) '
                     r'target_norm=([\deE+.-]+)', log)]
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)} for n, r in
          re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) '
                     r'relative_residual=([\deE+.-]+)', log)]
    hdf = sorted((directory / 'output').glob('*.fields.*.athdf'))
    with h5py.File(hdf[-1]) as final:
        last_output = {'file': str(hdf[-1]),
                       'minimum_Er': float(np.min(final['user_out_var'][0])),
                       'time': float(final.attrs['Time'])}
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1,
        'source_commit_before_run': '6bc87e3d',
        'changed_source_sha256': sha(ROOT / 'src/linear_solver/linearMG/linearMG.cpp'),
        'binary_sha256': sha(BINARY),
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'command': match(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1,
                        'ATHENA_NRFLD_FGMRES_NORM_AUDIT': int(audit)},
        'run_log': str(directory / 'run.log'),
        'exit_status': int(match(r'Exit status:\s*(\d+)', timing)),
        'wall_time': match(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(match(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'fgmres_history': fg, 'fgmres_norm_history': norm_rows,
        'fgmres_total_iterations': sum(x['iterations'] for x in fg),
        'fgmres_fatal': match(r'(FGMRES did not reach[^\n]*)', log),
        'last_output': last_output,
        'evaluation': 'FAIL_SOLVER' if 'FGMRES did not reach' in log else 'PASS',
    }


def equal_hdf(name1, name2):
    one = sorted((TRIALS / name1 / 'output').glob('*.fields.*.athdf'))
    two = sorted((TRIALS / name2 / 'output').glob('*.fields.*.athdf'))
    if len(one) != len(two):
        return False
    for path1, path2 in zip(one, two):
        with h5py.File(path1) as a, h5py.File(path2) as b:
            if not all(np.array_equal(a[key][:], b[key][:]) for key in a
                       if isinstance(a[key], h5py.Dataset)):
                return False
    return True


cases = [record('threelevel_coupled_variable_long5_norm_on_r1', 1, True),
         record('threelevel_coupled_variable_long5_norm_off_r1', 1, False),
         record('threelevel_coupled_variable_long5_norm_on_r2', 2, True),
         record('threelevel_coupled_variable_long5_max200_r1', 1, True)]
report = json.loads(REPORT.read_text())
report['wor25_fgmres_norm_floor_audit'] = {
    'scope': 'Opt-in logging only of already-computed RHS and true-residual L2 norms in the 5-step strict 3-level coupled variable-limiter case.',
    'cases': cases,
    'on_off_rank1_hdf_exact': equal_hdf(cases[0]['name'], cases[1]['name']),
    'on_off_rank1_fgmres_history_exact': cases[0]['fgmres_history']
                                          == cases[1]['fgmres_history'],
    'decision': 'This opt-in diagnostic exposes the absolute RHS and true-residual scales behind the FGMRES relative-residual failure without changing the target or failure policy. In the failed second step, RHS L2 is about 265.6 and true residual about 3e-6 versus target 2.66e-6; this alone does not prove machine-roundoff saturation. Raising the iteration cap to 200 does not reach 1e-8. ON/OFF solver and output controls are exact. Retain as a diagnostic only; do not relax the production stopping criterion.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps([(x['name'], x['fgmres_norm_history'][-1:] if x['fgmres_norm_history']
                   else [], x['evaluation']) for x in cases], indent=2))
