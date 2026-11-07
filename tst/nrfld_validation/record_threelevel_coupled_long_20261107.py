#!/usr/bin/env python3
"""Record five-step coupled three-level FGMRES and face-balance controls."""

import glob
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
BINARY = Path('/tmp/wor25_23_24_resolution_roundoff_production_testc_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match(pattern, source, default=None):
    found = re.search(pattern, source)
    return found.group(1) if found else default


def run(name, ranks, kind):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    files = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    pairs = compare([row for file in files for row in read_faces(Path(file))])
    by_cycle = defaultdict(list)
    for pair in pairs:
        by_cycle[pair['cycle']].append(pair)
    energy_by_cycle = defaultdict(list)
    for line in (line for file in files for line in Path(file).read_text().splitlines()):
        if not line.startswith('[NR_ENERGY_LEDGER]'):
            continue
        values = dict(re.findall(r'([A-Za-z_]+)=([\deE+.-]+)', line))
        energy_by_cycle[int(values['cycle'])].append({
            'rank': int(values['rank']),
            'interior_delta': float(values['interior_delta']),
            'absolute_relative_balance': float(values['absolute_relative_balance'])})
    cycles = []
    for cycle, entries in sorted(by_cycle.items()):
        local_energy = energy_by_cycle[cycle]
        worst = max(entries, key=lambda x: x['relative_mismatch'])
        cycles.append({
            'cycle': cycle, 'coarse_fine_face_pair_count': len(entries),
            'max_face_relative_mismatch': worst['relative_mismatch'],
            'worst_face_pair': worst,
            'max_face_absolute_mismatch': max(abs(x['mismatch']) for x in entries),
            'coarse_fine_mismatch_sum': sum(x['mismatch'] for x in entries),
            'coarse_fine_absolute_mismatch_sum': sum(abs(x['mismatch']) for x in entries),
            'global_interior_diffusion_sum': sum(x['interior_delta'] for x in local_energy),
            'energy_rank_rows': local_energy,
            'face_pass': all(x['relative_mismatch'] <= 1e-6 for x in entries)})
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)} for n, r in
          re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    newton = [{'status': status, 'iterations': int(n), 'cycle': int(cycle)}
              for status, n, cycle in re.findall(
                  r'\[NR\] status=(\w+) iterations=(\d+)[^\n]*cycle=(\d+)', log)]
    hdf = sorted((directory / 'output').glob('*.fields.*.athdf'))
    output = None
    if len(hdf) > 1:
        with h5py.File(hdf[-1]) as final:
            output = {'file': str(hdf[-1]),
                      'min_Er': float(np.min(final['user_out_var'][0])),
                      'max_Er': float(np.max(final['user_out_var'][0])),
                      'final_time': float(final.attrs['Time'])}
    exit_status = int(match(r'Exit status:\s*(\d+)', timing))
    target = 1.5e-8 if kind == 'tol15' else 1e-8
    solver_pass = exit_status == 0 and len([x for x in newton if
                    x['status'] == 'converged']) == 5 and all(
                        x['true_relative_residual'] < target for x in fg)
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1,
        'source_commit_before_run': 'c8136c81',
        'binary_sha256': sha(BINARY), 'input_path': str(INPUT),
        'input_sha256': sha(INPUT),
        'command': match(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'trial': kind, 'fgmres_target': target,
        'run_log': str(directory / 'run.log'), 'face_logs': files,
        'exit_status': exit_status,
        'wall_time': match(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(match(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_history': newton, 'fgmres_history': fg,
        'fgmres_total_iterations': sum(x['iterations'] for x in fg),
        'max_true_relative_residual': max(x['true_relative_residual'] for x in fg),
        'fgmres_fatal': match(r'(FGMRES did not reach[^\n]*)', log),
        'face_history': cycles, 'final_output': output,
        'solver_pass': solver_pass,
        'evaluation': 'FAIL_SOLVER' if not solver_pass else
                      'FAIL_FACE_RELATIVE_ONLY' if not all(x['face_pass'] for x in cycles)
                      else 'PASS',
    }


def equal_final(a, b):
    path_a = sorted((TRIALS / a / 'output').glob('*.fields.*.athdf'))[-1]
    path_b = sorted((TRIALS / b / 'output').glob('*.fields.*.athdf'))[-1]
    with h5py.File(path_a) as x, h5py.File(path_b) as y:
        return all(np.array_equal(x[key][:], y[key][:]) for key in x
                   if isinstance(x[key], h5py.Dataset))


cases = [run(f'threelevel_coupled_variable_long5{suffix}_r{rank}', rank, kind)
         for suffix, kind in [('', 'strict'), ('_v2', 'v2_strict'),
                              ('_tol15', 'tol15')]
         for rank in (1, 2)]
report = json.loads(REPORT.read_text())
report['threelevel_coupled_multistep_screen'] = {
    'scope': 'Five-step static 3-level coupled variable-limiter Test C, 1/2 MPI ranks. Strict FGMRES 1e-8, 2 MG V-cycles, and exploratory relaxed target 1.5e-8.',
    'cases': cases,
    'tol15_final_rank_hdf_exact': equal_final(
        'threelevel_coupled_variable_long5_tol15_r1',
        'threelevel_coupled_variable_long5_tol15_r2'),
    'decision': 'At the second time step, both ranks fail the 1e-8 FGMRES target around 1.12e-8; 2 MG V-cycles also fail around 1.13e-8. The 1.5e-8 exploratory setting permits five steps and records per-cycle face balance, but is not adopted as a production tolerance. Face mismatch persistence must be interpreted with this solver-setting caveat. No default or operator change.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
for case in cases:
    print(case['name'], case['evaluation'], case['fgmres_total_iterations'],
          case['fgmres_fatal'], [(x['cycle'], x['max_face_relative_mismatch'],
                                x['face_pass']) for x in case['face_history']])
