#!/usr/bin/env python3
"""Record opt-in face roundoff-scale diagnostic regression cases."""

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
TEST_INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
CHECKPOINT = TRIALS / 'collapse_postflddt_opacity_natural_r1/output/protostellar_core_fld_mayer5.00015.rst'
TEST_BINARY = Path('/tmp/wor25_23_24_resolution_roundoff_production_testc_build/bin/athena')
COLLAPSE_BINARY = Path('/tmp/wor25_23_24_resolution_roundoff_production_collapse_build/bin/athena')
BASE_TEST_BINARY = Path('/tmp/wor25_23_24_resolution_mode_k1_audit_build/bin/athena')
BASE_COLLAPSE_BINARY = Path('/tmp/wor25_23_24_resolution_collapse_postflddt_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match(pattern, source):
    found = re.search(pattern, source)
    if found is None:
        raise ValueError(pattern)
    return found.group(1)


def hdf_exact(name, baseline):
    files1 = sorted((TRIALS / name / 'output').glob('*.athdf'))
    files2 = sorted((TRIALS / baseline / 'output').glob('*.athdf'))
    if len(files1) != len(files2):
        return False
    for one, two in zip(files1, files2):
        with h5py.File(one) as a, h5py.File(two) as b:
            if not all(np.array_equal(a[key][:], b[key][:]) for key in a
                       if isinstance(a[key], h5py.Dataset)):
                return False
    return True


def face_data(rows):
    pairs = compare(rows)
    if not pairs:
        return {'pair_count': 0, 'pass': None}
    worst = max(pairs, key=lambda pair: pair['relative_mismatch'])
    coarse = next(row for row in rows if row['cycle'] == worst['cycle']
                  and row['stage'] == worst['stage'] and row['gid'] == worst['coarse_gid'])
    axis, side = worst['axis'], worst['side']
    opposite = 'max' if side == 'min' else 'min'
    fine = [row for row in rows if row['level'] == coarse['level'] + 1
            and row['cycle'] == coarse['cycle'] and row['stage'] == coarse['stage']
            and row[f'x{axis}{opposite}'] == coarse[f'x{axis}{side}']
            and all(row[f'x{other}min'] >= coarse[f'x{other}min']
                    and row[f'x{other}max'] <= coarse[f'x{other}max']
                    for other in range(1, 4) if other != axis)]
    left = f"{'xyz'[axis-1]}{'minus' if side == 'min' else 'plus'}_roundoff_scale"
    right = f"{'xyz'[axis-1]}{'plus' if side == 'min' else 'minus'}_roundoff_scale"
    scale = coarse[left] + sum(row[right] for row in fine)
    return {'pair_count': len(pairs), 'worst_pair': worst,
            'max_relative_mismatch': worst['relative_mismatch'],
            'roundoff_scale_at_worst_pair': scale,
            'mismatch_over_roundoff_scale': abs(worst['mismatch']) / scale,
            'relative_tolerance': 1e-6,
            'pass': worst['relative_mismatch'] <= 1e-6}


def record(name, baseline, kind, ranks, fixed=None, ledger=True):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    original = (TRIALS / baseline / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    stdout = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    rows = [row for file in stdout for row in read_faces(Path(file))]
    base_stdout = sorted(glob.glob(str(TRIALS / baseline / 'mpi/**/stdout'),
                                   recursive=True))
    base_rows = [row for file in base_stdout for row in read_faces(Path(file))]
    core_keys = ('cycle', 'stage', 'coarse_gid', 'axis', 'side', 'fine_count',
                 'coarse_flux', 'fine_flux', 'mismatch', 'relative_mismatch')
    core = lambda pair: tuple(pair[key] for key in core_keys)
    face_core_exact = ([core(pair) for pair in compare(rows)]
                       == [core(pair) for pair in compare(base_rows)]) if ledger else None
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)} for n, r in
          re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    newton = [{'status': status, 'iterations': int(n)} for status, n in
              re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log)]
    energy = [float(value) for value in re.findall(
        r'absolute_relative_balance=([\deE+.-]+)', log)]
    final = sorted((directory / 'output').glob('*.athdf'))[-1]
    with h5py.File(final) as hdf:
        output_minimum = (float(np.min(hdf['user_out_var'][0])) if kind == 'testc'
                          else float(np.min(hdf['prim'][0])))
    binary = TEST_BINARY if kind == 'testc' else COLLAPSE_BINARY
    base_binary = BASE_TEST_BINARY if kind == 'testc' else BASE_COLLAPSE_BINARY
    solver_pass = all(item['status'] in ('converged', 'dt_zero_initialization')
                      for item in newton) and all(item['true_relative_residual'] < 1e-8
                                                 for item in fg)
    face = face_data(rows) if ledger else {'pair_count': 0, 'pass': None}
    return {
        'name': name, 'kind': kind, 'ranks': ranks,
        'threads_per_rank': 1 if kind == 'testc' else 2,
        'source_base_commit': '211ace53',
        'changed_nr_fld_source_sha256': sha(ROOT / 'src/nr_multigrid/NRFLD.cpp'),
        'binary_sha256': sha(binary), 'base_binary_sha256': sha(base_binary),
        'input_path': str(TEST_INPUT) if kind == 'testc' else None,
        'input_sha256': sha(TEST_INPUT) if kind == 'testc' else None,
        'checkpoint_path': str(CHECKPOINT) if kind == 'collapse' else None,
        'checkpoint_sha256': sha(CHECKPOINT) if kind == 'collapse' else None,
        'overrides': {'fld/fixed_flux_limiter': fixed} if fixed is not None else
                     {'time/nlim': 17, 'problem/N_J': 7.616,
                      'nrfld/consistent_rb_colors': False,
                      'nrfld/symmetric_rb_sweeps': False},
        'environment': {'ATHENA_NRFLD_INTERFACE_LEDGER': int(ledger),
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1,
                        'OMP_NUM_THREADS': 1 if kind == 'testc' else 2,
                        'ATHENA_NRFLD_EXPERIMENTAL_COLLAPSE': int(kind == 'collapse')},
        'command': match(r'Command being timed: "([^"]+)"', timing),
        'run_log': str(directory / 'run.log'), 'face_logs': stdout,
        'exit_status': int(match(r'Exit status:\s*(\d+)', timing)),
        'wall_time': match(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(match(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_history': newton, 'fgmres_history': fg,
        'fgmres_total_iterations': sum(item['iterations'] for item in fg),
        'max_true_relative_residual': max(item['true_relative_residual'] for item in fg),
        'max_implicit_energy_absolute_relative_balance': max(energy),
        'output_minimum': output_minimum, 'face_ledger': face,
        'hdf_exact_vs_base': hdf_exact(name, baseline),
        'face_ledger_core_exact_vs_base': face_core_exact,
        'fgmres_history_exact_vs_base': re.findall(
            r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)
            == re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', original),
        'solver_pass': solver_pass,
        'evaluation': 'PASS_SOLVER_FACE_NOT_MEASURED' if not ledger and solver_pass else
                      'PASS' if solver_pass and face['pass'] else
                      'FAIL_FACE_RELATIVE_ONLY' if solver_pass else 'FAIL_SOLVER',
    }


cases = [
    record(f'threelevel_roundoff_prod_{mode}_r{rank}',
           f'testc_threelevel_shifted{"" if mode == "fixed" else "_variable"}_r{rank}',
           'testc', rank, mode == 'fixed')
    for mode in ('fixed', 'variable') for rank in (1, 2)
]
cases += [record(f'collapse_natural_roundoff_prod_r{rank}',
                 f'collapse_natural_postflddt_colors_default_r{rank}',
                 'collapse', rank) for rank in (1, 2)]
cases.append(record('threelevel_roundoff_prod_off_r1',
                    'testc_threelevel_shifted_r1', 'testc', 1, True, False))
report = json.loads(REPORT.read_text())
report['opt_in_face_roundoff_scale_diagnostic'] = {
    'scope': 'Production opt-in interface-ledger scale only; no Newton/linear operator or pass criterion change',
    'cases': cases,
    'decision': 'Retain as an opt-in diagnostic: all seven runs have exactly equal HDF numeric arrays and FGMRES histories versus base. Fixed/variable three-level face FAILs and natural-collapse rank-dependent PASS/FAIL are unchanged. Scale is a heuristic cancellation indicator, not a rigorous error bound; the face tolerance remains 1e-6. Off mode emits no face-ledger rows and converges identically.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps([(case['name'], case['hdf_exact_vs_base'],
                   case['fgmres_history_exact_vs_base'],
                   case['face_ledger'].get('max_relative_mismatch'),
                   case['evaluation']) for case in cases], indent=2))
