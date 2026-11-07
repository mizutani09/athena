#!/usr/bin/env python3
"""Record test-only arithmetic screens of the natural AMR collapse face."""

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
CHECKPOINT1 = TRIALS / 'collapse_postflddt_opacity_natural_r1/output/protostellar_core_fld_mayer5.00015.rst'
CHECKPOINT2 = TRIALS / 'collapse_postflddt_opacity_natural_r2/output/protostellar_core_fld_mayer5.00015.rst'
ROUND_BINARY = Path('/tmp/wor25_23_24_resolution_collapse_face_roundoff_build/bin/athena')
ARITH_BINARY = Path('/tmp/wor25_23_24_resolution_collapse_arithmetic_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match(pattern, source):
    found = re.search(pattern, source)
    if found is None:
        raise ValueError(pattern)
    return found.group(1)


def mass(hdf):
    volume = (np.diff(hdf['x3f'][:], axis=1)[:, :, None, None]
              * np.diff(hdf['x2f'][:], axis=1)[:, None, :, None]
              * np.diff(hdf['x1f'][:], axis=1)[:, None, None, :])
    return float(np.sum(hdf['prim'][0] * volume))


def face_result(rows):
    pairs = compare(rows)
    bad = max(pairs, key=lambda pair: pair['relative_mismatch'])
    coarse = next(row for row in rows if row['cycle'] == bad['cycle']
                  and row['stage'] == bad['stage'] and row['gid'] == bad['coarse_gid'])
    axis, side = bad['axis'], bad['side']
    opposite = 'max' if side == 'min' else 'min'
    fine = [row for row in rows if row['level'] == coarse['level'] + 1
            and row['cycle'] == coarse['cycle'] and row['stage'] == coarse['stage']
            and row[f'x{axis}{opposite}'] == coarse[f'x{axis}{side}']
            and all(row[f'x{other}min'] >= coarse[f'x{other}min']
                    and row[f'x{other}max'] <= coarse[f'x{other}max']
                    for other in range(1, 4) if other != axis)]
    coarse_key = f"{'xyz'[axis - 1]}{'minus' if side == 'min' else 'plus'}"
    fine_key = f"{'xyz'[axis - 1]}{'plus' if side == 'min' else 'minus'}"
    roundoff_scale = None
    if f'{coarse_key}_roundoff_scale' in coarse:
        roundoff_scale = coarse[f'{coarse_key}_roundoff_scale'] + sum(
            row[f'{fine_key}_roundoff_scale'] for row in fine)
    return {'pair_count': len(pairs), 'max_relative_mismatch': bad['relative_mismatch'],
            'bad_pair': bad, 'bad_pair_coarse_coordinates': {
                key: coarse[key] for key in ('x1min', 'x1max', 'x2min', 'x2max', 'x3min', 'x3max')},
            'roundoff_scale_at_bad_pair': roundoff_scale,
            'mismatch_over_roundoff_scale': abs(bad['mismatch']) / roundoff_scale
            if roundoff_scale else None,
            'relative_tolerance': 1e-6, 'pass': bad['relative_mismatch'] <= 1e-6}


def hdf_exact(name, baseline):
    files1 = sorted((TRIALS / name / 'output').glob('*.athdf'))
    files2 = sorted((TRIALS / baseline / 'output').glob('*.athdf'))
    if len(files1) != len(files2):
        return False
    for one, two in zip(files1, files2):
        with h5py.File(one) as a, h5py.File(two) as b:
            for key in a:
                if isinstance(a[key], h5py.Dataset) and not np.array_equal(a[key][:], b[key][:]):
                    return False
    return True


def run(name, ranks, checkpoint, binary, baseline, alternate=False):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    stdout = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    rows = [row for path in stdout for row in read_faces(Path(path))]
    face = face_result(rows)
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    nr = [{'status': status, 'iterations': int(n)} for status, n in
          re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log)]
    prim = sorted((directory / 'output').glob('*.prim.*.athdf'))
    with h5py.File(prim[0]) as first, h5py.File(prim[-1]) as last:
        m0, m1 = mass(first), mass(last)
        minima = {'rho': float(np.min(last['prim'][0])),
                  'pressure': float(np.min(last['prim'][1])),
                  'Er': float(np.min(last['prim'][8]))}
        levels = {str(int(level)): int(count) for level, count in
                  zip(*np.unique(last['Levels'][:], return_counts=True))}
    solver_pass = all(item['status'] == 'converged' for item in nr)
    solver_pass &= all(item['true_relative_residual'] < 1e-8 for item in fg)
    result = {
        'name': name, 'ranks': ranks, 'threads_per_rank': 2,
        'base_source_commit': '4a2d5e03', 'base_binary_sha256': sha(
            '/tmp/wor25_23_24_resolution_collapse_postflddt_build/bin/athena'),
        'test_binary_sha256': sha(binary), 'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': sha(checkpoint),
        'environment': {'OMP_NUM_THREADS': 2, 'ATHENA_NRFLD_EXPERIMENTAL_COLLAPSE': 1,
                        'ATHENA_NRFLD_INTERFACE_LEDGER': 1, 'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'overrides': {'mesh/num_threads': 2, 'time/nlim': 17, 'problem/N_J': 7.616,
                      'nrfld/global_jacobian_check': False,
                      'gravity/reset_fmg_color_each_solve': True,
                      'gravity/repeat_initial_fmg_after_restart': True,
                      'nrfld/consistent_rb_colors': False,
                      'nrfld/symmetric_rb_sweeps': False},
        'command': match(r'Command being timed: "([^"]+)"', timing),
        'run_log': str(directory / 'run.log'), 'face_logs': stdout,
        'exit_status': int(match(r'Exit status:\s*(\d+)', timing)),
        'wall_time': match(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(match(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_history': nr, 'fgmres_history': fg,
        'fgmres_total_iterations': sum(item['iterations'] for item in fg),
        'max_true_relative_residual': max(item['true_relative_residual'] for item in fg),
        'face_ledger': face,
        'max_implicit_energy_absolute_relative_balance': max(map(float,
            re.findall(r'absolute_relative_balance=([\deE+.-]+)', log))),
        'mass_volume_integral': {'initial': m0, 'final': m1,
                                 'relative_change': (m1 - m0) / m0},
        'final_minima': minima, 'final_level_counts': levels,
        'hdf_exact_vs_base': hdf_exact(name, baseline),
        'solver_pass': solver_pass,
        'evaluation': 'PASS' if solver_pass and face['pass'] else
                      'FAIL_FACE_RELATIVE_ONLY' if solver_pass else 'FAIL_SOLVER',
    }
    if alternate:
        alternatives = {}
        for variant in ('ld', 'moment'):
            transformed = []
            for file in stdout:
                text = Path(file).read_text()
                for axis in ('xminus', 'xplus', 'yminus', 'yplus', 'zminus', 'zplus'):
                    text = text.replace(axis + '_delta=', axis + '_delta_old=')
                    text = text.replace(axis + '_delta_' + variant + '=', axis + '_delta=')
                output = directory / f'{variant}_{Path(file).parent.name}_stdout'
                output.write_text(text)
                transformed.append(output)
            alternatives[variant] = face_result(
                [row for path in transformed for row in read_faces(path)])
        result['alternate_face_ledgers'] = alternatives
    return result


runs = [
    run('collapse_natural_face_roundoff_r1', 1, CHECKPOINT1, ROUND_BINARY,
        'collapse_natural_postflddt_colors_default_r1'),
    run('collapse_natural_face_roundoff_common_r2', 2, CHECKPOINT1, ROUND_BINARY,
        'collapse_natural_postflddt_colors_default_r2'),
    run('collapse_natural_arithmetic_r1', 1, CHECKPOINT1, ARITH_BINARY,
        'collapse_natural_postflddt_colors_default_r1', True),
]
separate_checkpoint = run('collapse_natural_face_roundoff_r2', 2, CHECKPOINT2,
                          ROUND_BINARY, 'collapse_natural_postflddt_colors_default_r2')
report = json.loads(REPORT.read_text())
report['collapse_natural_face_arithmetic_screen'] = {
    'scope': 'Natural 43-to-64-leaf collapse, cycle-15 restart with common checkpoint, default colors and thresholds; test-only roundoff scale and alternate face arithmetic',
    'roundoff_patch_sha256': sha(ROOT / 'tst/nrfld_validation/threelevel_roundoff_scale_test_only.patch'),
    'arithmetic_patch_sha256': sha(ROOT / 'tst/nrfld_validation/threelevel_arithmetic_ledger_test_only.patch'),
    'runs': runs, 'different_checkpoint_exploratory_run': separate_checkpoint,
    'decision': 'At the common checkpoint, 1-rank worst-face mismatch 0.352 on flux 1.35e5 is 5.78% of the epsilon-weighted local face scale 6.09; 2-rank worst mismatch 0.199 on flux 5.10e5 is 3.27% of scale 6.09. This heuristic is consistent with cancellation sensitivity but is not a rigorous bound. Long-double-only face recomputation leaves the 1-rank maximum 1.30039e-6 FAIL; direct shared-moment face diagnostic reduces it to 1.03567e-6 but still FAIL, and worsened the three-level Test C condition. Both alternatives are rejected as universal fixes. Diagnostic ON/OFF HDF arrays are exactly equal for matching checkpoint/rank. The separate 2-rank checkpoint trial is not used for rank comparison.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'common_checkpoint': [
    (r['name'], r['face_ledger']['max_relative_mismatch'],
     r['face_ledger']['mismatch_over_roundoff_scale'], r['evaluation']) for r in runs],
    'alternates': {key: value['max_relative_mismatch'] for key, value in
                   runs[2]['alternate_face_ledgers'].items()}}, indent=2))
