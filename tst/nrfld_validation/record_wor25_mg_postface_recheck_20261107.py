#!/usr/bin/env python3
"""Compare MG alone before/after the centered Test C face operator."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_mg_recheck_20261107')
OUT = ROOT / 'tst/nrfld_validation/wor25_mg_postface_recheck_20261107.json'
SOURCE_COMMIT = 'b0849726084aec3fc055adf3da31f36dc72404e2'
BINARY_SHA256 = '62c3a7f2ebc15af56062c9d90669deae7b5e60890ca33d3a51d413fd35c4d121'
RUNS = {
    'mg_centered_r1': ('fixed_q1e7', 'mg', 'centered', 1, 1),
    'mg_oldface_r1': ('fixed_q1e7', 'mg', 'old', 1, 1),
    'fgmres_centered_r1': ('fixed_q1e7', 'fgmres', 'centered', 1, 1),
    'mg_centered_coupled_r1': ('coupled_variable', 'mg', 'centered', 1, 1),
    'mg_oldface_coupled_r1': ('coupled_variable', 'mg', 'old', 1, 1),
    'mg_centered_coupled_2step_r1': ('coupled_variable', 'mg', 'centered', 1, 2),
    'mg_oldface_coupled_2step_r1': ('coupled_variable', 'mg', 'old', 1, 2),
    'mg_centered_coupled_2step_r2': ('coupled_variable', 'mg', 'centered', 2, 2),
    'mg_oldface_coupled_2step_r2': ('coupled_variable', 'mg', 'old', 2, 2),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(line, key, cast=float):
    match = re.search(r'(?:^|\s)' + key + r'=([\w.eE+-]+)', line)
    return cast(match.group(1)) if match else None


def numeric_digest(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        item = data[key]
        if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
            digest.update(key.encode())
            digest.update(item.dtype.str.encode())
            digest.update(str(item.shape).encode())
            digest.update(np.ascontiguousarray(item[...]).tobytes())
    return digest.hexdigest()


def parse(name, problem, solver, face, ranks, steps):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    input_name = ('wor25_mg_shifted_fixed_q1e7_20261107.in' if problem == 'fixed_q1e7'
                  else 'testc_threelevel_shifted_20261107.in')
    input_file = ROOT / 'tst/nrfld_validation' / input_name
    nr = []
    fg = []
    handoff = []
    for line in log.splitlines():
        if line.startswith('[NR] status='):
            if value(line, 'status', str) == 'dt_zero_initialization':
                continue
            nr.append({'status': value(line, 'status', str),
                       'iterations': value(line, 'iterations', int),
                       'final_l2': value(line, 'final_l2'),
                       'final_max': value(line, 'final_max')})
        elif line.startswith('[LINEAR_FGMRES]'):
            fg.append({'iterations': value(line, 'iterations', int),
                       'true_relative_residual': value(line, 'relative_residual'),
                       'roundoff_accepted': value(line, 'roundoff_accepted', int)})
        elif line.startswith('[NRFLD handoff]') and 'solve=0 ' in line and 'region=all ' in line:
            handoff.append({'stage': value(line, 'stage', str),
                            'common_relative_residual': value(line, 'common_rel'),
                            'delta_mean': value(line, 'delta_mean'),
                            'full_step_min': value(line, 'full_step_min'),
                            'nonpositive_cells': value(line, 'nonpositive', int)})
    face_pairs = compare([row for file in sorted((directory / 'mpi').rglob('stdout'))
                          for row in read_faces(file)])
    energy = [float(x) for x in re.findall(r'absolute_relative_balance=([\deE+.-]+)', log)]
    mg_limit_defects = [float(x) for x in re.findall(
        r'Stopping at the configured V-cycle limit: completed=30, final relative defect=([\deE+.-]+)', log)]
    mg_divergence_defects = [float(x) for x in re.findall(
        r'Rejecting divergent V-cycle sequence: defect norm = ([\deE+.-]+)', log)]
    hdfs = sorted((directory / 'output').glob('*.athdf'))
    final = None
    if len(hdfs) > 1:
        with h5py.File(hdfs[-1]) as data:
            final = {'path': str(hdfs[-1]), 'sha256': sha(hdfs[-1]),
                     'numeric_arrays_sha256': numeric_digest(data)}
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing)
    rss = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)
    exit_code = int(re.search(r'Exit status:\s*(\d+)', timing).group(1))
    command = re.search(r'Command being timed: "([^"]+)"', timing)
    face_max = max((p['relative_mismatch'] for p in face_pairs), default=None)
    return {'name': name, 'source_git_hash': SOURCE_COMMIT,
            'binary_sha256': BINARY_SHA256, 'problem': problem,
            'solver': solver, 'face_operator': face, 'ranks': ranks,
            'requested_steps': steps, 'input': str(input_file),
            'input_sha256': sha(input_file),
            'environment': {'OMP_NUM_THREADS': 1,
                            'ATHENA_NRFLD_CENTERED_FACE_TRIAL': 0 if face == 'old' else None,
                            'ATHENA_NRFLD_INTERFACE_LEDGER': True,
                            'ATHENA_NRFLD_ENERGY_LEDGER': True},
            'command': command.group(1) if command else None,
            'exit_status': exit_code,
            'elapsed': wall.group(1) if wall else None,
            'peak_rss_kb': int(rss.group(1)) if rss else None,
            'newton_history': nr, 'fgmres_history': fg,
            'fgmres_total_iterations': sum(x['iterations'] for x in fg),
            'mg_vcycle_limit_count': len(mg_limit_defects),
            'mg_vcycle_limit_defects': mg_limit_defects,
            'mg_divergence_count': len(mg_divergence_defects),
            'mg_divergence_defects': mg_divergence_defects,
            'first_solve_handoff_all': handoff,
            'face_pairs': face_pairs,
            'max_face_relative_mismatch': face_max,
            'face_pass_1e-6': face_max <= 1e-6 if face_max is not None else None,
            'energy_absolute_relative_balance_history': energy,
            'max_energy_absolute_relative_balance': max(energy, default=None),
            'final_output': final,
            'solver_pass': exit_code == 0 and len(nr) == steps and
                           all(x['status'] == 'converged' for x in nr),
            'log_path': str(directory / 'run.log')}


def hdf_comparison(left, right, left_name, right_name):
    with h5py.File(left) as a, h5py.File(right) as b:
        keys = [k for k in a if isinstance(a[k], h5py.Dataset)
                and np.issubdtype(a[k].dtype, np.number)]
        exact = all(np.array_equal(a[k][...], b[k][...]) for k in keys)
        delta = np.abs(a['user_out_var'][...]-b['user_out_var'][...])
        scale = float(np.max(np.abs(b['user_out_var'][...])))
    return {'left': left_name, 'right': right_name,
            'all_numeric_arrays_exact': exact,
            'max_abs_user_out_var': float(np.max(delta)),
            'max_rel_user_out_var': float(np.max(delta))/scale if scale else 0.0}


def main():
    runs = [parse(name, *config) for name, config in RUNS.items()]
    by_name = {x['name']: x for x in runs}
    comparisons = []
    for left, right in [('mg_centered_coupled_2step_r1', 'mg_centered_coupled_2step_r2')]:
        comparisons.append(hdf_comparison(by_name[left]['final_output']['path'],
                                          by_name[right]['final_output']['path'], left, right))
    fgmres = sorted((Path('/tmp/wor24_wor25_default_testc_trials/default_fixedcap_testc_r1/output')).glob('*.athdf'))[-1]
    comparisons.append(hdf_comparison(by_name['mg_centered_coupled_2step_r1']['final_output']['path'],
                                      fgmres, 'mg_centered_coupled_2step_r1', 'fgmres_centered_coupled_2step_r1'))
    OUT.write_text(json.dumps({'runs': runs, 'output_comparisons': comparisons}, indent=2) + '\n')
    for x in runs:
        print(x['name'], x['solver_pass'], x['mg_vcycle_limit_count'],
              x['mg_divergence_count'], x['max_face_relative_mismatch'])


if __name__ == '__main__':
    main()
