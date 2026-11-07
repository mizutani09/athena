#!/usr/bin/env python3
"""Record longer persistent-AMR and natural-collapse restart controls."""

import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
BINARY = Path('/tmp/wor25_23_24_resolution_collapse_postflddt_build/bin/athena')
INPUT = ROOT / 'tst/nrfld_validation/collapse_postflddt_static_partial_20261107.in'
NATURAL_CHECKPOINT = TRIALS / 'collapse_postflddt_opacity_natural_r1/output/protostellar_core_fld_mayer5.00015.rst'
STATIC_CHECKPOINT = TRIALS / 'collapse_static_long20_valid_r1/output/protostellar_core_fld_mayer5.00010.rst'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first(pattern, source):
    match = re.search(pattern, source)
    if match is None:
        raise ValueError(pattern)
    return match.group(1)


def mass(hdf):
    volume = (np.diff(hdf['x3f'][:], axis=1)[:, :, None, None]
              * np.diff(hdf['x2f'][:], axis=1)[:, None, :, None]
              * np.diff(hdf['x1f'][:], axis=1)[:, None, None, :])
    return float(np.sum(hdf['prim'][0] * volume))


def output_data(name):
    folder = TRIALS / name / 'output'
    files = sorted(folder.glob('*.prim.*.athdf'))
    with h5py.File(files[0]) as start, h5py.File(files[-1]) as final:
        initial_mass, final_mass = mass(start), mass(final)
        counts = {str(int(level)): int(count) for level, count in
                  zip(*np.unique(final['Levels'][:], return_counts=True))}
        minima = {'rho': float(np.min(final['prim'][0])),
                  'pressure': float(np.min(final['prim'][1])),
                  'Er': float(np.min(final['prim'][8]))}
        final_time = float(final.attrs['Time'])
    return {'initial_mass_volume_integral': initial_mass,
            'final_mass_volume_integral': final_mass,
            'relative_mass_change': (final_mass - initial_mass) / initial_mass,
            'final_level_counts': counts, 'final_minima': minima,
            'final_time': final_time, 'final_prim_file': str(files[-1])}


def run(name, ranks, kind, checkpoint=None):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    files = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    rows = [row for file in files for row in read_faces(Path(file))]
    pairs = compare(rows)
    ledger_by_cycle = defaultdict(list)
    for row in rows:
        ledger_by_cycle[int(row['cycle'])].append(int(row['level']))
    faces_by_cycle = defaultdict(list)
    for pair in pairs:
        faces_by_cycle[pair['cycle']].append(pair)
    cycles = []
    for cycle, levels in sorted(ledger_by_cycle.items()):
        cycle_pairs = faces_by_cycle[cycle]
        cycles.append({'cycle': cycle, 'leaf_count': len(levels),
                       'absolute_level_counts': {str(key): value for key, value in
                                                 sorted(Counter(levels).items())},
                       'coarse_fine_face_pair_count': len(cycle_pairs),
                       'max_face_relative_mismatch': max(
                           (p['relative_mismatch'] for p in cycle_pairs), default=None),
                       'face_pass': all(p['relative_mismatch'] <= 1e-6 for p in cycle_pairs)})
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)} for n, r in
          re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    newton = [{'status': status, 'iterations': int(n)} for status, n in
              re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log)]
    energy = [float(x) for x in re.findall(r'absolute_relative_balance=([\deE+.-]+)', log)]
    output = output_data(name)
    solver_pass = all(item['status'] in ('converged', 'dt_zero_initialization')
                      for item in newton) and all(item['true_relative_residual'] < 1e-8
                                                 for item in fg)
    face_pass = all(p['relative_mismatch'] <= 1e-6 for p in pairs)
    positive = all(value > 0 for value in output['final_minima'].values())
    return {
        'name': name, 'kind': kind, 'ranks': ranks, 'threads_per_rank': 2,
        'source_commit_before_run': '394d66c5', 'binary_sha256': sha(BINARY),
        'input_path': str(INPUT) if checkpoint is None else None,
        'input_sha256': sha(INPUT) if checkpoint is None else None,
        'checkpoint_path': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': sha(checkpoint) if checkpoint else None,
        'overrides': {'time/nlim': 25 if kind == 'natural' else 20},
        'environment': {'OMP_NUM_THREADS': 2, 'ATHENA_NRFLD_EXPERIMENTAL_COLLAPSE': 1,
                        'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'command': first(r'Command being timed: "([^"]+)"', timing),
        'run_log': str(directory / 'run.log'), 'face_logs': files,
        'exit_status': int(first(r'Exit status:\s*(\d+)', timing)),
        'wall_time': first(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(first(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_history': newton, 'fgmres_history': fg,
        'fgmres_total_iterations': sum(item['iterations'] for item in fg),
        'max_true_relative_residual': max(item['true_relative_residual'] for item in fg),
        'coarse_fine_face_pair_count': len(pairs),
        'max_face_relative_mismatch': max((p['relative_mismatch'] for p in pairs), default=None),
        'worst_face_pair': max(pairs, key=lambda p: p['relative_mismatch']) if pairs else None,
        'face_history': cycles,
        'max_implicit_energy_absolute_relative_balance': max(energy),
        'output': output, 'solver_pass': solver_pass, 'face_pass': face_pass,
        'positive': positive,
        'evaluation': 'PASS' if solver_pass and face_pass and positive else
                      'FAIL_FACE_RELATIVE_ONLY' if solver_pass and positive else 'FAIL_SOLVER_OR_POSITIVITY',
    }


def final_array(name, kind):
    file = sorted((TRIALS / name / 'output').glob(f'*.{kind}.*.athdf'))[-1]
    with h5py.File(file) as hdf:
        return hdf['prim' if kind == 'prim' else 'user_out_var'][:]


def relative_l1(name1, name2, kind):
    one, two = final_array(name1, kind), final_array(name2, kind)
    return float(np.sum(abs(one - two)) / np.sum(abs(one)))


natural = [run(f'collapse_natural_long25_r{r}', r, 'natural', NATURAL_CHECKPOINT)
           for r in (1, 2)]
static = [run(f'collapse_static_long20_valid_r{r}', r, 'persistent_static')
          for r in (1, 2)]
restart = [run(f'collapse_static_long20_restart10_r{r}', r, 'persistent_static_restart',
               STATIC_CHECKPOINT) for r in (1, 2)]
comparisons = {
    'static_continuous_rank1_vs_rank2': {kind: relative_l1(
        'collapse_static_long20_valid_r1', 'collapse_static_long20_valid_r2', kind)
        for kind in ('prim', 'fld')},
    'static_restart_rank1_vs_rank2': {kind: relative_l1(
        'collapse_static_long20_restart10_r1', 'collapse_static_long20_restart10_r2', kind)
        for kind in ('prim', 'fld')},
    'natural_rank1_vs_rank2': {kind: relative_l1(
        'collapse_natural_long25_r1', 'collapse_natural_long25_r2', kind)
        for kind in ('prim', 'fld')},
    'static_continuous_vs_restart': {
        str(rank): {kind: relative_l1(
            f'collapse_static_long20_valid_r{rank}',
            f'collapse_static_long20_restart10_r{rank}', kind)
            for kind in ('prim', 'fld')} for rank in (1, 2)},
}
report = json.loads(REPORT.read_text())
report['persistent_amr_long_duration_screen'] = {
    'scope': 'Natural 43-to-64-leaf extension to cycle 25 and persistent static 15-leaf collapse to cycle 20, fixed source and 1/2 MPI ranks, with cycle-10 restart',
    'natural_runs': natural, 'static_continuous_runs': static,
    'static_restart_runs': restart, 'comparisons_final_relative_l1': comparisons,
    'invalid_preliminary_invocations': [
        {'name': f'collapse_static_long20_r{rank}',
         'run_log': str(TRIALS / f'collapse_static_long20_r{rank}/run.log'),
         'reason': 'Command-line override nrfld/global_jacobian_check absent in input; ParameterInput fatal diagnostic despite process exit 0; not counted as a run.'}
        for rank in (1, 2)],
    'decision': 'Natural regrid: after cycle 15, the 43-leaf mixed-level mesh becomes 64 leaves all at the same level; cycles 16-24 therefore contain no coarse/fine pair and cannot establish persistence of the interface failure. All Newton/FGMRES solves converge. Persistent static 15-leaf AMR: all 60 face pairs across cycles 0-19 pass 1e-6 on both ranks; cycle-10 restart has 30 passing pairs across cycles 10-19, and final continuous/restart double-HDF relative L1 is O(1e-13). Positive fields and true residual <1e-8 throughout. This is a local, short 20-cycle validation, not a closed mass/energy budget or full-size collapse validation.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({
    'natural': [(r['name'], r['coarse_fine_face_pair_count'],
                 sum(c['coarse_fine_face_pair_count'] == 0 for c in r['face_history']))
                for r in natural],
    'static': [(r['name'], r['coarse_fine_face_pair_count'],
                r['max_face_relative_mismatch'], r['evaluation']) for r in static + restart],
    'comparisons_final_relative_l1': comparisons,
}, indent=2))
