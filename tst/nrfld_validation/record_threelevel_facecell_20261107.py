#!/usr/bin/env python3
"""Compare the same level-1/2 Test C face cells across one and two MPI ranks."""

import glob
import hashlib
import json
import math
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
BINARY = Path('/tmp/wor25_23_24_resolution_threelevel_facecell_build/bin/athena')
PATCH = ROOT / 'tst/nrfld_validation/threelevel_facecell_test_only.patch'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regex(pattern, source):
    result = re.search(pattern, source)
    if result is None:
        raise ValueError(pattern)
    return result.group(1)


def read_run(ranks, prefix='threelevel_facecell2'):
    name = f'{prefix}_r{ranks}'
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    stdout = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    cells = {}
    rows = []
    for file in stdout:
        path = Path(file)
        rows.extend(read_faces(path))
        for line in path.read_text().splitlines():
            if not line.startswith('[NR_FACE_CELL]'):
                continue
            entry = {key: float(value) for key, value in re.findall(
                r'(\w+)=([+-]?[\d.eE+-]+)', line)}
            key = (entry['x2min'], entry['x3min'], int(entry['k']), int(entry['j']))
            if key in cells:
                raise ValueError(f'duplicate face cell {key}')
            cells[key] = entry
    pairs = compare(rows)
    keyed_pairs = {}
    for pair in pairs:
        coarse = next(row for row in rows if row['gid'] == pair['coarse_gid']
                      and row['cycle'] == pair['cycle'] and row['stage'] == pair['stage'])
        key = (pair['cycle'], pair['stage'], pair['axis'], pair['side'],
               coarse['x1min'], coarse['x1max'], coarse['x2min'], coarse['x3min'])
        keyed_pairs[key] = pair
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    energy = [float(x) for x in re.findall(r'absolute_relative_balance=([\deE+.-]+)', log)]
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1,
        'fgmres_batched_reductions': prefix == 'threelevel_facecell2',
        'base_source_commit': '9ee735fa', 'binary_sha256': sha(BINARY),
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'command': regex(r'Command being timed: "([^"]+)"', timing),
        'environment': {'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'run_log': str(directory / 'run.log'), 'face_logs': stdout,
        'exit_status': int(regex(r'Exit status:\s*(\d+)', timing)),
        'wall_time': regex(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb': int(regex(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'newton_statuses': re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log),
        'fgmres_history': fg, 'max_true_relative_residual': max(f['true_relative_residual'] for f in fg),
        'max_implicit_energy_absolute_relative_balance': max(energy),
        'face_pair_count': len(pairs),
        'max_face_relative_mismatch': max(p['relative_mismatch'] for p in pairs),
        'evaluation': 'FAIL_FACE_RELATIVE_ONLY',
    }, cells, keyed_pairs


def hdf_equal(name1, name2):
    a = sorted((TRIALS / name1 / 'output').glob('*.athdf'))
    b = sorted((TRIALS / name2 / 'output').glob('*.athdf'))
    if len(a) != len(b):
        return False
    for one, two in zip(a, b):
        with h5py.File(one) as x, h5py.File(two) as y:
            for key in x:
                if isinstance(x[key], h5py.Dataset) and not np.array_equal(x[key][:], y[key][:]):
                    return False
    return True


run1, cells1, pairs1 = read_run(1)
run2, cells2, pairs2 = read_run(2)
assert cells1.keys() == cells2.keys() and len(cells1) == 256
assert pairs1.keys() == pairs2.keys()
field_differences = {}
for field in ('center', 'ghost', 'coefficient', 'dbar', 'debar', 'term'):
    delta = [cells1[key][field] - cells2[key][field] for key in cells1]
    field_differences[field] = {
        'nonzero_cell_count': sum(value != 0.0 for value in delta),
        'max_absolute': max(map(abs, delta)),
        'rms': math.sqrt(sum(value * value for value in delta) / len(delta)),
    }
gradient = [abs(cells1[key]['ghost'] - cells1[key]['center']) for key in cells1]
gradient_diff = [
    (cells1[key]['ghost'] - cells1[key]['center'])
    - (cells2[key]['ghost'] - cells2[key]['center']) for key in cells1]
dx = 0.0625 / 8.0
dt = 2e-4
weight = dt / (dx * dx) * (dx ** 3)
face_comparisons = []
for key in sorted(pairs1):
    if key[4] != 0.5 or key[3] != 'min' or key[2] != 1:
        continue
    one = pairs1[key]
    two = pairs2[key]
    transverse = key[6:8]
    matching = [cell for cell in cells1 if cell[:2] == transverse]
    predicted = weight * sum(cells1[cell]['term'] - cells2[cell]['term']
                             for cell in matching)
    actual = one['coarse_flux'] - two['coarse_flux']
    face_comparisons.append({
        'x2min': transverse[0], 'x3min': transverse[1],
        'coarse_flux_rank1': one['coarse_flux'], 'coarse_flux_rank2': two['coarse_flux'],
        'fine_flux_rank1': one['fine_flux'], 'fine_flux_rank2': two['fine_flux'],
        'coarse_flux_difference_from_ledger': actual,
        'coarse_flux_difference_from_cell_terms': predicted,
        'prediction_absolute_error': abs(predicted - actual),
        'relative_mismatch_rank1': one['relative_mismatch'],
        'relative_mismatch_rank2': two['relative_mismatch'],
    })
assert len(face_comparisons) == 4
unbatched1, ucells1, _ = read_run(1, 'threelevel_facecell_unbatched')
unbatched2, ucells2, _ = read_run(2, 'threelevel_facecell_unbatched')
assert ucells1.keys() == ucells2.keys() and len(ucells1) == 256
unbatched_rank_max_center_difference = max(
    abs(ucells1[key]['center'] - ucells2[key]['center']) for key in ucells1)
unbatched_rank_max_gradient_difference = max(
    abs((ucells1[key]['ghost'] - ucells1[key]['center'])
        - (ucells2[key]['ghost'] - ucells2[key]['center'])) for key in ucells1)
unbatched_rank_coefficient_exact = all(
    ucells1[key]['coefficient'] == ucells2[key]['coefficient']
    and ucells1[key]['dbar'] == ucells2[key]['dbar'] for key in ucells1)
report = json.loads(REPORT.read_text())
report['threelevel_rank_facecell_probe'] = {
    'scope': 'Fixed-limiter q=1e7 three-level shifted Test C, level-1/2 x=0.5 face, 1/2 MPI ranks; test-only diagnostic',
    'probe_patch_sha256': sha(PATCH), 'runs': [run1, run2],
    'compared_face_cell_count': len(cells1),
    'field_rank_differences': field_differences,
    'gradient_rank1_max_absolute': max(gradient),
    'gradient_rank_difference_max_absolute': max(map(abs, gradient_diff)),
    'gradient_rank_difference_nonzero_cell_count': sum(value != 0.0 for value in gradient_diff),
    'face_comparisons': face_comparisons,
    'rank_hdf_exact': hdf_equal('threelevel_facecell2_r1', 'threelevel_facecell2_r2'),
    'probe_hdf_exact_vs_base': {
        str(rank): hdf_equal(f'threelevel_facecell2_r{rank}',
                             f'testc_threelevel_shifted_r{rank}')
        for rank in (1, 2)},
    'unbatched_reductions_trial': {
        'runs': [unbatched1, unbatched2],
        'rank_max_center_difference': unbatched_rank_max_center_difference,
        'rank_max_gradient_difference': unbatched_rank_max_gradient_difference,
        'rank_coefficient_and_dbar_exact': unbatched_rank_coefficient_exact,
        'rank_hdf_exact': hdf_equal('threelevel_facecell_unbatched_r1',
                                    'threelevel_facecell_unbatched_r2'),
        'decision': 'REJECT_AS_FIX: Newton/FGMRES converge with 11+11 iterations, but 1/2-rank per-face maxima remain 9.72e-6/1.03e-5 above 1e-6. The internal double center rank difference grows to 8.45e-12 and gradient difference to 4.44e-16. Do not change batched-reduction default.'},
    'decision': 'Shared diffusion coefficient and Dbar are bitwise identical across all 256 coarse face cells. Double-precision center and ghost differ by about 1.2e-12 despite exactly equal float32 HDF arrays. The face gradient is only 3.0e-12 and differs by at most one binary64 ulp (2.22e-16) across ranks, producing the observed coarse flux differences. Fine integrated flux agrees across ranks in this fixed-limiter case. This localizes rank sensitivity to the actual double state and subtraction on a near-zero gradient; it does not establish why the tiny state difference arose, or prove exact face conservation. Per-face relative criterion remains FAIL. No production change.',
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({
    'cells': len(cells1), 'field_differences': field_differences,
    'gradient_rank_difference_max_absolute': max(map(abs, gradient_diff)),
    'face_comparisons': face_comparisons,
    'rank_hdf_exact': report['threelevel_rank_facecell_probe']['rank_hdf_exact'],
    'probe_hdf_exact_vs_base': report['threelevel_rank_facecell_probe']['probe_hdf_exact_vs_base'],
    'unbatched_reductions_trial': {
        'rank_max_center_difference': unbatched_rank_max_center_difference,
        'rank_max_gradient_difference': unbatched_rank_max_gradient_difference,
        'max_face_relative_mismatch': [unbatched1['max_face_relative_mismatch'],
                                       unbatched2['max_face_relative_mismatch']]},
}, indent=2))
