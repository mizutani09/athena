#!/usr/bin/env python3
"""Reconstruct one three-level coarse/fine face from stored binary64 values."""

import glob
import hashlib
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


getcontext().prec = 90
D = Decimal.from_float
ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
BINARY = Path('/tmp/wor25_23_24_resolution_face_exact_probe_build/bin/athena')
PATCH = ROOT / 'tst/nrfld_validation/face_exact_probe_test_only.patch'
COARSE_TRANSVERSE_DX = 0.0078125  # This input's level-4 x2/x3 cell width.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hdf_exact(name1, name2):
    files1 = sorted((TRIALS / name1 / 'output').glob('*.fields.*.athdf'))
    files2 = sorted((TRIALS / name2 / 'output').glob('*.fields.*.athdf'))
    if len(files1) != len(files2):
        return False
    for one, two in zip(files1, files2):
        with h5py.File(one) as a, h5py.File(two) as b:
            if not all(np.array_equal(a[key][:], b[key][:]) for key in a
                       if isinstance(a[key], h5py.Dataset)):
                return False
    return True


def record(name, ranks):
    directory = TRIALS / name
    log = (directory / 'run.log').read_text(errors='replace')
    timing = (directory / 'time.log').read_text(errors='replace')
    files = sorted(glob.glob(str(directory / 'mpi/**/stdout'), recursive=True))
    groups = defaultdict(lambda: {'coarse': [], 'fine': []})
    for file in files:
        for line in Path(file).read_text().splitlines():
            if not line.startswith('[NR_FACE_EXACT_PROBE]'):
                continue
            values = {key: float(value) for key, value in re.findall(
                r'([A-Za-z0-9_]+)=([\deE+.-]+)', line)}
            key = (math.floor(values['x2'] / COARSE_TRANSVERSE_DX),
                   math.floor(values['x3'] / COARSE_TRANSVERSE_DX))
            groups[key]['coarse' if int(values['level']) == 4 else 'fine'].append(values)
    assert len(groups) == 256
    assert all(len(x['coarse']) == 1 and len(x['fine']) == 4 for x in groups.values())
    components = {key: Decimal(0) for key in
                  ('stored_face', 'coarse_ghost_and_coefficient_rounding',
                   'fine_ghost_rounding', 'moment_restriction_rounding',
                   'shifted_moment_float_candidate')}
    by_gid = defaultdict(lambda: {key: Decimal(0) for key in components})
    exact_coarse_ref = 0
    exact_fine_coeff = 0
    for group in groups.values():
        coarse, fine = group['coarse'][0], group['fine']
        exact_coarse_ref += sum(x['coarse_ref'] == coarse['center'] for x in fine)
        exact_fine_coeff += sum(x['coefficient'] == x['moment_d'] for x in fine)
        scale = D(coarse['dt']) * D(coarse['idx2']) * D(coarse['volume'])
        coarse_stored = (scale * D(coarse['coefficient'])
                         * (D(coarse['ghost']) - D(coarse['center'])))
        fine_stored = sum((D(x['dt']) * D(x['idx2']) * D(x['volume'])
                           * D(x['coefficient']) * (D(x['ghost']) - D(x['center']))
                          for x in fine), Decimal(0))
        coarse_moment = (scale * Decimal(4) / 3
                         * (D(coarse['moment_de'])
                            - D(coarse['moment_d']) * D(coarse['center'])))
        fine_ideal = Decimal(0)
        for x in fine:
            ideal_ghost = ((Decimal(2)
                            * (D(coarse['center']) + D(x['offset']))
                            + D(x['center'])) / 3)
            fine_ideal += (D(x['dt']) * D(x['idx2']) * D(x['volume'])
                           * D(x['coefficient'])
                           * (ideal_ghost - D(x['center'])))
        # Offline candidate: subtract the *identical* coarse reference before
        # restricting D*Er. It is not a replacement for the full residual/Jv.
        shifted_moment = sum((x['moment_d'] *
                              ((x['center'] - x['offset']) - x['coarse_ref'])
                              for x in fine)) / 4.0
        shifted_coarse = scale * D(float(4.0/3.0 * shifted_moment))
        values = {
            'stored_face': coarse_stored + fine_stored,
            'coarse_ghost_and_coefficient_rounding': coarse_stored - coarse_moment,
            'fine_ghost_rounding': fine_stored - fine_ideal,
            'moment_restriction_rounding': coarse_moment + fine_ideal,
            'shifted_moment_float_candidate': shifted_coarse + fine_stored,
        }
        for key, value in values.items():
            components[key] += value
            by_gid[int(coarse['gid'])][key] += value
    assert exact_coarse_ref == 1024 and exact_fine_coeff == 1024
    assert abs(components['stored_face'] - sum(
        components[key] for key in ('coarse_ghost_and_coefficient_rounding',
                                    'fine_ghost_rounding',
                                    'moment_restriction_rounding'))) < Decimal('1e-70')
    rows = [row for file in files for row in read_faces(Path(file))]
    face_pairs = [pair for pair in compare(rows) if pair['axis'] == 1
                  and pair['coarse_gid'] in by_gid]
    pairs_by_gid = {pair['coarse_gid']: pair for pair in face_pairs}
    component_only_max_relative = {}
    for component in ('coarse_ghost_and_coefficient_rounding',
                      'fine_ghost_rounding', 'moment_restriction_rounding'):
        component_only_max_relative[component] = max(
            float(abs(values[component])) /
            (abs(pairs_by_gid[gid]['coarse_flux']) +
             abs(pairs_by_gid[gid]['fine_flux']))
            for gid, values in by_gid.items())
    shifted_relative_by_gid = {}
    for gid, values in by_gid.items():
        pair = pairs_by_gid[gid]
        new_mismatch = float(values['shifted_moment_float_candidate'])
        fine_flux = pair['fine_flux']
        new_coarse_flux = new_mismatch - fine_flux
        scale = abs(new_coarse_flux) + abs(fine_flux)
        shifted_relative_by_gid[str(gid)] = abs(new_mismatch) / scale if scale else 0.0
    fg = re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) '
                    r'relative_residual=([\deE+.-]+)', log)
    baseline = (TRIALS / f'threelevel_coupled_variable_r{ranks}' / 'run.log').read_text()
    baseline_fg = re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) '
                             r'relative_residual=([\deE+.-]+)', baseline)
    return {
        'name': name, 'ranks': ranks, 'threads_per_rank': 1,
        'source_commit_before_run': '4d3f2511',
        'test_only_patch_sha256': sha(PATCH), 'binary_sha256': sha(BINARY),
        'input_path': str(INPUT), 'input_sha256': sha(INPUT),
        'command': re.search(r'Command being timed: "([^"]+)"', timing).group(1),
        'environment': {'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_INTERFACE_LEDGER': 1,
                        'ATHENA_NRFLD_ENERGY_LEDGER': 1},
        'run_log': str(directory / 'run.log'), 'cell_logs': files,
        'exit_status': int(re.search(r'Exit status:\s*(\d+)', timing).group(1)),
        'wall_time': re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)',
                               timing).group(1),
        'peak_rss_kb': int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',
                                    timing).group(1)),
        'fgmres_history': [{'iterations': int(n), 'true_relative_residual': float(r)}
                           for n, r in fg],
        'fgmres_history_exact_vs_base': fg == baseline_fg,
        'hdf_numeric_arrays_exact_vs_base': hdf_exact(
            name, f'threelevel_coupled_variable_r{ranks}'),
        'coarse_cell_count': len(groups), 'fine_cell_count': exact_coarse_ref,
        'fine_coarse_reference_exact_count': exact_coarse_ref,
        'fine_coefficient_equals_moment_d_count': exact_fine_coeff,
        'aggregate_integrated_face_components': {key: float(value)
                                                 for key, value in components.items()},
        'integrated_face_components_by_coarse_gid': {
            str(gid): {key: float(value) for key, value in values.items()}
            for gid, values in sorted(by_gid.items())},
        'logged_face_pairs': face_pairs,
        'component_only_max_relative_using_logged_face_scale':
            component_only_max_relative,
        'shifted_offline_relative_mismatch_by_gid': shifted_relative_by_gid,
        'shifted_offline_max_relative_mismatch': max(shifted_relative_by_gid.values()),
        'evaluation': 'PASS_DIAGNOSTIC_ONLY',
    }


cases = [record(f'threelevel_coupled_variable_exactprobe_ref_r{rank}', rank)
         for rank in (1, 2)]
report = json.loads(REPORT.read_text())
report['wor24_face_exact_reconstruction_probe'] = {
    'scope': 'Test-only per-cell probe at x1=0.5 of 256 coarse cells and 1024 fine cells, coupled 3-level variable-limiter Test C, 1/2 MPI ranks; binary64 values reconstructed using Decimal.from_float.',
    'cases': cases,
    'decision': 'The actual fine-side coarse reference is bitwise identical to the coarse active center for all 1024 cells. Exact arithmetic on stored binary64 inputs decomposes the x1=0.5 imbalance into coarse ghost/coefficient, fine ghost and restricted D*Er moment rounding. Subtracting the common coarse reference before moment averaging substantially reduces the offline aggregate mismatch but does not pass the original 1e-6 relative face criterion at every coarse block. Even isolating fine ghost rounding leaves component/face-scale ratios above 1e-3, so correcting moment restriction alone cannot remove every observed contribution. This is a diagnostic candidate only: making it a solver fix requires the Newton residual, correction boundary and FGMRES matvec to use the same centered face-flux representation; no production operator change is made.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
for case in cases:
    print(case['name'], case['aggregate_integrated_face_components'],
          case['fgmres_history_exact_vs_base'],
          case['hdf_numeric_arrays_exact_vs_base'])
