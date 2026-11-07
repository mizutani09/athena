#!/usr/bin/env python3
"""Separate coarse/fine Jv accuracy from roundoff-sensitive face balance."""

import glob
import json
import re
from pathlib import Path

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
report = json.loads(REPORT.read_text())
screen = report['threelevel_coupled_screen']
cases = [x for x in screen['base_cases'] if x['name'] in (
    'threelevel_coupled_variable_r1', 'threelevel_coupled_variable_r2')]
cases.extend(screen['amplitude_trials'])
records = []

for case in cases:
    directory = TRIALS / case['name']
    log = (directory / 'run.log').read_text(errors='replace')
    rows = [row for file in glob.glob(str(directory / 'mpi/**/stdout'), recursive=True)
            for row in read_faces(Path(file))]
    pairs = compare(rows)
    face_keys = [f'{letter}{side}_delta' for letter in 'xyz'
                 for side in ('minus', 'plus')]
    all_block_boundary_sum = sum(row[key] for row in rows for key in face_keys)
    coarse_fine_sum = sum(pair['mismatch'] for pair in pairs)
    coarse_fine_absolute_sum = sum(abs(pair['mismatch']) for pair in pairs)
    coarse_fine_scale_sum = sum(pair['roundoff_scale'] for pair in pairs)
    energy_rank_rows = [float(value) for value in re.findall(
        r'\[NR_ENERGY_LEDGER\][^\n]*interior_delta=([\deE+.-]+)', log)]
    assert len(energy_rank_rows) == case['ranks']
    jv_sweep = []
    for direction, epsilon, value in re.findall(
            r'\[NRFLD coupled Jv\] direction=(\w+) epsilon=([\deE+.-]+) '
            r'region=2 cells=3200[^\n]*?rel2=([\deE+.-]+)', log):
        jv_sweep.append({'direction': direction, 'epsilon': float(epsilon),
                         'schur_relative_l2': float(value)})
    records.append({
        'name': case['name'], 'ranks': case['ranks'],
        'initial_pulse_amplitude': case['initial_pulse_amplitude'],
        'input_sha256': case['input_sha256'],
        'binary_sha256': case['binary_sha256'], 'source_commit':
        case['source_commit_before_run'], 'command': case['command'],
        'run_log': case['run_log'], 'face_logs': case['face_logs'],
        'exit_status': case['exit_status'], 'wall_time': case['wall_time'],
        'peak_rss_kb': case['peak_rss_kb'],
        'newton_history': case['newton_history'],
        'fgmres_history': case['fgmres_history'],
        'maximum_true_relative_residual': case['maximum_true_relative_residual'],
        'coarse_fine_face_pair_count': len(pairs),
        'coarse_fine_mismatch_sum': coarse_fine_sum,
        'coarse_fine_absolute_mismatch_sum': coarse_fine_absolute_sum,
        'coarse_fine_roundoff_scale_sum': coarse_fine_scale_sum,
        'absolute_mismatch_over_roundoff_scale_sum': (
            coarse_fine_absolute_sum / coarse_fine_scale_sum),
        'all_block_boundary_face_sum': all_block_boundary_sum,
        'same_level_and_periodic_remainder': all_block_boundary_sum - coarse_fine_sum,
        'global_interior_diffusion_sum_from_rank_ledgers': sum(energy_rank_rows),
        'face_sum_minus_global_diffusion': (all_block_boundary_sum
                                            - sum(energy_rank_rows)),
        'coarse_fine_schur_jv_epsilon_sweep': jv_sweep,
        'coarse_fine_ghost_tangent_h1e4': case['coupled_ghost_tangent_h1e4'],
        'max_face_relative_mismatch': case['max_face_relative_mismatch'],
        'evaluation': case['evaluation'],
    })

report['wor23_24_coupled_amplitude_cutline'] = {
    'scope': 'Same coupled 3-level variable-limiter conservative operator; only initial pulse amplitude 0.01/0.1/1.0 differs, each at 1/2 MPI ranks.',
    'records': records,
    'decision': 'At amplitude 1.0 both ranks pass the 1e-6 face relative test, while at 0.1/0.01 they fail despite converged Newton/FGMRES and small coarse/fine Schur Jv. The ghost tangent mismatch at h=1e-4 is O(1e-12) for all amplitudes. The sum of coarse/fine face mismatches accounts for the global interior diffusion imbalance in periodic Test C; same-level and periodic remainder is at accumulation roundoff. The observed face imbalance is therefore real in the reported discrete face terms, but it is not explained by a dominant Jv/tangent mismatch. The local epsilon-weighted roundoff scale is a heuristic, not a rigorous error bound or acceptance criterion. Keep WOR24 face consistency unresolved; WOR23 Jv task may be closed on its diagnostic scope.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
for record in records:
    print(record['name'], record['max_face_relative_mismatch'],
          record['coarse_fine_mismatch_sum'],
          record['same_level_and_periodic_remainder'],
          record['absolute_mismatch_over_roundoff_scale_sum'],
          record['evaluation'])
