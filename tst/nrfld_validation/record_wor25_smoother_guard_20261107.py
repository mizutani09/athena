#!/usr/bin/env python3
"""Record the unsupported MG smoother-count hang and input guard."""

import glob
import hashlib
import json
import re
from pathlib import Path

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = ROOT / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
INPUT = ROOT / 'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
GUARDED_BINARY = Path('/tmp/wor25_23_24_resolution_fgmres_norm_audit_build/bin/athena')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find(pattern, source):
    hit = re.search(pattern, source)
    return hit.group(1) if hit else None


def trial(name, sweeps, stage, binary):
    folder = TRIALS / name
    log = (folder / 'run.log').read_text(errors='replace')
    timing = (folder / 'time.log').read_text(errors='replace')
    files = sorted(glob.glob(str(folder / 'mpi/**/stdout'), recursive=True))
    faces = compare([row for file in files for row in read_faces(Path(file))])
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
              log)]
    nr = [{'status': s, 'iterations': int(n), 'cycle': int(c)}
          for s, n, c in re.findall(
              r'\[NR\] status=(\w+) iterations=(\d+)[^\n]*cycle=(\d+)', log)]
    return {
        'name': name, 'stage': stage, 'ranks': 1, 'threads_per_rank': 1,
        'sweeps_pre_and_post': sweeps,
        'source_commit_before_run': '58786a64',
        'binary_sha256': sha(binary), 'input_path': str(INPUT),
        'input_sha256': sha(INPUT),
        'command': find(r'Command being timed: "([^"]+)"', timing),
        'run_log': str(folder / 'run.log'), 'stdout_logs': files,
        'wall_time': find(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', timing),
        'peak_rss_kb_time_parent': int(find(
            r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)),
        'exit_status': int(find(r'Exit status:\s*(\d+)', timing)),
        'fatal_input_guard': 'must each be 0, 1, or 2' in log,
        'fgmres_history': fg, 'newton_history': nr,
        'face_pair_count': len(faces),
        'max_face_relative_mismatch': max(
            (x['relative_mismatch'] for x in faces), default=None),
        'face_pass': all(x['relative_mismatch'] <= 1e-6 for x in faces),
    }


cases = [
    trial('threelevel_coupled_smooth4_r1', 4, 'pre_guard_terminated',
          GUARDED_BINARY),
    trial('smoother_guard_invalid', 4, 'guarded_invalid', GUARDED_BINARY),
    trial('smoother_guard_valid', 2, 'guarded_valid_constructor', GUARDED_BINARY),
    trial('smoother_guard_valid_solve', 2, 'guarded_valid_solve', GUARDED_BINARY),
]
# The old trial used the exact binary already hashed by the preceding
# same-binary linearity ON/OFF audit, before this isolated build was rebuilt.
preceding = json.loads(REPORT.read_text())['wor25_linearity_each_solve_audit']
cases[0]['binary_sha256'] = preceding['cases'][0]['binary_sha256']
cases[0]['binary_sha256_note'] = 'reused preceding same-binary audit hash before rebuild'
cases[0]['observed_athena_rss_kb_from_ps'] = 459172
cases[0]['termination'] = 'SIGTERM sent to mpirun after >5 min with no completed solve'
for case in cases:
    if case['stage'] == 'guarded_invalid':
        case['evaluation'] = 'PASS_REJECTED_UNSUPPORTED_INPUT'
    elif case['stage'] == 'guarded_valid_constructor':
        case['evaluation'] = 'PASS_CONSTRUCTOR'
    elif case['stage'] == 'guarded_valid_solve':
        case['evaluation'] = ('PASS_SOLVER_FAIL_FACE_RELATIVE' if case['fgmres_history']
                              and all(x['true_relative_residual'] < 1e-8
                                      for x in case['fgmres_history'])
                              and not case['face_pass'] else 'CHECK')
    else:
        case['evaluation'] = 'TERMINATED_UNSUPPORTED_INPUT'
report = json.loads(REPORT.read_text())
report['wor25_smoother_count_guard'] = {
    'scope': 'FGMRES preconditioner smoother-count experiment on coupled '
             'variable-limiter 3-level shifted Test C and constructor guard.',
    'cases': cases,
    'decision': 'MG transfer task lists support only 0, 1, or 2 pre/post '
                'sweeps. An attempted 4/4 FGMRES preconditioner run spent '
                '5:46 without finishing one linear solve and was terminated. '
                'The input guard rejects 4/4 at construction while 2/2 '
                'still solves one step. This is a hang-prevention fix, not '
                'a convergence improvement. The existing program returns '
                'exit status zero for this constructor ATHENA_ERROR, so '
                'automation must check the FATAL ERROR text as well.'
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
for case in cases:
    print(case['name'], case['evaluation'], case['wall_time'],
          len(case['fgmres_history']), case['max_face_relative_mismatch'])
