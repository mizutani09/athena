#!/usr/bin/env python3
"""Record current-source Test C first-harmonic MG handoff measurements."""
import glob
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

W = Path(__file__).resolve().parents[2]
B = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = W / 'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
BINARY = Path('/tmp/wor25_23_24_resolution_mode_k1_audit_build/bin/athena')
INPUTS = {
    'uniform': W / 'tst/nrfld_validation/wor25_k1_uniform_20261107.in',
    'aligned': W / 'tst/nrfld_validation/wor25_k1_aligned_20261107.in',
    'shifted': B / 'mode_audit_mg_x1/input.used',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def field_integrals(run):
    files = sorted((B/run/'output').glob('*.fields.*.athdf'))
    rows = []
    for f in files:
        with h5py.File(f) as h:
            vol = (np.diff(h['x3f'][:], axis=1)[:, :, None, None]
                   * np.diff(h['x2f'][:], axis=1)[:, None, :, None]
                   * np.diff(h['x1f'][:], axis=1)[:, None, None, :])
            er = h['user_out_var'][0].astype(np.float64)
            rows.append({'file': f.name, 'leaf_blocks': len(h['Levels']),
                         'Er_volume_integral_from_float32_hdf': float(np.sum(er*vol)),
                         'Er_min': float(np.min(er))})
    return rows

def record(shape, method, ranks, diag_off=False):
    name = f'wor25_k1_current_{shape}_{method}' + ('_diagoff' if diag_off else '') + f'_r{ranks}'
    log = (B/name/'run.log').read_text(errors='replace')
    handoff = []
    for line in log.splitlines():
        if line.startswith('[NRFLD handoff]'):
            f = dict(re.findall(r'(\w+)=([^ ]+)', line))
            handoff.append({k: (float(v) if k in ('common_rel', 'delta_mean', 'delta_fluct_rms',
                                                  'delta_k1', 'rhs_k1', 'residual_k1',
                                                  'full_step_min', 'residual_mean') else v)
                            for k, v in f.items() if k in ('solve', 'stage', 'region', 'common_rel',
                                'delta_mean', 'delta_fluct_rms', 'delta_k1', 'rhs_k1',
                                'residual_k1', 'full_step_min', 'residual_mean')})
    statuses = [{'status': a, 'iterations': int(b)} for a,b in
                re.findall(r'\[NR\] status=(\w+) iterations=(\d+)', log)]
    fgmres = [{'iterations': int(a), 'true_relative_residual': float(b)} for a,b in
              re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)', log)]
    exit_status = int(re.search(r'Exit status:\s*(\d+)', log).group(1))
    return {'name': name, 'shape': shape, 'method': method, 'ranks': ranks, 'threads': 1,
            'source_commit_before_diagnostic': '94c32735',
            'diagnostic_source_sha256': sha(W/'src/linear_solver/linearMG/linearMG_handoff_diagnostic.cpp'),
            'binary_sha256': sha(BINARY), 'input_path': str(INPUTS[shape]),
            'input_sha256': sha(INPUTS[shape]),
            'command': re.search(r'Command being timed: "([^"]+)"', log).group(1),
            'overrides': {'nrfld/linear_solver': method,
                          'fld/const_opacity': 8.186332719786666e-8,
                          'nrfld/fgmres_linearity_check': False,
                          'nrfld/handoff_diagnostics': not diag_off},
            'run_log': str(B/name/'run.log'), 'exit_status': exit_status,
            'wall_time': re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', log).group(1),
            'peak_rss_kb': int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', log).group(1)),
            'newton_statuses': statuses, 'fgmres_history': fgmres,
            'handoff_history': handoff, 'field_outputs': field_integrals(name),
            'evaluation': 'FAIL_EXPECTED_MG_BACKTRACKING' if exit_status and method=='mg'
                          else 'PASS' if exit_status==0 and statuses[-1]['status']=='converged'
                          and all(f['true_relative_residual']<1e-8 for f in fgmres)
                          else 'FAIL'}

def equal_hdf(a,b):
    x = sorted((B/a/'output').glob('*.fields.*.athdf'))
    y = sorted((B/b/'output').glob('*.fields.*.athdf'))
    if len(x)!=len(y): return False
    for left,right in zip(x,y):
        with h5py.File(left) as p,h5py.File(right) as q:
            if not all(np.array_equal(p[k][:],q[k][:]) for k in p if isinstance(p[k],h5py.Dataset)):
                return False
    return True

runs = [record(shape, method, n) for shape in ('uniform','aligned','shifted')
        for method in ('mg','fgmres') for n in ([1,2] if shape=='shifted' else [1])]
runs.append(record('shifted','fgmres',1,diag_off=True))
first = {(x['shape'],x['method'],x['ranks']): next((h for h in x['handoff_history']
    if h['solve']=='0' and h['region']=='all' and h['stage']
    == ('fine_sweeps' if x['method']=='mg' else 'boundary')),None)
    for x in runs if x['handoff_history']}
summary = {'uniform_mg_to_fgmres_k1_ratio': first['uniform','mg',1]['delta_k1']/first['uniform','fgmres',1]['delta_k1'],
           'aligned_mg_to_fgmres_k1_ratio': first['aligned','mg',1]['delta_k1']/first['aligned','fgmres',1]['delta_k1'],
           'shifted_mg_to_fgmres_k1_ratio': first['shifted','mg',1]['delta_k1']/first['shifted','fgmres',1]['delta_k1'],
           'shifted_mg_to_fgmres_boundary_residual_k1_ratio': first['shifted','mg',1]['residual_k1']/first['shifted','fgmres',1]['residual_k1'],
           'shifted_fgmres_rank_hdf_exact': equal_hdf('wor25_k1_current_shifted_fgmres_r1','wor25_k1_current_shifted_fgmres_r2'),
           'shifted_fgmres_diag_on_off_hdf_exact': equal_hdf('wor25_k1_current_shifted_fgmres_r1','wor25_k1_current_shifted_fgmres_diagoff_r1')}
report = json.loads(REPORT.read_text())
report['wor25_first_harmonic_handoff_audit'] = {'scope': 'Actual production MG/Newton correction on uniform, aligned, and shifted Test C; x1 first Fourier harmonic is a projection diagnostic, not a composite-operator eigenmode.',
                                              'binary_sha256': sha(BINARY), 'runs': runs,
                                              'summary': summary,
                                              'decision': 'Keep opt-in diagnostic. Do not change MG algorithm: shifted AMR MG still fails; FGMRES passes.'}
REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
