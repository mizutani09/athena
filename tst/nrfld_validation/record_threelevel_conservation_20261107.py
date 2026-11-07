#!/usr/bin/env python3
"""Record local three-level Test C face-ledger screening without hiding failures."""
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

W = Path(__file__).resolve().parents[2]
B = Path('/tmp/wor25_23_24_resolution_trials')
REPORT = W/'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
LEDGER = W/'tst/nrfld_validation/collapse_interface_ledger.py'
BINARY = Path('/tmp/wor25_23_24_resolution_mode_k1_audit_build/bin/athena')
INPUT3 = W/'tst/nrfld_validation/testc_threelevel_shifted_20261107.in'
INPUT2 = W/'tst/nrfld_validation/testc_twolevel_shifted_20261107.in'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fields(name):
    result=[]
    for f in sorted((B/name/'output').glob('*.fields.*.athdf')):
        with h5py.File(f) as h:
            vol=(np.diff(h['x3f'][:],axis=1)[:,:,None,None]
                 *np.diff(h['x2f'][:],axis=1)[:,None,:,None]
                 *np.diff(h['x1f'][:],axis=1)[:,None,None,:])
            er=h['user_out_var'][0].astype(np.float64)
            levels,counts=np.unique(h['Levels'][:],return_counts=True)
            result.append({'file':f.name,'levels':{str(int(a)):int(b) for a,b in zip(levels,counts)},
                           'Er_min':float(er.min()),'Er_volume_integral_from_float32_hdf':float(np.sum(er*vol))})
    return result

def record(name, input_path, fixed, opacity, threshold, ranks):
    d=B/name;log=(d/'run.log').read_text(errors='replace')
    fg=[{'iterations':int(a),'true_relative_residual':float(b)} for a,b in
        re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',log)]
    nr=[{'status':a,'iterations':int(b)} for a,b in
        re.findall(r'\[NR\] status=(\w+) iterations=(\d+)',log)]
    jv=[]
    for line in log.splitlines():
        if line.startswith('[NRFLD global Jv]') and 'region=coarse_fine_face' in line:
            f=dict(re.findall(r'(\w+)=([^ ]+)',line))
            jv.append({'direction':f.get('direction'),'h':float(f.get('epsilon')),
                       'cells':int(f.get('cells')),'relative_l2':float(f.get('l2_rel')),
                       'tangent_relative_l2':float(f.get('tangent_l2_rel')),
                       'ghost_relative_l2':float(f.get('ghost_l2_rel'))})
    logs=sorted((d/'mpi').glob('**/rank.*/stdout'))
    result=subprocess.run(['python3',str(LEDGER),'--tolerance','1e-6',*map(str,logs)],
                          capture_output=True,text=True)
    face=json.loads(result.stdout) if result.stdout else {'pairs':[],'pass':False}
    pairs=face.get('pairs',[])
    bad=max(pairs,key=lambda p:p['relative_mismatch']) if pairs else None
    final=fields(name)
    coarse_level=None
    if bad and final:
        hfile=next((d/'output').glob('*.fields.00001.athdf'))
        with h5py.File(hfile) as h:coarse_level=int(h['Levels'][bad['coarse_gid']])
    energy=[float(x) for x in re.findall(r'absolute_relative_balance=([\deE+.-]+)',log)]
    exit_status=int(re.search(r'Exit status:\s*(\d+)',log).group(1))
    solver_pass=exit_status==0 and nr[-1]['status']=='converged' and all(v['true_relative_residual']<threshold for v in fg)
    return {'name':name,'source_commit_before_run':'9ee735fa','binary_sha256':sha(BINARY),
            'input_path':str(input_path),'input_sha256':sha(input_path),'ranks':ranks,'threads':1,
            'overrides':{'nrfld/linear_solver':'fgmres','fld/fixed_flux_limiter':fixed,
                         'fld/const_opacity':opacity,'nrfld/fgmres_threshold':threshold,
                         'nrfld/fgmres_linearity_check':False},
            'command':re.search(r'Command being timed: "([^"]+)"',log).group(1),
            'run_log':str(d/'run.log'),'exit_status':exit_status,
            'wall_time':re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)',log).group(1),
            'peak_rss_kb':int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',log).group(1)),
            'newton_statuses':nr,'fgmres_history':fg,'fgmres_total_iterations':sum(v['iterations'] for v in fg),
            'global_coarse_fine_jv':jv,'field_outputs':final,
            'face_ledger':{'pair_count':len(pairs),'max_relative_mismatch':face.get('max_relative_mismatch'),
                           'max_absolute_mismatch':max((abs(p['mismatch']) for p in pairs),default=None),
                           'bad_pair':bad,'bad_pair_coarse_level':coarse_level,
                           'max_abs_over_max_flux':max((abs(p['mismatch']) for p in pairs),default=0)
                               /max((max(abs(p['coarse_flux']),abs(p['fine_flux'])) for p in pairs),default=1),
                           'relative_tolerance':1e-6,'pass':face.get('pass'),'logs':list(map(str,logs))},
            'max_implicit_energy_absolute_relative_balance':max(energy) if energy else None,
            'solver_pass':solver_pass,'face_pass':face.get('pass',False),
            'evaluation':('PASS' if solver_pass and face.get('pass') else
                          'FAIL_FACE_RELATIVE_ONLY' if solver_pass else 'FAIL_LINEAR_OR_NEWTON')}

def rank_equal(a,b):
    x=sorted((B/a/'output').glob('*.fields.*.athdf'))
    y=sorted((B/b/'output').glob('*.fields.*.athdf'))
    if len(x)!=len(y):return False
    for u,v in zip(x,y):
        with h5py.File(u) as h,h5py.File(v) as k:
            if not all(np.array_equal(h[n][:],k[n][:]) for n in h if isinstance(h[n],h5py.Dataset)):
                return False
    return True

runs=[]
for fixed,mode in [(True,'fixed'),(False,'variable')]:
    for rank in (1,2):
        name='testc_threelevel_shifted_r'+str(rank) if fixed else f'testc_threelevel_shifted_variable_r{rank}'
        runs.append(record(name,INPUT3,fixed,8.186332719786666e-8,1e-8,rank))
for mode,fixed in [('fixed',True),('variable',False)]:
    runs.append(record(f'testc_threelevel_{mode}_strict1e9_r1',INPUT3,fixed,8.186332719786666e-8,1e-9,1))
    for rank in (1,2):
        runs.append(record(f'testc_threelevel_{mode}_opacity10_r{rank}',INPUT3,fixed,8.186332719786666e-7,1e-8,rank))
    runs.append(record(f'testc_twolevel_{mode}_ledger_r1',INPUT2,fixed,8.186332719786666e-8,1e-8,1))
summary={'original_threelevel_face_relative_fail_count':sum(r['evaluation']=='FAIL_FACE_RELATIVE_ONLY' for r in runs if r['name'].startswith('testc_threelevel_shifted')),
         'twolevel_fixed_face_relative':next(r['face_ledger']['max_relative_mismatch'] for r in runs if r['name']=='testc_twolevel_fixed_ledger_r1'),
         'fixed_original_rank_hdf_exact':rank_equal('testc_threelevel_shifted_r1','testc_threelevel_shifted_r2'),
         'variable_original_rank_hdf_exact':rank_equal('testc_threelevel_shifted_variable_r1','testc_threelevel_shifted_variable_r2'),
         'fixed_opacity10_rank_hdf_exact':rank_equal('testc_threelevel_fixed_opacity10_r1','testc_threelevel_fixed_opacity10_r2'),
         'variable_opacity10_rank_hdf_exact':rank_equal('testc_threelevel_variable_opacity10_r1','testc_threelevel_variable_opacity10_r2')}
x=json.loads(REPORT.read_text())
x['threelevel_small_flux_conservation_screen']={'scope':'92-leaf 3-level shifted Test C, periodic radiation-only, fixed/variable limiter, 1/2 MPI ranks',
 'runs':runs,'summary':summary,
 'decision':'Original strong-diffusion cases FAIL the 1e-6 per-face relative ledger criterion on tiny 1e-9 flux; report failure. Absolute mismatch 1e-14 to 1e-13, global implicit energy balance near 1e-11 or better, rank HDF exact. Stricter FGMRES did not resolve; 10x opacity makes criterion pass but changes physics and is not a code fix. No production operator change.'}
REPORT.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
