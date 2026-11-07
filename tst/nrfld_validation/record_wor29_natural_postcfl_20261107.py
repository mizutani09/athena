#!/usr/bin/env python3
"""Record RBGS color order versus FGMRES threshold on natural AMR transition."""
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

W=Path(__file__).resolve().parents[2]
B=Path('/tmp/wor25_23_24_resolution_trials')
REPORT=W/'tst/nrfld_validation/wor25_23_24_resolution_20261107.json'
LEDGER=W/'tst/nrfld_validation/collapse_interface_ledger.py'
BINARY=Path('/tmp/wor25_23_24_resolution_collapse_postflddt_build/bin/athena')
CHECKPOINT=B/'collapse_postflddt_opacity_natural_r1/output/protostellar_core_fld_mayer5.00015.rst'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mass(h):
    vol=(np.diff(h['x3f'][:],axis=1)[:,:,None,None]
         *np.diff(h['x2f'][:],axis=1)[:,None,:,None]
         *np.diff(h['x1f'][:],axis=1)[:,None,None,:])
    return float(np.sum(h['prim'][0]*vol))

def run(mode,ranks,strict):
    suffix='_strict1e9' if strict else ''
    name=f'collapse_natural_postflddt_colors_{mode}{suffix}_r{ranks}'
    d=B/name;log=(d/'run.log').read_text()
    fg=[{'iterations':int(n),'true_relative_residual':float(v)} for n,v in
        re.findall(r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',log)]
    newton=[{'status':s,'iterations':int(n)} for s,n in
            re.findall(r'\[NR\] status=(\w+) iterations=(\d+)',log)]
    logs=sorted((d/'mpi').glob('**/rank.*/stdout'))
    raw=subprocess.run(['python3',str(LEDGER),'--tolerance','1e-6',*map(str,logs)],
                       capture_output=True,text=True,check=True)
    face=json.loads(raw.stdout)
    bad=max(face['pairs'],key=lambda p:p['relative_mismatch'])
    files=sorted((d/'output').glob('*.prim.*.athdf'))
    with h5py.File(files[0]) as start,h5py.File(files[-1]) as end:
        m0,m1=mass(start),mass(end)
        minima={'rho':float(np.min(end['prim'][0])),
                'press':float(np.min(end['prim'][1])),
                'Er':float(np.min(end['prim'][8]))}
        levels={str(int(x)):int(y) for x,y in zip(*np.unique(end['Levels'][:],return_counts=True))}
    colors={'default':(False,False),'consistent':(True,False),'symmetric':(False,True)}[mode]
    target=1e-9 if strict else 1e-8
    solver_pass=all(v['true_relative_residual']<target for v in fg) and all(n['status']=='converged' for n in newton)
    return {'name':name,'ranks':ranks,'threads':2,
            'source_commit_before_run':'4a2d5e03',
            'binary_sha256':sha(BINARY),'checkpoint_path':str(CHECKPOINT),'checkpoint_sha256':sha(CHECKPOINT),
            'overrides':{'nrfld/consistent_rb_colors':colors[0],
                         'nrfld/symmetric_rb_sweeps':colors[1],
                         'nrfld/fgmres_threshold':target,
                         'nrfld/global_jacobian_check':False,
                         'gravity/reset_fmg_color_each_solve':True,
                         'gravity/repeat_initial_fmg_after_restart':True,
                         'problem/N_J':7.6160,'time/nlim':17},
            'environment':{'ATHENA_NRFLD_EXPERIMENTAL_COLLAPSE':1,
                           'ATHENA_NRFLD_INTERFACE_LEDGER':1,
                           'ATHENA_NRFLD_ENERGY_LEDGER':1,'OMP_NUM_THREADS':2},
            'command':re.search(r'Command being timed: "([^"]+)"',log).group(1),
            'run_log':str(d/'run.log'),'exit_status':int(re.search(r'Exit status:\s*(\d+)',log).group(1)),
            'wall_time':re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)',log).group(1),
            'peak_rss_kb':int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',log).group(1)),
            'newton_history':newton,'fgmres_history':fg,'fgmres_total_iterations':sum(v['iterations'] for v in fg),
            'max_true_relative_residual':max(v['true_relative_residual'] for v in fg),
            'face_ledger':{'pair_count':len(face['pairs']),'max_relative_mismatch':face['max_relative_mismatch'],
                           'bad_pair':bad,'relative_tolerance':1e-6,'pass':face['pass'],'logs':list(map(str,logs))},
            'max_implicit_energy_absolute_relative_balance':max(map(float,re.findall(r'absolute_relative_balance=([\deE+.-]+)',log))),
            'mass_volume_integral':{'initial':m0,'final':m1,'relative_change':(m1-m0)/m0},
            'final_minima':minima,'final_level_counts':levels,
            'solver_pass':solver_pass,'evaluation':'PASS' if solver_pass and face['pass'] else 'FAIL_FACE_RELATIVE_ONLY' if solver_pass else 'FAIL_SOLVER'}

def final(name,kind):
    f=next((B/name/'output').glob(f'*.{kind}.00017.athdf'))
    with h5py.File(f) as h:return h['prim' if kind=='prim' else 'user_out_var'][:]

def relative(a,b,kind):
    x,y=final(a,kind),final(b,kind)
    return float(np.sum(abs(x-y))/np.sum(abs(x)))

runs=[run(mode,ranks,strict) for strict in (False,True)
      for mode in ('default','consistent','symmetric') for ranks in (1,2)]
comparisons={}
for strict in (False,True):
    suffix='_strict1e9' if strict else ''
    for mode in ('default','consistent','symmetric'):
        a=f'collapse_natural_postflddt_colors_{mode}{suffix}_r1'
        b=f'collapse_natural_postflddt_colors_{mode}{suffix}_r2'
        comparisons[mode+suffix]={kind:relative(a,b,kind) for kind in ('prim','fld')}
report=json.loads(REPORT.read_text())
report['wor29_natural_postcfl_color_threshold']={
    'scope':'Natural 43-to-64-leaf regrid from common cycle-15 checkpoint with post-implicit CFL fix; both RBGS flags explicitly overridden.',
    'binary_sha256':sha(BINARY),'checkpoint_sha256':sha(CHECKPOINT),'runs':runs,
    'rank_comparisons_final_relative_l1':comparisons,
    'decision':'At 1e-8 default uses 50 FGMRES iterations but 1-rank face criterion fails (1.30e-6); consistent uses 52 and both ranks pass; symmetric uses 54 and both pass. At 1e-9 default and consistent use 56 and both ranks pass; symmetric uses 59 but 1-rank face criterion fails (1.09e-6). No robust color-order advantage. Stricter threshold is an input-level mitigation for this collapse case, not a global default because strong Test C reaches a residual floor.'}
REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'evaluation':[r['evaluation'] for r in runs],
                  'iterations':[r['fgmres_total_iterations'] for r in runs],
                  'face':[r['face_ledger']['max_relative_mismatch'] for r in runs]},indent=2))
