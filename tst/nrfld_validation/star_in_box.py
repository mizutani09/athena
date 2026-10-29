#!/usr/bin/env python3
"""Build an isolated RSG commissioning executable and verify its numerical contracts.

Includes uncommitted FLD sources. Does not reconfigure the parent repository.
No result from this short suite demonstrates astrophysical reproduction.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'utils'), str(ROOT / 'vis/python')]
import athena_read  # noqa: E402
from diagnose_star_in_box import analyze, read_input, snapshot_pair  # noqa: E402


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def materialize(source, target, overrides):
    text, block, seen = [], None, set()
    for line in source.read_text().splitlines():
        clean = line.split('#', 1)[0].strip()
        if clean.startswith('<'):
            block = clean.strip('<>')
        elif '=' in clean:
            key = block + '/' + clean.split('=', 1)[0].strip()
            if key in overrides:
                line = clean.split('=', 1)[0] + '= ' + str(overrides[key]); seen.add(key)
        text.append(line)
    if seen != set(overrides):
        raise ValueError(f'unknown override keys: {set(overrides)-seen}')
    target.write_text('\n'.join(text) + '\n')


def history(path, fallback_names=None):
    lines = path.read_text().splitlines()
    names = re.findall(r'\[\d+\]=([^\s]+)', lines[1])
    if not names:
        names = fallback_names
    a = np.loadtxt(path, ndmin=2)
    if names is None or len(names) != a.shape[1]:
        raise ValueError('missing or inconsistent history labels')
    return dict(zip(names, a.T))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'work_runs' /
                        ('star_in_box_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--max-cores', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = {'commands': [], 'checks': {}, 'status': 'running',
                'parent_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'dirty_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def command(cmd, cwd, logfile, expected=0):
        record = {'argv': [str(x) for x in cmd], 'cwd': str(cwd), 'log': str(logfile)}
        manifest['commands'].append(record)
        with logfile.open('w') as log:
            proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=args.timeout)
        record['exit_code'] = proc.returncode
        if expected == 0 and proc.returncode != 0:
            raise RuntimeError(f'command failed; see {logfile}')
        return logfile.read_text()

    def check(name, value):
        manifest['checks'][name] = bool(value)
        if not value:
            raise AssertionError(name)

    try:
        if args.binary:
            binary = args.binary.resolve()
        else:
            build = out / 'build_source'; build.mkdir()
            for name in ('configure.py', 'Makefile.in'):
                shutil.copy2(ROOT / name, build / name)
            shutil.copytree(ROOT / 'src', build / 'src', ignore=shutil.ignore_patterns('defs.hpp'))
            manifest['source_sha256'] = {str(p.relative_to(build)): digest(p)
                                        for p in sorted(build.rglob('*')) if p.is_file()}
            command([sys.executable, 'configure.py', '--prob=nrfld_star_in_box',
                     '--eos=adiabatic', '-nrmgfld', '-hdf5', '-mpi'], build, out / 'configure.log')
            command(['make', f'-j{args.max_cores}'], build, out / 'build.log')
            binary = build / 'bin/athena'
        config = command([str(binary), '-c'], out, out / 'binary_config.log')
        check('binary_problem', bool(re.search(r'Problem generator:\s+nrfld_star_in_box\b', config)))
        check('binary_mpi_hdf5', bool(re.search(r'MPI parallelism:\s+ON', config)) and bool(re.search(r'HDF5 output:\s+ON', config)))
        manifest['binary_sha256'] = digest(binary)
        source = ROOT / 'inputs/radiation/athinput.nrfld_star_in_box'
        manifest['input_template_sha256'] = digest(source)

        def run(name, overrides=None, ranks=1, restart=None, reject=False):
            d = out / name; d.mkdir()
            opts = {f'mesh/nx{a}': 16 for a in (1, 2, 3)}
            opts.update({f'meshblock/nx{a}': 8 for a in (1, 2, 3)})
            opts.update({'time/nlim': 4, 'time/tlim': 1.0,
                         'output2/dt': 1e-8, 'output3/dt': 1e-8, 'output4/dt': 1e-8})
            opts.update(overrides or {})
            inp = d / 'input.used'; materialize(source, inp, opts)
            cmd = ['mpiexec', '-n', str(ranks), str(binary)]
            if restart:
                cmd += ['-r', str(restart)]
            cmd += ['-i', str(inp), '-d', str(d)]
            log = command(cmd, d, d / 'run.log', expected=1 if reject else 0)
            if reject:
                check(name + '_did_not_evolve', 'Setup complete' not in log)
            else:
                check(name + '_terminated', bool(re.search(r'Terminating on (cycle|time) limit', log)))
                check(name + '_no_solver_warning', not re.search(r'WARNING|FATAL|failed to converge', log, re.I))
                statuses = re.findall(r'\[NR\] status=(\S+)', log)
                check(name + '_newton_status', bool(statuses) and set(statuses) <= {'converged', 'dt_zero_initialization'})
                residuals = [float(v) for v in re.findall(r'(?<!_)final_max=(\S+)', log)]
                check(name + '_newton_tolerance', bool(residuals) and max(residuals) <= float(read_input(inp)['nrfld/nr_threshold']))
                stats, _ = analyze(d, inp, bins=8)
                end = re.findall(r'^time=([^ ]+) cycle=', log, re.M)
                check(name + '_snapshot_is_final', bool(end) and np.isclose(stats['time_code'], float(end[-1]), rtol=2e-7, atol=1e-12))
                manifest.setdefault('runs', {})[name] = stats
                check(name + '_finite_positive', all(np.isfinite(v) for v in stats.values() if isinstance(v, (float, int))))
            return d

        init = run('initial', {'time/nlim': 0})
        cfg = read_input(init / 'input.used')
        stats, _ = analyze(init, init / 'input.used', 8)
        check('gas_mass_normalized', abs(stats['gas_mass_msun'] / 3.0 - 1) < 2e-6)
        check('heating_normalized', abs(stats['Lheat_lsun'] / 90000.0 - 1) < 2e-6)
        # Changing MeshBlocks leaves the initial seed and all physical fields identical.
        blocked = run('initial_one_block', {'time/nlim': 0, **{f'meshblock/nx{a}': 16 for a in (1, 2, 3)}})

        def difference(a, b):
            worst = 0.0
            for af, bf in zip(snapshot_pair(a), snapshot_pair(b)):
                aa, bb = athena_read.athdf(str(af)), athena_read.athdf(str(bf))
                check(a.name + '_' + b.name + '_same_time', abs(float(aa['Time'])-float(bb['Time'])) < 1e-7)
                for key in aa['VariableNames']:
                    key = key.decode() if isinstance(key, bytes) else key
                    scale = max(float(np.max(abs(aa[key]))), 1e-20)
                    worst = max(worst, float(np.max(abs(aa[key]-bb[key]))) / scale)
            return worst

        check('block_independent_initial_state', difference(init, blocked) == 0.0)
        direct = run('evolve')
        h = history(next(direct.glob('*.hst')))
        check('evolve_reached_four_cycles', len(h['time']) >= 5 and h['time'][-1] > 0)
        check('drag_sink_positive_monotonic', h['Edrag_removed'][-1] > 0 and np.all(np.diff(h['Edrag_removed']) >= 0))
        half = run('before_restart', {'time/nlim': 2})
        rst = sorted(half.glob('*.rst'), key=lambda p: p.stat().st_mtime)[-1]
        resumed = run('restart', restart=rst)
        err = difference(direct, resumed); manifest['restart_relative_max'] = err
        check('restart_fields', err < 3e-5)
        hr = history(next(resumed.glob('*.hst')), list(h))
        check('restart_drag_counter', np.isclose(h['Edrag_removed'][-1], hr['Edrag_removed'][-1], rtol=3e-5, atol=1e-14))
        if args.max_cores >= 2:
            parallel = run('mpi2', ranks=2)
            err = difference(direct, parallel); manifest['mpi_relative_max'] = err
            check('mpi_fields', err < 3e-5)
        # A synthetic constant table verifies the existing table callback and units,
        # without claiming to supply physical stellar opacity data.
        table = out / 'constant_opacity.h5'
        with h5py.File(table, 'w') as f:
            f['axes/log10_temperature'] = np.linspace(1, 8, 8)
            f['axes/log10_density'] = np.linspace(-30, 0, 8)
            f['kappa/log10_planck'] = np.full((8, 8), -2.0)
            f['kappa/log10_rosseland'] = np.full((8, 8), -2.0)
        manifest['synthetic_opacity_sha256'] = digest(table)
        tabrun = run('opacity_table', {'fld/use_opacity_table': 'true',
                                      'fld/opacity_table_file': str(table)})
        err = difference(direct, tabrun); manifest['table_relative_max'] = err
        check('table_matches_constant', err < 3e-5)
        run('reject_boundary', {'nrfld/linearMG_ix1_bc': 'zerofixed'}, reject=True)
        check('boundary_guard_message', 'boundaries must be user' in (out / 'reject_boundary/run.log').read_text())
        # Hydro-only equilibrium drift at two resolutions; radiation operators disabled.
        mach = []
        for n in (16, 32):
            case = run(f'hse{n}', {**{f'mesh/nx{a}': n for a in (1, 2, 3)},
                'time/nlim': -1, 'time/tlim': 0.01, 'time/cfl_number': 0.2, 'problem/polytropic_index': 3.0,
                'problem/luminosity_lsun': 0, 'problem/velocity_perturb_fraction': 0,
                'problem/core_drag_timescale': 0, 'fld/is_couple': 'false', 'fld/cut_diff': 'true'})
            mach.append(manifest['runs'][case.name]['mach_rms_mass'])
        manifest['hse_mach_rms_16_32'] = mach
        check('hse_drift_decreases', mach[1] < mach[0])
        manifest['status'] = 'passed'
    except Exception as exc:
        manifest['status'] = 'failed'; manifest['error'] = repr(exc)
        raise
    finally:
        (out / 'summary.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{manifest["status"]}: {out / "summary.json"}')


if __name__ == '__main__':
    main()
