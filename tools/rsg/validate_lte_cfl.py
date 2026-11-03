#!/usr/bin/env python3
"""Short material-EOS RSG full-FLD CFL comparison for WOR-52.

Run from the repository root after scripts/configure_rsg.sh. Outputs remain in
the requested directory, with an exact command and input copy for each case.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CFL = (0.01, 0.03, 0.1, 0.3)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_fields(directory):
    import sys
    sys.path.insert(0, str(ROOT / 'utils'))
    import athena_read
    fields = {}
    for tag in ('prim', 'uov'):
        files = sorted(directory.glob(f'*.{tag}.*.athdf'))
        if not files:
            raise ValueError(f'missing {tag} output in {directory}')
        data = athena_read.athdf(str(files[-1]))
        fields[tag] = {name: np.asarray(data[name], dtype=float)
                       for name in ('rho', 'press', 'vel1', 'vel2', 'vel3')
                       if tag == 'prim' and name in data}
        if tag == 'uov':
            fields[tag] = {name: np.asarray(data[name], dtype=float)
                           for name in ('Tgas', 'Trad', 'Er') if name in data}
        fields[tag]['time'] = float(data['Time'])
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tlim', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--mode', choices=('full', 'coupled_local', 'fixed_local'),
                        default='full')
    parser.add_argument('--one-step', action='store_true',
                        help='test each requested CFL at its untruncated first timestep')
    parser.add_argument('--energy-ledger', action='store_true',
                        help='emit and check the implicit gas/radiation energy ledger')
    args = parser.parse_args()
    if args.energy_ledger and args.mode == 'fixed_local':
        parser.error('energy ledger requires evolving gas and radiation')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = ROOT / 'bin/athena'
    config = subprocess.check_output([str(executable), '-c'], cwd=ROOT, text=True)
    if not re.search(r'Problem generator:\s+rsg_star_in_box\b', config) \
            or not re.search(r'Equation of state:\s+general/eos_table\b', config):
        raise ValueError('requires the production RSG general-table-EOS build')
    source = ROOT / 'inputs/rsg/athinput.st35gm03n07_fld'
    summary = {'binary_sha256': digest(executable), 'input_sha256': digest(source),
               'tlim': args.tlim, 'mode': args.mode, 'one_step': args.one_step,
               'energy_ledger': args.energy_ledger,
               'cases': {}, 'status': 'running'}
    (output / 'binary_config.txt').write_text(config)
    environment = dict(os.environ, OMP_NUM_THREADS='1')
    if args.energy_ledger:
        environment['ATHENA_NRFLD_ENERGY_LEDGER'] = '1'
    try:
        for cfl in CFL:
            tag = f'cfl_{cfl:.2f}'
            case = output / tag
            case.mkdir()
            input_text = source.read_text()
            if input_text.count('<output1>\n') != 1:
                raise ValueError('missing output1 block in RSG input')
            case_input = case / 'input.used'
            case_input.write_text(input_text.replace(
                '<output1>\n', '<output1>\ndata_format = %.16e\n', 1))
            command = [str(executable), '-i', str(case_input), '-d', str(case),
                       'mesh/nx1=32', 'mesh/nx2=32', 'mesh/nx3=32',
                       'meshblock/nx1=32', 'meshblock/nx2=32',
                       'meshblock/nx3=32', f'time/cfl_number={cfl:.17g}',
                       f'time/tlim={100. if args.one_step else args.tlim:.17g}',
                       f'time/nlim={1 if args.one_step else 1000}',
                       f'output1/dt={args.tlim:.17g}',
                       f'output2/dt={args.tlim:.17g}',
                       f'output3/dt={args.tlim:.17g}',
                       'output4/dt=1e30', f'job/problem_id={tag}']
            if args.mode != 'full':
                command.append('fld/cut_diff=true')
            if args.mode == 'fixed_local':
                command.append('fld/fixed_u_rad=true')
            with (case / 'run.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout)
            text = (case / 'run.log').read_text(errors='replace')
            statuses = re.findall(r'\[NR\] status=(\S+)', text)
            final_residuals = [float(value) for value in re.findall(
                r'\bfinal_max=([0-9.eE+-]+)', text)]
            ledgers = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                       for line in text.splitlines()
                       if line.startswith('[NR_ENERGY_LEDGER]')]
            cycles = re.findall(r'time=([0-9.eE+-]+) cycle=(\d+)', text)
            first_dt = re.search(r'^cycle=0 time=[^ ]+ dt=([0-9.eE+-]+)',
                                 text, re.MULTILINE)
            record = {'command': command, 'exit_code': result.returncode,
                      'cycles': int(cycles[-1][1]) if cycles else None,
                      'time': float(cycles[-1][0]) if cycles else None,
                      'newton_calls': len(statuses),
                      'newton_converged': bool(statuses) and all(
                          status in ('converged', 'dt_zero_initialization')
                          for status in statuses),
                      'max_final_residual': max(final_residuals, default=None),
                      'energy_ledger_records': len(ledgers),
                      'max_energy_relative_balance': max((
                          abs(float(item['relative_balance']))
                          for item in ledgers), default=None),
                      'max_energy_absolute_relative_balance': max((
                          abs(float(item['absolute_relative_balance']))
                          for item in ledgers), default=None),
                      'marshak_delta': [float(item['marshak_delta'])
                                        for item in ledgers],
                      'first_dt': float(first_dt.group(1)) if first_dt else None,
                      'opacity_violations': bool(re.search(
                          r'FLD_OPACITY_DIAGNOSTICS .*?(?:out_of_domain|clamped|negative_results|zero_results|nonfinite_results)=[1-9]',
                          text))}
            summary['cases'][tag] = record
            (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            if (result.returncode or not record['newton_converged']
                    or record['opacity_violations']
                    or record['max_final_residual'] is None
                    or record['max_final_residual'] > 1.e-8
                    or (args.energy_ledger and (
                        not ledgers
                        or record['max_energy_relative_balance'] > 1.e-8
                        or record['max_energy_absolute_relative_balance'] > 1.e-8
                        or (args.mode == 'full' and not any(
                            abs(value) > 0. for value in record['marshak_delta']))
                        or (args.mode == 'coupled_local' and any(
                            value != 0. for value in record['marshak_delta']))))
                    or (args.one_step and record['cycles'] != 1)
                    or (not args.one_step and not np.isclose(
                        record['time'], args.tlim, rtol=1e-12))):
                raise RuntimeError(f'{tag} failed: {case / "run.log"}')
            print(f'{tag}: {record["cycles"]} steps, '
                  f'{record["newton_calls"]} Newton calls, '
                  f'max final residual {record["max_final_residual"]:.3e}', flush=True)
        if args.one_step:
            baseline = summary['cases']['cfl_0.01']['first_dt'] / 0.01
            for cfl in CFL:
                actual = summary['cases'][f'cfl_{cfl:.2f}']['first_dt']
                if not np.isclose(actual/cfl, baseline, rtol=1.e-8):
                    raise ValueError('first timestep was capped, not set by requested CFL')
        if not args.one_step:
            reference = final_fields(output / 'cfl_0.01')
            for cfl in CFL[1:]:
                tag = f'cfl_{cfl:.2f}'
                fields = final_fields(output / tag)
                differences = {}
                for group in ('prim', 'uov'):
                    if not np.isclose(fields[group]['time'], args.tlim, rtol=1e-6):
                        raise ValueError(f'{tag} snapshot at wrong time')
                    for name, values in reference[group].items():
                        if name == 'time':
                            continue
                        other = fields[group][name]
                        scale = max(float(np.max(np.abs(values))), 1.e-30)
                        differences[f'{group}/{name}'] = float(np.max(np.abs(other-values))/scale)
                summary['cases'][tag]['max_field_difference_vs_cfl_0.01'] = differences
                print(f'{tag} max field difference vs 0.01: '
                      f'{max(differences.values()):.3e}', flush=True)
        summary['status'] = 'completed'
    except Exception as error:
        summary['status'] = 'failed'
        summary['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
