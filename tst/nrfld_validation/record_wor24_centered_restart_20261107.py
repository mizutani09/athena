#!/usr/bin/env python3
"""Record the isolated centered-face coupled AMR restart comparison."""

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from collapse_interface_ledger import compare, read_faces


ROOT = Path(__file__).resolve().parents[2]
TRIALS = Path('/tmp/wor24_centered_wor25_roundoff_restart')
OUT = ROOT / 'tst/nrfld_validation/wor24_centered_restart_20261107.json'
SOURCE = 'e786abc429684cf413d23f423b6d48df5838437d'
BINARY = '1b47730af10734229823cd2347a3c003e8dfd1370ee5c32997092049d8fc9d66'
BINARY_PATH = '/tmp/wor24_centered_wor25_roundoff_final_v2_build/bin/athena'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_log(path):
    text = path.read_text(errors='replace')
    fg = [{'iterations': int(n), 'true_relative_residual': float(r)}
          for n, r in re.findall(
              r'\[LINEAR_FGMRES\] iterations=(\d+) relative_residual=([\deE+.-]+)',
              text)]
    nr = []
    for line in text.splitlines():
        if line.startswith('[NR] status='):
            fields = dict(re.findall(r'(\w+)=([\w.eE+-]+)', line))
            nr.append({'status': fields['status'], 'iterations': int(fields['iterations']),
                       'final_l2': float(fields['final_l2']),
                       'final_max': float(fields['final_max'])})
    pairs = compare(read_faces(path))
    resource_path = path.with_suffix('.resources')
    resource = resource_path.read_text() if resource_path.exists() else ''
    return {
        'log': str(path), 'log_sha256': sha(path),
        'operator_path': re.search(r'\[NRFLD_INTERFACE_MODE\] path=(\w+)', text).group(1),
        'newton_history': nr, 'fgmres_history': fg,
        'fgmres_total_iterations': sum(row['iterations'] for row in fg),
        'max_true_relative_residual': max((row['true_relative_residual'] for row in fg),
                                          default=None),
        'face_pair_count': len(pairs),
        'max_face_relative_mismatch': max((row['relative_mismatch'] for row in pairs),
                                          default=None),
        'max_implicit_energy_absolute_relative_balance': max(
            (float(v) for v in re.findall(r'absolute_relative_balance=([\deE+.-]+)',
                                          text)), default=None),
        'elapsed_seconds': float(re.search(r'elapsed_s=([\d.]+)', resource).group(1)),
        'peak_rss_kib': int(re.search(r'max_rss_kib=(\d+)', resource).group(1)),
        'completed': 'Terminating on cycle limit' in text and 'FATAL' not in text
    }


def parse_case(folder):
    name = folder.name
    mode, layout, ranks = name.split('_')
    direct_ranks, restart_ranks = map(int, ranks.split('to'))
    input_path = folder / 'input.used'
    direct = parse_log(folder / 'direct.log')
    restarted = parse_log(folder / 'restart.log')
    direct_files = sorted((folder / 'direct').glob('*.fields.*.athdf'))
    restart_files = sorted((folder / 'restart').glob('*.fields.*.athdf'))
    block_history = []
    for file in direct_files:
        with h5py.File(file) as data:
            block_history.append({'file': file.name,
                                  'cycle': int(re.search(r'\.(\d+)\.athdf$', file.name).group(1)),
                                  'blocks': int(data.attrs['NumMeshBlocks'])})
    with h5py.File(direct_files[-1]) as a, h5py.File(restart_files[-1]) as b:
        keys = [key for key in a if isinstance(a[key], h5py.Dataset)
                and np.issubdtype(a[key].dtype, np.number)]
        numeric_exact = all(np.array_equal(a[key][...], b[key][...]) for key in keys)
        field_difference = float(np.max(np.abs(a['user_out_var'][...]
                                               - b['user_out_var'][...])))
        time_equal = float(a.attrs['Time']) == float(b.attrs['Time'])
        topology_equal = np.array_equal(a['LogicalLocations'][...],
                                        b['LogicalLocations'][...])
    options = [
        'time/nlim=5', 'fld/is_couple=true', 'fld/only_rad=false',
        'fld/fixed_flux_limiter=false', 'fld/const_opacity=0.004',
        'problem/density_high=1.15', 'problem/transverse_amplitude=0.3'
    ]
    if layout == 'adaptive':
        options.append('problem/test_regrid_cycle=2')
    direct_prefix = f'mpirun -n {direct_ranks} ' if direct_ranks > 1 else ''
    restart_prefix = f'mpirun -n {restart_ranks} ' if restart_ranks > 1 else ''
    checkpoint = folder / 'direct/nrfld_diff_amr_interface.00001.rst'
    face_ok = direct['max_face_relative_mismatch'] is not None and (
        direct['max_face_relative_mismatch'] <= 1e-6)
    restart_ok = (direct['completed'] and restarted['completed'] and time_equal
                  and topology_equal and field_difference <= 1e-10)
    return {
        'case': name, 'mode': mode, 'layout': layout, 'direct_ranks': direct_ranks,
        'restart_ranks': restart_ranks, 'source_git_hash': SOURCE,
        'binary_sha256': BINARY, 'input': str(input_path),
        'input_sha256': sha(input_path),
        'environment': {
            'OMP_NUM_THREADS': 1, 'ATHENA_NRFLD_CENTERED_FACE_TRIAL': True,
            'ATHENA_NRFLD_INTERFACE_LEDGER': True,
            'ATHENA_NRFLD_ENERGY_LEDGER': True,
            'ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE': mode == 'candidate',
            'ATHENA_NRFLD_LEGACY_INTERFACE': mode == 'baseline',
            'ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF': mode == 'candidate',
            'ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV': False if mode == 'candidate' else None,
            'ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION': mode == 'candidate',
            'ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER': mode == 'candidate',
            'ATHENA_NRFLD_EXPERIMENTAL_REGRID': mode == 'candidate',
            'ATHENA_NRFLD_EXPERIMENTAL_COUPLED': mode == 'candidate'},
        'direct_command': (f'{direct_prefix}{BINARY_PATH} -i {input_path} '
                           f'-d {folder / "direct"} ' + ' '.join(options)),
        'restart_command': (f'{restart_prefix}{BINARY_PATH} -r {checkpoint} '
                            f'-d {folder / "restart"} time/nlim=5'),
        'direct': direct, 'restart': restarted,
        'direct_block_history': block_history,
        'final_direct_hdf_sha256': sha(direct_files[-1]),
        'final_restart_hdf_sha256': sha(restart_files[-1]),
        'final_numeric_arrays_exact': numeric_exact,
        'final_field_max_abs_difference': field_difference,
        'final_time_equal': time_equal, 'final_topology_equal': topology_equal,
        'face_pass': face_ok, 'restart_pass': restart_ok
    }


def main():
    cases = [parse_case(folder) for folder in sorted(TRIALS.iterdir())
             if (folder / 'restart.log').exists()
             and (folder / 'restart.log').stat().st_size > 0
             and list((folder / 'restart').glob('*.fields.*.athdf'))]
    OUT.write_text(json.dumps({'cases': cases}, indent=2) + '\n')
    for case in cases:
        print(case['case'], case['restart_pass'], case['face_pass'],
              case['final_field_max_abs_difference'])


if __name__ == '__main__':
    main()
