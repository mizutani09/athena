#!/usr/bin/env python3
"""Audit first-stage RSG gas + FLD + analytic pgen gravity force.

The selected FLD_FORCE_VOLUME block must be emitted with
fld/force_diagnostic_verbosity=3. Forces are code force densities.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


G_CGS = 6.67430e-8
MSUN_CGS = 1.98847e33
RSUN_CGS = 6.957e10


def parameters(path):
    result = {}
    for line in path.read_text().splitlines():
        clean = line.split('#', 1)[0].strip()
        if '=' in clean:
            key, value = clean.split('=', 1)
            result[key.strip()] = value.strip()
    return result


def force_rows(path):
    rows = [[float(x) for x in line.split()[1:]]
            for line in path.read_text().splitlines()
            if line.startswith('FLD_FORCE_VOLUME ')]
    if not rows:
        raise ValueError(f'No FLD_FORCE_VOLUME rows in {path}')
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--gid', type=int, required=True)
    parser.add_argument('--compare-log', type=Path)
    args = parser.parse_args()

    p = parameters(args.input)
    length = float(p['leng_unit'])
    velocity2 = float(p['egas_unit']) / float(p['rho_unit'])
    gm = G_CGS * float(p['potential_mass_msun']) * MSUN_CGS / (length*velocity2)
    radius = float(p['radius_rsun']) * RSUN_CGS / length
    core = float(p['core_radius_fraction']) * radius
    outer = float(p['outer_flattening_fraction']) * radius

    rows = force_rows(args.log)
    with h5py.File(args.initial) as f:
        rho = np.asarray(f['prim'][0, args.gid], dtype=float).reshape(-1)
        coords = np.stack(np.meshgrid(f['x1v'][args.gid],
                                      f['x2v'][args.gid],
                                      f['x3v'][args.gid],
                                      indexing='ij'), axis=-1)
        # Athena writes block arrays in (z,y,x) order.
        coords = np.transpose(coords, (2, 1, 0, 3)).reshape(-1, 3)
    if len(rows) != len(rho) or not np.allclose(rows[:, :3], coords,
                                               rtol=2e-7, atol=1e-5):
        raise ValueError('Diagnostic rows do not match the initial HDF5 block')

    xyz = rows[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    s = (r/outer)**8
    a = core**4 + r**4/np.sqrt(1.0 + s)
    gmag = gm*r**3 / ((1.0+s)**1.5*a**1.25)
    gravity = -rho[:, None] * gmag[:, None] * xyz/r[:, None]
    rad = rows[:, 3:6]
    gas = rows[:, 6:9]
    total = gas + rad + gravity
    total_norm = np.linalg.norm(total, axis=1)
    support = (np.linalg.norm(gas, axis=1)
               + np.linalg.norm(rad, axis=1)
               + np.linalg.norm(gravity, axis=1))
    for label, selected in [('core', r < core), ('star', r < radius)]:
        idx = np.flatnonzero(selected)
        peak = idx[np.argmax(total_norm[idx])]
        print(f'{label}: cells={len(idx)} '
              f'relative_L1={total_norm[idx].sum()/support[idx].sum():.12e} '
              f'max_abs={total_norm[peak]:.12e} '
              f'max_radius={r[peak]:.9f} '
              f'max_xyz={xyz[peak].tolist()}')
    if args.compare_log:
        other = force_rows(args.compare_log)
        if other.shape != rows.shape or not np.array_equal(other[:, :3], xyz):
            raise ValueError('Comparison diagnostic cells differ')
        difference = np.linalg.norm(rad-other[:, 3:6], axis=1)
        print(f'radiation_difference_sum_relative='
              f'{difference.sum()/np.linalg.norm(other[:, 3:6],axis=1).sum():.12e}')


if __name__ == '__main__':
    main()
