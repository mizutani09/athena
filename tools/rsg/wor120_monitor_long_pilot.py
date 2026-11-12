#!/usr/bin/env python3
"""Summarize the completed snapshots of a running WOR-120 RSG pilot."""

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def history(path):
    lines = path.read_text().splitlines()
    names = re.findall(r'\[\d+\]=([^ ]+)', lines[1])
    rows = []
    for line in lines[2:]:
        try:
            row = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(row) == len(names):
            rows.append(row)
    data = np.asarray(rows)
    return {name: data[:, col] for col, name in enumerate(names)}


def snapshot(path, star_radius):
    with h5py.File(path) as f:
        names = [name.decode() for name in f.attrs['VariableNames']]
        state = np.asarray(f['prim'], dtype=float)
        rho = state[names.index('rho')]
        pressure = state[names.index('press')]
        velocity = [state[names.index(f'vel{axis}')] for axis in (1, 2, 3)]
        x = np.asarray(f['x1v'])[:, None, None, :]
        y = np.asarray(f['x2v'])[:, None, :, None]
        z = np.asarray(f['x3v'])[:, :, None, None]
        radius = np.sqrt(x*x+y*y+z*z)
        speed = np.sqrt(sum(component**2 for component in velocity))
        radial_velocity = (velocity[0]*x+velocity[1]*y+velocity[2]*z)/radius
        # Diagnostic proxy only: this model uses a tabulated general EOS.
        proxy_mach = speed/np.sqrt((5.0/3.0)*pressure/rho)
        inside = radius < star_radius
        peak = np.unravel_index(np.argmax(proxy_mach), proxy_mach.shape)
        edges = np.arange(0, 1051, 50)
        shell_velocity = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (radius >= lo) & (radius < hi)
            shell_velocity.append(float(np.sum(rho[mask]*radial_velocity[mask])
                                        /np.sum(rho[mask])))
        core = radius < 166
        core_axis = core & (np.maximum.reduce(np.broadcast_arrays(
            np.abs(x)/radius, np.abs(y)/radius, np.abs(z)/radius))
                            > np.cos(np.deg2rad(15)))
        return {'time_code': float(f.attrs['Time']),
                'mass_sum': float(np.sum(rho)),
                'outside_star_mass_fraction': float(np.sum(rho[~inside])/np.sum(rho)),
                'inside_proxy_supersonic_mass_fraction': float(
                    np.sum(rho[inside & (proxy_mach > 1)])/np.sum(rho[inside])),
                'core_mean_speed': float(np.mean(speed[core])),
                'core_axis_mean_radial_velocity': float(np.mean(radial_velocity[core_axis])),
                'surface_750_830_mass_weighted_vr': float(
                    np.sum(rho[(radius >= 750) & inside]
                           *radial_velocity[(radius >= 750) & inside])
                    /np.sum(rho[(radius >= 750) & inside])),
                'inside_max_speed': float(np.max(speed[inside])),
                'outside_max_speed': float(np.max(speed[~inside])),
                'peak_proxy_mach': float(proxy_mach[peak]),
                'peak_proxy_mach_radius': float(radius[peak]),
                'peak_proxy_mach_density': float(rho[peak]),
                'shell_edges': edges.tolist(),
                'shell_mass_weighted_vr': shell_velocity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--star-radius', type=float, default=830.0)
    parser.add_argument('--year-code', type=float, default=1434.5)
    args = parser.parse_args()
    files = sorted(args.run_dir.glob('*.prim.*.athdf'))
    if not files:
        raise ValueError('No primitive snapshots found')
    snapshots = [snapshot(path, args.star_radius) for path in files]
    initial_mass = snapshots[0]['mass_sum']
    for row, path in zip(snapshots, files):
        row['file'] = path.name
        row['time_year'] = row['time_code']/args.year_code
        row['domain_mass_change_fraction'] = row['mass_sum']/initial_mass-1.0
    hist = history(next(args.run_dir.glob('*.hst')))
    result = {'snapshot_count': len(snapshots),
              'latest_history_time_code': float(hist['time'][-1]),
              'latest_history_max_mach': float(hist['Mach_max'][-1]),
              'latest_history_dt': float(hist['dt'][-1]),
              'note': 'proxy Mach uses sqrt((5/3) P/rho), not the tabulated EOS',
              'snapshots': snapshots}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    years = hist['time']/args.year_code
    axes[0, 0].plot(years, hist['Mach_max'], color='#aa3e48')
    axes[0, 0].set(ylabel='maximum Mach (tabulated EOS)', xlabel='time [yr]')
    axes[0, 1].semilogy(years, hist['dt'], color='#4d6392')
    axes[0, 1].set(ylabel='CFL step [code time]', xlabel='time [yr]')
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
    for color, row in zip(colors, snapshots):
        centers = 0.5*(np.asarray(row['shell_edges'][:-1])
                       +np.asarray(row['shell_edges'][1:]))
        axes[1, 0].plot(centers, row['shell_mass_weighted_vr'],
                        color=color, label=f'{row["time_year"]:.3f} yr')
    axes[1, 0].axhline(0, color='black', linewidth=0.6)
    axes[1, 0].axvline(args.star_radius, color='gray', linestyle='--', linewidth=0.7)
    axes[1, 0].set(xlabel='radius [Rsun]', ylabel='mass-weighted radial velocity [code]')
    axes[1, 0].legend(fontsize=7, ncol=2)
    times = [row['time_year'] for row in snapshots]
    axes[1, 1].plot(times, [100*row['outside_star_mass_fraction'] for row in snapshots],
                    'o-', label='mass at r ≥ Rstar')
    axes[1, 1].plot(times, [100*row['inside_proxy_supersonic_mass_fraction']
                             for row in snapshots], 'o-',
                    label='mass with proxy Mach > 1 at r < Rstar')
    axes[1, 1].set(xlabel='time [yr]', ylabel='fraction of domain mass [%]')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle('WOR-120 extended-gravity 64³ pilot: partial progress')
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    print(args.figure)
    print(args.json)


if __name__ == '__main__':
    main()
