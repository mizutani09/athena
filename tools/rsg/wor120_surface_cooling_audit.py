#!/usr/bin/env python3
"""Compare RSG surface cooling controls with the imported HSE flux profile."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RSUN = 6.957e10
LSUN = 3.828e33
SHELLS = ((0, 166), (650, 750), (750, 800), (800, 830), (750, 830), (830, 900))


def fields(primitive, user_output):
    with h5py.File(primitive) as f, h5py.File(user_output) as u:
        names = [item.decode() for item in f.attrs['VariableNames']]
        state = np.asarray(f['prim'], dtype=float)
        rho = state[names.index('rho')]
        v = [state[names.index(f'vel{i}')] for i in (1, 2, 3)]
        x = np.asarray(f['x1v'])[:, None, None, :]
        y = np.asarray(f['x2v'])[:, None, :, None]
        z = np.asarray(f['x3v'])[:, :, None, None]
        radius = np.sqrt(x*x+y*y+z*z)
        vr = (v[0]*x+v[1]*y+v[2]*z)/radius
        return float(f.attrs['Time']), radius, rho, vr, np.asarray(
            u['user_out_var'][0], dtype=float)


def snapshot(directory, time_code):
    paths = sorted(directory.glob('*.prim.*.athdf'))
    if not paths:
        raise ValueError(f'No snapshots in {directory}')
    candidates = []
    for path in paths:
        with h5py.File(path) as f:
            candidates.append((abs(float(f.attrs['Time'])-time_code), path))
    delta, path = min(candidates)
    if delta > 1e-4:
        raise ValueError(f'No snapshot at {time_code} in {directory}')
    return fields(path, Path(str(path).replace('.prim.', '.uov.')))


def shell_stats(data):
    time, radius, rho, vr, temp = data
    result = {'time_code': time}
    for lo, hi in SHELLS:
        mask = (radius >= lo) & (radius < hi)
        key = f'{lo}-{hi}'
        result[key] = {
            'cells': int(mask.sum()),
            'mass_weighted_temperature_K': float(np.average(temp[mask], weights=rho[mask])),
            'mass_weighted_radial_velocity_code': float(np.average(vr[mask], weights=rho[mask]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True,
                        help='label:run-directory')
    parser.add_argument('--time-code', type=float, default=60.12841)
    parser.add_argument('--luminosity-lsun', type=float, default=91932.0)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()

    profile = np.loadtxt(args.profile)
    radius = profile[:, 0]/RSUN
    light_fraction = (4*np.pi*profile[:, 0]**2*profile[:, 9]
                      /(args.luminosity_lsun*LSUN))
    selected = (radius > 810) & (radius < 825)
    transition_r = radius[selected]
    transition_l = light_fraction[selected]
    crossings = {str(level): float(np.interp(level, transition_l, transition_r))
                 for level in (0.1, 0.5, 0.9)}
    cooling = np.gradient(4*np.pi*profile[:, 0]**2*profile[:, 9], profile[:, 0])
    cooling /= 4*np.pi*profile[:, 0]**2
    nearest_816 = int(np.argmin(np.abs(radius-816)))
    cooling_days_816 = (float(profile[nearest_816, 2]/cooling[nearest_816]/86400)
                        if cooling[nearest_816] > 0 else None)
    cases = {}
    for item in args.case:
        label, directory = item.split(':', 1)
        cases[label] = shell_stats(snapshot(Path(directory), args.time_code))
    result = {'profile': {'radiative_luminosity_fraction': {
                  str(r): float(np.interp(r, radius, light_fraction))
                  for r in (750, 800, 810, 815, 820, 830)},
                  'transition_rsun': crossings,
                  'transition_width_10_90_rsun': crossings['0.9']-crossings['0.1'],
                  'grid_width_64_rsun': 2021.0/64,
                  'local_cooling_time_near_816_days': cooling_days_816},
              'cases': cases}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(radius, light_fraction, color='#b54e4e')
    axes[0, 0].set(xlim=(740, 835), ylim=(0, 1.1),
                   xlabel='radius [Rsun]', ylabel='initial radiative luminosity / Lstar')
    axes[0, 0].axvspan(crossings['0.1'], crossings['0.9'], color='orange', alpha=0.2)
    axes[0, 1].semilogy(radius, profile[:, 8], color='#4d6d92')
    axes[0, 1].set(xlim=(780, 835), xlabel='radius [Rsun]',
                   ylabel='Rosseland opacity [cm²/g]')
    labels = list(cases)
    positions = np.arange(len(labels))
    width = 0.4
    for ax, field, ylabel in ((axes[1, 0], 'mass_weighted_temperature_K',
                              'mass-weighted gas temperature [K]'),
                             (axes[1, 1], 'mass_weighted_radial_velocity_code',
                              'mass-weighted radial velocity [code]')):
        for offset, shell, color in ((-width/2, '750-800', '#518eaa'),
                                     (width/2, '800-830', '#db8a4a')):
            ax.bar(positions+offset, [cases[label][shell][field] for label in labels],
                   width, label=shell+' Rsun', color=color)
        ax.set_xticks(positions, labels, rotation=20)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        if field.endswith('velocity_code'):
            ax.axhline(0, color='black', linewidth=0.7)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle('RSG surface cooling controls at code time 60.12841 (0.042 yr)')
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    print(args.figure)
    print(args.json)


if __name__ == '__main__':
    main()
