#!/usr/bin/env python3
"""Compare matched RSG snapshots by radius and core flow direction."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def snapshot(path):
    with h5py.File(path) as f:
        names = [x.decode() for x in f.attrs['VariableNames']]
        state = np.asarray(f['prim'], dtype=float)
        rho = state[names.index('rho')]
        velocity = [state[names.index(f'vel{d}')] for d in (1, 2, 3)]
        x = np.asarray(f['x1v'], dtype=float)[:, None, None, :]
        y = np.asarray(f['x2v'], dtype=float)[:, None, :, None]
        z = np.asarray(f['x3v'], dtype=float)[:, :, None, None]
        radius = np.sqrt(x*x+y*y+z*z)
        radial_velocity = (velocity[0]*x+velocity[1]*y+velocity[2]*z)/radius
        speed = np.sqrt(sum(component**2 for component in velocity))
        max_cos = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))/radius
        return {'time': float(f.attrs['Time']),
                'locations': np.asarray(f['LogicalLocations']),
                'rho': rho.ravel(), 'r': radius.ravel(),
                'vr': radial_velocity.ravel(), 'speed': speed.ravel(),
                'max_cos': max_cos.ravel()}


def stats(state, mask):
    rho, vr, speed = (state[key][mask] for key in ('rho', 'vr', 'speed'))
    return {'cells': int(mask.sum()),
            'mean_vr': float(np.mean(vr)),
            'mean_abs_vr': float(np.mean(np.abs(vr))),
            'mean_speed': float(np.mean(speed)),
            'mass_weighted_vr': float(np.sum(rho*vr)/np.sum(rho))}


def binned(state, edges):
    r = state['r']
    index = np.searchsorted(edges, r, side='right')-1
    valid = (index >= 0) & (index < len(edges)-1)
    counts = np.bincount(index[valid], minlength=len(edges)-1)
    mean = np.divide(np.bincount(index[valid], weights=state['vr'][valid],
                                minlength=len(edges)-1), counts,
                     out=np.full(len(edges)-1, np.nan), where=counts > 0)
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--er-only', type=Path, required=True)
    parser.add_argument('--extended', type=Path)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    cases = {'baseline': snapshot(args.baseline),
             'Er average only': snapshot(args.er_only),
             'Er + core gravity': snapshot(args.pair)}
    if args.extended:
        cases['Er + gravity to 3 Rcore'] = snapshot(args.extended)
    first = cases['baseline']
    for label, state in cases.items():
        if (abs(state['time']-first['time']) > 1e-9
                or not np.array_equal(state['locations'], first['locations'])
                or not np.array_equal(state['r'], first['r'])):
            raise ValueError(f'{label} does not match the baseline grid/time')

    r, cos = first['r'], first['max_cos']
    intervals = [(0, 50), (50, 100), (100, 166), (166, 250),
                 (250, 500), (500, 830), (830, 1010)]
    masks = {f'{lo}-{hi}': (r >= lo) & (r < hi) for lo, hi in intervals}
    masks['core'] = r < 166
    masks['core_axis15'] = (r < 166) & (cos > np.cos(np.deg2rad(15)))
    masks['core_diagonal'] = (r < 166) & (cos < 0.70)
    result = {'time_code': first['time'],
              'definition': 'velocity in code units; equal-volume means except mass_weighted_vr',
              'cases': {label: {region: stats(state, mask)
                                for region, mask in masks.items()}
                        for label, state in cases.items()}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2)+'\n')

    edges = np.arange(0, 1025, 25, dtype=float)
    centers = 0.5*(edges[:-1]+edges[1:])
    means = {name: binned(state, edges) for name, state in cases.items()}
    colors = {'baseline': '#4c607b', 'Er average only': '#e09c3a',
              'Er + core gravity': '#b54858',
              'Er + gravity to 3 Rcore': '#7954a5'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col',
                             constrained_layout=True)
    for column, (lo, hi) in enumerate(((0, 300), (300, 1010))):
        top, bottom = axes[:, column]
        for name, mean in means.items():
            top.plot(centers, mean, marker='o', markersize=2.5,
                     label=name, color=colors[name], linewidth=1.3)
        baseline = means['baseline']
        for name in cases:
            if name == 'baseline':
                continue
            bottom.plot(centers, means[name]-baseline,
                        marker='o', markersize=2.5, label=name+' − baseline',
                        color=colors[name], linewidth=1.3)
        for ax in (top, bottom):
            ax.set_xlim(lo, hi)
            ax.axhline(0, color='black', linewidth=0.6)
            ax.grid(alpha=0.25)
            for landmark in (166, 830):
                if lo < landmark < hi:
                    ax.axvline(landmark, color='gray', linestyle='--', linewidth=0.8)
        bottom.set_xlabel('radius [Rsun]')
        top.set_ylabel('mean radial velocity [code]')
        bottom.set_ylabel('difference [code]')
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f'RSG matched 64³ runs at code time {first["time"]:.4f}')
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    print(args.figure)
    print(args.json)


if __name__ == '__main__':
    main()
