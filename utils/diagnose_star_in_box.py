#!/usr/bin/env python3
"""Volume/mass weighted spherical diagnostics for the uniform-grid RSG baseline.

The spherical optical depth is an estimator, not a ray-traced photosphere.
No bolometric image or conserved face radiation luminosity is inferred from Er.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'vis/python'))
import athena_read  # noqa: E402


def read_input(path):
    result = {}
    block = None
    for raw in Path(path).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line.strip('<>')
        elif '=' in line:
            key, val = (s.strip() for s in line.split('=', 1))
            result[f'{block}/{key}'] = val
    return result


def snapshot_pair(directory, first=False):
    pairs = []
    for p in sorted(Path(directory).glob('*.prim.*.athdf')):
        u = p.with_name(p.name.replace('.prim.', '.uov.'))
        if u.exists():
            pairs.append((p, u))
    if not pairs:
        raise ValueError('no paired prim/uov HDF5 snapshots')
    return pairs[0 if first else -1]


def analyze(directory, input_path, bins=64, first=False):
    cfg = read_input(input_path)
    number = lambda k: float(cfg[k])
    ru, eu, lu = (number('hydro/' + k) for k in ('rho_unit', 'egas_unit', 'leng_unit'))
    vu = np.sqrt(eu / ru)
    gamma = number('hydro/gamma')
    pfile, ufile = snapshot_pair(directory, first)
    p, u = athena_read.athdf(str(pfile)), athena_read.athdf(str(ufile))
    if p['Time'] != u['Time']:
        raise ValueError('snapshot times differ')
    for data in (p, u):
        for key in data['VariableNames']:
            key = key.decode() if isinstance(key, bytes) else key
            if not np.isfinite(data[key]).all():
                raise ValueError(f'nonfinite field {key}')
    if any(np.min(a) <= 0 for a in (p['rho'], p['press'], u['Er'])):
        raise ValueError('nonpositive density, gas pressure or radiation energy')
    z, y, x = np.meshgrid(p['x3v'], p['x2v'], p['x1v'], indexing='ij')
    r = np.sqrt(x*x + y*y + z*z) * lu
    vol = (np.diff(p['x3f'])[:, None, None] * np.diff(p['x2f'])[None, :, None]
           * np.diff(p['x1f'])[None, None, :]) * lu**3
    rho, pg, vr = p['rho'] * ru, p['press'] * eu, u['vr'] * vu
    # Only complete spherical shells; corners contribute to total mass, not shells.
    rmax = min(abs(p[f'x{a}f'][b]) for a in (1, 2, 3) for b in (0, -1)) * lu
    edges = np.linspace(0, rmax, bins + 1)
    rc = (edges[1:] + edges[:-1]) / 2
    shell = np.digitize(r.ravel(), edges) - 1
    valid = (shell >= 0) & (shell < bins)

    def total(a):
        return np.bincount(shell[valid], weights=(a * vol).ravel()[valid], minlength=bins)

    volume, mass = total(np.ones_like(r)), total(rho)
    filled = volume > 0

    def divide(a, b):
        return np.divide(a, b, out=np.full(bins, np.nan), where=b > 0)

    mean_rho, mean_vr = divide(mass, volume), divide(total(rho * vr), mass)
    # Subtract mass-weighted shell means to remove bulk radial pulsation.
    h = gamma / (gamma - 1) * pg / rho
    mean_h = divide(total(rho * h), mass)
    lconv = 4 * np.pi * rc**2 * (divide(total(rho * vr * h), volume)
                                   - divide(total(rho * vr), volume) * mean_h)
    pturb = divide(total(rho * vr**2), volume) - mean_rho * mean_vr**2
    vel2 = sum((p[f'vel{a}'] * vu)**2 for a in (1, 2, 3))
    mach = np.sqrt(vel2 * rho / (gamma * pg))
    stats = {
        'time_code': float(p['Time']), 'time_year': float(p['Time'] * lu / vu / 31557600),
        'snapshot': str(pfile), 'gas_mass_msun': float(np.sum(rho * vol) / 1.98847e33),
        'Lheat_lsun': float(np.sum(u['heat'] * vol) * eu * vu / lu / 3.828e33),
        'mach_max': float(mach.max()),
        'mach_rms_mass': float(np.sqrt(np.sum(rho * vol * mach**2) / np.sum(rho * vol))),
        'rho_min_cgs': float(rho.min()), 'pressure_min_cgs': float(pg.min()),
        'Er_min_cgs': float(u['Er'].min() * eu),
        'shells_populated': int(filled.sum()),
        'interpretation': 'commissioning diagnostics; not evidence of relaxed stellar convection',
    }
    # Mark empty bins explicitly; do not silently replace them with zero.
    profile = np.column_stack((rc, mean_rho, divide(total(pg), volume),
        divide(total(u['Tgas']), volume), divide(total(u['Er'] * eu), volume),
        mean_vr, pturb, lconv, volume))
    return stats, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--bins', type=int, default=32)
    parser.add_argument('--first', action='store_true')
    parser.add_argument('--output', type=Path, required=True, help='output filename prefix')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.bins < 2:
        parser.error('--bins must be at least two')
    stats, profile = analyze(args.directory, args.input, args.bins, args.first)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(stats, indent=2) + '\n')
    np.savetxt(args.output.with_suffix('.csv'), profile, delimiter=',',
               header='r_cm,rho_cgs,Pgas_cgs,Tgas_K,Er_cgs,vr_cm_s,Pturb_radial_cgs,Lconv_erg_s,volume_cm3')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))
        r = profile[:, 0] / 6.957e10
        ax[0, 0].semilogy(r, profile[:, 1]); ax[0, 0].set_ylabel('Density [g cm$^{-3}$]')
        ax[0, 1].semilogy(r, profile[:, 3]); ax[0, 1].set_ylabel('Gas temperature [K]')
        ax[1, 0].plot(r, profile[:, 5] / 1e5); ax[1, 0].set_ylabel('Mean radial velocity [km/s]')
        ax[1, 1].plot(r, profile[:, 7] / 3.828e33); ax[1, 1].set_ylabel(r'Convective luminosity [$L_\odot$]')
        for a in ax.flat:
            a.set_xlabel(r'Radius [$R_\odot$]'); a.grid(alpha=0.2)
        fig.suptitle(f'RSG commissioning, t = {stats["time_year"]:.4f} yr (unrelaxed)')
        fig.tight_layout(); fig.savefig(args.output.with_suffix('.png'), dpi=160); plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
