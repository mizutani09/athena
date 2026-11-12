#!/usr/bin/env python3
"""Compare a 3D variable-limiter Athena++ force profile to cell quadrature.

Use inputs/radiation/athinput.nrfld_force_3d and pass N:LOG for runs with
mesh/nx1=mesh/nx2=mesh/nx3=N and cubic MeshBlocks of N/2 cells.
"""

import argparse
from pathlib import Path

import numpy as np


def read_force(path):
    rows = []
    fallback = None
    for line in path.read_text().splitlines():
        if line.startswith('FLD_FORCE_VOLUME '):
            rows.append([float(v) for v in line.split()[1:]])
        elif line.startswith('FLD_FORCE_MAX ') and 'high_order_fallbacks=' in line:
            fallback = int(line.split('high_order_fallbacks=')[1].split()[0])
    if not rows or fallback is None:
        raise ValueError(f'Missing force-volume data or fallback count: {path}')
    return np.asarray(rows), fallback


def reference(x, h, opacity_mass_coeff, density_amplitude):
    # All quantities use the code units in athinput.nrfld_force_3d.
    e0 = 7.5657e-15 * (1.0e7)**4 / 4.28129270484549e14
    wave = 2.0 * np.pi / 1024.0
    nodes, weights = np.polynomial.legendre.leggauss(8)
    q = np.array(np.meshgrid(nodes, nodes, nodes, indexing='ij')).reshape(3, -1).T
    w = np.prod(np.array(np.meshgrid(weights, weights, weights,
                                   indexing='ij')).reshape(3, -1).T, axis=1) / 8.0
    output = np.zeros((len(x), 3))
    for start in range(0, len(x), 1000):
        stop = min(start + 1000, len(x))
        pos = x[None, start:stop, :] + 0.5 * h * q[:, None, :]
        phase = wave * pos
        cosine, sine = np.cos(phase), np.sin(phase)
        product = np.prod(cosine, axis=2)
        erad = e0 * (1.0 + 0.5 * product)
        sigma = opacity_mass_coeff * 1.2 * (1.0 + density_amplitude*product)
        grad = np.empty((*erad.shape, 3))
        for direction in range(3):
            others = [a for a in range(3) if a != direction]
            grad[:, :, direction] = (-e0 * 0.5 * wave
                * sine[:, :, direction]
                * cosine[:, :, others[0]] * cosine[:, :, others[1]])
        ratio = np.linalg.norm(grad, axis=2) / (sigma * erad)
        limiter = (2.0 + ratio) / (6.0 + 3.0 * ratio + ratio**2)
        output[start:stop] = -np.sum(
            w[:, None, None] * limiter[:, :, None] * grad, axis=0)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True,
                        help='N:run.log')
    parser.add_argument('--opacity-mass-coeff', type=float, default=0.0025)
    parser.add_argument('--density-amplitude', type=float, default=0.0)
    args = parser.parse_args()
    results = []
    for case in args.case:
        n_text, path = case.split(':', 1)
        n = int(n_text)
        data, fallback = read_force(Path(path))
        expected = reference(data[:, :3], 1024.0 / n,
                             args.opacity_mass_coeff, args.density_amplitude)
        numerical = data[:, 3:6]
        error = np.sum(np.abs(numerical - expected)) / np.sum(np.abs(expected))
        print(f'N={n:4d} samples={len(data):6d} relative_L1={error:.9e} '
              f'fallbacks={fallback}')
        if len(data) != (n // 2)**3 or fallback:
            raise SystemExit('Incomplete 3D block or high-order fallback activated')
        results.append((n, error))
    results.sort()
    for (n0, e0), (n1, e1) in zip(results, results[1:]):
        rate = np.log2(e0 / e1)
        print(f'{n0}->{n1} observed_order={rate:.3f}')
    if len(results) >= 2 and np.log2(results[-2][1] / results[-1][1]) < 3.5:
        raise SystemExit('Fourth-order 3D force convergence gate failed')


if __name__ == '__main__':
    main()
