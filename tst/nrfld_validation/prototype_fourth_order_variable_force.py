#!/usr/bin/env python3
"""Prototype a fourth-order cell-average -lambda(R) dEr/dx operator.

This is a one-dimensional analytic study, not an Athena++ solver path.  It
uses only five neighboring cell averages (i-2 through i+2), keeps one Er
unknown per cell, and compares against direct continuum quadrature.  The
analytic density varies with Er; the opacity per unit mass is constant.
"""

import numpy as np


A_RAD = 7.5657e-15
T0, T1, WIDTH = 1.0e7, 2.0e7, 80.0
RGAS, MU, RHO0, KAPPA = 8.3144621e7, 2.33, 1.2, 0.01
PTOTAL = RHO0 * RGAS / MU * T0 + A_RAD * T0**4 / 3.0
AVG_NODES, AVG_WEIGHTS = np.polynomial.legendre.leggauss(16)
REF_NODES, REF_WEIGHTS = np.polynomial.legendre.leggauss(32)
GAUSS2 = np.array((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))

# Map the five neighboring cell averages to the coefficients of a quartic in
# s=(x-x_i)/h.  Each row is the exact average of s**power over cell i+offset.
AVERAGE_MATRIX = np.array([
    [((offset + 0.5)**(power + 1) - (offset - 0.5)**(power + 1))
     / (power + 1) for power in range(5)]
    for offset in range(-2, 3)
])
AVERAGE_TO_POLYNOMIAL = np.linalg.inv(AVERAGE_MATRIX)


def temperature(x):
    return T0 + (T1 - T0) * np.exp(-0.5 * (x / WIDTH)**2)


def energy(x):
    return A_RAD * temperature(x)**4


def density_from_energy(erad):
    return (PTOTAL - erad / 3.0) / (RGAS / MU * (erad / A_RAD)**0.25)


def limiter(erad, gradient):
    ratio = np.abs(gradient) / (KAPPA * density_from_energy(erad) * erad)
    return (2.0 + ratio) / (6.0 + 3.0 * ratio + ratio**2)


def analytic_cell_force(x, h):
    positions = x[None, :] + 0.5 * h * REF_NODES[:, None]
    profile = np.exp(-0.5 * (positions / WIDTH)**2)
    temp = T0 + (T1 - T0) * profile
    temp_gradient = -(T1 - T0) * positions / WIDTH**2 * profile
    erad = A_RAD * temp**4
    gradient = 4.0 * A_RAD * temp**3 * temp_gradient
    return -0.5 * np.sum(
        REF_WEIGHTS[:, None] * limiter(erad, gradient) * gradient, axis=0)


def relative_error(n):
    h = 1024.0 / n
    x = (np.arange(n) + 0.5) * h - 512.0
    averages = np.array([
        0.5 * np.sum(
            AVG_WEIGHTS[:, None] * energy(
                x[None, :] + (offset + 0.5 * AVG_NODES[:, None]) * h),
            axis=0)
        for offset in range(-2, 3)
    ])
    coefficients = AVERAGE_TO_POLYNOMIAL @ averages
    erad = sum(coefficients[p, None, :] * GAUSS2[:, None]**p
               for p in range(5))
    gradient = sum(p * coefficients[p, None, :]
                   * GAUSS2[:, None]**(p - 1) / h for p in range(1, 5))
    force = -0.5 * np.sum(limiter(erad, gradient) * gradient, axis=0)
    selected = (x > 0.0) & (x < 64.0)
    reference = analytic_cell_force(x[selected], h)
    return (np.sum(np.abs(force[selected] - reference))
            / np.sum(np.abs(reference)))


def main():
    previous = None
    rates = []
    for n in (64, 128, 256, 512, 1024, 2048):
        error = relative_error(n)
        rate = float('nan') if previous is None else np.log2(previous / error)
        print(f'N={n:4d} relative_L1={error:.10e} rate={rate:.3f}')
        if previous is not None:
            rates.append(rate)
        previous = error
    if not (3.8 < rates[-2] < 4.2 and 3.8 < rates[-1] < 4.2):
        raise SystemExit('Prototype did not reach fourth-order convergence')


if __name__ == '__main__':
    main()
