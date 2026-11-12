#!/usr/bin/env python3
"""Compare FLD_FORCE_PROFILE with the analytic cell-averaged radiation force.

Use inputs/radiation/athinput.nrfld_fourth_order_force with mesh/nx1=N,
mesh/x2min=mesh/x3min=-8192/N, mesh/x2max=mesh/x3max=8192/N,
fld/force_diagnostic_gid=N/32, and time/xorder=2 or 4. Pass each run log as
--case N:ORDER:PATH. The comparison interval is 0 < x < 64 cm.
"""

import argparse
from pathlib import Path

import numpy as np


def read_parameters(path):
    sections = {}
    section = None
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<') and line.endswith('>'):
            section = line[1:-1]
            sections[section] = {}
        elif section and '=' in line:
            key, value = line.split('=', 1)
            sections[section][key.strip()] = value.strip()
    return sections


def measure(log, n, params):
    rows = []
    for line in log.read_text().splitlines():
        if line.startswith('FLD_FORCE_PROFILE '):
            rows.append([float(v) for v in line.split()[1:]])
    if not rows:
        raise ValueError(f'No FLD_FORCE_PROFILE rows in {log}')
    data = np.asarray(rows)
    x = data[:, 0]
    a_rad = data[:, 3]
    selected = (x > 0.0) & (x < 64.0)
    if not np.any(selected):
        raise ValueError(f'No force samples in 0 < x < 64 for {log}')
    x = x[selected]
    a_rad = a_rad[selected]

    hydro = params['hydro']
    problem = params['problem']
    mesh = params['mesh']
    rho_unit = float(hydro['rho_unit'])
    egas_unit = float(hydro['egas_unit'])
    mu = float(hydro['mu'])
    t0 = float(problem['T0'])
    t1 = float(problem['T1'])
    width = float(problem['w'])
    h = (float(mesh['x1max']) - float(mesh['x1min'])) / n
    a_cgs = 7.5657e-15
    rgas_cgs = 8.3144621e7
    p_total = rho_unit * rgas_cgs / mu * t0 + a_cgs * t0**4 / 3.0

    def erad(xpos):
        temperature = t0 + (t1 - t0) * np.exp(-0.5 * (xpos / width)**2)
        return a_cgs * temperature**4

    nodes, weights = np.polynomial.legendre.leggauss(16)
    erad_average = 0.5 * np.sum(
        weights[:, None] * erad(x[None, :] + 0.5*h*nodes[:, None]), axis=0)
    temperature_average = (erad_average / a_cgs)**0.25
    rho_code = ((p_total - erad_average/3.0)
                / (rgas_cgs/mu * temperature_average) / rho_unit)
    force_density_numeric = a_rad * rho_code
    if params['fld']['fixed_flux_limiter'].lower() == 'true':
        force_density_exact = -(erad(x + 0.5*h) - erad(x - 0.5*h)) / (3.0*h*egas_unit)
    else:
        # Integrate -lambda(R) dE/dx over each cell.  The reference uses the
        # analytic continuum gradient, not the code's centered differences.
        kappa_mass = float(problem['opacity_mass_coeff'])
        xp = x[None, :] + 0.5*h*nodes[:, None]
        profile = np.exp(-0.5*(xp/width)**2)
        temperature = t0 + (t1 - t0)*profile
        dtemperature = -(t1 - t0)*xp/(width*width)*profile
        energy = a_cgs*temperature**4
        denergy = 4*a_cgs*temperature**3*dtemperature
        density = ((p_total - energy/3.0)
                   / (rgas_cgs/mu * temperature))
        ratio = np.abs(denergy)/(kappa_mass*density*energy)
        limiter = (2.0 + ratio)/(6.0 + 3.0*ratio + ratio*ratio)
        force_density_exact = -0.5*np.sum(
            weights[:, None]*limiter*denergy, axis=0)/egas_unit
    error = np.sum(np.abs(force_density_numeric - force_density_exact))
    norm = np.sum(np.abs(force_density_exact))
    return error / norm, len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True,
                        help='N:ORDER:PATH_TO_LOG (repeat for each run)')
    args = parser.parse_args()
    params = read_parameters(args.input)
    results = {}
    for case in args.case:
        n_text, order_text, path_text = case.split(':', 2)
        n, order = int(n_text), int(order_text)
        error, count = measure(Path(path_text), n, params)
        results.setdefault(order, []).append((n, error))
        print(f'N={n:4d} xorder={order} L1_relative={error:.9e} samples={count}')
    for order, values in sorted(results.items()):
        values.sort()
        for (n0, e0), (n1, e1) in zip(values, values[1:]):
            rate = np.log(e0/e1) / np.log(n1/n0)
            print(f'xorder={order} {n0}->{n1} observed_order={rate:.3f}')
        if (order == 4 and len(values) >= 2
                and params['fld']['fixed_flux_limiter'].lower() == 'true'):
            (n0, e0), (n1, e1) = values[-2:]
            if np.log(e0/e1)/np.log(n1/n0) < 3.5:
                raise SystemExit('Fourth-order force convergence gate failed')


if __name__ == '__main__':
    main()
