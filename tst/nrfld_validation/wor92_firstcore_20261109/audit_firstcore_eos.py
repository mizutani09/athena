#!/usr/bin/env python3
"""Check first-core Tomida table against an analytic H2/He pressure limit."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


SOURCE_SHA = "742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43"
KB = 1.380649e-16
MH = 1.6735575e-24
X, Y = 0.70, 0.28


def audit(table, output):
    if hashlib.sha256(table.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("unexpected EOS input")
    rows = [line for line in table.open() if line.strip() and not line.startswith("#")]
    nv, nx, nr = map(int, rows[0].split())
    xlo, xhi = map(float, rows[1].split())
    rlo, rhi = map(float, rows[2].split())
    ratio = np.fromstring(rows[3], sep=" ")
    values = np.array([np.fromstring(row, sep=" ") for row in rows[4:]])
    values = values.reshape(nv, nx, nr)
    axis_x = np.linspace(xlo, xhi, nx)
    axis_r = np.linspace(rlo, rhi, nr)
    field = [RegularGridInterpolator((axis_x, axis_r), values[k])
             for k in (0, 1, 3)]
    gas_constant = KB / MH * (X / 2 + Y / 4)
    temperatures = np.geomspace(14.0, 431.3, 60)
    densities = np.geomspace(1e-16, 1e-10, 8)
    pressure_error = []
    energy_inverse_error = []
    cv = []
    temperature_log_derivative_error = []
    for rho in densities:
        lr = np.log10(rho)
        cv_this = []
        for temperature in temperatures:
            q = brentq(lambda u: field[2]([u, lr])[0] - np.log10(temperature),
                       xlo, xhi)
            e = rho * 10.0**q
            p = e * 10.0**field[0]([q, lr])[0]
            p_analytic = rho * gas_constant * temperature
            pressure_error.append(p / p_analytic - 1.0)
            pcoord = np.log10(p / rho) + np.log10(ratio[1])
            e_from_p = p * 10.0**field[1]([pcoord, lr])[0]
            energy_inverse_error.append(e_from_p / e - 1.0)
            # Compare the exact piecewise-bilinear d ln T/d ln e at fixed rho
            # against an independent centered difference inside one table cell.
            cell = min(nx - 2, max(0, np.searchsorted(axis_x, q) - 1))
            frac = (q - axis_x[cell]) / (axis_x[cell + 1] - axis_x[cell])
            if 0.02 < frac < 0.98:
                lr_cell = min(nr - 2, max(0, np.searchsorted(axis_r, lr) - 1))
                wr = (lr - axis_r[lr_cell]) / (axis_r[lr_cell + 1] - axis_r[lr_cell])
                slope0 = (values[3, cell+1, lr_cell] - values[3, cell, lr_cell]) / (
                    axis_x[cell+1] - axis_x[cell])
                slope1 = (values[3, cell+1, lr_cell+1] - values[3, cell, lr_cell+1]) / (
                    axis_x[cell+1] - axis_x[cell])
                exact = (1-wr)*slope0 + wr*slope1
                eps = 1e-6
                numerical = (field[2]([q+eps, lr])[0] - field[2]([q-eps, lr])[0])/(2*eps)
                temperature_log_derivative_error.append(numerical-exact)
            dq = 1e-4
            t_lo = 10.0**field[2]([q-dq, lr])[0]
            t_hi = 10.0**field[2]([q+dq, lr])[0]
            cv_this.append((rho*10.0**(q+dq) - rho*10.0**(q-dq))
                           / (rho*(t_hi-t_lo)))
        cv.append(cv_this)
    cv = np.asarray(cv)
    if (np.any(cv <= 0) or np.max(np.abs(pressure_error)) > 1e-8
            or np.max(np.abs(temperature_log_derivative_error)) > 1e-8):
        raise ValueError("first-core EOS analytic or stability check failed")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for index in (0, 3, 7):
        ax.plot(temperatures, cv[index], label=rf"$\rho={densities[index]:.1e}$ g cm$^{{-3}}$")
    ax.axhline(1.5 * gas_constant, color="black", ls="--", lw=1,
               label=r"Low-$T$ H$_2$/He translational limit")
    ax.set_xscale("log")
    ax.set_ylabel(r"Specific heat $c_v$ [erg g$^{-1}$ K$^{-1}$]")
    ax.set_xlabel("Gas temperature [K]")
    ax.set_title("EOS table in the visited first-core temperature range")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    summary = {
        "source_table_sha256": SOURCE_SHA,
        "density_range_g_cm3": [float(densities[0]), float(densities[-1])],
        "temperature_range_K": [float(temperatures[0]), float(temperatures[-1])],
        "samples": len(pressure_error),
        "analytic_pressure": "rho*k_B*T/m_H*(X/2+Y/4), X=0.70, Y=0.28",
        "pressure_max_absolute_relative_error": float(np.max(np.abs(pressure_error))),
        "energy_p_to_e_inverse_max_absolute_relative_error": float(
            np.max(np.abs(energy_inverse_error))),
        "cv_min_erg_g_K": float(np.min(cv)),
        "cv_max_erg_g_K": float(np.max(cv)),
        "temperature_log_derivative_samples": len(temperature_log_derivative_error),
        "temperature_log_derivative_max_absolute_error": float(np.max(
            np.abs(temperature_log_derivative_error))),
        "low_temperature_analytic_cv_erg_g_K": 1.5 * gas_constant,
        "plot": str(output),
        "scope": "first core 14–431.3 K; dissociation and second collapse not tested",
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eos_table", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.eos_table, args.output_png), indent=2))


if __name__ == "__main__":
    main()
