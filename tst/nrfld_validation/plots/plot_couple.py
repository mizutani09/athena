"""Analytic solution and history plot for the NRFLD coupling test."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nrfld_validation.io import read_history


def coupling_analytic(case: str) -> tuple[np.ndarray, np.ndarray]:
    if case not in ("low", "high"):
        raise ValueError(f"unknown coupling case: {case}")
    # Match the constants and parameters used by nrfld_couple.cpp and its input.
    a_rad = 7.5657e-15
    r_gas = 8.31451e7
    c_light = 2.99792458e10
    mu = 0.6
    gamma = 5.0 / 3.0
    rho = 1.0e-7
    erad = 1.0e12
    sigma = 4.0e-8
    egas0 = 1.0e2 if case == "low" else 1.0e10
    temperature0 = (gamma - 1.0) * mu * egas0 / (rho * r_gas)
    bcoef = (a_rad / erad) ** 0.25
    acoef = (
        bcoef * (gamma - 1.0) * mu * c_light * sigma * erad / (rho * r_gas)
    )
    x0 = bcoef * temperature0
    # Resolve both the very early transient and the equilibrium tail. A grid
    # only in |x-1| misses the low-energy transient when x0 is near zero.
    if x0 > 1.0:
        outer = np.geomspace(x0, 1.1, 20000)
        tail = 1.0 + np.geomspace(0.1, 1.0e-14, 30000)[1:]
    else:
        outer = np.geomspace(x0, 0.9, 20000)
        tail = 1.0 - np.geomspace(0.1, 1.0e-14, 30000)[1:]
    x = np.concatenate((outer, tail))

    def primitive(value: np.ndarray | float) -> np.ndarray | float:
        return 0.25 * np.log(np.abs(value - 1.0) / np.abs(value + 1.0)) \
            - 0.5 * np.arctan(value)

    time = -(primitive(x) - primitive(x0)) / acoef
    temperature = x / bcoef
    egas = rho * r_gas * temperature / ((gamma - 1.0) * mu)
    order = np.argsort(time)
    ordered_time = time[order]
    ordered_egas = egas[order]
    # Floating-point rounding can collapse adjacent samples near equilibrium.
    unique_time, first_indices = np.unique(ordered_time, return_index=True)
    return unique_time, ordered_egas[first_indices]


def coupling_equilibrium() -> float:
    a_rad = 7.5657e-15
    r_gas = 8.31451e7
    gamma = 5.0 / 3.0
    mu = 0.6
    rho = 1.0e-7
    erad = 1.0e12
    return rho * r_gas * (erad / a_rad)**0.25 / ((gamma - 1.0) * mu)


def coupling_analytic_at(times: np.ndarray, case: str) -> np.ndarray:
    reference_time, reference_egas = coupling_analytic(case)
    initial = 1.0e2 if case == "low" else 1.0e10
    return np.interp(times, reference_time, reference_egas,
                     left=initial, right=coupling_equilibrium())


def plot_coupling(output_dir: Path, figure: Path, case: str = "low") -> None:
    history = read_history(output_dir)
    analytic_time, analytic_egas = coupling_analytic(case)
    mask = (history["Rtime"] > 0.0) & (history["e_gas"] > 0.0)
    reference_mask = (analytic_time > 0.0) & np.isfinite(analytic_time)
    reference = np.exp(np.interp(
        np.log(history["Rtime"][mask]),
        np.log(analytic_time[reference_mask]),
        np.log(analytic_egas[reference_mask]),
    ))
    relative_error = history["e_gas"][mask] / reference - 1.0
    radiation_drift = history["E_rad"][mask] / 1.0e12 - 1.0
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, (axis, error_axis) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axis.loglog(analytic_time[reference_mask], analytic_egas[reference_mask],
                color="black", label="analytic gas energy")
    axis.scatter(history["Rtime"][mask], history["e_gas"][mask], s=12,
                 label="NRFLD gas energy")
    axis.set_ylabel(r"$e_{\rm gas}\;[\mathrm{erg\,cm^{-3}}]$")
    axis.legend()
    error_axis.semilogx(history["Rtime"][mask], relative_error,
                        label="gas relative error")
    error_axis.semilogx(history["Rtime"][mask], radiation_drift,
                        label="fixed radiation relative drift")
    error_axis.axhline(0.0, color="black", linewidth=0.7)
    error_axis.set_xlabel(r"$t\;[\mathrm{s}]$")
    error_axis.set_ylabel("relative difference")
    error_axis.legend()
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
