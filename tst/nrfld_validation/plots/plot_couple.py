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
    a_rad = 7.5646e-15
    k_b = 1.380658e-16
    m_h = 1.6733e-24
    c_light = 2.99792458e10
    mu = 0.6
    gamma = 5.0 / 3.0
    rho = 1.0e-7
    erad = 1.0e12
    sigma = 4.0e-8
    egas0 = 1.0e2 if case == "low" else 1.0e10
    temperature0 = (gamma - 1.0) * mu * m_h * egas0 / (rho * k_b)
    bcoef = (a_rad / erad) ** 0.25
    acoef = (
        bcoef * (gamma - 1.0) * mu * m_h * c_light * sigma * erad / (rho * k_b)
    )
    x0 = bcoef * temperature0
    epsilon = 1.0e-14
    if x0 > 1.0:
        x = np.logspace(np.log10(1.0 + epsilon), np.log10(x0 - epsilon), 4000)
    else:
        x = np.logspace(np.log10(x0 + epsilon), np.log10(1.0 - epsilon), 4000)

    def primitive(value: np.ndarray | float) -> np.ndarray | float:
        return 0.25 * np.log(np.abs(value - 1.0) / np.abs(value + 1.0)) \
            - 0.5 * np.arctan(value)

    time = -(primitive(x) - primitive(x0)) / acoef
    temperature = x / bcoef
    egas = rho * k_b * temperature / ((gamma - 1.0) * mu * m_h)
    return time, egas


def plot_coupling(output_dir: Path, figure: Path, case: str = "low") -> None:
    history = read_history(output_dir)
    analytic_time, analytic_egas = coupling_analytic(case)
    mask = (history["Rtime"] > 0.0) & (history["e_gas"] > 0.0)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.loglog(analytic_time, analytic_egas, color="black", label="analytic")
    axis.scatter(history["Rtime"][mask], history["e_gas"][mask], s=18, label="NRFLD")
    axis.set_xlabel(r"$t\;[\mathrm{s}]$", fontsize=20)
    axis.set_ylabel(r"$e_{\rm gas}\;[\mathrm{erg\,cm^{-3}}]$", fontsize=20)
    axis.tick_params(axis="both", which="major", labelsize=18)
    axis.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
