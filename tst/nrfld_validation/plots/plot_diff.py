"""Analytic solution and profile plot for the NRFLD diffusion test."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nrfld_validation.io import latest_athdf, read_athdf


def diffusion_analytic(x_cm: np.ndarray, time_s: float) -> np.ndarray:
    rho_unit = 1.0e-7
    egas_unit = 1.0e7
    length_unit = 4.0e10
    light_speed = 2.99792458e10
    opacity = 0.4
    initial_time = 5.337025523170433e1
    total_time = time_s + initial_time
    diffusion = light_speed / (3.0 * rho_unit * opacity)
    e0 = 1.0e5 * egas_unit
    prefactor = e0 / (2.0 * np.sqrt(np.pi * diffusion * total_time / length_unit**2))
    return prefactor * np.exp(
        -np.square(x_cm - 0.5 * length_unit) / (4.0 * diffusion * total_time)
    )


def plot_diffusion(output_dir: Path, figure: Path) -> None:
    data = read_athdf(latest_athdf(output_dir, "uov_x_slice"), num_ghost=2)
    length_unit = 4.0e10
    time_unit = length_unit / np.sqrt(1.0e7 / 1.0e-7)
    x = np.asarray(data["x1v"], dtype=float) * length_unit
    erad = np.asarray(data["E_rad"][0, 0], dtype=float)
    analytic = diffusion_analytic(x, float(data["Time"]) * time_unit)

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(x, erad, s=18, label="NRFLD")
    axis.plot(x, analytic, color="black", label="analytic")
    axis.set_xlabel(r"$x\;[\mathrm{cm}]$", fontsize=20)
    axis.set_ylabel(r"$E_{\rm rad}\;[\mathrm{erg\,cm^{-3}}]$", fontsize=20)
    axis.tick_params(axis="both", which="major", labelsize=18)
    axis.xaxis.get_offset_text().set_fontsize(15)
    axis.yaxis.get_offset_text().set_fontsize(15)
    axis.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(figure, dpi=200)
    plt.close(fig)
