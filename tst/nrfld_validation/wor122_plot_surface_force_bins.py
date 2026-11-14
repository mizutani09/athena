#!/usr/bin/env python3
"""Plot first-stage RSG surface-force balance from WOR-122 JSON audits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    shells = json.loads(path.read_text())["shells"]
    keys = [key for key in shells if int(key.split("-")[1]) <= 850]
    radius = np.array([np.mean([float(part) for part in key.split("-")])
                       for key in keys])
    return radius, [shells[key] for key in keys]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cases = (
        ("force128_axis.json", "128³ axis block", "#2563eb", "o"),
        ("force256_axis.json", "256³ axis block", "#059669", "s"),
        ("force256_diagonal.json", "256³ diagonal block", "#f97316", "^"),
    )
    fig, (residual_ax, component_ax) = plt.subplots(
        2, 1, figsize=(8.0, 6.5), sharex=True)
    for filename, label, color, marker in cases:
        radius, rows = load(args.data_dir / filename)
        measured = np.array([
            row["mass_weighted_mean_radial_acceleration_over_g"]["residual"]
            for row in rows])
        residual_ax.plot(radius, 100 * measured, color=color, marker=marker,
                         linewidth=1.8, label=label)
        if filename == "force256_axis.json":
            continuum = np.array([
                row["continuum_mean_radial_acceleration_over_g"]["residual"]
                for row in rows])
            observed = np.array([
                row["observed_first_step_radial_acceleration_over_g"]["mass_weighted_mean"]
                for row in rows])
            residual_ax.plot(radius, 100 * continuum, color="#374151",
                             linestyle=":", linewidth=1.5, label="1D profile")
            residual_ax.scatter(radius, 100 * observed, marker="x", s=40,
                                color="#111827", label="256³ measured Δv/Δt")
        if filename == "force256_diagonal.json":
            continue
        for term, linestyle in (("gas", "-"), ("radiation", "--")):
            difference = np.array([
                row["mass_weighted_mean_radial_acceleration_over_g"][term]
                - row["continuum_mean_radial_acceleration_over_g"][term]
                for row in rows])
            component_ax.plot(radius, 100 * difference, color=color,
                              linestyle=linestyle, marker=marker, linewidth=1.6,
                              label=f"{label}: {term}")

    for axis in (residual_ax, component_ax):
        axis.axhline(0, color="#111827", linewidth=0.9)
        axis.axvspan(813, 817, color="#f59e0b", alpha=0.13)
        axis.set_xlim(750, 850)
        axis.grid(alpha=0.22)
    residual_ax.set_ylabel("mean radial force residual / g [%]")
    component_ax.set_ylabel("3D − 1D force component / g [%]")
    component_ax.set_xlabel("radius [R☉]")
    residual_ax.legend(ncol=2, fontsize=8, frameon=False)
    component_ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle("RSG initial surface force: Godunov + FLD + gravity", fontsize=12)
    fig.text(0.5, 0.008,
             "One selected block per case; shaded band is the thin radiative transition.",
             ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
