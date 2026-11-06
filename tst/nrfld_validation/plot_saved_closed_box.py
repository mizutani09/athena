#!/usr/bin/env python3
"""Replot a closed-box report from saved CSV, summary, and manifest only."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_report(root: Path):
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    runs = {row["case"]: row for row in manifest["runs"]}
    if summary["status"] != "passed":
        raise ValueError("report is not passed")
    histories = {}
    for tag, metrics in summary["cases"].items():
        command = runs[tag]["command"]
        overrides = dict(token.split("=", 1) for token in command if "=" in token)
        target = float(overrides["time/tlim"])
        nlim = int(overrides["time/nlim"])
        with (root / tag / "history.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        data = {field: np.array([float(row[field]) for row in rows])
                for field in ("time_code", "mass_relative_drift", "energy_relative_drift")}
        if not np.all(np.isfinite(np.concatenate(list(data.values())))):
            raise ValueError(f"non-finite history: {tag}")
        if not np.isclose(data["time_code"][-1], target, rtol=1e-10):
            raise ValueError(f"time/tlim mismatch: {tag}")
        if metrics["cycles"] > nlim:
            raise ValueError(f"time/nlim exceeded: {tag}")
        for field, metric in (("mass_relative_drift", "mass_max_relative_drift"),
                              ("energy_relative_drift", "energy_max_relative_drift")):
            measured = float(np.max(np.abs(data[field])))
            if not np.isclose(measured, metrics[metric], rtol=1e-8, atol=1e-15):
                raise ValueError(f"summary/history mismatch: {tag} {field}")
        histories[tag] = data
    return summary, histories


def plot(root: Path, output: Path):
    summary, histories = load_report(root)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    cfl_cases = ("cfl_0.1", "tol_1e-08", "cfl_0.4")
    colors = ("tab:blue", "tab:orange", "tab:green")
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True,
                             layout="constrained")
    for tag, color, marker in zip(cfl_cases, colors, markers):
        data = histories[tag]
        cfl = summary["cases"][tag]["cfl_number"]
        for ax, field in zip(axes, ("mass_relative_drift", "energy_relative_drift")):
            ax.plot(data["time_code"], data[field], color=color, marker=marker,
                    markersize=3, linewidth=1.2, label=f"CFL {cfl:g}")
    for ax, label in zip(axes, ("Mass", "Gas + radiation energy")):
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_ylabel(f"{label} relative drift")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Code time")
    fig.savefig(output / "conservation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
    cfl = np.array([summary["cases"][tag]["cfl_number"] for tag in cfl_cases])
    for field, color, marker, label in (
        ("mass_max_relative_drift", "tab:blue", "o", "Mass"),
        ("energy_max_relative_drift", "tab:orange", "s", "Gas + radiation energy"),
    ):
        values = [summary["cases"][tag][field] for tag in cfl_cases]
        ax.plot(cfl, values, color=color, marker=marker, markersize=5,
                linewidth=1.4, label=label)
    ax.set(xlabel="CFL number", ylabel="Maximum absolute relative drift",
           xticks=cfl)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.savefig(output / "cfl_dependence.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report_root, args.output_dir)


if __name__ == "__main__":
    main()
