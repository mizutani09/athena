#!/usr/bin/env python3
"""Plot controlled RSG surface-seed retention experiments."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_cases(directory):
    result = {}
    for path in directory.glob("*modes.json"):
        result.update(json.loads(path.read_text()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = read_cases(args.directory)
    configs = [
        ("wor122_entropy_budget_full", "FLD, surface heat", "#1f77b4", "-"),
        ("wor122_entropy_budget_closedtop", "FLD, nearly closed top", "#9467bd", "-"),
        ("wor122_entropy_budget_nodiff", "No diffusion, surface heat", "#ff7f0e", "-"),
        ("wor122_entropy_hydro_source_only", "No diffusion, heat, no velocity noise", "#ff7f0e", "--"),
        ("wor122_entropy_hydro_velocity_only", "No diffusion, no heat, velocity noise", "#2ca02c", "--"),
        ("wor122_entropy_hydro_coarse", "No diffusion, no heat or noise", "#2ca02c", "-"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label, color, style in configs:
        cases = data[key]
        time = [case["time_code"] * 0.25465 for case in cases]
        tmode = [case["shell"]["mode_temp_fraction_over_initial"] for case in cases]
        entropy_proxy = [case["shell"]["mode_adiabatic_residual_over_initial"]
                         for case in cases]
        axes[0].plot(time, tmode, style, color=color, marker="o", label=label)
        axes[1].plot(time, entropy_proxy, style, color=color, marker="o")
    axes[0].set_ylabel("Temperature seed mode / initial")
    axes[1].set_ylabel("Adiabatic-residual mode / initial")
    axes[1].set_xlabel("Elapsed time [days]")
    axes[0].set_ylim(0, 1.12)
    axes[1].set_ylim(0, 1.12)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.axhline(1.0, color="black", lw=0.6, alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle("RSG surface-patch entropy-seed retention, 10 solar radii")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
