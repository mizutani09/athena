#!/usr/bin/env python3
"""Compare matched-density positive-x first-core profiles with the old control."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def arrays(rows, x_key, rho_key, temp_key, velocity_key):
    x = np.array([float(row[x_key]) for row in rows])
    mask = (x >= 0.0) & (x <= 25.0)
    return (x[mask], np.array([float(row[rho_key]) for row in rows])[mask],
            np.array([float(row[temp_key]) for row in rows])[mask],
            np.array([float(row[velocity_key]) for row in rows])[mask])


def compare(control, table, output):
    base = arrays(read_csv(control), "x_au", "rho_g_cm3", "Tgas_K", "vr_km_s")
    trial = arrays(read_csv(table), "x_au", "rho_g_cm3", "temperature_K", "vx_km_s")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for values, label, color, linestyle in (
            (base, "Ideal gas + fixed opacity; baseline", "#bd6828", "--"),
            (trial, "Tomida EOS + Semenov; WOR-92", "#086b83", "-")):
        x, rho, temp, vr = values
        axes[0].plot(x, rho, label=label, color=color, ls=linestyle)
        axes[1].plot(x, temp, color=color, ls=linestyle)
        axes[2].plot(x, -vr, color=color, ls=linestyle)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Gas density [g cm$^{-3}$]")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("Gas temperature [K]")
    axes[2].set_ylabel("Infall speed [km/s]")
    axes[2].set_xlabel("Positive x-axis radius [au]")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 25)
    fig.suptitle(r"First-core axis profiles at $\rho_{\max}\simeq10^{-10}$ g cm$^{-3}$")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    summary = {
        "control_csv": str(control), "table_csv": str(table),
        "control_time_code": float(read_csv(control)[0]["time_code"]),
        "control_max_rho_g_cm3": float(np.max(base[1])),
        "table_max_rho_g_cm3_on_axis": float(np.max(trial[1])),
        "control_max_axis_temperature_K": float(np.max(base[2])),
        "table_max_axis_temperature_K": float(np.max(trial[2])),
        "control_min_dx_au": min(float(row["dx_au"]) for row in read_csv(control)),
        "table_min_positive_x_spacing_au": float(np.min(np.diff(trial[0]))),
        "plot": str(output),
        "limitation": "Different EOS, opacity, AMR resolution, and execution paths; not a one-variable causal comparison",
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("wor92_axis_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline_csv, args.wor92_axis_csv,
                             args.output_png), indent=2))


if __name__ == "__main__":
    main()
