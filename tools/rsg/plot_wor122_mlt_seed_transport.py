#!/usr/bin/env python3
"""Summarize the matched Henyey-MLT local seed and boundary tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "full": "full FLD, open top",
    "nodiff": "FLD diffusion off",
    "closedtop": "full FLD, nearly closed top",
    "fixedtop": "full FLD, fixed-T top",
}


def load_case(run_dir: Path) -> tuple[str, list[dict]]:
    name = run_dir.name.removeprefix("wor122_mlt_seed_")
    transport = json.loads((run_dir / "transport.json").read_text())
    modes = json.loads((run_dir / "mode.json").read_text())[run_dir.name]
    snaps = transport["snapshots"]
    if len(snaps) != len(modes):
        raise ValueError(f"snapshot count differs for {run_dir}")
    rows = []
    for snap, mode in zip(snaps, modes):
        if abs(snap["time_code"] - mode["time_code"]) > 1e-5:
            raise ValueError(f"snapshot time differs for {run_dir}")
        transition = snap["shells"]["813-817"]
        outer = snap["shells"]["850-860"]
        rows.append({
            "time_code": snap["time_code"],
            "time_days": snap["time_code"] * 6.111101578275392 / 24.0,
            "transition_T_over_initial": transition["Tgas_over_initial"],
            "outer_T_over_initial": outer["Tgas_over_initial"],
            "temperature_mode_over_initial": mode["shell"]["mode_temp_fraction_over_initial"],
            "convective_proxy_over_Fstar": transition["Fenth_mass_corrected_over_Fstar"],
            "transition_max_Mach": transition["max_mach"],
            "mass_drift": snap["mass_drift_from_hdf"],
        })
    return name, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-figure", required=True, type=Path)
    args = parser.parse_args()

    cases = dict(load_case(run_dir) for run_dir in args.run_dirs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"cases": cases}, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    fields = (
        ("transition_T_over_initial", "T / T₀, 813–817 R☉"),
        ("temperature_mode_over_initial", "Temperature seed amplitude / initial"),
        ("outer_T_over_initial", "T / T₀, 850–860 R☉"),
    )
    for name in LABELS:
        if name not in cases:
            continue
        rows = cases[name]
        for ax, (field, ylabel) in zip(axes, fields):
            ax.plot([r["time_days"] for r in rows], [r[field] for r in rows],
                    marker="o", label=LABELS[name])
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time [days]")
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
