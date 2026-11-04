#!/usr/bin/env python3
"""Compare two solar-convection runs at matched physical times."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import read_history


def load_case(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray]]:
    history = read_history(next(path.glob("*.hst")))
    summary = json.loads((path / "summary.json").read_text())
    tau = json.loads((path / "tau1_slice.json").read_text())["records"]
    tau_series = {
        "time_code": np.asarray([record["time_code"] for record in tau]),
        "T_tau23_mean_K": np.asarray([record["T_tau23_mean_K"] for record in tau]),
    }
    if not summary["log"]["completed"]:
        raise ValueError(f"Run is incomplete: {path}")
    return history, summary, tau_series


def early_metrics(h: dict[str, np.ndarray], summary: dict[str, object],
                  tau: dict[str, np.ndarray]) -> dict[str, float]:
    early = (h["time"] >= 0) & (h["time"] < 2)
    late = (h["time"] >= 8) & (h["time"] <= 10)
    if early.sum() < 5 or late.sum() < 5:
        raise ValueError("Comparison requires history through t=10")
    j = int(np.argmin(abs(h["time"] - 0.5)))
    return {
        "net_at_t05_code": float(h["BottomMdotNet"][j]),
        "mass_at_t05_percent": float(100 * h["MassDrift"][j]),
        "bottom_flux_at_t05_solar": float(h["FhydBottom"][j] / 6.3e10),
        "net_rms_t0_2_code": float(np.sqrt(np.mean(h["BottomMdotNet"][early]**2))),
        "max_mass_drift_t0_2_percent": float(100 * np.max(abs(h["MassDrift"][early]))),
        "late_top_flux_solar": float(np.mean(h["FtotTop"][late]) / 6.3e10),
        "last_tau23_T_K": float(tau["T_tau23_mean_K"][-1]),
        "cycles": int(summary["log"]["final_cycle"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="HD2 boundary initialization comparison")
    args = parser.parse_args()
    old = load_case(args.old)
    new = load_case(args.new)
    data = {"old": old, "fixed": new}
    metrics = {label: early_metrics(*values) for label, values in data.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for label, (h, summary, tau) in data.items():
        t = h["time"]
        ax[0, 0].plot(t, h["BottomMdotNet"], label=label)
        ax[0, 1].plot(t, 100 * h["MassDrift"], label=label)
        ax[1, 0].plot(t, h["FtotTop"] / 6.3e10, label=label)
        ax[1, 1].plot(tau["time_code"], tau["T_tau23_mean_K"], label=label)
    ax[0, 0].set(ylabel="bottom net mass flux [code]", xlim=(0, 10))
    ax[0, 1].set(ylabel="mass drift [%]", xlim=(0, 10))
    ax[1, 0].set(ylabel="top flux / solar", xlabel="time [code]", xlim=(0, 10))
    ax[1, 1].set(ylabel="slice T at tau=2/3 [K]", xlabel="time [code]",
                 xlim=(0, 10))
    for axis in ax.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(args.title)
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)

    names = {
        "net_at_t05_code": "Bottom net flux near t=0.5 [code]",
        "mass_at_t05_percent": "Mass drift near t=0.5 [%]",
        "bottom_flux_at_t05_solar": "Bottom hydro flux near t=0.5 / solar",
        "net_rms_t0_2_code": "Bottom net flux RMS, t=0–2 [code]",
        "max_mass_drift_t0_2_percent": "Maximum |mass drift|, t=0–2 [%]",
        "late_top_flux_solar": "Mean top flux, t=8–10 / solar",
        "last_tau23_T_K": "Last xz-slice T at tau=2/3 [K]",
        "cycles": "Completed cycles",
    }
    rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{metrics['old'][key]:.4g}</td>"
        f"<td>{metrics['fixed'][key]:.4g}</td></tr>"
        for key, name in names.items())
    links = " ".join(
        f"<a href='{html.escape(os.path.relpath(path / 'index.html', args.output))}'>"
        f"{label} case</a>" for label, path in (("Old", args.old), ("Fixed", args.new)))
    (args.output / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        f"<title>{html.escape(args.title)}</title>"
        "<style>body{font:19px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}h1{font-size:2rem}img{max-width:100%}"
        "table{border-collapse:collapse}th,td{padding:.5rem .8rem;"
        "border-bottom:1px solid #ddd;text-align:left}</style>"
        f"<h1>{html.escape(args.title)}</h1>"
        "<p><strong>Same grid, physics tables, CFL and MPI placement.</strong>"
        " These are startup diagnostics; neither run is statistically steady.</p>"
        f"<p>{links} · <a href='comparison.json'>JSON</a></p>"
        f"<table><tr><th>Metric</th><th>Old</th><th>Fixed</th></tr>{rows}</table>"
        "<figure><img src='comparison.png' alt='comparison'></figure></html>"
    )
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
