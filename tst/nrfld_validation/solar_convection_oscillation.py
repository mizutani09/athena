#!/usr/bin/env python3
"""Compare the lower-boundary oscillation in matched solar-convection windows.

Each --case is LABEL RUN_DIR START_CODE END_CODE. The spectral peak is a
diagnostic of periodic behavior, not proof of its physical or numerical cause.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import input_values, physical_time_unit, read_history


def analyze(label: str, run_dir: Path, start: float, end: float,
            header: Path, flux_ref: float) -> tuple[dict, dict]:
    history = read_history(next(run_dir.glob("*.hst")), header)
    selected = (history["time"] >= start) & (history["time"] < end)
    t = history["time"][selected]
    if len(t) < 20:
        raise ValueError(f"{label}: need at least 20 history samples in window")
    units = input_values(run_dir / "athinput.generated")
    velocity = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity)
    dt = float(np.median(np.diff(t)))
    uniform_t = np.arange(t[0], t[-1] + 0.5*dt, dt)
    curves = {}
    for key, scale in (("FhydBottom", flux_ref), ("FtotTop", flux_ref),
                       ("MassDrift", 0.01), ("PBNDScale", 0.01)):
        curves[key] = np.interp(uniform_t, t, history[key][selected]) / scale
    y = curves["FhydBottom"]
    trend = np.polyval(np.polyfit(uniform_t, y, 1), uniform_t)
    power = np.abs(np.fft.rfft((y - trend)*np.hanning(len(y))))**2
    frequency = np.fft.rfftfreq(len(y), dt)
    possible = (frequency >= 1.0/30.0) & (frequency <= 1.0/3.0)
    peak = int(np.argmax(np.where(possible, power, -np.inf)))
    result = {
        "label": label, "run_dir": str(run_dir),
        "window_start_code": float(t[0]), "window_end_code": float(t[-1]),
        "sample_count": len(t),
        "time_unit_seconds": float(time_unit),
        "dominant_period_code": float(1.0/frequency[peak]),
        "dominant_period_minutes": float(time_unit/frequency[peak]/60.0),
        "bottom_hydro_flux_mean_over_solar": float(np.mean(curves["FhydBottom"])),
        "bottom_hydro_flux_std_over_solar": float(np.std(curves["FhydBottom"])),
        "top_flux_mean_over_solar": float(np.mean(curves["FtotTop"])),
        "top_flux_std_over_solar": float(np.std(curves["FtotTop"])),
        "mass_drift_std_percent": float(np.std(curves["MassDrift"])),
        "pressure_scale_std_percent": float(np.std(curves["PBNDScale"])),
    }
    return result, {"time": uniform_t - uniform_t[0], "curves": curves,
                    "period": np.divide(1.0, frequency, out=np.full_like(
                        frequency, np.inf), where=frequency > 0),
                    "power": power}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=4, action="append", required=True,
                        metavar=("LABEL", "RUN_DIR", "START", "END"))
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-flux", type=float, default=6.3e10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for label, path, start, end in args.case:
        result, data = analyze(label, Path(path), float(start), float(end),
                               args.header_from, args.reference_flux)
        rows.append(result)
        curves.append(data)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row, data in zip(rows, curves):
        label = row["label"]
        t = data["time"]
        axes[0, 0].plot(t, data["curves"]["FhydBottom"], label=label)
        axes[0, 1].plot(t, data["curves"]["MassDrift"], label=label)
        axes[1, 0].plot(t, data["curves"]["PBNDScale"] - 100.0, label=label)
        period = data["period"]
        power = data["power"]
        keep = (period >= 3) & (period <= 30)
        axes[1, 1].plot(period[keep], power[keep]/max(np.sum(power[keep]),
                                                    1e-100), label=label)
    axes[0, 0].set(title="Bottom hydro energy flux", ylabel=r"$F/F_\odot$")
    axes[0, 1].set(title="Total mass drift", ylabel="%")
    axes[1, 0].set(title="Lower-boundary pressure scale", ylabel="deviation from 1 [%]")
    axes[1, 1].set(title="Bottom-flux periodogram", ylabel="fraction of selected power",
                   xlim=(3, 30))
    for axis in axes.flat:
        axis.set_xlabel("code time since window start" if axis is not axes[1, 1]
                        else "period [code time]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=10)
    fig.savefig(args.output_dir / "oscillation_comparison.png", dpi=150)
    plt.close(fig)
    cells = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in (
            row["label"],
            f"{row['window_start_code']:.1f}–{row['window_end_code']:.1f}",
            f"{row['dominant_period_code']:.2f}",
            f"{row['dominant_period_minutes']:.2f}",
            f"{row['bottom_hydro_flux_std_over_solar']:.1f}",
            f"{row['mass_drift_std_percent']:.3f}",
            f"{row['pressure_scale_std_percent']:.3f}",
        )) + "</tr>" for row in rows)
    (args.output_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar convection oscillation</title><style>"
        "body{font:19px/1.5 system-ui;max-width:1200px;margin:2rem auto;"
        "padding:0 1rem;color:#172b3a}table{border-collapse:collapse}"
        "th,td{padding:.5rem .75rem;border-bottom:1px solid #ddd;text-align:left}"
        "img{max-width:100%}</style><h1>Lower-boundary oscillation diagnostic</h1>"
        "<p>The spectral peak establishes a periodic signal; it does not "
        "identify its cause. Flux amplitude uses the solar reference flux.</p>"
        "<table><tr><th>Case</th><th>Window</th><th>Period [code]</th>"
        "<th>Period [min]</th><th>Bottom flux std / solar</th>"
        "<th>Mass drift std [%]</th><th>PBND scale std [%]</th></tr>"
        f"{cells}</table><img src='oscillation_comparison.png' alt='oscillation'>"
        "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
