#!/usr/bin/env python3
"""Quantify coherent vertical motion in a stitched solar-convection movie."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from plot_simple_convection_movies import read_plane  # noqa: E402
from plot_simple_convection_tau import opacity_interpolator  # noqa: E402
from solar_convection_campaign_movies import collect, load  # noqa: E402
from solar_convection_report import input_values, physical_time_unit  # noqa: E402


def harmonic_amplitude(time: np.ndarray, values: np.ndarray,
                       period: float) -> float:
    angle = 2*np.pi*time/period
    matrix = np.column_stack((np.ones(len(time)), time - np.mean(time),
                              np.sin(angle), np.cos(angle)))
    coefficients = np.linalg.lstsq(matrix, values, rcond=None)[0]
    return float(np.hypot(coefficients[2], coefficients[3]))


def dominant_period(time: np.ndarray, values: np.ndarray) -> float:
    step = float(np.median(np.diff(time)))
    uniform_time = np.arange(time[0], time[-1] + 0.5*step, step)
    series = np.interp(uniform_time, time, values)
    series -= np.polyval(np.polyfit(uniform_time, series, 1), uniform_time)
    frequency = np.fft.rfftfreq(len(series), step)
    power = np.abs(np.fft.rfft(series * np.hanning(len(series))))**2
    candidate = (frequency >= 1/30) & (frequency <= 1/8)
    return float(1/frequency[np.argmax(np.where(candidate, power, -np.inf))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--extra-run", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True,
                        help="Rosseland opacity table used to locate tau=1")
    parser.add_argument("--start-time", type=float, default=800.0)
    parser.add_argument("--end-time", type=float, default=1600.0)
    parser.add_argument("--window-width", type=float, default=100.0)
    parser.add_argument("--period-code", type=float,
                        help="Fit the same known period in every window; "
                             "useful after the coherent mode becomes weak")
    args = parser.parse_args()
    if not np.isfinite(args.window_width) or args.window_width <= 0:
        parser.error("--window-width must be finite and positive")
    span = (args.end_time - args.start_time) / args.window_width
    if (not np.isfinite(span) or span < 1
            or not np.isclose(span, round(span))):
        parser.error("Time range must contain whole, positive windows")
    if args.period_code is not None and (not np.isfinite(args.period_code)
                                         or args.period_code <= 0):
        parser.error("--period-code must be finite and positive")
    summary = json.loads(args.source_summary.read_text())
    runs = [Path(item).resolve() for item in summary["source_runs"]]
    runs += [item.resolve() for item in args.extra_run]
    units = input_values(runs[0] / "athinput.generated")
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    opacity = opacity_interpolator(args.opacity_table.resolve())
    surface_rows = []
    for item in collect(runs, "xz", end_time=args.end_time):
        if item.time < args.start_time:
            continue
        frame = load(item, "xz", velocity_unit, units["leng_unit"],
                     units["rho_unit"], opacity)
        height_km = np.asarray(frame["tau1_height_mm"]) * 1000.0
        if not np.isfinite(height_km).all():
            raise ValueError(f"Incomplete tau=1 slice at {item.time}")
        surface_rows.append((item.time, float(np.mean(height_km)),
                             float(np.std(height_km))))
    surface = np.asarray(surface_rows)

    plane_rows = []
    for item in collect(runs, "xy", end_time=args.end_time):
        if item.time < args.start_time:
            continue
        prim, _, _ = read_plane(item.prim, "prim", "xy")
        velocity = prim[4] * velocity_unit / 1e5
        plane_rows.append((item.time, float(np.mean(velocity)),
                           float(np.std(velocity)),
                           float(np.mean(velocity**2))))
    plane = np.asarray(plane_rows)

    late_start = args.end_time - args.window_width
    late_surface = surface[(surface[:, 0] >= late_start) &
                           (surface[:, 0] <= args.end_time)]
    late_plane = plane[(plane[:, 0] >= late_start) &
                       (plane[:, 0] <= args.end_time)]
    if len(late_surface) < 20 or len(late_plane) < 20:
        raise ValueError("Need at least 20 samples in the final window")
    late_peak = dominant_period(late_surface[:, 0], late_surface[:, 1])
    period = args.period_code if args.period_code is not None else late_peak
    windows = []
    for index in range(round(span)):
        start = args.start_time + index*args.window_width
        end = start + args.window_width
        z = surface[(surface[:, 0] >= start) & (surface[:, 0] < end)]
        v = plane[(plane[:, 0] >= start) & (plane[:, 0] < end)]
        if len(z) < 20 or len(v) < 20:
            raise ValueError(f"Too few samples in t={start}–{end}")
        windows.append({
            "start_time_code": start, "end_time_code": end,
            "tau1_mean_height_km": float(np.mean(z[:, 1])),
            "tau1_height_std_km": float(np.std(z[:, 1])),
            "tau1_harmonic_amplitude_km": harmonic_amplitude(
                z[:, 0], z[:, 1], period),
            "tau1_horizontal_corrugation_mean_km": float(np.mean(z[:, 2])),
            "xy_mean_vz_std_km_s": float(np.std(v[:, 1])),
            "xy_mean_vz_harmonic_amplitude_km_s": harmonic_amplitude(
                v[:, 0], v[:, 1], period),
            "xy_velocity_fluctuation_rms_km_s": float(np.mean(v[:, 2])),
            "xy_coherent_velocity_energy_fraction": float(np.mean(
                v[:, 1]**2 / v[:, 3])),
        })

    dzdt = np.gradient(late_surface[:, 1], late_surface[:, 0]*time_unit)
    optical_speed = np.interp(late_plane[:, 0], late_surface[:, 0], dzdt)
    correlation = float(np.corrcoef(late_plane[:, 1], optical_speed)[0, 1])
    result = {
        "description": "One xz-slice Rosseland tau=1 surface and one fixed-z "
                       "xy-slice mean velocity; correlation does not prove cause",
        "source_runs": [str(run) for run in runs],
        "tau1_opacity_table": str(args.opacity_table.resolve()),
        "time_unit_seconds": time_unit,
        "analyzed_start_code": args.start_time,
        "analyzed_end_code": args.end_time,
        "window_width_code": args.window_width,
        "late_period_code": period,
        "late_period_minutes": period*time_unit/60,
        "late_spectral_peak_code": late_peak,
        "fit_period_origin": ("fixed_reference" if args.period_code is not None
                              else "late_spectral_peak"),
        "late_xy_mean_vz_to_tau1_height_speed_correlation": correlation,
        "late_tau1_height_range_km": [float(np.min(late_surface[:, 1])),
                                      float(np.max(late_surface[:, 1]))],
        "windows": windows,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                             constrained_layout=True)
    axes[0].plot(surface[:, 0], surface[:, 1], lw=1)
    axes[0].set(ylabel="Mean τ=1 height [km]", title="One xz slice")
    axes[1].plot(plane[:, 0], plane[:, 1], lw=1)
    axes[1].set(ylabel="XY mean $v_z$ [km/s]", title="Fixed height near photosphere")
    axes[2].plot(plane[:, 0], plane[:, 1]**2 / plane[:, 3], lw=1)
    axes[2].set(xlabel="Time [code]", ylabel="Coherent velocity-energy fraction")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output / "surface_mode_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    ax.plot(late_plane[:, 0], late_plane[:, 1], label="XY mean $v_z$")
    ax.plot(late_plane[:, 0], optical_speed, label="d(z at τ=1)/dt", alpha=0.85)
    ax.set(xlabel="Time [code]", ylabel="Speed [km/s]",
           title=f"Late coherent motion; r={correlation:.2f}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "late_surface_speed.png", dpi=160)
    plt.close(fig)

    table = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in (
            f"{w['start_time_code']}–{w['end_time_code']}",
            f"{w['tau1_harmonic_amplitude_km']:.1f}",
            f"{w['xy_mean_vz_harmonic_amplitude_km_s']:.3f}",
            f"{100*w['xy_coherent_velocity_energy_fraction']:.1f}%",
            f"{w['tau1_horizontal_corrugation_mean_km']:.1f}",
        )) + "</tr>" for w in windows)
    (output / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Solar vertical surface mode</title>"
        "<style>body{font:22px/1.55 system-ui;max-width:1200px;"
        "margin:2rem auto;padding:0 1rem;color:#172b3a}"
        "h1{font-size:2.2rem}img{max-width:100%}"
        "table{border-collapse:collapse}th,td{padding:.5rem .7rem;"
        "border-bottom:1px solid #ddd;text-align:left}</style>"
        "<h1>Coherent vertical motion in the solar box</h1>"
        f"<p><strong>{'Reference fit period' if args.period_code is not None else 'Late spectral peak'}: "
        f"{period*time_unit/60:.2f} min.</strong> "
        "The mean motion and optical-depth-one surface both oscillate. "
        "Their amplitudes decline over this run; the physical or numerical "
        "excitation mechanism remains under investigation.</p>"
        "<p>The surface height is measured in one xz slice; the velocity "
        "is the mean of one fixed-height xy slice. Optical-depth surfaces "
        "can move because of thermodynamic and opacity changes as well as "
        "fluid motion.</p>"
        f"<p><strong>Late velocity/height-speed correlation:</strong> "
        f"{correlation:.2f}. "
        "<a href='summary.json'>JSON</a> · "
        "<a href='../index.html'>Campaign movies and runs</a></p>"
        "<div style='overflow-x:auto'><table><tr><th>Time [code]</th>"
        "<th>τ=1 amplitude [km]</th>"
        "<th>XY mean <i>v</i><sub>z</sub> amplitude [km/s]</th>"
        "<th>Coherent velocity-energy fraction</th>"
        "<th>Mean horizontal τ=1 corrugation [km]</th></tr>"
        f"{table}</table></div>"
        "<figure><img src='surface_mode_history.png' alt='surface history'></figure>"
        "<figure><img src='late_surface_speed.png' alt='late surface speed'></figure>"
        "</html>"
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
