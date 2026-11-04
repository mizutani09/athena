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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import (effective_problem_bool,
                                     effective_problem_float, input_values,
                                     physical_time_unit, read_history)


def acoustic_one_way_minutes(path: Path) -> float:
    """Initial-profile sound travel time across the simulated vertical layer."""
    with path.open() as source:
        source.readline()
        columns = source.readline().lstrip("# ").split()
    values = np.loadtxt(path)
    z = values[:, columns.index("z_cm")]
    pressure = values[:, columns.index("pressure")]
    density = values[:, columns.index("rho")]
    gamma1 = values[:, columns.index("gamma1")]
    if not (np.all(np.isfinite(z)) and np.all(pressure > 0)
            and np.all(density > 0) and np.all(gamma1 > 0)):
        raise ValueError(f"{path}: invalid acoustic profile")
    order = np.argsort(z)
    sound_speed = np.sqrt(gamma1[order]*pressure[order]/density[order])
    return float(np.trapezoid(1.0/sound_speed, z[order])/60.0)


def vertical_mode_coherence(run_dir: Path, start: float,
                            end: float) -> tuple[float | None, int]:
    """Fraction of vertical-velocity variance in the horizontal mean mode."""
    fractions = []
    for path in sorted(run_dir.glob("*.prim_full.*.athdf")):
        with h5py.File(path, "r") as source:
            time = float(source.attrs["Time"])
            if not start <= time < end:
                continue
            if np.any(source["Levels"][...] != 0):
                raise ValueError(f"{path}: coherence currently requires uniform grid")
            nz = int(source.attrs["RootGridSize"][2])
            names = [v.decode() for v in source.attrs["VariableNames"]]
            vz_index = names.index("vel3")
            blocks = source["prim"][vz_index]
            locations = source["LogicalLocations"][...]
            bz, by, bx = blocks.shape[1:]
            sums = np.zeros(nz)
            squares = np.zeros(nz)
            counts = np.zeros(nz)
            for block, location in zip(blocks, locations):
                z = int(location[2])*bz
                sums[z:z+bz] += np.sum(block, axis=(1, 2))
                squares[z:z+bz] += np.sum(block**2, axis=(1, 2))
                counts[z:z+bz] += bx*by
            mean_z = sums/counts
            fractions.append(float(np.mean(mean_z**2)
                                   / (np.sum(squares)/np.sum(counts))))
    return (float(np.mean(fractions)) if fractions else None,
            len(fractions))


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
    halfway = 0.5*(uniform_t[0] + uniform_t[-1])
    first_std = float(np.std(y[uniform_t < halfway]))
    second_std = float(np.std(y[uniform_t >= halfway]))
    trend = np.polyval(np.polyfit(uniform_t, y, 1), uniform_t)
    power = np.abs(np.fft.rfft((y - trend)*np.hanning(len(y))))**2
    frequency = np.fft.rfftfreq(len(y), dt)
    possible = (frequency >= 1.0/30.0) & (frequency <= 1.0/3.0)
    peak = int(np.argmax(np.where(possible, power, -np.inf)))
    coherence, snapshots = vertical_mode_coherence(run_dir, start, end)
    result = {
        "label": label, "run_dir": str(run_dir),
        "window_start_code": float(t[0]), "window_end_code": float(t[-1]),
        "sample_count": len(t),
        "time_unit_seconds": float(time_unit),
        "dominant_period_code": float(1.0/frequency[peak]),
        "dominant_period_minutes": float(time_unit/frequency[peak]/60.0),
        "bottom_hydro_flux_mean_over_solar": float(np.mean(curves["FhydBottom"])),
        "bottom_hydro_flux_std_over_solar": float(np.std(curves["FhydBottom"])),
        "bottom_hydro_flux_first_half_std_over_solar": first_std,
        "bottom_hydro_flux_second_half_std_over_solar": second_std,
        "bottom_hydro_flux_second_to_first_std_ratio":
            second_std/first_std if first_std > 0 else None,
        "top_flux_mean_over_solar": float(np.mean(curves["FtotTop"])),
        "top_flux_std_over_solar": float(np.std(curves["FtotTop"])),
        "mass_drift_std_percent": float(np.std(curves["MassDrift"])),
        "pressure_scale_std_percent": float(np.std(curves["PBNDScale"])),
        "global_mass_weighted_vz_std_km_s": float(np.std(
            history["3-mom"][selected] / history["mass"][selected])
            * velocity / 1.0e5),
        "horizontal_mean_vz_variance_fraction": coherence,
        "full_volume_snapshot_count": snapshots,
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
    parser.add_argument("--reference-profile", type=Path,
                        help="Initial 1D profile for a rough sound travel time")
    args = parser.parse_args()
    fixed_keys = ("mass_balance_relax_time", "mass_balance_gain",
                  "mass_balance_min_scale", "mass_balance_max_scale", "Cdmp")
    controls = []
    for label, raw_path, _, _ in args.case:
        run_dir = Path(raw_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        overrides = manifest.get("problem_overrides", [])
        source = run_dir / "athinput.generated"
        values = {key: effective_problem_float(source, overrides, key)
                  for key in fixed_keys}
        values["mass_balance_on"] = effective_problem_bool(
            source, overrides, "mass_balance_on")
        values["mass_balance_rate_gain"] = effective_problem_float(
            source, overrides, "mass_balance_rate_gain")
        controls.append((label, values))
    for label, values in controls[1:]:
        for key in (*fixed_keys, "mass_balance_on"):
            if values[key] != controls[0][1][key]:
                raise ValueError(f"{label}: {key}={values[key]} differs from "
                                 f"{controls[0][0]}={controls[0][1][key]}; "
                                 "not a matched gain sweep")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for (label, path, start, end), (_, settings) in zip(args.case, controls):
        result, data = analyze(label, Path(path), float(start), float(end),
                               args.header_from, args.reference_flux)
        result.update(settings)
        rows.append(result)
        curves.append(data)
    acoustic_minutes = (acoustic_one_way_minutes(args.reference_profile)
                        if args.reference_profile else None)
    if acoustic_minutes is not None:
        for row in rows:
            row["initial_acoustic_one_way_minutes"] = acoustic_minutes
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
            f"{row['mass_balance_rate_gain']:.0f}",
            f"{row['mass_balance_relax_time']:.1f}",
            f"{row['window_start_code']:.1f}–{row['window_end_code']:.1f}",
            f"{row['dominant_period_code']:.2f}",
            f"{row['dominant_period_minutes']:.2f}",
            f"{row['bottom_hydro_flux_std_over_solar']:.1f}",
            f"{row['bottom_hydro_flux_second_to_first_std_ratio']:.2f}",
            f"{row['mass_drift_std_percent']:.3f}",
            f"{row['pressure_scale_std_percent']:.3f}",
            f"{row['global_mass_weighted_vz_std_km_s']:.3f}",
            (f"{row['horizontal_mean_vz_variance_fraction']:.3f} "
             f"({row['full_volume_snapshot_count']} dumps)"
             if row["horizontal_mean_vz_variance_fraction"] is not None
             else "—"),
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
        + (f"<p>Initial 1D sound travel time across the layer: "
           f"<strong>{acoustic_minutes:.2f} min one-way</strong>. "
           "This estimate does not identify an eigenmode.</p>"
           if acoustic_minutes is not None else "") +
        "<table><tr><th>Case</th><th>Rate gain</th>"
        "<th>Relax time [code]</th><th>Window</th><th>Period [code]</th>"
        "<th>Period [min]</th><th>Bottom flux std / solar</th>"
        "<th>Late/early flux amplitude</th>"
        "<th>Mass drift std [%]</th><th>PBND scale std [%]</th>"
        "<th>Global vz std [km/s]</th>"
        "<th>Mean-vz variance fraction</th></tr>"
        f"{cells}</table><img src='oscillation_comparison.png' alt='oscillation'>"
        "<p><a href='summary.json'>Numerical results</a></p></html>"
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
