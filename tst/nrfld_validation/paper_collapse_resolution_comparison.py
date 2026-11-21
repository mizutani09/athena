#!/usr/bin/env python3
"""Compare two leaf-block collapse snapshots at the same density regime."""

import argparse
import csv
import html
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AU_CM = 1.495978707e13
MSUN_G = 1.98847e33


def read_case(path, edges):
    with h5py.File(path) as hdf:
        names = [value.decode() for value in hdf.attrs["VariableNames"]]
        data = hdf["hydro"]

        def field(name):
            return np.asarray(data[names.index(name)], dtype=np.float64)

        rho = field("rho_cgs")
        temperature = field("Tgas_K")
        pressure = field("press")
        density_code = field("dens")
        velocity = [field("vel{}".format(i)) for i in (1, 2, 3)]
        peak = np.unravel_index(np.argmax(rho), rho.shape)
        center = np.array([
            hdf["x1v"][peak[0], peak[3]],
            hdf["x2v"][peak[0], peak[2]],
            hdf["x3v"][peak[0], peak[1]],
        ])
        x1v = np.asarray(hdf["x1v"][:], dtype=np.float64)
        x2v = np.asarray(hdf["x2v"][:], dtype=np.float64)
        x3v = np.asarray(hdf["x3v"][:], dtype=np.float64)
        x = x1v[:, None, None, :]
        y = x2v[:, None, :, None]
        z = x3v[:, :, None, None]
        dx, dy, dz = x-center[0], y-center[1], z-center[2]
        radius = np.broadcast_to(np.sqrt(dx*dx + dy*dy + dz*dz), rho.shape)
        vr = ((velocity[0]*dx + velocity[1]*dy + velocity[2]*dz)
              / np.maximum(radius, 1.0e-30))
        speed_ratio = vr / np.sqrt(pressure/density_code)

        dx1 = np.diff(hdf["x1f"][:], axis=1)[:, None, None, :]
        dx2 = np.diff(hdf["x2f"][:], axis=1)[:, None, :, None]
        dx3 = np.diff(hdf["x3f"][:], axis=1)[:, :, None, None]
        volume_au3 = np.broadcast_to(dx1*dx2*dx3, rho.shape)
        mass_g = rho * volume_au3 * AU_CM**3
        levels = np.asarray(hdf["Levels"][:], dtype=np.int64)
        cycle = int(hdf.attrs["NumCycles"])
        time = float(hdf.attrs["Time"])

    flat_r = radius.ravel()
    flat_volume = volume_au3.ravel()
    indices = np.searchsorted(edges, flat_r)-1
    valid = (indices >= 0) & (indices < len(edges)-1)
    indices, weights = indices[valid], flat_volume[valid]
    volume = np.bincount(indices, weights=weights, minlength=len(edges)-1)

    def average(values):
        total = np.bincount(indices, weights=values.ravel()[valid]*weights,
                            minlength=len(edges)-1)
        return np.divide(total, volume, out=np.full_like(total, np.nan),
                         where=volume > 0.0)

    profile = {
        "rho": average(rho),
        "temperature": average(temperature),
        "vr_scale": average(speed_ratio),
        "volume": volume,
    }
    metrics = {
        "cycle": cycle,
        "time_code": time,
        "blocks": int(len(levels)),
        "level_index_max": int(levels.max()),
        "peak_xyz_au": center.tolist(),
        "rho_peak_g_cm3": float(rho[peak]),
        "T_at_density_peak_K": float(temperature[peak]),
    }
    for radius_au in (5.0, 10.0, 20.0):
        mask = radius <= radius_au
        metrics["mass_within_{}au_Msun".format(int(radius_au))] = float(
            mass_g[mask].sum()/MSUN_G)
    return metrics, profile


def crossing_radius(mid, values, threshold):
    finite = np.isfinite(values)
    candidates = np.where(finite & (values < threshold))[0]
    candidates = candidates[candidates > np.where(finite)[0][0]] if candidates.size else candidates
    if not candidates.size:
        return None
    index = int(candidates[0])
    return float(mid[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    edges = np.geomspace(0.08, 100.0, 72)
    mid = np.sqrt(edges[:-1]*edges[1:])
    cases = {}
    profiles = {}
    for label, path in (("low", args.low), ("high", args.high)):
        cases[label], profiles[label] = read_case(path, edges)
        cases[label]["source_hdf"] = str(path.resolve())
        cases[label]["rho_1e-12_radius_au"] = crossing_radius(
            mid, profiles[label]["rho"], 1.0e-12)

    comparison = {}
    for key in ("rho_peak_g_cm3", "T_at_density_peak_K",
                "mass_within_5au_Msun", "mass_within_10au_Msun",
                "mass_within_20au_Msun", "rho_1e-12_radius_au"):
        low, high = cases["low"][key], cases["high"][key]
        comparison[key] = {"low": low, "high": high,
                           "relative_change": high/low-1.0}
    comparison["time_code_difference"] = (
        cases["high"]["time_code"]-cases["low"]["time_code"])
    for field in ("rho", "temperature", "vr_scale"):
        low, high = profiles["low"][field], profiles["high"][field]
        valid = np.isfinite(low) & np.isfinite(high)
        comparison[field+"_profile_relative_l1"] = float(
            np.sum(np.abs(high[valid]-low[valid]))
            / np.sum(np.abs(low[valid])))

    summary = {"status": "passed", "cases": cases, "comparison": comparison}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (args.output/"profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["radius_au", "rho_low", "rho_high", "T_low", "T_high",
                         "vr_scale_low", "vr_scale_high"])
        for index, radius in enumerate(mid):
            writer.writerow([radius, profiles["low"]["rho"][index],
                             profiles["high"]["rho"][index],
                             profiles["low"]["temperature"][index],
                             profiles["high"]["temperature"][index],
                             profiles["low"]["vr_scale"][index],
                             profiles["high"]["vr_scale"][index]])

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for label, color in (("low", "#d95f02"), ("high", "#1b9e77")):
        name = "AMR level index {}".format(cases[label]["level_index_max"])
        axes[0].plot(mid, profiles[label]["rho"], color=color, label=name)
        axes[1].plot(mid, profiles[label]["temperature"], color=color)
        axes[2].plot(mid, profiles[label]["vr_scale"], color=color)
    axes[0].set(xscale="log", yscale="log", xlabel="Radius [au]",
                ylabel="Volume-mean density [g cm$^{-3}$]")
    axes[1].set(xscale="log", yscale="log", xlabel="Radius [au]",
                ylabel="Volume-mean gas temperature [K]")
    axes[2].set(xscale="log", xlabel="Radius [au]",
                ylabel=r"$v_r / \sqrt{P/\rho}$")
    axes[2].axhline(0.0, color="0.5", lw=0.8)
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0.3, 30.0)
    figure.savefig(args.output/"resolution_profiles.svg")
    figure.savefig(args.output/"resolution_profiles.png", dpi=180)
    plt.close(figure)

    rows = "".join(
        "<tr><th>{}</th><td>{:.8g}</td><td>{:.8g}</td><td>{:+.3%}</td></tr>".format(
            html.escape(key), value["low"], value["high"], value["relative_change"])
        for key, value in comparison.items() if isinstance(value, dict))
    page = """<!doctype html><meta charset='utf-8'><title>Collapse resolution comparison</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}}table{{border-collapse:collapse}}th,td{{padding:.35rem .7rem;border-bottom:1px solid #ccc;text-align:right}}th{{text-align:left}}img{{width:100%}}code{{background:#eee;padding:.1rem .25rem}}</style>
<h1>Protostellar-collapse resolution comparison</h1>
<p>Both snapshots are centered independently on their density maximum and compared in the same radial bins. The low and high snapshots have peak densities within one percent.</p>
<table><thead><tr><th>Metric</th><th>low</th><th>high</th><th>change</th></tr></thead><tbody>{}</tbody></table>
<p><img src='resolution_profiles.svg' alt='Radial resolution comparison'></p>
<p><a href='profiles.csv'>profiles.csv</a> · <a href='summary.json'>summary.json</a> ·
<a href='../additional_core_analysis/index.html'>additional core diagnostics</a> ·
<a href='../movies/index.html'>collapse movies</a> ·
<a href='../solver_audit.json'>solver audit</a> ·
<a href='../final_energy_audit.json'>energy audit</a> ·
<a href='../progress.csv'>segment progress</a></p>""".format(rows)
    (args.output/"index.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
