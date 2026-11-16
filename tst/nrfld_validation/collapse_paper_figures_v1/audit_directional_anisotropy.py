#!/usr/bin/env python3
"""Measure xy-plane axis/diagonal differences from native c1080 AMR cells."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tst/nrfld_validation/wor92_firstcore_20261109"))
from analyze_core_profile import eos_temperature  # noqa: E402

RHO_UNIT = 1e-18
PRESS_UNIT = 1e-9
A_RAD = 7.5657e-15
HDF_SHA = "13767189008f53ffb9e802b7ef66dcd97faff7045adbdd59051d19269a0354ac"
EOS_SHA = "742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_plane(hdf):
    columns = {name: [] for name in ("x", "y", "z", "xl", "xr", "yl", "yr",
                                     "area", "level", "rho", "press", "er")}
    with h5py.File(hdf) as h:
        names = [v.decode() for v in h.attrs["VariableNames"]]
        prim_names = names[:int(h.attrs["NumVariables"][0])]
        if int(h.attrs["NumCycles"]) != 1080:
            raise ValueError("expected c1080")
        for block in range(h["prim"].shape[1]):
            zf = np.asarray(h["x3f"][block], dtype=float)
            if not (zf[0] <= 0 < zf[-1]):
                continue
            k = int(np.searchsorted(zf, 0, side="right") - 1)
            xf = np.asarray(h["x1f"][block], dtype=float)
            yf = np.asarray(h["x2f"][block], dtype=float)
            if xf[-1] < -32 or xf[0] > 32 or yf[-1] < -32 or yf[0] > 32:
                continue
            x, y = np.meshgrid((xf[1:] + xf[:-1]) / 2,
                               (yf[1:] + yf[:-1]) / 2)
            xl, yl = np.meshgrid(xf[:-1], yf[:-1])
            xr, yr = np.meshgrid(xf[1:], yf[1:])
            area = np.outer(np.diff(yf), np.diff(xf))
            values = {
                "x": x, "y": y, "z": np.full_like(x, (zf[k] + zf[k + 1]) / 2),
                "xl": xl, "xr": xr, "yl": yl, "yr": yr,
                "area": area,
                "level": np.full_like(x, int(h["Levels"][block])),
                "rho": np.asarray(h["prim"][prim_names.index("rho"), block, k]),
                "press": np.asarray(h["prim"][prim_names.index("press"), block, k]),
                "er": np.asarray(h["prim"][prim_names.index("Er"), block, k]),
            }
            for name, value in values.items():
                columns[name].append(value.ravel())
    result = {name: np.concatenate(parts) for name, parts in columns.items()}
    result["rho"] *= RHO_UNIT
    result["press"] *= PRESS_UNIT
    result["er"] *= PRESS_UNIT
    return result


def calculate(data, eos, angular_half_width_deg=10):
    data["Tgas"], outside = eos_temperature(eos, data["rho"], data["press"])
    if outside:
        raise ValueError("EOS table domain exceeded")
    data["Trad"] = (data["er"] / A_RAD) ** 0.25
    r = np.hypot(data["x"], data["y"])
    angle = np.arctan2(abs(data["y"]), abs(data["x"]))
    tolerance = np.deg2rad(angular_half_width_deg)
    axis = np.minimum(angle, np.pi / 2 - angle) <= tolerance
    diagonal = abs(angle - np.pi / 4) <= tolerance
    rows = []
    for lo in np.arange(1, 31, 0.5):
        hi = lo + 0.5
        ring = (r >= lo) & (r < hi)
        values = {}
        for direction, sector in (("axis", axis), ("diagonal", diagonal)):
            selected = ring & sector
            weights = data["area"][selected]
            if len(weights) < 4:
                break
            values[direction] = {
                "cells": int(selected.sum()),
                "Tgas": float(np.average(data["Tgas"][selected], weights=weights)),
                "Trad": float(np.average(data["Trad"][selected], weights=weights)),
                "rho": float(np.average(data["rho"][selected], weights=weights)),
                "level_min": int(data["level"][selected].min()),
                "level_max": int(data["level"][selected].max()),
                "mean_radius": float(np.average(r[selected], weights=weights)),
            }
        if len(values) == 2:
            a, d = values["axis"], values["diagonal"]
            rows.append({"r_inner_au": lo, "r_outer_au": hi,
                         "axis_cells": a["cells"], "diag_cells": d["cells"],
                         "axis_r_au": a["mean_radius"], "diag_r_au": d["mean_radius"],
                         "axis_Tgas_K": a["Tgas"], "diag_Tgas_K": d["Tgas"],
                         "Tgas_axis_minus_diag_K": a["Tgas"] - d["Tgas"],
                         "Tgas_frac": (a["Tgas"] - d["Tgas"]) / ((a["Tgas"] + d["Tgas"]) / 2),
                         "Trad_frac": (a["Trad"] - d["Trad"]) / ((a["Trad"] + d["Trad"]) / 2),
                         "rho_frac": (a["rho"] - d["rho"]) / ((a["rho"] + d["rho"]) / 2),
                         "axis_level_min": a["level_min"], "axis_level_max": a["level_max"],
                         "diag_level_min": d["level_min"], "diag_level_max": d["level_max"]})
    return rows


def ray_profiles(data, out):
    radii = np.arange(1.0, 30.0001, 0.1)
    grouped = {}
    for group, degrees in (("axis", (0, 90, 180, 270)),
                           ("diagonal", (45, 135, 225, 315))):
        values = []
        for degrees_i in degrees:
            theta = np.deg2rad(degrees_i)
            ray = {name: [] for name in ("rho", "Tgas", "Trad", "level")}
            for r in radii:
                x, y = r * np.cos(theta), r * np.sin(theta)
                inside = ((data["xl"] <= x) & (x < data["xr"])
                          & (data["yl"] <= y) & (y < data["yr"]))
                matches = np.flatnonzero(inside)
                if len(matches) != 1:
                    raise ValueError(f"ray {degrees_i} deg, r={r}: {len(matches)} cells")
                i = matches[0]
                for name in ray:
                    ray[name].append(data[name][i])
            values.append(ray)
        grouped[group] = {name: np.mean([v[name] for v in values], axis=0)
                          for name in values[0]}
    rows = []
    for i, r in enumerate(radii):
        a, d = grouped["axis"], grouped["diagonal"]
        rows.append({"radius_au": r,
                     "axis_Tgas_K": a["Tgas"][i], "diag_Tgas_K": d["Tgas"][i],
                     "axis_Trad_K": a["Trad"][i], "diag_Trad_K": d["Trad"][i],
                     "axis_rho": a["rho"][i], "diag_rho": d["rho"][i],
                     "axis_level_mean": a["level"][i],
                     "diag_level_mean": d["level"][i]})
    with (out / "directional_rays.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True, layout="constrained")
    for group, color in (("axis", "#b35331"), ("diagonal", "#5a497d")):
        v = grouped[group]
        axes[0].plot(radii, v["Tgas"], color=color, label=group)
        axes[1].semilogy(radii, v["rho"], color=color)
        axes[2].plot(radii, v["level"], color=color)
    axes[0].set(ylabel="Gas temperature [K]")
    axes[1].set(ylabel=r"Density [g cm$^{-3}$]")
    axes[2].set(xlabel="Radius [au]", ylabel="Mean AMR level", xlim=(1, 30))
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.axvspan(7.507, 7.874, color="#b35331", alpha=0.12)
    fig.savefig(out / "directional_rays.pdf")
    fig.savefig(out / "directional_rays.png", dpi=200)
    plt.close(fig)
    summary = {}
    for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 30)):
        selected = (radii >= lo) & (radii < hi)
        a, d = grouped["axis"], grouped["diagonal"]
        summary[f"{lo}_{hi}_au"] = {
            "mean_abs_Tgas_frac": float(np.mean(np.abs((a["Tgas"][selected] - d["Tgas"][selected])
                                                      / ((a["Tgas"][selected] + d["Tgas"][selected]) / 2)))),
            "max_abs_Tgas_frac": float(np.max(np.abs((a["Tgas"][selected] - d["Tgas"][selected])
                                                    / ((a["Tgas"][selected] + d["Tgas"][selected]) / 2)))),
            "mean_abs_rho_frac": float(np.mean(np.abs((a["rho"][selected] - d["rho"][selected])
                                                    / ((a["rho"][selected] + d["rho"][selected]) / 2))))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--eos", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    if sha(args.hdf) != HDF_SHA or sha(args.eos) != EOS_SHA:
        raise ValueError("input SHA mismatch")
    data = read_plane(args.hdf)
    rows = calculate(data, args.eos)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "directional_anisotropy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    r = np.asarray([(x["r_inner_au"] + x["r_outer_au"]) / 2 for x in rows])
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True, layout="constrained")
    for name, color in (("Tgas_frac", "#b35331"), ("Trad_frac", "#5a497d"),
                        ("rho_frac", "#165b78")):
        axes[0].plot(r, 100 * np.asarray([x[name] for x in rows]), label=name,
                     color=color)
    axes[0].axhline(0, color="0.4", lw=0.7)
    axes[0].set(ylabel="Axis − diagonal / mean [%]")
    axes[0].legend(frameon=False)
    axes[1].plot(r, [x["axis_Tgas_K"] for x in rows], label="Axis", color="#b35331")
    axes[1].plot(r, [x["diag_Tgas_K"] for x in rows], label="Diagonal", color="#5a497d")
    axes[1].set(xlabel="Projected radius [au]", ylabel="Gas temperature [K]",
                xlim=(1, 30))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axvspan(7.507, 7.874, color="#b35331", alpha=0.12)
        ax.grid(alpha=0.2)
    fig.savefig(args.out / "directional_anisotropy.pdf")
    fig.savefig(args.out / "directional_anisotropy.png", dpi=200)
    plt.close(fig)
    ray_summary = ray_profiles(data, args.out)
    summary = {"source_hdf_sha256": HDF_SHA, "eos_sha256": EOS_SHA,
               "angular_half_width_deg": 10, "ring_width_au": 0.5,
               "plane_cells": int(len(data["x"])), "rings": len(rows),
               "matched_radius_rays": ray_summary}
    for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 30)):
        segment = [row for row in rows if lo <= row["r_inner_au"] < hi]
        summary[f"{lo}_{hi}_au"] = {
            "mean_abs_Tgas_frac": float(np.mean(np.abs([v["Tgas_frac"] for v in segment]))),
            "max_abs_Tgas_frac": float(np.max(np.abs([v["Tgas_frac"] for v in segment]))),
            "mean_abs_Trad_frac": float(np.mean(np.abs([v["Trad_frac"] for v in segment]))),
            "mean_abs_rho_frac": float(np.mean(np.abs([v["rho_frac"] for v in segment]))),
            "levels": sorted(set(v["axis_level_min"] for v in segment)
                             | set(v["axis_level_max"] for v in segment)
                             | set(v["diag_level_min"] for v in segment)
                             | set(v["diag_level_max"] for v in segment)),
        }
    (args.out / "directional_anisotropy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
