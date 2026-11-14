#!/usr/bin/env python3
"""Post-process Mach 5 outputs without modifying the simulation.

Each --profile is LABEL=FILE. Supply all three controlled cases to compare
resolution and timestep; this script does not infer convergence from one case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from plots.plot_radiative_shock import (
    decode_names, jump_center, load_analytic_profile, load_numerical_profile, merge_profile,
    variable_index,
)
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_at(x, analytic, field, center):
    """Interpolate separately across the embedded gas/density discontinuity."""
    if field == "trad":
        return np.interp(x, analytic["x"], analytic[field])
    result = np.empty_like(x)
    for left in (True, False):
        target = x < center if left else x >= center
        source = analytic["x"] < center if left else analytic["x"] >= center
        result[target] = np.interp(x[target], analytic["x"][source],
                                   analytic[field][source])
    return result


def peak_width(x, y, peak, level):
    """Width of the connected excursion containing the peak; linear crossings."""
    left = right = peak
    while left > 0 and y[left - 1] >= level:
        left -= 1
    while right < len(x) - 1 and y[right + 1] >= level:
        right += 1
    clipped = left == 0 or right == len(x) - 1
    lo = float(x[left]) if left == 0 else float(np.interp(
        level, [y[left - 1], y[left]], [x[left - 1], x[left]]))
    hi = float(x[right]) if right == len(x) - 1 else float(np.interp(
        level, [y[right + 1], y[right]], [x[right + 1], x[right]]))
    return {"width_cm": hi - lo, "left_cm": lo, "right_cm": hi,
            "clipped_at_comparison_boundary": clipped}


def measure(path, analytic, half_window, domain):
    # The shared loader removes invalid samples when merging. Reject them here
    # instead of silently improving a scientific comparison by dropping cells.
    with h5py.File(path) as raw:
        names = decode_names(raw.attrs["VariableNames"])
        for candidates in (("rho", "density"), ("T_gas", "Tgas"), ("T_rad", "Trad")):
            if not np.all(np.isfinite(raw["user_out_var"][variable_index(names, candidates)])):
                raise ValueError(f"nonfinite {candidates[0]} in {path}")
        if not np.all(np.isfinite(raw["x1v"])):
            raise ValueError(f"nonfinite coordinates in {path}")
        shape = raw["user_out_var"][0].shape
        raw_x = np.broadcast_to(np.asarray(raw["x1v"])[:, None, None, :], shape).reshape(-1)
        raw_dx = np.broadcast_to(np.diff(np.asarray(raw["x1f"], dtype=float), axis=-1)
                                 [:, None, None, :], shape).reshape(-1)
        width_x, (cell_widths,) = merge_profile(raw_x, [raw_dx])
    numerical = load_numerical_profile(path)
    if not np.array_equal(width_x, numerical["x"]):
        raise ValueError(f"cell-width/profile coordinate mismatch in {path}")
    if not all(np.all(np.isfinite(numerical[k])) for k in ("rho", "tgas", "trad")):
        raise ValueError(f"nonfinite profile: {path}")
    center = jump_center(analytic["x"], analytic["rho"])
    raw_center = jump_center(numerical["x"], numerical["rho"])
    x = numerical["x"] + center - raw_center
    region = ((x >= max(domain[0], analytic["x"][0])) &
              (x <= min(domain[1], analytic["x"][-1])))
    widths = cell_widths
    near = region & (np.abs(x - center) <= half_window)
    outside = region & ~near
    reference = {k: reference_at(x, analytic, k, center)
                 for k in ("rho", "tgas", "trad")}
    residual = numerical["trad"] - reference["trad"]
    relative = residual / reference["trad"]
    peak = np.flatnonzero(near)[np.argmax(residual[near])]
    height = float(residual[peak])
    shock_index = int(np.argmax(abs(np.diff(numerical["rho"]))))
    dx = float(x[shock_index + 1] - x[shock_index])
    metrics = {
        "file": str(path.resolve()), "sha256": digest(path),
        "time_s": numerical["time"], "profile_points": len(x),
        "shock_raw_cm": raw_center, "shock_alignment_shift_cm": center - raw_center,
        "shock_dx_cm": dx,
        "near_shock_cell_width_min_max_cm": [float(min(widths[near])), float(max(widths[near]))],
        "bump_peak_cell_width_cm": float(widths[peak]),
        "Trad_near_shock_max_abs_relative_percent": float(100 * max(abs(relative[near]))),
        "bump_height_K": max(0., height),
        "bump_height_relative_percent": max(0., float(100 * relative[peak])),
        "bump_peak_offset_from_shock_cm": float(x[peak] - center) if height > 0 else None,
        "Trad_peak_near_shock_K": float(max(numerical["trad"][near])),
        "bump_fwhm": None,
        "bump_width_above_one_percent": None,
        "agreement": {},
    }
    if height > 0:
        for key, curve, level in (("bump_fwhm", residual, height / 2),
                                  ("bump_width_above_one_percent", relative, .01)):
            if curve[peak] >= level:
                width = peak_width(x[region] - center, curve[region],
                                   int(np.count_nonzero(region[:peak])), level)
                width["width_cells_at_shock"] = width["width_cm"] / dx
                metrics[key] = width
    for field in ("rho", "tgas", "trad"):
        difference = abs(numerical[field] - reference[field])
        metrics["agreement"][field] = {
            "global_relative_L1_percent": float(100 * np.sum(difference[region] * widths[region]) /
                                                np.sum(abs(reference[field][region]) * widths[region])),
            "outside_shock_max_relative_percent": float(100 * max(difference[outside] /
                                                                   abs(reference[field][outside]))),
        }
    arrays = {"x_from_shock_cm": x[region] - center,
              "cell_width_cm": widths[region]}
    for field in reference:
        arrays[field] = numerical[field][region]
        arrays[field + "_reference"] = reference[field][region]
    arrays["Trad_residual_K"] = residual[region]
    arrays["Trad_relative_error_percent"] = 100 * relative[region]
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True, help="LABEL=ATHDF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shock-half-window", type=float, default=60.)
    parser.add_argument("--domain", nargs=2, type=float, default=(-3000., 1000.))
    args = parser.parse_args()
    analytic = load_analytic_profile(args.analytic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "analytic_file": str(args.analytic.resolve()), "analytic_sha256": digest(args.analytic),
        "definitions": {
            "shock": "midpoint of largest adjacent density jump; translated in post-processing",
            "shock_window_cm": [-args.shock_half_window, args.shock_half_window],
            "comparison_domain_cm": args.domain,
            "bump": "positive Trad_numerical - Trad_reference peak in shock window",
            "width": "connected excursion containing residual peak; interpolated FWHM crossings",
            "position_precision": "cell sampling and shock alignment each have order dx/2 uncertainty",
            "global_L1": "sum |numerical-reference| dx / sum |reference| dx",
        }, "cases": {},
    }
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    center = jump_center(analytic["x"], analytic["rho"])
    for i, field in enumerate(("rho", "tgas", "trad")):
        axes[i].plot(analytic["x"] - center, analytic[field], "k--", lw=1, label="semi-analytic")
    for item in args.profile:
        label, filename = item.split("=", 1)
        if not label.replace("_", "").replace("-", "").isalnum() or label in result["cases"]:
            raise ValueError("use unique alphanumeric labels with optional _ or -")
        metrics, data = measure(Path(filename), analytic, args.shock_half_window, args.domain)
        result["cases"][label] = metrics
        with (args.output_dir / f"{label}_aligned.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(data)
            writer.writerows(zip(*data.values()))
        for i, field in enumerate(("rho", "tgas", "trad")):
            axes[i].plot(data["x_from_shock_cm"], data[field], lw=1, label=label)
        axes[3].plot(data["x_from_shock_cm"], data["Trad_relative_error_percent"], ".-", ms=2, label=label)
    times = [case["time_s"] for case in result["cases"].values()]
    result["same_physical_time"] = max(times) - min(times) < 1e-7
    result["convergence_classification"] = "not inferred automatically; verify inputs, binary and dt history"
    for i, axis in enumerate(axes):
        axis.axvline(0, color="grey", ls=":")
        axis.grid(alpha=.25)
        axis.legend()
        axis.set_xlim(*(args.domain if i < 2 else (-args.shock_half_window, args.shock_half_window)))
        axis.set_ylabel(("density [g/cm3]", "Tgas [K]", "Trad [K]", "Trad error [%]")[i])
        axis.set_xlabel("x - shock [cm]")
        if i >= 2:
            visible = []
            for line in axis.lines:
                if line.get_transform() != axis.transData:
                    continue
                lx, ly = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
                visible.extend(ly[np.abs(lx) <= args.shock_half_window])
            lo, hi = min(visible), max(visible)
            margin = max((hi - lo) * .1, .01)
            axis.set_ylim(lo - margin, hi + margin)
    axes[3].axhline(0, color="black", lw=.5)
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    fig.savefig(args.output_dir / "comparison.pdf")
    plt.close(fig)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
