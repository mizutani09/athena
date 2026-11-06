#!/usr/bin/env python3
"""Compare exact lower-face upflow/downflow hydro energy fluxes.

The diagnostic columns split the same Godunov energy flux used by FhydBottom
according to the sign of the face mass flux. They are distinct from a
cell-centered estimate one layer above the boundary.
"""

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

from solar_convection_report import input_values, read_history


F_SUN = 6.3e10
FIELDS = ("FhydUpBottom", "FhydDownBottom", "FhydBottom", "FtopTotal")


def mean(time: np.ndarray, value: np.ndarray) -> float:
    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("Need at least two distinct history samples")
    return float(np.trapezoid(value, time) / (time[-1] - time[0]))


def load_case(run: Path, header: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((run / "manifest.json").read_text())
    history = read_history(next(run.glob("*.hst")), header)
    needed = (set(FIELDS) - {"FtopTotal"}) | {"FtotTop"}
    missing = needed - set(history)
    if missing:
        raise ValueError(f"{run}: missing directional diagnostics {sorted(missing)}")
    history["FtopTotal"] = history["FtotTop"] + history.get("FadvRadTop", 0)
    closure = np.max(np.abs(history["FhydUpBottom"]
                            + history["FhydDownBottom"]
                            - history["FhydBottom"])) / F_SUN
    if closure > 1e-8:
        raise ValueError(f"{run}: directional fluxes do not close: {closure:g}")
    return manifest, history


def controls(manifest: dict) -> list[str]:
    problem = [item for item in manifest["problem_overrides"]
               if not item.startswith("s_in=")]
    numerical_prefixes = ("mesh/nx", "time/cfl_number=", "time/tlim=",
                          "time/nlim=", "output1/dt=", "output6/dt=")
    numerical = [item for item in manifest["command"]
                 if item.startswith(numerical_prefixes)]
    return sorted(problem + numerical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--raised", required=True, type=Path)
    parser.add_argument("--header-from", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    fixed, raised = args.fixed.resolve(), args.raised.resolve()
    fixed_manifest, base = load_case(fixed, args.header_from)
    raised_manifest, test = load_case(raised, args.header_from)
    for key in ("binary_sha256", "base_input_sha256", "data_sha256", "restart"):
        if fixed_manifest[key] != raised_manifest[key]:
            raise ValueError(f"Matched runs differ in {key}")
    if controls(fixed_manifest) != controls(raised_manifest):
        raise ValueError("Matched runs differ in controls other than s_in")
    if any(not np.isfinite(values).all() for h in (base, test)
           for values in h.values()):
        raise ValueError("Nonfinite history")
    start = max(base["time"][0], test["time"][0])
    end = min(base["time"][-1], test["time"][-1])
    if end - start < 5:
        raise ValueError("Need at least five code-time units of overlap")
    selected = (test["time"] >= start) & (test["time"] <= end)
    time = test["time"][selected]
    if len(time) < 20:
        raise ValueError("Need at least 20 samples")
    units = input_values(fixed / "athinput.generated")
    mass_flux_unit = units["rho_unit"] * np.sqrt(
        units["egas_unit"] / units["rho_unit"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {"fixed": base, "raised": test}
    average = {}
    closure = {}
    mass_flux = {}
    specific_energy = {}
    for label, history in cases.items():
        mask = (history["time"] >= start) & (history["time"] <= end)
        t = history["time"][mask]
        average[label] = {field: mean(t, history[field][mask])/F_SUN
                          for field in FIELDS}
        mass_flux[label] = {
            "up_cgs": mean(t, history["BottomMdotUp"][mask])*mass_flux_unit,
            "down_cgs": mean(t, history["BottomMdotDown"][mask])*mass_flux_unit}
        specific_energy[label] = {
            "up_erg_g": average[label]["FhydUpBottom"]*F_SUN
                        /mass_flux[label]["up_cgs"],
            "down_erg_g": average[label]["FhydDownBottom"]*F_SUN
                          /mass_flux[label]["down_cgs"]}
        closure[label] = float(np.max(np.abs(
            history["FhydUpBottom"] + history["FhydDownBottom"]
            - history["FhydBottom"])) / F_SUN)
    delta = {field: mean(time, test[field][selected]
                         - np.interp(time, base["time"], base[field]))/F_SUN
             for field in FIELDS}
    report = {"method": "Exact Godunov lower-face energy flux split by face mass-flux sign; trapezoidal time averages",
              "fixed_run": str(fixed), "raised_run": str(raised),
              "start_code": float(start), "end_code": float(end),
              "mean_over_solar": average, "raised_minus_fixed_over_solar": delta,
              "mass_flux_g_cm2_s": mass_flux,
              "specific_hydro_energy_erg_g": specific_energy,
              "max_directional_closure_over_solar": closure,
              "fixed_binary_sha256": fixed_manifest["binary_sha256"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                             constrained_layout=True)
    colors = {"FhydUpBottom": "tab:blue", "FhydDownBottom": "tab:orange",
              "FhydBottom": "tab:green", "FtopTotal": "tab:purple"}
    for label, history in cases.items():
        t = history["time"]
        for field in ("FhydUpBottom", "FhydDownBottom"):
            axes[0].plot(t-start, history[field]/F_SUN,
                         label=f"{label} {field}", color=colors[field],
                         ls="-" if label == "fixed" else "--", alpha=0.85)
        axes[2].plot(t-start, history["FhydBottom"]/F_SUN,
                     label=f"{label} bottom", color=colors["FhydBottom"],
                     ls="-" if label == "fixed" else "--")
        axes[2].plot(t-start, history["FtopTotal"]/F_SUN,
                     label=f"{label} top", color=colors["FtopTotal"],
                     ls="-" if label == "fixed" else "--")
    for field in ("FhydUpBottom", "FhydDownBottom", "FhydBottom"):
        difference = (test[field][selected]
                      - np.interp(time, base["time"], base[field]))/F_SUN
        axes[1].plot(time-start, difference, label=field,
                     color=colors[field])
    axes[0].set_ylabel("Lower-face hydro flux / solar")
    axes[1].set_ylabel("Raised − fixed / solar")
    axes[2].set(ylabel="Net face flux / solar", xlabel="Code time since restart")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output / "directional_flux.png", dpi=150)
    plt.close(fig)

    labels = (("Upward hydro", "FhydUpBottom"),
              ("Downward hydro", "FhydDownBottom"),
              ("Net lower hydro", "FhydBottom"),
              ("Top total", "FtopTotal"))
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{average['fixed'][field]:+.6f}</td>"
        f"<td>{average['raised'][field]:+.6f}</td>"
        f"<td>{delta[field]:+.6f}</td></tr>"
        for name, field in labels)
    transport_rows = "".join(
        f"<tr><td>{label}</td>"
        f"<td>{mass_flux[label]['up_cgs']:+.6f}</td>"
        f"<td>{mass_flux[label]['down_cgs']:+.6f}</td>"
        f"<td>{specific_energy[label]['up_erg_g']:.6e}</td>"
        f"<td>{specific_energy[label]['down_erg_g']:.6e}</td></tr>"
        for label in ("fixed", "raised"))
    fixed_link = html.escape(os.path.relpath(fixed / "index.html", output))
    raised_link = html.escape(os.path.relpath(raised / "index.html", output))
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Solar lower-face directional flux</title>"
            "<style>body{font:21px/1.5 system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem}"
            "img{max-width:100%}table{border-collapse:collapse}"
            "th,td{padding:.5rem .8rem;border-bottom:1px solid #ccc;text-align:right}"
            "th:first-child,td:first-child{text-align:left}strong{color:#18446f}</style>"
            "<h1>Exact lower-face directional hydro flux</h1>"
            "<p>The same Godunov energy face flux is split by the sign of its "
            "mass face flux. Upward + downward equals the net lower hydro flux "
            "in HST. Values below are in units of solar flux.</p>"
            f"<p>Matched restart interval {start:.1f}–{end:.1f} code time. "
            f"<a href='{fixed_link}'>Fixed case</a> · "
            f"<a href='{raised_link}'>Raised case</a> · "
            "<a href='summary.json'>JSON data</a>.</p>"
            "<table><tr><th>Quantity</th><th>Fixed</th><th>Raised</th>"
            f"<th>Difference</th></tr>{rows}</table>"
            "<h2>Mass circulation and energy per mass</h2>"
            "<table><tr><th>Case</th><th>Up mass [g cm⁻² s⁻¹]</th>"
            "<th>Down mass [g cm⁻² s⁻¹]</th>"
            "<th>Up hydro energy [erg g⁻¹]</th>"
            f"<th>Down hydro energy [erg g⁻¹]</th></tr>{transport_rows}</table>"
            "<p><strong>These large opposite contributions nearly cancel.</strong> "
            "The difference is the energy passed into the box at its lower face; "
            "it does not by itself measure eventual photospheric output.</p>"
            "<img src='directional_flux.png' alt='Directional lower hydro flux'>"
            "</html>")
    (output / "index.html").write_text(page + "\n")
    print(output / "index.html")


if __name__ == "__main__":
    main()
