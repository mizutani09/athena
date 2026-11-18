#!/usr/bin/env python3
"""Summarize Athena++ MG gravity and force-balance audits for WOR-161."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = json.loads((ROOT/"data/ic/rsg21_scf_omega4e-8.json").read_text(
    encoding="utf-8"))


def history(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("# [1]="))
    names = re.findall(r"\[\d+\]=([^ ]+)", header)
    records = [line for line in lines if line and not line.startswith("#")]
    return dict(zip(names, (float(value) for value in records[-1].split())))


def metrics(record: dict[str, float]) -> dict[str, float]:
    gamma = 5.0/3.0
    expected_energy = (SOURCE["pressure_volume_integral_erg"]/(gamma-1.0)
                       +SOURCE["rotational_kinetic_energy_erg"])
    return {
        "stellar_mass_relative_error": (
            record["mapped_stellar_mass_g"]/SOURCE["gas_mass_g"]-1.0),
        "angular_momentum_relative_error": (
            record["mapped_Jz_g_cm2_s"]
            /SOURCE["angular_momentum_g_cm2_s"]-1.0),
        "hd_energy_relative_error": record["mapped_E_erg"]/expected_energy-1.0,
        "gas_potential_l2_relative": math.sqrt(
            record["phi_error2"]/record["phi_norm2"]),
        "force_l2_relative": math.sqrt(
            record["force_error2"]/record["force_norm2"]),
        "force_max_relative": record["force_relative_max"],
        "atmosphere_mass_g": record["atmosphere_mass_g"],
        "core_power_erg_s": record["core_power_erg_s"],
        "gravity_torque_erg": record["gravity_torque_erg"],
        "core_potential_energy_erg": record["core_potential_energy_erg"],
        "self_potential_energy_erg": record["self_potential_energy_erg"],
        "core_interaction_energy_relative_error": (
            record["core_potential_energy_erg"]
            /SOURCE["core_interaction_energy_erg"]-1.0),
        "self_gravity_energy_relative_error": (
            record["self_potential_energy_erg"]
            /SOURCE["self_gravitational_virial_erg"]-1.0),
    }


def read_tabs(directory: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in sorted(directory.glob("*.uov.*.tab")):
        data = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            data.append([float(value) for value in fields])
        x_values = sorted({item[1] for item in data})
        dx = x_values[1]-x_values[0] if len(x_values) > 1 else float("nan")
        for item in data:
            force_scale = item[12]
            source_phi = item[9]
            rows.append({
                "block": float(re.search(r"block(\d+)", path.name).group(1)),
                "x_rsun": item[1],
                "y_rsun": item[3],
                "dx_rsun": dx,
                "temperature_k": item[4],
                "stellar_fraction": item[5],
                "radiative_fraction": item[6],
                "phi_self_mg": item[7],
                "phi_core": item[8],
                "phi_self_source": source_phi,
                "phi_self_error": item[10],
                "phi_relative_error": (abs(item[10]/source_phi)
                                       if source_phi != 0.0 else float("nan")),
                "force_residual": item[11],
                "force_scale": force_scale,
                "force_relative": (item[11]/force_scale
                                   if force_scale > 0.0 else float("nan")),
                "pressure_accel": item[13],
                "gravity_accel": item[14],
                "core_work_density": item[15],
                "gravity_torque_density": item[16],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tab-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "root32_l5": {"directory": "final_root32_l5",
                       "root_points": 32, "rho_atm": 1.0e-14},
        "root64_l5": {"directory": "final_root64_l5",
                       "root_points": 64, "rho_atm": 1.0e-14},
        "root64_low_atm": {"directory": "final_root64_low_atm",
                            "root_points": 64, "rho_atm": 1.0e-15},
        "root64_high_atm": {"directory": "final_root64_high_atm",
                             "root_points": 64, "rho_atm": 1.0e-13},
    }
    summary: dict[str, object] = {"schema_version": 1, "cases": {}}
    for name, controls in cases.items():
        record = history(HERE/str(controls["directory"])/"scf_rsg_map.hst")
        public_controls = {key: value for key, value in controls.items()
                           if key != "directory"}
        summary["cases"][name] = {**public_controls, **metrics(record)}
    step_lines = (HERE/"final_root32_step1/scf_rsg_map.hst").read_text(
        encoding="utf-8").splitlines()
    header = next(line for line in step_lines if line.startswith("# [1]="))
    names = re.findall(r"\[\d+\]=([^ ]+)", header)
    step_records = [dict(zip(names, (float(value) for value in line.split())))
                    for line in step_lines if line and not line.startswith("#")]
    initial, final = step_records[0], step_records[-1]
    summary["one_step_root32"] = {
        "delta_mass_relative": final["mapped_stellar_mass_g"]
            /initial["mapped_stellar_mass_g"]-1.0,
        "delta_J_relative": final["mapped_Jz_g_cm2_s"]
            /initial["mapped_Jz_g_cm2_s"]-1.0,
        "delta_HD_energy_relative": final["mapped_E_erg"]
            /initial["mapped_E_erg"]-1.0,
        "delta_mechanical_energy_relative": (
            (final["mapped_E_erg"]+final["core_potential_energy_erg"]
             +final["self_potential_energy_erg"])
            /(initial["mapped_E_erg"]+initial["core_potential_energy_erg"]
              +initial["self_potential_energy_erg"])-1.0),
        "final_core_power_erg_s": final["core_power_erg_s"],
        "final_gravity_torque_erg": final["gravity_torque_erg"],
    }

    rows = read_tabs(args.tab_dir)
    fields = list(rows[0])
    with (HERE/"gravity_slice_root64_l5.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    stellar = [item for item in rows if item["stellar_fraction"] > 0.999
               and math.isfinite(item["force_relative"])]
    summary["slice"] = {
        "stellar_cells": len(stellar),
        "force_relative_median": float(np.median(
            [item["force_relative"] for item in stellar])),
        "force_relative_p90": float(np.percentile(
            [item["force_relative"] for item in stellar], 90.0)),
        "force_relative_p99": float(np.percentile(
            [item["force_relative"] for item in stellar], 99.0)),
    }
    (HERE/"gravity_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    patches = [Rectangle((item["x_rsun"]-0.5*item["dx_rsun"],
                          item["y_rsun"]-0.5*item["dx_rsun"]),
                         item["dx_rsun"], item["dx_rsun"]) for item in stellar]
    values = np.clip([item["force_relative"] for item in stellar], 1.0e-4, 1.0)
    collection = PatchCollection(
        patches, cmap="viridis", norm=colors.LogNorm(vmin=1.0e-4, vmax=1.0),
        linewidth=0.0)
    collection.set_array(values)
    axes[0].add_collection(collection)
    axes[0].set(xlim=(-1050, 1050), ylim=(-1050, 1050), aspect="equal",
                xlabel=r"$x/R_\odot$", ylabel=r"$y/R_\odot$",
                title="Athena++ discrete force residual")
    figure.colorbar(collection, ax=axes[0], label="relative force residual")

    radii = np.array([math.hypot(item["x_rsun"], item["y_rsun"])
                      for item in stellar])
    force = np.array([item["force_relative"] for item in stellar])
    phi = np.array([item["phi_relative_error"] for item in stellar])
    edges = np.linspace(0.0, 1000.0, 41)
    centers = 0.5*(edges[:-1]+edges[1:])
    for values_array, label, style, percentile in (
            (force, "force median", "-", 50),
            (force, "force p90", "--", 90),
            (force, "force p99", ":", 99),
            (phi, "potential median", "-.", 50)):
        binned = [np.percentile(values_array[(radii >= left) & (radii < right)],
                                percentile)
                  if np.any((radii >= left) & (radii < right)) else np.nan
                  for left, right in zip(edges[:-1], edges[1:])]
        axes[1].semilogy(centers, binned, style, label=label)
    axes[1].set(xlabel=r"cylindrical radius / $R_\odot$",
                ylabel="relative error", ylim=(1.0e-5, 2.0),
                title="Radial residual distribution")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()
    figure.savefig(HERE/"scf_gravity_residual.png", dpi=180)

    work_figure, work_axes = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, field, title in (
            (work_axes[0], "core_work_density", "Core gravitational work density"),
            (work_axes[1], "gravity_torque_density", "Gravitational torque density")):
        signed = np.array([item[field] for item in stellar])
        maximum = max(float(np.max(np.abs(signed))), 1.0e-300)
        signed_collection = PatchCollection(
            patches, cmap="coolwarm",
            norm=colors.SymLogNorm(linthresh=maximum*1.0e-6,
                                   vmin=-maximum, vmax=maximum),
            linewidth=0.0)
        signed_collection.set_array(signed)
        axis.add_collection(signed_collection)
        axis.set(xlim=(-1050, 1050), ylim=(-1050, 1050), aspect="equal",
                 xlabel=r"$x/R_\odot$", ylabel=r"$y/R_\odot$", title=title)
        work_figure.colorbar(signed_collection, ax=axis, label="code units")
    work_figure.savefig(HERE/"scf_gravity_work_torque.png", dpi=180)


if __name__ == "__main__":
    main()
