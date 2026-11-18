#!/usr/bin/env python3
"""Summarize WOR-160 C++ Cartesian/SMR mapping audits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
SOURCE_METADATA = HERE.parent.parent/"data/ic/rsg21_scf_omega4e-8.json"
with SOURCE_METADATA.open(encoding="utf-8") as handle:
    source = json.load(handle)
CORE_SUPPORT_CM = float(source["core_support_radius_cm"])
SOURCE_INTERFACE_EQUATOR_CM = float(
    source["interface_equatorial_radius_cm"])


def row(path: Path) -> dict[str, float | str]:
    with path.open(encoding="utf-8") as handle:
        item = next(csv.DictReader(handle))
    return {key: (value if key == "label" else float(value))
            for key, value in item.items()}


audits = [row(HERE/name) for name in (
    "smr_0.csv", "smr_2.csv", "smr_4.csv", "smr_6.csv",
    "smr_root64_l4.csv", "smr_root64_l5.csv",
)]
final = audits[-1]

with (HERE/"smr_slice_root64_l5.csv").open(encoding="utf-8") as handle:
    slice_rows = [{key: float(value) for key, value in item.items()}
                  for item in csv.DictReader(handle)]

radial_points = sorted(
    (math.hypot(item["x_cm"], item["y_cm"]),
     item["radiative_fraction"])
    for item in slice_rows if item["x_cm"] >= 0.0
)
mapped_interface = float("nan")
for left, right in zip(radial_points, radial_points[1:]):
    if left[1] >= 0.5 and right[1] < 0.5:
        fraction = (left[1]-0.5)/(left[1]-right[1])
        mapped_interface = left[0]+fraction*(right[0]-left[0])
        break

summary = {
    "schema_version": 1,
    "source": {
        "profile_radial_points": int(source["radial_points"]),
        "profile_angular_points": int(source["angular_points"]),
        "angular_velocity_rad_s": float(source["angular_velocity_rad_s"]),
        "core_support_radius_cm": CORE_SUPPORT_CM,
        "interface_equatorial_radius_cm": SOURCE_INTERFACE_EQUATOR_CM,
    },
    "recommended_mapping": {
        key: final[key] for key in (
            "root_points", "refinement_levels", "quadrature", "min_dx_cm",
            "mass_relative_error", "discrete_J_relative_error",
            "energy_relative_error", "pressure_recovery_l1_relative",
            "density_center_l1_relative", "temperature_center_l1_relative",
            "force_l2_relative", "force_max_relative",
            "atmosphere_mass_g", "radiative_mass_g",
        )
    },
    "mapped_interface_equatorial_radius_cm": mapped_interface,
    "interface_relative_error": (
        mapped_interface/SOURCE_INTERFACE_EQUATOR_CM-1.0),
}
(HERE/"mapping_metrics.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")

figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for root, marker in ((32.0, "o"), (64.0, "s")):
    selected = sorted((item for item in audits if item["root_points"] == root),
                      key=lambda item: item["min_dx_cm"], reverse=True)
    x = [item["min_dx_cm"]/CORE_SUPPORT_CM for item in selected]
    for field, label in (
        ("mass_relative_error", "mass"),
        ("discrete_J_relative_error", r"$J_z$"),
        ("energy_relative_error", "HD energy"),
        ("force_l2_relative", "force L2"),
    ):
        axes[0].loglog(x, [abs(item[field]) for item in selected],
                       marker+"-", label=f"{label}, root {int(root)}")
axes[0].invert_xaxis()
axes[0].set(xlabel=r"minimum cell width / $h$",
            ylabel="relative error", title="Cartesian/SMR mapping convergence")
axes[0].grid(True, which="both", alpha=0.25)
axes[0].legend(fontsize=8, ncol=2)

patches = []
density = []
for item in slice_rows:
    width = item["dx_cm"]
    patches.append(Rectangle((item["x_cm"]-0.5*width,
                              item["y_cm"]-0.5*width), width, width))
    density.append(item["density_g_cm3"])
collection = PatchCollection(
    patches, cmap="magma", norm=colors.LogNorm(vmin=1.0e-12, vmax=max(density)),
    linewidth=0.0)
collection.set_array(density)
axes[1].add_collection(collection)
interface = [item for item in slice_rows
             if 0.1 < item["radiative_fraction"] < 0.9]
axes[1].scatter([item["x_cm"] for item in interface],
                [item["y_cm"] for item in interface], s=2, c="cyan",
                label="radiative/convective interface")
radius = 7.1e13
axes[1].set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal",
            xlabel="x [cm]", ylabel="y [cm]", title="Mapped equatorial density")
axes[1].legend(fontsize=8, loc="upper right")
figure.colorbar(collection, ax=axes[1], label=r"density [g cm$^{-3}$]")
figure.savefig(HERE/"scf_cartesian_mapping.png", dpi=180)
