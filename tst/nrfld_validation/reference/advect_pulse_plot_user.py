#!/usr/bin/env python3
"""Analyze the Zhang et al. (2011) Section 6.7 advecting radiation pulse test."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


# Variable-name mapping. Adjust these first if your output names change.
DEFAULT_VAR_MAP = {
    "rho": "rho",
    "vel1": "vel1",
    "temperature": "Tgas",
    "erad": "Erad",
    "pressure": "Pgas",
    "radiation_temperature": "Trad",
}

DEFAULT_GAS_CONSTANT = 8.31451e7  # erg mol^-1 K^-1
DEFAULT_RADIATION_CONSTANT = 7.5657e-15  # erg cm^-3 K^-4

# Plot appearance settings. Adjust these values before running the script.
PLOT_FONT_SIZE = 20
PLOT_TICK_SIZE = 15
PLOT_LEGEND_SIZE = 15
PLOT_TITLE_SIZE = 14
PLOT_LINE_WIDTH = 1.8
PLOT_GRID_ALPHA = 0.25
PLOT_DPI = 300
FIG11_SIZE = (8.0, 9.0)
FIG12_SIZE = (12.0, 4.5)
# The uov fields are written by the pgen in physical cgs units.
PLOT_DENSITY_SCALE = 1.0       # g cm^-3
PLOT_VELOCITY_SCALE = 1.0e-5  # cm s^-1 -> km s^-1
PLOT_TEMPERATURE_SCALE = 1.0e-7  # K -> 10^7 K

plt.rcParams.update({
    "font.size": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_TITLE_SIZE,
    "xtick.labelsize": PLOT_TICK_SIZE,
    "ytick.labelsize": PLOT_TICK_SIZE,
    "legend.fontsize": PLOT_LEGEND_SIZE,
})


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save both the raster preview and a vector PDF with the same stem."""
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def import_athena_read() -> object:
    """Find and import vis/python/athena_read.py without hard-coding the repo root."""
    here = Path(__file__).resolve()
    search_roots = [here.parent] + list(here.parents)
    candidates = []
    for root in search_roots:
        candidates.append(root / "athena" / "vis" / "python")
        candidates.append(root / "vis" / "python")
    for candidate in candidates:
        target = candidate / "athena_read.py"
        if target.is_file():
            sys.path.insert(0, str(candidate))
            import athena_read  # type: ignore

            return athena_read
    raise RuntimeError(
        "Could not locate vis/python/athena_read.py. Run this script from within the "
        "repository tree or place it somewhere with an ancestor containing "
        "'athena/vis/python/athena_read.py'."
    )


ATHENA_READ = import_athena_read()


def wrap_periodic(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    length = x_max - x_min
    return x_min + np.mod(x - x_min, length)


def trim_ghost_1d(arr: np.ndarray, n_active: int, num_ghost: int) -> np.ndarray:
    if arr.shape[0] == n_active + 2 * num_ghost:
        return arr[num_ghost:-num_ghost]
    return arr


def trim_ghost_3d(arr: np.ndarray, root_grid_size: np.ndarray, num_ghost: int) -> np.ndarray:
    slices = []
    for axis, n_active in enumerate(root_grid_size[::-1]):
        axis_size = arr.shape[axis]
        if axis_size == n_active + 2 * num_ghost:
            slices.append(slice(num_ghost, axis_size - num_ghost))
        else:
            slices.append(slice(None))
    return arr[tuple(slices)]


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return path

    athdf_files = sorted(path.rglob("*.athdf"))
    preferred_athdf = [p for p in athdf_files if "uov" in p.name.lower()]
    if preferred_athdf:
        return preferred_athdf[-1]
    if athdf_files:
        return athdf_files[-1]

    text_files = []
    for pattern in ("*.csv", "*.tab", "*.txt", "*.dat"):
        text_files.extend(sorted(path.rglob(pattern)))
    if text_files:
        return text_files[-1]
    raise FileNotFoundError(f"No supported data files found under directory: {path}")


def load_text_profile(path: Path, var_map: dict[str, str]) -> dict[str, np.ndarray]:
    if pd is None:
        raise RuntimeError("pandas is required to read text/CSV profile files.")
    sep = "," if path.suffix.lower() == ".csv" else None
    df = pd.read_csv(path, sep=sep, engine="python", comment="#")
    return {key: df[val].to_numpy() for key, val in var_map.items() if val in df.columns}


def load_athdf_profile(
    path: Path,
    num_ghost: int,
    var_map: dict[str, str],
    mu: float | None,
    gas_constant: float,
) -> dict[str, np.ndarray]:
    data = ATHENA_READ.athdf(str(path), raw=True)
    root_grid_size = np.array(data["RootGridSize"], dtype=int)
    block_size_full = np.array(data["MeshBlockSize"], dtype=int)
    block_size = block_size_full - 2 * num_ghost if num_ghost > 0 else block_size_full
    if np.any(block_size <= 0):
        raise RuntimeError(
            f"Invalid active block size after removing ghost zones from {path}. "
            f"MeshBlockSize={block_size_full}, num_ghost={num_ghost}"
        )

    nmb = int(data["NumMeshBlocks"])
    logical_locations = np.asarray(data["LogicalLocations"], dtype=int)
    x_global = np.empty(root_grid_size[0], dtype=float)
    x2_global = np.empty(root_grid_size[1], dtype=float)
    x3_global = np.empty(root_grid_size[2], dtype=float)

    profile: dict[str, np.ndarray] = {
        "x": x_global,
        "x2": x2_global,
        "x3": x3_global,
        "time_detected": float(data.get("Time", np.nan)),
        "root_grid_size": root_grid_size.astype(int),
        "source_path": np.array([str(path)]),
    }

    for logical_name, file_name in var_map.items():
        if file_name in data:
            profile[logical_name] = np.empty(root_grid_size[0], dtype=float)

    active_x1v = np.asarray(data["x1v"], dtype=float)
    active_x2v = np.asarray(data["x2v"], dtype=float)
    active_x3v = np.asarray(data["x3v"], dtype=float)
    if num_ghost > 0:
        active_x1v = active_x1v[:, num_ghost:-num_ghost]
        active_x2v = active_x2v[:, num_ghost:-num_ghost]
        active_x3v = active_x3v[:, num_ghost:-num_ghost]

    accumulators = {key: np.zeros(root_grid_size[0], dtype=float) for key in profile if key in var_map}
    counts = np.zeros(root_grid_size[0], dtype=int)

    for mb in range(nmb):
        loc = logical_locations[mb]
        ix0 = loc[0] * block_size[0]
        iy0 = loc[1] * block_size[1]
        iz0 = loc[2] * block_size[2]
        ix1 = ix0 + block_size[0]
        iy1 = iy0 + block_size[1]
        iz1 = iz0 + block_size[2]

        x_global[ix0:ix1] = active_x1v[mb]
        x2_global[iy0:iy1] = active_x2v[mb]
        x3_global[iz0:iz1] = active_x3v[mb]
        counts[ix0:ix1] += block_size[1] * block_size[2]

        for logical_name, file_name in var_map.items():
            if file_name not in data:
                continue
            arr = np.asarray(data[file_name][mb], dtype=float)
            if num_ghost > 0:
                arr = arr[num_ghost:-num_ghost, num_ghost:-num_ghost, num_ghost:-num_ghost]
            yz_sum = arr.sum(axis=(0, 1))
            accumulators[logical_name][ix0:ix1] += yz_sum

    positive = counts > 0
    for logical_name in accumulators:
        values = np.zeros(root_grid_size[0], dtype=float)
        values[positive] = accumulators[logical_name][positive] / counts[positive]
        profile[logical_name] = values

    if "temperature" not in profile and "pressure" in profile and "rho" in profile and mu is not None:
        r_over_mu = gas_constant / mu
        profile["temperature"] = profile["pressure"] / np.maximum(profile["rho"] * r_over_mu, 1.0e-300)

    if "radiation_temperature" not in profile and "erad" in profile:
        profile["radiation_temperature"] = np.power(
            np.maximum(profile["erad"], 0.0) / DEFAULT_RADIATION_CONSTANT, 0.25
        )

    return profile


def load_profile(
    path_str: str,
    num_ghost: int,
    var_map: dict[str, str],
    mu: float | None,
    gas_constant: float,
) -> dict[str, np.ndarray]:
    path = resolve_input_path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".athdf":
        return load_athdf_profile(path, num_ghost, var_map, mu, gas_constant)
    if suffix in (".csv", ".tab", ".txt", ".dat"):
        return load_text_profile(path, var_map)
    raise RuntimeError(f"Unsupported file type: {path}")


def require_field(profile: dict[str, np.ndarray], key: str, run_label: str, mapping_name: str) -> np.ndarray:
    if key not in profile:
        raise RuntimeError(
            f"Required field '{key}' is missing for {run_label}. "
            f"Adjust the variable mapping for '{mapping_name}'."
        )
    return np.asarray(profile[key], dtype=float)


def periodic_shift_profile(
    x: np.ndarray,
    y: np.ndarray,
    shift: float,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_x = wrap_periodic(x - shift, x_min, x_max)
    order = np.argsort(shifted_x)
    return shifted_x[order], y[order]


def periodic_interpolate_to_reference_grid(
    x_ref: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    length = x_max - x_min
    order = np.argsort(x_src)
    xs = x_src[order]
    ys = y_src[order]
    xs_ext = np.concatenate(([xs[-1] - length], xs, [xs[0] + length]))
    ys_ext = np.concatenate(([ys[-1]], ys, [ys[0]]))
    return np.interp(x_ref, xs_ext, ys_ext)


def restrict_to_reference_grid(
    x_ref: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, str]:
    n_ref = len(x_ref)
    n_src = len(x_src)
    length = x_max - x_min
    if n_src % n_ref == 0:
        ratio = n_src // n_ref
        dx_ref = length / n_ref
        dx_src = length / n_src
        aligned = np.allclose(dx_ref, ratio * dx_src, rtol=0.0, atol=1.0e-12)
        if aligned:
            order = np.argsort(x_src)
            ys = y_src[order]
            return ys.reshape(n_ref, ratio).mean(axis=1), f"bin-average (ratio={ratio})"
    return periodic_interpolate_to_reference_grid(x_ref, x_src, y_src, x_min, x_max), "interpolation"


def compute_relative_error(test: np.ndarray, reference: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    return (test - reference) / np.where(np.abs(reference) > floor, reference, np.sign(reference) * floor + floor)


def plot_fig11_profiles(
    output_path: Path,
    x_ref: np.ndarray,
    profile_data: dict[str, dict[str, np.ndarray]],
    velocity_mode: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=FIG11_SIZE, sharex=True)
    panels = [("rho", r"$\rho$ [g cm$^{-3}$]"),
              ("velocity_plot", r"$v_x$ [km s$^{-1}$]" if velocity_mode == "raw"
               else r"$\delta v_x$ [km s$^{-1}$]"),
              ("temperature", r"$T_{\rm gas}$ [$10^7$ K]")]
    style_map = {
        "static512": {"label": "static", "linestyle": "-", "color": "tab:blue"},
        "moving512": {"label": "advected", "linestyle": "--", "color": "tab:orange"},
        "static4096": {"label": "_nolegend_", "linestyle": ":", "color": "tab:green"},
    }

    for ax, (field, ylabel) in zip(axes, panels):
        for key in ("static512", "moving512", "static4096"):
            if key not in profile_data or field not in profile_data[key]:
                continue
            style = style_map[key]
            ax.plot(x_ref, profile_data[key][field], linestyle=style["linestyle"], color=style["color"],
                    linewidth=PLOT_LINE_WIDTH, label=style["label"])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
    axes[0].legend(loc="lower right", fontsize=PLOT_LEGEND_SIZE)
    axes[-1].set_xlabel(r"$x$ [cm]")
    fig.subplots_adjust(hspace=0.0, left=0.14, right=0.98, top=0.99, bottom=0.08)
    save_plot(fig, output_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_fig12_errors(
    output_path: Path,
    x_ref: np.ndarray,
    errors: dict[str, dict[str, np.ndarray]],
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIG12_SIZE, sharey=True)
    panel_titles = {
        "static_vs_highres": "static 512 vs static 4096",
        "moving_vs_static": "advected 512 vs static 512",
    }
    for ax, key in zip(axes, ("static_vs_highres", "moving_vs_static")):
        if key not in errors:
            ax.set_visible(False)
            continue
        ax.plot(x_ref, errors[key]["rho"], linestyle="-", color="tab:blue", linewidth=PLOT_LINE_WIDTH, label=r"$\rho$")
        ax.plot(x_ref, errors[key]["temperature"], linestyle="-", color="tab:orange", linewidth=PLOT_LINE_WIDTH,
                label=r"$T_{\rm gas}$")
        # ax.set_title(panel_titles[key], fontsize=PLOT_TITLE_SIZE)
        ax.set_xlabel(r"$x$ [cm]")
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
    visible_axes = [ax for ax in axes if ax.get_visible()]
    if visible_axes:
        error_ax = visible_axes[0]
        # error_ax.set_ylabel(r"$(\mathrm{test}-\mathrm{ref})/\mathrm{ref}$")
        error_ax.set_ylabel(r"Relative difference")
        error_ax.tick_params(axis="y", labelleft=True)
        handles, labels = error_ax.get_legend_handles_labels()
        if handles:
            error_ax.legend(handles, labels, loc="best", fontsize=PLOT_LEGEND_SIZE)
    fig.tight_layout()
    save_plot(fig, output_path)
    if show:
        plt.show()
    plt.close(fig)


def write_processed_profiles_csv(output_path: Path, x: np.ndarray, data: dict[str, dict[str, np.ndarray]]) -> None:
    rows: dict[str, np.ndarray] = {"x": x}
    for run_key, run_data in data.items():
        for field, values in run_data.items():
            rows[f"{run_key}_{field}"] = values
    if pd is not None:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        headers = list(rows.keys())
        writer.writerow(headers)
        for i in range(len(x)):
            writer.writerow([rows[key][i] for key in headers])


def write_relative_errors_csv(output_path: Path, x: np.ndarray, errors: dict[str, dict[str, np.ndarray]]) -> None:
    rows: dict[str, np.ndarray] = {"x": x}
    for comp_key, comp_data in errors.items():
        for field, values in comp_data.items():
            rows[f"{comp_key}_{field}"] = values
    if pd is not None:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        headers = list(rows.keys())
        writer.writerow(headers)
        for i in range(len(x)):
            writer.writerow([rows[key][i] for key in headers])


def add_common_mapping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rho-name", default=DEFAULT_VAR_MAP["rho"])
    parser.add_argument("--vel1-name", default=DEFAULT_VAR_MAP["vel1"])
    parser.add_argument("--temperature-name", default=DEFAULT_VAR_MAP["temperature"])
    parser.add_argument("--erad-name", default=DEFAULT_VAR_MAP["erad"])
    parser.add_argument("--pressure-name", default=DEFAULT_VAR_MAP["pressure"])
    parser.add_argument("--trad-name", default=DEFAULT_VAR_MAP["radiation_temperature"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static512", required=True, help="Path to the low-resolution static run file or directory.")
    parser.add_argument("--moving512", required=True, help="Path to the low-resolution moving run file or directory.")
    parser.add_argument("--static4096", help="Path to the high-resolution static run file or directory.")
    parser.add_argument("--time", type=float, default=4.8e-5, help="Physical comparison time in seconds.")
    parser.add_argument("--v0-moving", type=float, default=1.0e6, help="Moving-run background velocity in cm s^-1.")
    parser.add_argument("--x-min", type=float, default=-512.0)
    parser.add_argument("--x-max", type=float, default=512.0)
    parser.add_argument("--output-prefix", default="advect_pulse")
    parser.add_argument("--plot-fig11", action="store_true")
    parser.add_argument("--plot-fig12", action="store_true")
    parser.add_argument("--velocity-mode", choices=("raw", "perturbation"), default="perturbation")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--num-ghost", type=int, default=2)
    parser.add_argument("--mu", type=float, default=None, help="Needed only if temperature must be reconstructed from pressure and density.")
    parser.add_argument("--gas-constant", type=float, default=DEFAULT_GAS_CONSTANT)
    add_common_mapping_args(parser)
    return parser


def main(**kwargs: object) -> None:
    if not kwargs["plot_fig11"] and not kwargs["plot_fig12"]:
        kwargs["plot_fig11"] = True
        kwargs["plot_fig12"] = True

    var_map = {
        "rho": kwargs["rho_name"],
        "vel1": kwargs["vel1_name"],
        "temperature": kwargs["temperature_name"],
        "erad": kwargs["erad_name"],
        "pressure": kwargs["pressure_name"],
        "radiation_temperature": kwargs["trad_name"],
    }

    static512 = load_profile(kwargs["static512"], kwargs["num_ghost"], var_map, kwargs["mu"], kwargs["gas_constant"])
    moving512 = load_profile(kwargs["moving512"], kwargs["num_ghost"], var_map, kwargs["mu"], kwargs["gas_constant"])
    static4096 = None
    if kwargs["static4096"] is not None:
        static4096 = load_profile(kwargs["static4096"], kwargs["num_ghost"], var_map, kwargs["mu"], kwargs["gas_constant"])

    x_ref = require_field(static512, "x", "static512", "x")
    rho_ref = require_field(static512, "rho", "static512", "rho-name")
    vel_ref = require_field(static512, "vel1", "static512", "vel1-name")
    temp_ref = require_field(static512, "temperature", "static512", "temperature-name")

    x_min = float(kwargs["x_min"])
    x_max = float(kwargs["x_max"])
    shift = float(kwargs["v0_moving"]) * float(kwargs["time"])

    moving_x = require_field(moving512, "x", "moving512", "x")
    moving_rho = require_field(moving512, "rho", "moving512", "rho-name")
    moving_vel = require_field(moving512, "vel1", "moving512", "vel1-name")
    moving_temp = require_field(moving512, "temperature", "moving512", "temperature-name")

    moving_x_shifted, moving_rho_shifted = periodic_shift_profile(moving_x, moving_rho, shift, x_min, x_max)
    _, moving_vel_shifted = periodic_shift_profile(moving_x, moving_vel, shift, x_min, x_max)
    _, moving_temp_shifted = periodic_shift_profile(moving_x, moving_temp, shift, x_min, x_max)

    moving_rho_on_ref = periodic_interpolate_to_reference_grid(x_ref, moving_x_shifted, moving_rho_shifted, x_min, x_max)
    moving_vel_on_ref = periodic_interpolate_to_reference_grid(x_ref, moving_x_shifted, moving_vel_shifted, x_min, x_max)
    moving_temp_on_ref = periodic_interpolate_to_reference_grid(x_ref, moving_x_shifted, moving_temp_shifted, x_min, x_max)

    highres_on_ref: dict[str, np.ndarray] = {}
    highres_method = None
    if static4096 is not None:
        static4096_x = require_field(static4096, "x", "static4096", "x")
        static4096_rho = require_field(static4096, "rho", "static4096", "rho-name")
        static4096_vel = require_field(static4096, "vel1", "static4096", "vel1-name")
        static4096_temp = require_field(static4096, "temperature", "static4096", "temperature-name")
        highres_on_ref["rho"], highres_method = restrict_to_reference_grid(x_ref, static4096_x, static4096_rho, x_min, x_max)
        highres_on_ref["vel1"], _ = restrict_to_reference_grid(x_ref, static4096_x, static4096_vel, x_min, x_max)
        highres_on_ref["temperature"], _ = restrict_to_reference_grid(x_ref, static4096_x, static4096_temp, x_min, x_max)

    if kwargs["velocity_mode"] == "perturbation":
        vel_static_plot = vel_ref
        vel_moving_plot = moving_vel_on_ref - float(kwargs["v0_moving"])
        vel_high_plot = highres_on_ref.get("vel1", np.array([]))
    else:
        vel_static_plot = vel_ref
        vel_moving_plot = moving_vel_on_ref
        vel_high_plot = highres_on_ref.get("vel1", np.array([]))

    # Convert only the Fig. 11 presentation data. Relative errors above use
    # the unscaled physical arrays and therefore remain dimensionless.
    vel_static_plot = vel_static_plot * PLOT_VELOCITY_SCALE
    vel_moving_plot = vel_moving_plot * PLOT_VELOCITY_SCALE
    if vel_high_plot.size:
        vel_high_plot = vel_high_plot * PLOT_VELOCITY_SCALE

    fig11_data: dict[str, dict[str, np.ndarray]] = {
        "static512": {"rho": rho_ref * PLOT_DENSITY_SCALE,
                       "temperature": temp_ref * PLOT_TEMPERATURE_SCALE,
                       "velocity_plot": vel_static_plot},
        "moving512": {"rho": moving_rho_on_ref * PLOT_DENSITY_SCALE,
                       "temperature": moving_temp_on_ref * PLOT_TEMPERATURE_SCALE,
                       "velocity_plot": vel_moving_plot},
    }
    if static4096 is not None:
        fig11_data["static4096"] = {
            "rho": highres_on_ref["rho"] * PLOT_DENSITY_SCALE,
            "temperature": highres_on_ref["temperature"] * PLOT_TEMPERATURE_SCALE,
            "velocity_plot": vel_high_plot,
        }

    errors: dict[str, dict[str, np.ndarray]] = {}
    if static4096 is not None:
        errors["static_vs_highres"] = {
            "rho": compute_relative_error(rho_ref, highres_on_ref["rho"]),
            "temperature": compute_relative_error(temp_ref, highres_on_ref["temperature"]),
        }
    errors["moving_vs_static"] = {
        "rho": compute_relative_error(moving_rho_on_ref, rho_ref),
        "temperature": compute_relative_error(moving_temp_on_ref, temp_ref),
    }

    print("Detected final time attributes:")
    print(f"  static512 : {static512.get('time_detected', np.nan)}")
    print(f"  moving512 : {moving512.get('time_detected', np.nan)}")
    if static4096 is not None:
        print(f"  static4096: {static4096.get('time_detected', np.nan)}")
    print(f"Shift used for moving run: v0 * t = {shift:.16e} cm")
    print(f"Shifted x range: [{moving_x_shifted.min():.16e}, {moving_x_shifted.max():.16e}]")
    print("Grid sizes:")
    print(f"  static512 : {len(x_ref)}")
    print(f"  moving512 : {len(moving_x)}")
    if static4096 is not None:
        print(f"  static4096: {len(static4096_x)}")
        print(f"High-resolution restriction method: {highres_method}")
        print("Max abs relative errors, static 512 vs static 4096:")
        print(f"  density     : {np.max(np.abs(errors['static_vs_highres']['rho'])):.16e}")
        print(f"  temperature : {np.max(np.abs(errors['static_vs_highres']['temperature'])):.16e}")
    print("Max abs relative errors, advected vs static 512:")
    print(f"  density     : {np.max(np.abs(errors['moving_vs_static']['rho'])):.16e}")
    print(f"  temperature : {np.max(np.abs(errors['moving_vs_static']['temperature'])):.16e}")
    vel_delta = moving_vel_on_ref - vel_ref
    if kwargs["velocity_mode"] == "perturbation":
        vel_delta = (moving_vel_on_ref - float(kwargs["v0_moving"])) - vel_ref
    print(f"Max abs velocity comparison difference: {np.max(np.abs(vel_delta)):.16e}")

    output_prefix = Path(str(kwargs["output_prefix"]))
    if kwargs["plot_fig11"]:
        plot_fig11_profiles(
            output_prefix.with_name(output_prefix.name + "_fig11_profiles.png"),
            x_ref,
            fig11_data,
            str(kwargs["velocity_mode"]),
            bool(kwargs["show"]),
        )
    if kwargs["plot_fig12"]:
        plot_fig12_errors(
            output_prefix.with_name(output_prefix.name + "_fig12_relative_errors.png"),
            x_ref,
            errors,
            bool(kwargs["show"]),
        )
    if kwargs["save_csv"]:
        write_processed_profiles_csv(
            output_prefix.with_name(output_prefix.name + "_processed_profiles.csv"),
            x_ref,
            fig11_data,
        )
        write_relative_errors_csv(
            output_prefix.with_name(output_prefix.name + "_relative_errors.csv"),
            x_ref,
            errors,
        )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(**vars(args))
