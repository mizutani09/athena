#!/usr/bin/env python3
"""Plot downloaded or locally generated NRFLD radiative-shock profiles."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nrfld-validation")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from nrfld_validation.io import latest_athdf


A_RAD = 7.5657e-15
CASE_XLIMS = {"mach2": (-800.0, 400.0), "mach5": (-3000.0, 1000.0)}
CASE_RHO_YLIMS = {"mach2": (0.5e-12, 1.3e-12), "mach5": (0.4e-12, 2.1e-12)}
CASE_TEMP_YLIMS = {"mach2": (90.0, 230.0), "mach5": (0.0, 1200.0)}
ANALYTIC_CANDIDATES = {
    "mach2": (
        "mach2_semianalytic_bvp.csv",
        "mach2_semianalytic_radiative_shock.csv",
        "mach2_semianalytic.csv",
    ),
    "mach5": (
        "mach5_semianalytic.csv",
        "mach5_semianalytic_bvp.csv",
        "mach5_semianalytic_radiative_shock.csv",
    ),
}


def decode_names(values: Iterable[object]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def variable_index(names: list[str], candidates: Iterable[str]) -> int:
    lowered = {name.lower(): index for index, name in enumerate(names)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"none of {list(candidates)} found in variables {names}")


def merge_profile(x: np.ndarray, arrays: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    finite = np.isfinite(x)
    for values in arrays:
        finite &= np.isfinite(values)
    x = x[finite]
    arrays = [values[finite] for values in arrays]
    order = np.argsort(x)
    x = x[order]
    arrays = [values[order] for values in arrays]

    rounded = np.round(x, decimals=10)
    unique_x: list[float] = []
    merged: list[list[float]] = [[] for _ in arrays]
    start = 0
    while start < len(rounded):
        stop = start + 1
        while stop < len(rounded) and rounded[stop] == rounded[start]:
            stop += 1
        unique_x.append(float(np.mean(x[start:stop])))
        for index, values in enumerate(arrays):
            merged[index].append(float(np.mean(values[start:stop])))
        start = stop
    return np.asarray(unique_x), [np.asarray(values) for values in merged]


def load_numerical_profile(path: Path) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as data:
        time = float(data.attrs["Time"])
        names = decode_names(data.attrs["VariableNames"])
        values = np.asarray(data["user_out_var"], dtype=float)
        rho = values[variable_index(names, ("rho", "density"))]
        tgas = values[variable_index(names, ("T_gas", "Tgas", "gas_temperature"))]
        try:
            trad = values[
                variable_index(names, ("T_rad", "Trad", "radiation_temperature"))
            ]
        except KeyError:
            erad = values[variable_index(names, ("E_rad", "Erad", "radiation_energy"))]
            trad = np.power(np.maximum(erad, 0.0) / A_RAD, 0.25)

        x1v = np.asarray(data["x1v"], dtype=float)
        x = np.broadcast_to(x1v[:, None, None, :], rho.shape).reshape(-1)
        rho = rho.reshape(-1)
        tgas = tgas.reshape(-1)
        trad = trad.reshape(-1)

    x, (rho, tgas, trad) = merge_profile(x, [rho, tgas, trad])
    return {"time": time, "x": x, "rho": rho, "tgas": tgas, "trad": trad}


def find_analytic(run_dir: Path, case: str, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for filename in ANALYTIC_CANDIDATES[case]:
        path = run_dir / filename
        if path.is_file():
            return path
    return None


def named_column(data: np.ndarray, candidates: Iterable[str]) -> np.ndarray:
    if data.dtype.names is None:
        raise ValueError("analytic CSV must contain a named header")
    lowered = {name.lower(): name for name in data.dtype.names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return np.asarray(data[lowered[candidate.lower()]], dtype=float)
    raise KeyError(f"none of {list(candidates)} found in {data.dtype.names}")


def load_analytic_profile(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    data = np.genfromtxt(path, names=True, delimiter=",", comments="#", encoding="utf-8")
    profile = {
        "x": named_column(data, ("x_cm", "x", "position")),
        "rho": named_column(data, ("rho_g_cm3", "rho", "density")),
        "tgas": named_column(data, ("Tgas_K", "T_gas", "Tgas")),
        "trad": named_column(data, ("Trad_K", "T_rad", "Trad")),
    }
    order = np.argsort(profile["x"], kind="stable")
    return {key: values[order] for key, values in profile.items()}


def jump_center(x: np.ndarray, rho: np.ndarray) -> float:
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else 0.0
    index = int(np.nanargmax(np.abs(np.diff(rho))))
    return 0.5 * float(x[index] + x[index + 1])


def add_temperature_inset(
    axis: plt.Axes,
    numerical: dict[str, np.ndarray | float],
    analytic: dict[str, np.ndarray] | None,
    case: str,
) -> None:
    x = np.asarray(numerical["x"])
    tgas = np.asarray(numerical["tgas"])
    trad = np.asarray(numerical["trad"])
    center = float(x[int(np.nanargmax(tgas))])
    width = 30.0 if case == "mach2" else 120.0
    xmin, xmax = center - 0.5 * width, center + 0.5 * width
    mask = (x >= xmin) & (x <= xmax)
    if np.count_nonzero(mask) < 3:
        return
    inset = inset_axes(axis, width="43%", height="43%", loc="lower right")
    inset.plot(x[mask], tgas[mask], "+", color="tab:orange")
    inset.plot(x[mask], trad[mask], "s", ms=3, mfc="none", color="tab:green")
    if analytic is not None:
        analytic_mask = (analytic["x"] >= xmin) & (analytic["x"] <= xmax)
        inset.plot(analytic["x"][analytic_mask], analytic["tgas"][analytic_mask], "--",
                   lw=1.2, color="grey")
        inset.plot(analytic["x"][analytic_mask], analytic["trad"][analytic_mask], "-.",
                   lw=1.2, color="grey")
    inset.set_xlim(xmin, xmax)
    inset.tick_params(axis="both", which="major", labelsize=9)
    mark_inset(axis, inset, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.8)


def plot_radiative_shock(
    run_dir: Path,
    figure: Path,
    case: str,
    analytic_path: Path | None = None,
    output_dir: Path | None = None,
    align_shock: bool = True,
    inset: bool | None = None,
) -> dict[str, float | str | bool]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir or run_dir / "output"
    if not output_dir.is_dir():
        output_dir = run_dir
    athdf_path = latest_athdf(output_dir, "uov_x")
    numerical = load_numerical_profile(athdf_path)
    resolved_analytic = find_analytic(run_dir, case, analytic_path)
    analytic = load_analytic_profile(resolved_analytic)

    numerical_shift = 0.0
    if align_shock and analytic is not None:
        numerical_shift = jump_center(analytic["x"], analytic["rho"]) - jump_center(
            np.asarray(numerical["x"]), np.asarray(numerical["rho"])
        )
        numerical["x"] = np.asarray(numerical["x"]) + numerical_shift

    x = np.asarray(numerical["x"])
    rho = np.asarray(numerical["rho"])
    tgas = np.asarray(numerical["tgas"])
    trad = np.asarray(numerical["trad"])
    finite = bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(rho))
        and np.all(np.isfinite(tgas))
        and np.all(np.isfinite(trad))
    )

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, (rho_axis, temp_axis) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.15)},
    )
    rho_axis.plot(x, rho, "o", ms=3.5, mfc="none", label=r"$\rho$ NRFLD")
    temp_axis.plot(x, tgas, "+", ms=6, color="tab:orange", label=r"$T_{\rm gas}$ NRFLD")
    temp_axis.plot(x, trad, "s", ms=3.5, mfc="none", color="tab:green",
                   label=r"$T_{\rm rad}$ NRFLD")
    if analytic is not None:
        rho_axis.plot(analytic["x"], analytic["rho"], color="grey", lw=1.8,
                      label=r"$\rho$ semi-analytic")
        temp_axis.plot(analytic["x"], analytic["tgas"], "--", color="grey", lw=1.8,
                       label=r"$T_{\rm gas}$ semi-analytic")
        temp_axis.plot(analytic["x"], analytic["trad"], "-.", color="grey", lw=1.8,
                       label=r"$T_{\rm rad}$ semi-analytic")

    rho_axis.set_xlim(*CASE_XLIMS[case])
    rho_axis.set_ylim(*CASE_RHO_YLIMS[case])
    temp_axis.set_ylim(*CASE_TEMP_YLIMS[case])
    temp_axis.set_xlabel(r"$x\;[\mathrm{cm}]$", fontsize=20)
    rho_axis.set_ylabel(r"$\rho\;[\mathrm{g\,cm^{-3}}]$", fontsize=20)
    temp_axis.set_ylabel(r"$T\;[\mathrm{K}]$", fontsize=20)
    for axis in (rho_axis, temp_axis):
        axis.tick_params(axis="both", which="major", labelsize=18)
        axis.xaxis.get_offset_text().set_fontsize(15)
        axis.yaxis.get_offset_text().set_fontsize(15)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=15)
    label = "Mach 2 subcritical" if case == "mach2" else "Mach 5 supercritical"
    fig.suptitle(rf"NRFLD radiative shock: {label}, $t={numerical['time']:.4g}\,$s",
                 fontsize=18)
    use_inset = inset if inset is not None else case == "mach5"
    if use_inset:
        add_temperature_inset(temp_axis, numerical, analytic, case)
    fig.savefig(figure, dpi=200)
    plt.close(fig)

    return {
        "passed": finite,
        "finite": finite,
        "time": float(numerical["time"]),
        "numerical_file": str(athdf_path),
        "analytic_file": str(resolved_analytic) if resolved_analytic else "",
        "shock_alignment_shift": float(numerical_shift),
        "profile_points": int(len(x)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("mach2", "mach5"), required=True)
    parser.add_argument("--analytic", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--no-align-shock", action="store_true")
    parser.add_argument(
        "--inset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the default (enabled for Mach 5, disabled for Mach 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure = args.figure or args.run_dir / "figures" / f"radiative_shock_{args.case}.png"
    metrics = plot_radiative_shock(
        args.run_dir,
        figure,
        args.case,
        analytic_path=args.analytic,
        output_dir=args.output_dir,
        align_shock=not args.no_align_shock,
        inset=args.inset,
    )
    print(f"saved: {figure}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
