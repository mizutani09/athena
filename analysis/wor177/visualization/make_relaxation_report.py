#!/usr/bin/env python3
"""Build the WOR-177 HD relaxation figures and standalone HTML report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


TDYN_CODE = 470.06031
RHO_UNIT = 1.0e-8
VELOCITY_UNIT_KMS = math.sqrt(1.0e5 / RHO_UNIT) / 1.0e5
GAMMA = 5.0 / 3.0
CORE_RADIUS_RSUN = 4.0548
SURFACE_RADIUS_RSUN = 931.9
TURN_LO = 1.365203400
TURN_HI = 1.373209704

SOURCE_PATHS = {
    "l6_diag": "corecut6p28_l6_t1_344891/continuation/relaxation_diagnostics.csv",
    "l7_diag": "corecut6p28_l7_t1p5_stage3_345480/continuation/relaxation_diagnostics.csv",
    "hist_065": "corecut6p28_l7_t065_344983/continuation/mesa_scf_rsg_corecut6p28_l7_4wz.hst",
    "hist_100": "corecut6p28_l7_t1_stage2_345249/continuation/mesa_scf_rsg_corecut6p28_l7_4wz.hst",
    "hist_150": "corecut6p28_l7_t1p5_stage3_345480/continuation/mesa_scf_rsg_corecut6p28_l7_4wz.hst",
}

SNAPSHOTS = [
    ("0.025", "corecut6p28_l6_l7_344840/l7", "00005"),
    ("0.65", "corecut6p28_l7_t065_344983/continuation", "00011"),
    ("1.00", "corecut6p28_l7_t1_stage2_345249/continuation", "00015"),
    ("1.50", "corecut6p28_l7_t1p5_stage3_345480/continuation", "00020"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d8dee8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#172033",
    )


def add_turning_band(ax: plt.Axes, label: bool = False) -> None:
    ax.axvspan(TURN_LO, TURN_HI, color="#2f9e6f", alpha=0.20, lw=0)
    if label:
        ax.text(
            0.5 * (TURN_LO + TURN_HI),
            0.96,
            "radial turn",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#176546",
            fontsize=8,
        )


def read_diagnostics(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    l6 = pd.read_csv(source / SOURCE_PATHS["l6_diag"])
    l7 = pd.read_csv(source / SOURCE_PATHS["l7_diag"])
    return l6, l7


def read_radial_history(source: Path) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    for key in ("hist_065", "hist_100", "hist_150"):
        data = np.loadtxt(source / SOURCE_PATHS[key])
        if data.ndim == 1:
            data = data[np.newaxis, :]
        rows.append(data)
    raw = np.vstack(rows)
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    _, unique = np.unique(raw[:, 0], return_index=True)
    raw = raw[np.sort(unique)]
    return pd.DataFrame(
        {
            "time_over_tdyn": raw[:, 0] / TDYN_CODE,
            "mean_radial_velocity_km_s": raw[:, 27] / raw[:, 14] / 1.0e5,
            "angular_momentum": raw[:, 15],
        }
    )


def infer_spacing(values: np.ndarray) -> float:
    unique = np.unique(values)
    delta = np.diff(unique)
    delta = delta[delta > 1.0e-10]
    if delta.size == 0:
        raise RuntimeError("cannot infer cell spacing from tab file")
    return float(np.median(delta))


def read_snapshot(source: Path, directory: str, snapshot: str) -> dict[str, np.ndarray | float]:
    paths = sorted((source / directory).glob(f"*.out3.{snapshot}.tab"))
    if not paths:
        raise FileNotFoundError(f"no tab files for {directory} snapshot {snapshot}")
    chunks: list[np.ndarray] = []
    spacings: list[np.ndarray] = []
    time_code = math.nan
    time_pattern = re.compile(r"time=([0-9.eE+-]+)")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
        match = time_pattern.search(first)
        if match:
            time_code = float(match.group(1))
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        dx = infer_spacing(data[:, 1])
        dy = infer_spacing(data[:, 3])
        chunks.append(data)
        spacings.append(np.column_stack((np.full(data.shape[0], dx), np.full(data.shape[0], dy))))
    data = np.vstack(chunks)
    spacing = np.vstack(spacings)
    x = data[:, 1]
    y = data[:, 3]
    radius = np.hypot(x, y)
    rho_code = data[:, 4]
    pressure_code = data[:, 5]
    vel_x = data[:, 6]
    vel_y = data[:, 7]
    radial_velocity = np.divide(
        vel_x * x + vel_y * y,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    sound_speed = np.sqrt(np.maximum(GAMMA * pressure_code / rho_code, 1.0e-300))
    return {
        "time_code": time_code,
        "time_over_tdyn": time_code / TDYN_CODE,
        "x": x,
        "y": y,
        "dx": spacing[:, 0],
        "dy": spacing[:, 1],
        "radius": radius,
        "rho": rho_code * RHO_UNIT,
        "rho_code": rho_code,
        "radial_velocity_km_s": radial_velocity * VELOCITY_UNIT_KMS,
        "radial_mach": radial_velocity / sound_speed,
    }


def rectangles(snapshot: dict[str, np.ndarray | float], mask: np.ndarray) -> np.ndarray:
    x = np.asarray(snapshot["x"])[mask]
    y = np.asarray(snapshot["y"])[mask]
    hx = 0.5 * np.asarray(snapshot["dx"])[mask]
    hy = 0.5 * np.asarray(snapshot["dy"])[mask]
    vertices = np.empty((x.size, 4, 2), dtype=float)
    vertices[:, 0, :] = np.column_stack((x - hx, y - hy))
    vertices[:, 1, :] = np.column_stack((x + hx, y - hy))
    vertices[:, 2, :] = np.column_stack((x + hx, y + hy))
    vertices[:, 3, :] = np.column_stack((x - hx, y + hy))
    return vertices


def draw_cell_map(
    ax: plt.Axes,
    snapshot: dict[str, np.ndarray | float],
    field: str,
    extent: float,
    cmap: str,
    norm: matplotlib.colors.Normalize,
) -> PolyCollection:
    x = np.asarray(snapshot["x"])
    y = np.asarray(snapshot["y"])
    dx = np.asarray(snapshot["dx"])
    dy = np.asarray(snapshot["dy"])
    mask = (np.abs(x) - 0.5 * dx <= extent) & (np.abs(y) - 0.5 * dy <= extent)
    collection = PolyCollection(
        rectangles(snapshot, mask),
        array=np.asarray(snapshot[field])[mask],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
    )
    ax.add_collection(collection)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.grid(False)
    return collection


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_global_evolution(l7: pd.DataFrame, output: Path) -> None:
    t = l7["time_over_tdyn"]
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.0), sharex=True)
    axes = axes.ravel()

    axes[0].plot(t, 1.0e5 * l7["mass_relative_drift"], color="#2457a6", lw=1.8)
    axes[0].axhline(10.0, color="#b33a3a", ls="--", lw=1.0, label="gate")
    axes[0].axhline(-10.0, color="#b33a3a", ls="--", lw=1.0)
    axes[0].set_ylabel(r"mass drift [$10^{-5}$]")
    axes[0].legend(loc="upper left")

    axes[1].plot(t, 100.0 * l7["mechanical_energy_relative_drift"], label="mechanical energy", color="#2457a6")
    axes[1].plot(t, 100.0 * l7["virial_relative_residual"], label="virial residual", color="#d2691e")
    axes[1].axhline(1.0, color="#b33a3a", ls="--", lw=0.9)
    axes[1].axhline(-1.0, color="#b33a3a", ls="--", lw=0.9)
    axes[1].set_ylabel("relative residual [%]")
    axes[1].legend(loc="best")

    axes[2].semilogy(t, l7["radial_ke_over_abs_W"], label="radial", color="#d2691e")
    axes[2].semilogy(t, l7["nonradial_perturbation_ke_over_abs_W"], label="nonradial", color="#6a4ca3")
    axes[2].axhline(5.0e-4, color="#b33a3a", ls="--", lw=1.0, label="gate")
    axes[2].set_ylabel(r"kinetic energy / $|W|$")
    axes[2].set_ylim(1.0e-6, 1.0e-3)
    axes[2].legend(loc="best")

    axes[3].semilogy(t, np.maximum(l7["radial_mach_rms"], 1.0e-4), label="radial Mach RMS", color="#2457a6")
    axes[3].semilogy(t, np.maximum(l7["radial_mach_gt_0p1_mass_fraction"], 1.0e-5), label="mass fraction, Mach > 0.1", color="#2f9e6f")
    axes[3].semilogy(t, np.maximum(l7["radial_mach_gt_0p3_mass_fraction"], 1.0e-5), label="mass fraction, Mach > 0.3", color="#6a4ca3")
    axes[3].axhline(0.03, color="#b33a3a", ls="--", lw=1.0, label="historical Mach gate")
    axes[3].set_ylabel("Mach / mass fraction")
    axes[3].set_ylim(1.0e-5, 2.0)
    axes[3].legend(loc="best")

    r0 = l7["mean_radius_cm"].iloc[0]
    rr0 = l7["rms_radius_cm"].iloc[0]
    axes[4].plot(t, 100.0 * (l7["mean_radius_cm"] / r0 - 1.0), label="mean radius", color="#2457a6")
    axes[4].plot(t, 100.0 * (l7["rms_radius_cm"] / rr0 - 1.0), label="RMS radius", color="#d2691e")
    axes[4].axhline(-1.0, color="#b33a3a", ls="--", lw=1.0, label="historical 1% gate")
    axes[4].set_ylabel("radius change [%]")
    axes[4].legend(loc="lower left")

    axes[5].plot(t, l7["meshblocks_total"], label="MeshBlocks", color="#2457a6")
    second = axes[5].twinx()
    second.plot(t, 100.0 * l7["wenoz_plm_fallback_stellar_mass_fraction"], label="PLM fallback mass", color="#d2691e")
    axes[5].set_ylabel("MeshBlocks")
    second.set_ylabel("PLM fallback stellar mass [%]", color="#d2691e")
    axes[5].legend(loc="lower left")
    second.legend(loc="upper right")

    for index, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + index))
        add_turning_band(ax, label=index == 0)
        ax.set_xlim(0.025, 1.5)
    axes[4].set_xlabel(r"time [$t_{dyn}$]")
    axes[5].set_xlabel(r"time [$t_{dyn}$]")
    fig.suptitle("Accepted L7 isolated-star relaxation: global diagnostics", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output / "global-evolution.png")


def plot_resolution_comparison(l6: pd.DataFrame, l7: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    pairs = [
        ("nonradial_perturbation_ke_over_abs_W", "nonradial kinetic energy / |W|", True),
        ("radial_ke_over_abs_W", "radial kinetic energy / |W|", True),
        ("mechanical_energy_relative_drift", "mechanical-energy drift", False),
        ("radial_mach_rms", "radial Mach RMS", True),
    ]
    for index, (ax, (column, ylabel, log_scale)) in enumerate(zip(axes.ravel(), pairs)):
        for frame, label, color in ((l6, "L6: 7.55 cells/support", "#b33a3a"), (l7, "L7: 15.10 cells/support", "#2457a6")):
            values = np.abs(frame[column]) if column == "mechanical_energy_relative_drift" else frame[column]
            ax.plot(frame["time_over_tdyn"], values, label=label, color=color, lw=1.8)
        if log_scale:
            ax.set_yscale("log")
        ax.axvline(0.2903, color="#b33a3a", ls=":", lw=1.0)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.025, 1.0)
        add_panel_label(ax, chr(ord("a") + index))
    axes[0, 0].axhline(5.0e-4, color="#555d6d", ls="--", lw=1.0, label="nonradial gate")
    axes[0, 1].axhline(5.0e-4, color="#555d6d", ls="--", lw=1.0, label="radial gate")
    axes[1, 0].axhline(1.0e-2, color="#555d6d", ls="--", lw=1.0, label="energy gate")
    axes[1, 1].axhline(3.0e-2, color="#555d6d", ls="--", lw=1.0, label="historical Mach gate")
    for ax in axes.ravel():
        ax.legend(loc="best")
    axes[1, 0].set_xlabel(r"time [$t_{dyn}$]")
    axes[1, 1].set_xlabel(r"time [$t_{dyn}$]")
    fig.suptitle("Core-resolution discriminator: L6 instability versus stable L7", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, output / "l6-l7-comparison.png")


def plot_turning_point(l7: pd.DataFrame, radial: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.6), sharex=True)
    axes[0].plot(radial["time_over_tdyn"], radial["mean_radial_velocity_km_s"], color="#2457a6", lw=2.0)
    axes[0].axhline(0.0, color="#424a59", lw=0.9)
    axes[0].set_ylabel(r"$\langle v_r\rangle_M$ [km s$^{-1}$]")

    r0 = l7["mean_radius_cm"].iloc[0]
    rr0 = l7["rms_radius_cm"].iloc[0]
    axes[1].plot(l7["time_over_tdyn"], 100.0 * (l7["mean_radius_cm"] / r0 - 1.0), label="mean radius", color="#2457a6")
    axes[1].plot(l7["time_over_tdyn"], 100.0 * (l7["rms_radius_cm"] / rr0 - 1.0), label="RMS radius", color="#d2691e")
    axes[1].set_ylabel("radius change [%]")
    axes[1].legend(loc="best")

    axes[2].semilogy(l7["time_over_tdyn"], l7["radial_ke_over_abs_W"], label="radial KE", color="#d2691e")
    axes[2].semilogy(l7["time_over_tdyn"], l7["nonradial_perturbation_ke_over_abs_W"], label="nonradial KE", color="#6a4ca3")
    axes[2].set_ylabel(r"kinetic energy / $|W|$")
    axes[2].set_xlabel(r"time [$t_{dyn}$]")
    axes[2].legend(loc="best")

    for index, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + index))
        add_turning_band(ax, label=index == 0)
        ax.set_xlim(0.6, 1.5)
    fig.suptitle("First radial turning point of the relaxed L7 model", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, output / "radial-turning-point.png")


def plot_maps(snapshots: list[tuple[str, dict[str, np.ndarray | float]]], output: Path, extent: float, name: str) -> None:
    density_norm = LogNorm(vmin=1.0e-14, vmax=6.0e-3)
    mach_norm = SymLogNorm(linthresh=0.03, linscale=0.7, vmin=-3.0, vmax=3.0, base=10)
    fig, axes = plt.subplots(2, len(snapshots), figsize=(14.8, 7.6), sharex=True, sharey=True)
    density_collection = None
    mach_collection = None
    for column, (label, snapshot) in enumerate(snapshots):
        density_collection = draw_cell_map(axes[0, column], snapshot, "rho", extent, "magma", density_norm)
        mach_collection = draw_cell_map(axes[1, column], snapshot, "radial_mach", extent, "RdBu_r", mach_norm)
        axes[0, column].set_title(rf"$t={label}\,t_{{dyn}}$")
        for row in range(2):
            if extent > 100.0:
                axes[row, column].add_patch(Circle((0.0, 0.0), SURFACE_RADIUS_RSUN, fill=False, ec="white" if row == 0 else "#252b36", lw=0.7, ls="--", alpha=0.8))
            else:
                axes[row, column].add_patch(Circle((0.0, 0.0), CORE_RADIUS_RSUN, fill=False, ec="white" if row == 0 else "#252b36", lw=0.9, ls="--", alpha=0.9))
            axes[row, column].tick_params(labelleft=column == 0)
        axes[1, column].set_xlabel(r"$x$ [$R_\odot$]")
        if column == 0:
            axes[0, column].set_ylabel(r"$y$ [$R_\odot$]")
            axes[1, column].set_ylabel(r"$y$ [$R_\odot$]")
    if extent > 100.0:
        for ax in axes.ravel():
            ax.set_xticks([-1000.0, 0.0, 1000.0])
            ax.set_yticks([-1000.0, 0.0, 1000.0])
    for column in range(len(snapshots)):
        tick_labels = axes[1, column].get_xticklabels()
        if column > 0:
            tick_labels[0].set_visible(False)
        if column < len(snapshots) - 1:
            tick_labels[-1].set_visible(False)
    axes[0, 0].text(0.03, 0.94, "density", transform=axes[0, 0].transAxes, color="white", ha="left", va="top")
    axes[1, 0].text(0.03, 0.94, "signed radial Mach", transform=axes[1, 0].transAxes, color="#172033", ha="left", va="top")
    assert density_collection is not None and mach_collection is not None
    fig.subplots_adjust(left=0.06, right=0.875, bottom=0.07, top=0.93, wspace=0.08, hspace=0.16)
    density_cax = fig.add_axes((0.895, 0.545, 0.012, 0.36))
    mach_cax = fig.add_axes((0.895, 0.095, 0.012, 0.36))
    cbar_density = fig.colorbar(density_collection, cax=density_cax)
    cbar_density.set_label(r"density [g cm$^{-3}$]")
    cbar_mach = fig.colorbar(mach_collection, cax=mach_cax, extend="both", ticks=[-1.0, -0.1, -0.03, 0.0, 0.03, 0.1, 1.0])
    cbar_mach.set_label("signed radial Mach")
    title = "Equatorial structure of the full star" if extent > 100.0 else "Equatorial core zoom (dashed circle: particle support)"
    fig.suptitle(title, fontsize=14, y=0.995)
    save_figure(fig, output / name)


def radial_bin_profile(snapshot: dict[str, np.ndarray | float], edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius = np.asarray(snapshot["radius"])
    rho = np.asarray(snapshot["rho"])
    mach = np.asarray(snapshot["radial_mach"])
    area = np.asarray(snapshot["dx"]) * np.asarray(snapshot["dy"])
    centers = np.sqrt(edges[:-1] * edges[1:])
    mean_rho = np.full(centers.size, np.nan)
    mean_mach = np.full(centers.size, np.nan)
    rms_mach = np.full(centers.size, np.nan)
    index = np.digitize(radius, edges) - 1
    for bin_index in range(centers.size):
        mask = index == bin_index
        if not np.any(mask):
            continue
        area_weight = area[mask]
        mass_weight = rho[mask] * area_weight
        mean_rho[bin_index] = np.sum(rho[mask] * area_weight) / np.sum(area_weight)
        mean_mach[bin_index] = np.sum(mach[mask] * mass_weight) / np.sum(mass_weight)
        rms_mach[bin_index] = math.sqrt(np.sum(np.square(mach[mask]) * mass_weight) / np.sum(mass_weight))
    return centers, mean_rho, mean_mach, rms_mach


def plot_radial_profiles(snapshots: list[tuple[str, dict[str, np.ndarray | float]]], output: Path) -> None:
    edges = np.geomspace(0.15, 1100.0, 150)
    colors = ["#818a98", "#2f9e6f", "#d2691e", "#2457a6"]
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 10.0), sharex=True)
    for (label, snapshot), color in zip(snapshots, colors):
        radius, rho, mach, mach_rms = radial_bin_profile(snapshot, edges)
        axes[0].loglog(radius, rho, label=rf"$t={label}\,t_{{dyn}}$", color=color, lw=1.7)
        axes[1].semilogx(radius, mach, color=color, lw=1.7)
        axes[2].loglog(radius, np.maximum(mach_rms, 1.0e-5), color=color, lw=1.7)
    axes[0].set_ylabel(r"equatorial mean $\rho$ [g cm$^{-3}$]")
    axes[0].legend(loc="best", ncol=2)
    axes[1].axhline(0.0, color="#424a59", lw=0.8)
    axes[1].set_ylabel(r"density-weighted $\langle \mathcal{M}_r\rangle$")
    axes[2].axhline(0.1, color="#b33a3a", ls="--", lw=0.9)
    axes[2].set_ylabel(r"density-weighted RMS $\mathcal{M}_r$")
    axes[2].set_xlabel(r"equatorial radius [$R_\odot$]")
    for index, ax in enumerate(axes):
        ax.axvline(CORE_RADIUS_RSUN, color="#6a4ca3", ls=":", lw=1.0)
        ax.axvline(SURFACE_RADIUS_RSUN, color="#555d6d", ls=":", lw=1.0)
        add_panel_label(ax, chr(ord("a") + index))
        ax.set_xlim(edges[0], edges[-1])
    fig.suptitle("Equatorial radial profiles across the relaxation", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, output / "radial-profiles.png")


def snapshot_time(snapshot: int) -> float:
    if snapshot == 11:
        return 0.65
    return 0.1 * (snapshot - 4)


def read_localization(source: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    directories = [
        "corecut6p28_l7_t065_344983/continuation",
        "corecut6p28_l7_t1_stage2_345249/continuation",
        "corecut6p28_l7_t1p5_stage3_345480/continuation",
    ]
    for directory in directories:
        for path in sorted((source / directory).glob("slice_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            snap = int(payload["snapshot"])
            payload["time_over_tdyn"] = snapshot_time(snap)
            rows.append(payload)
    return pd.DataFrame(rows).sort_values("time_over_tdyn").drop_duplicates("time_over_tdyn", keep="last")


def plot_localization(localization: pd.DataFrame, output: Path) -> None:
    t = localization["time_over_tdyn"]
    regions = [
        ("r_lt_20_rsun", r"$r<20\,R_\odot$", "#2457a6"),
        ("20_to_100_rsun", r"$20<r<100\,R_\odot$", "#2f9e6f"),
        ("100_to_800_rsun", r"$100<r<800\,R_\odot$", "#d2691e"),
        ("r_gt_800_rsun", r"$r>800\,R_\odot$", "#6a4ca3"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True)
    for suffix, label, color in regions:
        axes[0].plot(t, localization[f"radial_ke_fraction_{suffix}"], marker="o", ms=3.0, label=label, color=color)
        axes[1].plot(t, localization[f"nonradial_ke_fraction_{suffix}"], marker="o", ms=3.0, label=label, color=color)
    axes[0].set_ylabel("fraction of radial KE")
    axes[1].set_ylabel("fraction of nonradial KE")
    axes[1].set_xlabel(r"time [$t_{dyn}$]")
    axes[0].legend(loc="best", ncol=2)
    axes[1].legend(loc="best", ncol=2)
    axes[1].set_ylim(0.0, 1.02)
    for index, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + index))
        add_turning_band(ax, label=index == 0)
        ax.set_xlim(t.min(), t.max())
    fig.suptitle("Where the kinetic energy resides in the equatorial slice", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, output / "energy-localization.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digest(source: Path, directory: str, snapshot: str) -> dict[str, str | int]:
    paths = sorted((source / directory).glob(f"*.out3.{snapshot}.tab"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return {
        "relative_directory": directory,
        "snapshot": snapshot,
        "file_count": len(paths),
        "aggregate_sha256": digest.hexdigest(),
    }


def write_html(output: Path) -> None:
    html = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WOR-177 HD緩和計算 可視化レポート</title>
<style>
:root{color-scheme:light dark;--bg:#f4f6fa;--paper:#fff;--text:#182033;--muted:#5f6878;--line:#dfe4ec;--accent:#2457a6;--good:#176546;--soft:#edf2f9}
@media(prefers-color-scheme:dark){:root{--bg:#11151d;--paper:#1a202b;--text:#edf1f7;--muted:#aeb7c6;--line:#343d4d;--accent:#82aef4;--good:#79d7aa;--soft:#202a39}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--accent)}header{background:linear-gradient(135deg,#142746,#2457a6);color:#fff;padding:54px 24px 46px}header>div,main{max-width:1220px;margin:auto}.eyebrow{letter-spacing:.1em;text-transform:uppercase;font-size:12px;opacity:.78}h1{font-size:clamp(30px,5vw,54px);line-height:1.08;margin:8px 0 12px;font-weight:650}header p{max-width:800px;font-size:18px;margin:0;opacity:.9}.badges{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px}.badge{padding:5px 11px;border-radius:999px;background:#ffffff20;border:1px solid #ffffff33;font-size:13px}nav{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--paper) 92%,transparent);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}nav div{max-width:1220px;margin:auto;display:flex;gap:18px;overflow:auto;padding:11px 22px}nav a{color:var(--muted);text-decoration:none;white-space:nowrap;font-size:13px}main{padding:28px 22px 70px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:13px;margin-bottom:30px}.metric{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:17px}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{display:block;font-size:25px;font-weight:650;margin:3px 0}.metric em{font-style:normal;color:var(--good);font-size:12px}section{scroll-margin-top:58px;margin-top:42px}h2{font-size:25px;margin:0 0 8px}h3{font-size:18px;margin:0 0 6px}.lead{color:var(--muted);max-width:900px;margin:0 0 18px}.figure{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:15px;margin:17px 0 26px}.figure img{display:block;width:100%;height:auto;border-radius:7px;background:#fff}.figure figcaption{color:var(--muted);font-size:13px;margin-top:11px}.decision{background:color-mix(in srgb,var(--good) 10%,var(--paper));border-left:4px solid var(--good);padding:17px 19px;border-radius:9px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}.grid .figure{margin:0}table{width:100%;border-collapse:collapse;background:var(--paper);border-radius:12px;overflow:hidden}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}th{background:var(--soft);font-size:12px;color:var(--muted)}td:nth-child(n+2){font-variant-numeric:tabular-nums}code{background:var(--soft);padding:2px 5px;border-radius:5px;overflow-wrap:anywhere}details{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:13px 16px;margin-top:12px}summary{font-weight:600}footer{color:var(--muted);font-size:12px;margin-top:48px;border-top:1px solid var(--line);padding-top:18px}@media(max-width:650px){header{padding-top:36px}.grid{grid-template-columns:1fr}.figure{padding:8px}th,td{padding:8px 7px;font-size:12px}}
</style>
</head>
<body>
<header><div><div class="eyebrow">WOR-177 · modified LE + BSCF · Athena++ RK4+4WZ</div><h1>HD緩和計算の検証</h1><p>拡散コア赤色超巨星モデルについて、解像度選定と1.5動力学時間までの安定性をまとめた可視化レポートです。</p><div class="badges"><span class="badge">L7 採用</span><span class="badge">L6 棄却</span><span class="badge">コア支持半径あたり 15.10 cells</span><span class="badge">1.5 t<sub>dyn</sub></span></div></div></header>
<nav><div><a href="#decision">判定</a><a href="#history">時間発展</a><a href="#resolution">L6 / L7</a><a href="#turning">反転点</a><a href="#maps">空間分布</a><a href="#profiles">動径構造</a><a href="#provenance">再現性</a></div></nav>
<main>
<div class="metrics"><div class="metric"><span>最初の動径反転</span><strong>1.369 t<sub>dyn</sub></strong><em>収縮から膨張へ</em></div><div class="metric"><span>非動径運動エネルギー最大値</span><strong>9.47×10<sup>−6</sup> |W|</strong><em>判定閾値の1/52</em></div><div class="metric"><span>最終動径運動エネルギー</span><strong>1.27×10<sup>−5</sup> |W|</strong><em>最初の極大値の11.8%</em></div><div class="metric"><span>最終動径Mach RMS</span><strong>0.0299</strong><em>0.03未満へ復帰</em></div></div>
<section id="decision"><h2>判定</h2><p class="lead">L6ではコアに局在した非動径不安定性が成長します。支持半径の解像度をL7へ倍増するとこのモードは消え、残る運動は外層の有界な動径緩和です。エネルギー成長を伴わず最初の反転点まで到達しています。</p><div class="decision"><strong>主シードとして採用。</strong> 点源伴星を導入する計算にはjob 345480のrestartを使用します。無減衰の孤立星計算をこれ以上延長する必要はありません。</div><table><thead><tr><th>判定量</th><th>測定値</th><th>評価</th></tr></thead><tbody><tr><td>質量ドリフト</td><td>3.05×10<sup>−5</sup></td><td>合格</td></tr><tr><td>力学的エネルギードリフト</td><td>0.254%</td><td>合格</td></tr><tr><td>動径 KE / |W| 最大値</td><td>1.08×10<sup>−4</sup></td><td>合格</td></tr><tr><td>非動径 KE / |W| 最大値</td><td>9.47×10<sup>−6</sup></td><td>合格</td></tr><tr><td>平均半径の最大収縮</td><td>3.256%</td><td>緩和後の平衡移動として許容</td></tr><tr><td>最終規格化角運動量</td><td>1.95×10<sup>−7</sup></td><td>無回転と整合</td></tr></tbody></table></section>
<section id="history"><h2>全体量の時間発展</h2><p class="lead">緑色の帯が最初の動径反転区間です。半径が収縮・反転する間も、質量とエネルギーは有界に保たれます。</p><figure class="figure"><a href="assets/global-evolution.png"><img src="assets/global-evolution.png" alt="質量、エネルギー、運動エネルギー、Mach数、半径、メッシュブロック数の時間発展"></a><figcaption>0.025–1.5 t<sub>dyn</sub>のL7診断量。一般的な過去の閾値も比較用に示しています。低温テーパー領域では形式的なMach数が大きくなりますが、運動エネルギーは小さいままです。</figcaption></figure></section>
<section id="resolution"><h2>L7が必要な理由</h2><p class="lead">同一の拡散コアモデルでも、支持半径を7.55セルで解像するL6は不安定、15.10セルのL7は安定です。</p><figure class="figure"><a href="assets/l6-l7-comparison.png"><img src="assets/l6-l7-comparison.png" alt="L6とL7の運動エネルギー、エネルギードリフト、動径Mach数の比較"></a><figcaption>L6は0.2903 t<sub>dyn</sub>で非動径閾値を超え、0.046 |W|まで成長します。L7は1.5 t<sub>dyn</sub>まで9.47×10<sup>−6</sup> |W|以下です。</figcaption></figure></section>
<section id="turning"><h2>最初の動径反転点</h2><p class="lead">質量加重平均した動径速度は1.365–1.373 t<sub>dyn</sub>でゼロを横切ります。この時点で動径運動エネルギーはすでに強く減衰しています。</p><figure class="figure"><a href="assets/radial-turning-point.png"><img src="assets/radial-turning-point.png" alt="最初の動径反転点付近における動径速度、半径、運動エネルギー"></a><figcaption>反転は質量加重平均動径速度の符号と、平均半径・RMS半径の両方で確認できます。</figcaption></figure></section>
<section id="maps"><h2>空間分布の時間発展</h2><p class="lead">密度分布はほぼ球対称を維持しています。符号付き動径Mach数では、静かなコアに対して外層の収縮と反転後の膨張が明瞭です。</p><figure class="figure"><a href="assets/equatorial-full-star.png"><img src="assets/equatorial-full-star.png" alt="4時刻における全星赤道面の密度と符号付き動径Mach数"></a><figcaption>全星の赤道面。Mach数は青が内向き、赤が外向きです。破線円は初期表面931.9 R<sub>⊙</sub>を示します。</figcaption></figure><figure class="figure"><a href="assets/equatorial-core-zoom.png"><img src="assets/equatorial-core-zoom.png" alt="4時刻における赤道面中心部の密度と動径Mach数"></a><figcaption>中心±20 R<sub>⊙</sub>。破線円はL7で15.10セル解像される軟化粒子の支持半径4.05 R<sub>⊙</sub>です。</figcaption></figure></section>
<section id="profiles"><h2>動径構造と運動エネルギーの局在</h2><div class="grid"><figure class="figure"><a href="assets/radial-profiles.png"><img src="assets/radial-profiles.png" alt="4時刻の動径密度・動径Machプロファイル"></a><figcaption>赤道面上の面積加重密度と密度加重Mach数。縦線はコア支持半径と初期星表面です。表面外の大きなMach数は低密度大気に対応します。</figcaption></figure><figure class="figure"><a href="assets/energy-localization.png"><img src="assets/energy-localization.png" alt="動径・非動径運動エネルギー割合の動径分布"></a><figcaption>非動径エネルギーの99%以上は20 R<sub>⊙</sub>以内ですが、絶対振幅は非常に小さいままです。動径エネルギーは有界な脈動に伴って外層と中心側の間を移動します。</figcaption></figure></div></section>
<section id="provenance"><h2>再現性と出典</h2><p class="lead">XDのjobs 344891、344983、345249、345480が生成した累積診断量と、代表4時刻の赤道面スライスを使用しています。</p><details><summary>採用restart</summary><p><code>/work/mizutaniks/wor177/corecut6p28_l7_t1p5_stage3_345480/continuation/mesa_scf_rsg_corecut6p28_l7_4wz.final.rst</code></p><p>SHA-256: <code>62ee1c4b9c2ec4635d039b7e942108b944d90f6fd2211065256d76cfde90a50b</code></p></details><details><summary>モデルと数値手法</summary><p>MESAコアを6.2798 M<sub>⊙</sub>で拡散的に切除、軟化コア粒子6.2112 M<sub>⊙</sub>、支持半径4.0548 R<sub>⊙</sub>。root64 + 中心L7 + 表面テーパーL2、RK4 + 4WZ、FMG重力。本検証にはFLDを含めていません。</p></details></section>
<footer>リポジトリcommit 91fc12a87と採用シードmanifest <code>analysis/wor177/accepted_hd_primary_seed.json</code>に基づいて生成しました。</footer>
</main>
</body>
</html>
"""
    (output / "index.html").write_text(html, encoding="utf-8")


def write_manifest(source: Path, output: Path) -> None:
    assets = sorted((output / "assets").glob("*.png"))
    manifest = {
        "report": "WOR-177 HD relaxation audit",
        "accepted_restart_sha256": "62ee1c4b9c2ec4635d039b7e942108b944d90f6fd2211065256d76cfde90a50b",
        "source_root_on_xd": "/work/mizutaniks/wor177",
        "source_files": {key: {"relative_path": value, "sha256": sha256(source / value)} for key, value in SOURCE_PATHS.items()},
        "snapshot_groups": {
            label: snapshot_digest(source, directory, snapshot)
            for label, directory, snapshot in SNAPSHOTS
        },
        "figures": {asset.name: sha256(asset) for asset in assets},
    }
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True, help="Local copy of selected XD output files")
    parser.add_argument("--output", type=Path, required=True, help="Report output directory")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    configure_style()
    l6, l7 = read_diagnostics(source)
    radial = read_radial_history(source)
    snapshots = [(label, read_snapshot(source, directory, snapshot)) for label, directory, snapshot in SNAPSHOTS]
    localization = read_localization(source)

    plot_global_evolution(l7, assets)
    plot_resolution_comparison(l6, l7, assets)
    plot_turning_point(l7, radial, assets)
    plot_maps(snapshots, assets, 1000.0, "equatorial-full-star.png")
    plot_maps(snapshots, assets, 20.0, "equatorial-core-zoom.png")
    plot_radial_profiles(snapshots, assets)
    plot_localization(localization, assets)
    write_html(output)
    write_manifest(source, output)


if __name__ == "__main__":
    main()
