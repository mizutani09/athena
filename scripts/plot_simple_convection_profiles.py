#!/usr/bin/env python3
"""Plot the initial 1D profile and the time-height radiative cooling rate."""
from pathlib import Path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

from plot_simple_convection_movies import pairs, read_plane


def read_profile(path):
    names = path.read_text().splitlines()[1].lstrip("# ").split()
    data = np.loadtxt(path, comments="#")
    return {name: data[:, i] for i, name in enumerate(names)}


def plot_initial(profile, outdir):
    imgdir = Path(outdir) / "imgs"
    imgdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax[0, 0].plot(profile["z"], profile["optical_depth"])
    ax[0, 0].set(yscale="log", xlabel="z", ylabel="optical depth", title="Optical depth from top")
    ax[0, 1].plot(profile["z"], profile["superadiabaticity"], label=r"$\nabla-\nabla_{ad}$")
    ax[0, 1].axhline(0.0, color="k", lw=0.8)
    ax[0, 1].set(xlabel="z", ylabel=r"$\nabla-\nabla_{ad}$", title="Initial superadiabaticity")
    ax[1, 0].plot(profile["z"], profile["rho"], label="rho")
    ax[1, 0].set(yscale="log", xlabel="z", ylabel="density", title="Initial density")
    ax[1, 1].plot(profile["z"], profile["entropy_proxy"])
    ax[1, 1].set(xlabel="z", ylabel="entropy proxy", title="Initial entropy")
    fig.savefig(imgdir / "simple_convection_initial_profiles.png", dpi=180)
    plt.close(fig)


def cooling_history(outdir):
    paths = sorted(Path(outdir).glob("*.uov_xz_slice.*.athdf"))
    if not paths:
        raise RuntimeError("No uov_xz_slice files found")
    values, times, z = [], [], None
    for path in paths:
        with h5py.File(path, "r") as h:
            if h["user_out_var"].shape[0] < 5:
                raise RuntimeError("uov output has no radiative_cooling variable; rerun with the updated pgen")
            times.append(float(h.attrs["Time"]))
        uov, _, z = read_plane(path, "user_out_var", "xz")
        values.append(np.nanmean(uov[4], axis=1))
    return np.asarray(times), z, np.asarray(values)


def plot_cooling(outdir):
    imgdir = Path(outdir) / "imgs"
    imgdir.mkdir(exist_ok=True)
    times, z, q = cooling_history(outdir)
    lo, hi = np.nanpercentile(q, [1, 99])
    scale = max(abs(lo), abs(hi), 1.0e-30)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.pcolormesh(times, z, q.T, shading="auto", cmap="RdBu_r",
                       vmin=-scale, vmax=scale)
    ax.set(xlabel="time", ylabel="z", title="Horizontal-mean radiative cooling")
    fig.colorbar(im, ax=ax, label=r"$q_{rad}$ [erg cm$^{-3}$ s$^{-1}$]")
    fig.savefig(imgdir / "simple_convection_radiative_cooling_time_height.png", dpi=180)
    plt.close(fig)

    n0 = max(1, len(times)//5)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(q[:n0].mean(axis=0), z, label="early mean")
    ax.plot(q[-n0:].mean(axis=0), z, label="late mean")
    ax.plot(q.mean(axis=0), z, label="full-time mean")
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set(xlabel=r"$q_{rad}$ [erg cm$^{-3}$ s$^{-1}$]", ylabel="z",
           title="Height-dependent radiative cooling")
    ax.legend()
    fig.savefig(imgdir / "simple_convection_radiative_cooling_height_average.png", dpi=180)
    plt.close(fig)


def plot_slice_stratification(outdir, initial, temperature_unit=None):
    """Show how one xz slice evolves; this is not a full 3D horizontal mean."""
    frames = pairs(outdir, "xz")
    if len(frames) < 2:
        return
    width = max(1, len(frames)//5)
    selected = {"early": frames[:width], "late": frames[-width:]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    z = None
    for label, frame_pairs in selected.items():
        rho, temp, vz2 = [], [], []
        for prim_path, uov_path in frame_pairs:
            prim, _, z = read_plane(prim_path, "prim", "xz")
            uov, _, _ = read_plane(uov_path, "user_out_var", "xz")
            rho.append(np.nanmean(prim[0], axis=1))
            temp.append(np.nanmean(uov[2], axis=1))
            vz2.append(np.nanmean(prim[4]**2, axis=1))
        ax[0].plot(np.nanmean(temp, axis=0), z, label=label)
        ax[1].plot(np.nanmean(rho, axis=0), z, label=label)
        ax[2].plot(np.sqrt(np.nanmean(vz2, axis=0)), z, label=label)
    if temperature_unit is not None:
        ax[0].plot(np.interp(z, initial["z"], initial["temperature"])
                   * temperature_unit, z, color="black", ls="--", label="initial")
    ax[1].plot(np.interp(z, initial["z"], initial["rho"]), z,
               color="black", ls="--", label="initial")
    ax[0].set(xlabel="gas temperature [K]", ylabel="z [code]")
    ax[1].set(xlabel="density [code]", xscale="log")
    ax[2].set(xlabel="vertical RMS speed [code]")
    for a in ax:
        a.grid(alpha=0.25)
        a.legend(fontsize=10)
    fig.suptitle("Mean along one xz slice; not a full 3D horizontal mean")
    fig.savefig(Path(outdir) / "imgs" / "simple_convection_slice_stratification.png",
                dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--profile", type=Path, default=None)
    ap.add_argument("--temperature-unit", type=float,
                    help="Kelvin per code temperature for the initial profile")
    args = ap.parse_args()
    profile = args.profile or next(args.output_dir.glob("*profiles*.txt"))
    initial = read_profile(profile)
    plot_initial(initial, args.output_dir)
    plot_cooling(args.output_dir)
    plot_slice_stratification(args.output_dir, initial, args.temperature_unit)
    print(f"Wrote profile and cooling plots to {args.output_dir}")


if __name__ == "__main__":
    main()
