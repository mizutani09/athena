#!/usr/bin/env python3
"""Quick-look plots for the minimal NR-FLD convection benchmark."""
from pathlib import Path
import argparse
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def read_hst(path):
    names = None
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# ["):
            names = [x.split("]=")[-1] for x in line[2:].split()]
        elif line.strip() and not line.startswith("#"):
            rows.append([float(x) for x in line.split()])
    return names, np.asarray(rows)


def plot_history(outdir):
    imgdir = Path(outdir) / "imgs"
    imgdir.mkdir(parents=True, exist_ok=True)
    hst = next(Path(outdir).glob("*.hst"))
    names, data = read_hst(hst)
    col = {name: i for i, name in enumerate(names)}
    time = data[:, col["time"]]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(time, data[:, col["VzRms"]], label="VzRms")
    if "VzRms2Mean" in col:
        ax[0, 0].plot(time, np.sqrt(np.maximum(data[:, col["VzRms2Mean"]], 0.0)),
                      label="sqrt(<Vz^2>)")
    ax[0, 0].set(xlabel="time", ylabel="velocity", title="Vertical velocity")
    ax[0, 0].legend()

    ax[0, 1].plot(time, data[:, col["MaxSpeed"]], label="MaxSpeed")
    ax[0, 1].set(xlabel="time", ylabel="speed", title="Maximum speed")

    if "MassDrift" in col:
        ax[1, 0].plot(time, data[:, col["MassDrift"]], label="MassDrift")
    ax[1, 0].axhline(0.0, color="0.5", linewidth=0.8)
    ax[1, 0].set(xlabel="time", ylabel="relative mass drift",
                 title="HD2 mass control")
    if "PBNDScale" in col:
        ax_scale = ax[1, 0].twinx()
        ax_scale.plot(time, data[:, col["PBNDScale"]], color="tab:orange",
                      label="PBNDScale")
        ax_scale.set_ylabel("PBND scale")

    if "BottomMdotNet" in col:
        ax[1, 1].plot(time, data[:, col["BottomMdotNet"]],
                      color="tab:blue", label="bottom net")
    if "BottomMdotUp" in col:
        ax[1, 1].plot(time, data[:, col["BottomMdotUp"]],
                      color="tab:green", alpha=0.8, label="bottom upflow")
    if "BottomMdotDown" in col:
        ax[1, 1].plot(time, data[:, col["BottomMdotDown"]],
                      color="tab:red", alpha=0.8, label="bottom downflow")
    if "TopMdotOut" in col:
        ax[1, 1].plot(time, data[:, col["TopMdotOut"]],
                      color="tab:orange", label="top")
    ax[1, 1].axhline(0.0, color="0.5", linewidth=0.8)
    ax[1, 1].set(xlabel="time", ylabel="mass flux / area",
                 title="Boundary mass flux")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(imgdir / "simple_convection_history.png", dpi=160)
    plt.close(fig)


def read_slice(path, dataset):
    with h5py.File(path, "r") as h:
        data = h[dataset][...]
        locations = h["LogicalLocations"][...]
        root_x = h.attrs["RootGridX1"]
        root_z = h.attrs["RootGridX3"]
        root_nx = h.attrs["RootGridSize"]
    # x1v/x3v are local coordinates repeated in every meshblock. Rebuild the
    # global slice from LogicalLocations instead of overwriting blocks that
    # happen to have the same local coordinates.
    bx = data.shape[-1]
    bz = data.shape[-3]
    nx_blocks = int(np.max(locations[:, 0])) + 1
    nz_blocks = int(np.max(locations[:, 2])) + 1
    fields = np.full((data.shape[0], nz_blocks * bz, nx_blocks * bx), np.nan)
    dx = (root_x[1] - root_x[0]) / root_nx[0]
    dz = (root_z[1] - root_z[0]) / root_nx[2]
    x = root_x[0] + (np.arange(nx_blocks * bx) + 0.5) * dx
    z = root_z[0] + (np.arange(nz_blocks * bz) + 0.5) * dz
    for b, (ix_block, _, iz_block) in enumerate(locations):
        ix0 = int(ix_block) * bx
        iz0 = int(iz_block) * bz
        fields[:, iz0:iz0 + bz, ix0:ix0 + bx] = data[:, b, :, 0, :]
    return fields, x, z


def plot_snapshot(outdir):
    imgdir = Path(outdir) / "imgs"
    imgdir.mkdir(parents=True, exist_ok=True)
    prim_paths = sorted(Path(outdir).glob("*.prim_xz_slice.*.athdf"))
    uov_paths = sorted(Path(outdir).glob("*.uov_xz_slice.*.athdf"))
    prim_path, uov_path = None, None
    for candidate_p, candidate_u in zip(reversed(prim_paths), reversed(uov_paths)):
        with h5py.File(candidate_p, "r") as hp, h5py.File(candidate_u, "r") as hu:
            if np.isfinite(hp["prim"][...]).all() and np.isfinite(hu["user_out_var"][...]).all():
                prim_path, uov_path = candidate_p, candidate_u
                break
    if prim_path is None:
        raise RuntimeError("No finite xz-slice snapshot was found")
    prim, x, z = read_slice(prim_path, "prim")
    prim0, _, _ = read_slice(prim_paths[0], "prim")
    uov, _, _ = read_slice(uov_path, "user_out_var")
    uov0, _, _ = read_slice(uov_paths[0], "user_out_var")
    t0 = np.nanmean(uov0[2], axis=1)
    dtemp = (uov[2] - t0[:, None]) / t0[:, None]
    drho = (prim[0] - prim0[0]) / np.maximum(np.abs(prim0[0]), 1.0e-30)
    dpress = (prim[1] - prim0[1]) / np.maximum(np.abs(prim0[1]), 1.0e-30)
    # Athena++ primitive output order is rho, press, vel1, vel2, vel3.
    fields = [(prim[4], "vz", "RdBu_r"),
              (drho, "(rho - rho0) / rho0", "RdBu_r"),
              (dpress, "(P - P0) / P0", "RdBu_r"),
              (dtemp, "(Tgas - T0) / T0", "RdBu_r")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (field, title, cmap) in zip(axes.flat, fields):
        image = axis.pcolormesh(x, z, field, shading="auto", cmap=cmap)
        axis.set(xlabel="x", ylabel="z", title=title)
        fig.colorbar(image, ax=axis)
    fig.savefig(imgdir / "simple_convection_snapshot.png", dpi=160)
    plt.close(fig)

    # The raw fields are dominated by their horizontally averaged stratification.
    # Plot horizontal fluctuations separately so weak cellular structure is visible.
    vz_fluct = prim[4] - np.nanmean(prim[4], axis=1)[:, None]
    drho_fluct = drho - np.nanmean(drho, axis=1)[:, None]
    dpress_fluct = dpress - np.nanmean(dpress, axis=1)[:, None]
    dtemp_fluct = dtemp - np.nanmean(dtemp, axis=1)[:, None]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, field, title in (
            (axes[0, 0], vz_fluct, "vz - horizontal mean"),
            (axes[0, 1], dtemp_fluct, "dT fluctuation"),
            (axes[1, 0], drho_fluct, "density relative fluctuation"),
            (axes[1, 1], dpress_fluct, "pressure relative fluctuation")):
        scale = max(np.nanmax(np.abs(field)), 1.0e-12)
        image = axis.pcolormesh(x, z, field, shading="auto", cmap="RdBu_r",
                                vmin=-scale, vmax=scale)
        axis.set(xlabel="x", ylabel="z", title=title)
        fig.colorbar(image, ax=axis)
    fig.savefig(imgdir / "simple_convection_fluctuations.png", dpi=160)
    plt.close(fig)


def make_movie(outdir, fps=6):
    moviedir = Path(outdir) / "movies"
    moviedir.mkdir(parents=True, exist_ok=True)
    prim_paths = sorted(Path(outdir).glob("*.prim_xz_slice.*.athdf"))
    uov_paths = sorted(Path(outdir).glob("*.uov_xz_slice.*.athdf"))
    frames = []
    for prim_path, uov_path in zip(prim_paths, uov_paths):
        prim, x, z = read_slice(prim_path, "prim")
        uov, _, _ = read_slice(uov_path, "user_out_var")
        if np.isfinite(prim).all() and np.isfinite(uov).all():
            frames.append((prim[4], uov[2], prim_path.name))
    if not frames:
        raise RuntimeError("No finite xz-slice frames were found")
    vmax = max(np.nanmax(np.abs(frame[0])) for frame in frames)
    t0, _, _ = read_slice(uov_paths[0], "user_out_var")
    t0 = np.nanmean(t0[2], axis=1)
    dtemp_frames = [(vz, (temp - t0[:, None]) / t0[:, None], name)
                    for vz, temp, name in frames]
    tmin, tmax = np.nanpercentile(
        np.concatenate([frame[1].ravel() for frame in dtemp_frames]), [1, 99])
    tlim = max(abs(tmin), abs(tmax), 1.0e-12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im_vz = axes[0].pcolormesh(x, z, frames[0][0], shading="auto", cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax)
    im_t = axes[1].pcolormesh(x, z, dtemp_frames[0][1], shading="auto", cmap="RdBu_r",
                              vmin=-tlim, vmax=tlim)
    axes[0].set(xlabel="x", ylabel="z", title="vz")
    axes[1].set(xlabel="x", ylabel="z", title="(Tgas - T0) / T0")
    fig.colorbar(im_vz, ax=axes[0]); fig.colorbar(im_t, ax=axes[1])
    title = fig.suptitle(frames[0][2])

    def update(index):
        im_vz.set_array(dtemp_frames[index][0].ravel())
        im_t.set_array(dtemp_frames[index][1].ravel())
        title.set_text(dtemp_frames[index][2])
        return im_vz, im_t, title

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps,
                              blit=False)
    animation.save(moviedir / "simple_convection_xz.mp4",
                   writer=FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print(f"Movie frames: {len(frames)} finite / {len(prim_paths)} total")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    plot_history(args.output_dir)
    plot_snapshot(args.output_dir)
    if args.movie:
        make_movie(args.output_dir, args.fps)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
