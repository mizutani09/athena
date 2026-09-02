#!/usr/bin/env python3
"""Make absolute, relative, xz and xy movies from simple-convection slices."""
from pathlib import Path
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def read_plane(path, dataset, plane):
    with h5py.File(path, "r") as h:
        data = h[dataset][...]
        loc = h["LogicalLocations"][...]
        root_x = np.asarray(h.attrs["RootGridX1"])
        root_y = np.asarray(h.attrs["RootGridX2"])
        root_z = np.asarray(h.attrs["RootGridX3"])
        root_n = np.asarray(h.attrs["RootGridSize"], dtype=int)
    bx, by, bz = data.shape[-1], data.shape[-2], data.shape[-3]
    nx = int(np.max(loc[:, 0])) + 1
    ny = int(np.max(loc[:, 1])) + 1
    nz = int(np.max(loc[:, 2])) + 1
    dx = (root_x[1] - root_x[0]) / root_n[0]
    dy = (root_y[1] - root_y[0]) / root_n[1]
    dz = (root_z[1] - root_z[0]) / root_n[2]
    if plane == "xz":
        out = np.full((data.shape[0], nz * bz, nx * bx), np.nan)
        a = root_x[0] + (np.arange(nx * bx) + 0.5) * dx
        b = root_z[0] + (np.arange(nz * bz) + 0.5) * dz
        for q, (ib, _, kb) in enumerate(loc):
            out[:, int(kb)*bz:(int(kb)+1)*bz,
                int(ib)*bx:(int(ib)+1)*bx] = data[:, q, :, 0, :]
    elif plane == "xy":
        out = np.full((data.shape[0], ny * by, nx * bx), np.nan)
        a = root_x[0] + (np.arange(nx * bx) + 0.5) * dx
        b = root_y[0] + (np.arange(ny * by) + 0.5) * dy
        for q, (ib, jb, _) in enumerate(loc):
            out[:, int(jb)*by:(int(jb)+1)*by,
                int(ib)*bx:(int(ib)+1)*bx] = data[:, q, 0, :, :]
    else:
        raise ValueError(plane)
    return out, a, b


def pairs(outdir, plane):
    prim = sorted(Path(outdir).glob(f"*.prim_{plane}_slice.*.athdf"))
    uov = sorted(Path(outdir).glob(f"*.uov_{plane}_slice.*.athdf"))
    umap = {p.name.split(".uov_")[-1]: p for p in uov}
    return [(p, umap[p.name.split(".prim_")[-1]]) for p in prim
            if p.name.split(".prim_")[-1] in umap]


def load_frames(outdir, plane):
    frames = []
    for pp, up in pairs(outdir, plane):
        p, a, b = read_plane(pp, "prim", plane)
        u, _, _ = read_plane(up, "user_out_var", plane)
        if np.isfinite(p).all() and np.isfinite(u).all():
            frames.append({"rho": p[0], "press": p[1], "temp": u[2],
                           "intensity": u[1], "vz": p[4], "a": a, "b": b,
                           "name": pp.name})
    if not frames:
        raise RuntimeError(f"no finite {plane} frames found")
    return frames


def add_relative(frames, plane):
    for f in frames:
        for name in ("rho", "press", "temp", "intensity"):
            q = f[name]
            mean = np.nanmean(q, axis=1)[:, None] if plane == "xz" else np.nanmean(q)
            f[name + "_rel"] = (q - mean) / np.maximum(np.abs(mean), 1.0e-30)


def limits(frames, names, symmetric):
    q = np.concatenate([f[n].ravel() for f in frames for n in names])
    lo, hi = np.nanpercentile(q, [1, 99])
    if symmetric:
        s = max(abs(lo), abs(hi), 1.0e-30)
        return -s, s
    return lo, hi


def make_movie(frames, names, labels, output, plane, fps):
    ncol = 2
    fig, axes = plt.subplots(2, ncol, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    images = []
    for ax, name, label in zip(axes, names, labels):
        sym = name == "vz" or name.endswith("_rel")
        lo, hi = limits(frames, [name], sym)
        im = ax.pcolormesh(frames[0]["a"], frames[0]["b"], frames[0][name],
                           shading="auto", cmap="RdBu_r" if sym else "viridis",
                           vmin=lo, vmax=hi)
        ax.set(xlabel="x", ylabel="z" if plane == "xz" else "y", title=label)
        fig.colorbar(im, ax=ax)
        images.append(im)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    title = fig.suptitle(frames[0]["name"])

    def update(i):
        for im, name in zip(images, names):
            im.set_array(frames[i][name].ravel())
        title.set_text(frames[i]["name"])
        return (*images, title)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps,
                        blit=False)
    ani.save(output, writer=FFMpegWriter(fps=fps, bitrate=2200))
    plt.close(fig)
    print(f"Wrote {output} ({len(frames)} frames)")


def make_plane(outdir, plane, fps):
    frames = load_frames(outdir, plane)
    add_relative(frames, plane)
    root = Path(outdir)
    moviedir = root / "movies"
    moviedir.mkdir(exist_ok=True)
    make_movie(frames, ["rho", "press", "temp", "intensity"],
               ["density", "pressure", "Tgas", "radiation intensity (E_rad)"],
               moviedir / f"simple_convection_{plane}_absolute.mp4", plane, fps)
    make_movie(frames, ["rho_rel", "press_rel", "temp_rel", "intensity_rel"],
               ["density relative deviation", "pressure relative deviation",
                "Tgas relative deviation", "intensity relative deviation"],
               moviedir / f"simple_convection_{plane}_relative.mp4", plane, fps)
    if plane == "xz":
        make_movie(frames, ["intensity", "vz"],
                   ["radiation intensity (E_rad)", "v_z"],
                   moviedir / "simple_convection_xz_intensity_vz.mp4", plane, fps)
    else:
        make_movie(frames, ["temp_rel", "intensity_rel", "rho_rel", "vz"],
                   ["Tgas granulation", "intensity granulation",
                    "density granulation", "v_z"],
                   moviedir / "simple_convection_xy_granulation.mp4", plane, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--xz", action="store_true")
    ap.add_argument("--xy", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    planes = ["xz", "xy"] if args.all or not (args.xz or args.xy) else []
    if args.xz: planes.append("xz")
    if args.xy: planes.append("xy")
    for plane in dict.fromkeys(planes):
        try:
            make_plane(args.output_dir, plane, args.fps)
        except RuntimeError as exc:
            print(f"Skipping {plane}: {exc}")


if __name__ == "__main__":
    main()
