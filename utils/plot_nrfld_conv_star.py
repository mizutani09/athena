#!/usr/bin/env python3
"""Diagnose nrfld_conv_star output and make an equatorial-slice movie."""

import argparse
import os
import shutil
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

FUNC_DIR = "/home/kosuke/simulation/mhd/utils/func"
ATHENA_VIS = "/home/kosuke/simulation/mhd/athena_st/vis/python"
for path in (FUNC_DIR, ATHENA_VIS):
    if path not in sys.path:
        sys.path.insert(0, path)
from useful_functions_for_plot import (  # noqa: E402
    get_athdf_file_paths, make_movie_from_imgs_general)

RHO_UNIT = 6.23925895e-5
EGAS_UNIT = 5.52235370e9
VEL_UNIT = np.sqrt(EGAS_UNIT/RHO_UNIT)
TIME_UNIT_DAYS = 43.6498577473
GAMMA = 5.0/3.0

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def center_xy(filename, dataset):
    """Assemble the z=0-nearest XY plane from level-0 Athena meshblocks."""
    with h5py.File(filename, "r") as f:
        names = [x.decode() for x in f.attrs["VariableNames"]]
        raw = f[dataset]
        loc = f["LogicalLocations"][:]
        nx, ny, nz = [int(x) for x in f.attrs["RootGridSize"]]
        bx, by, bz = [int(x) for x in f.attrs["MeshBlockSize"]]
        kg = nz//2
        plane = {name: np.empty((ny, nx)) for name in names}
        for b, (lx, ly, lz) in enumerate(loc):
            k0 = int(lz)*bz
            if not (k0 <= kg < k0+bz):
                continue
            kk = kg-k0
            i0, j0 = int(lx)*bx, int(ly)*by
            for q, name in enumerate(names):
                plane[name][j0:j0+by, i0:i0+bx] = raw[q, b, kk]
        xmin, xmax, _ = f.attrs["RootGridX1"]
        ymin, ymax, _ = f.attrs["RootGridX2"]
        time = float(f.attrs["Time"])
    x = np.linspace(xmin+0.5*(xmax-xmin)/nx, xmax-0.5*(xmax-xmin)/nx, nx)
    y = np.linspace(ymin+0.5*(ymax-ymin)/ny, ymax-0.5*(ymax-ymin)/ny, ny)
    return plane, x, y, time


def read_hst(filename):
    arr = np.loadtxt(filename, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    result = {"time": arr[:, 0], "dt": arr[:, 1], "mass": arr[:, 2],
              "ke": arr[:, 6]+arr[:, 7]+arr[:, 8], "etot": arr[:, 9],
              "mach": arr[:, 10]}
    # Newer outputs separate the unresolved softened core from the envelope
    # diagnostic and store the moments needed for a mass-weighted RMS Mach.
    if arr.shape[1] >= 14:
        result["mach_env_max"] = arr[:, 11]
        result["mach_env_rms"] = np.sqrt(
            np.maximum(arr[:, 12], 0.0)/np.maximum(arr[:, 13], 1.0e-99))
    return result


def make_history_plot(hst, output):
    t = hst["time"]*TIME_UNIT_DAYS
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(t, hst["dt"]); axes[0, 0].set_ylabel("dt [code]")
    axes[0, 1].plot(t, hst["mach"], alpha=0.55, label=r"all $r<0.9R_*$")
    if "mach_env_max" in hst:
        axes[0, 1].plot(t, hst["mach_env_max"], label="envelope max")
        axes[0, 1].plot(t, hst["mach_env_rms"], label="envelope mass-weighted RMS")
        axes[0, 1].legend()
    axes[0, 1].set_ylabel("Mach number")
    axes[1, 0].plot(t, hst["mass"]/hst["mass"][0]-1); axes[1, 0].set_ylabel("mass / mass(0) - 1")
    axes[1, 1].plot(t, hst["ke"]/hst["etot"]); axes[1, 1].set_ylabel("kinetic / total energy")
    for ax in axes.flat:
        ax.set_xlabel("time [day]"); ax.grid(alpha=0.3)
    fig.savefig(output, dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--dest", default="star_conv_diagnostics")
    ap.add_argument("--repeat", type=int, default=5,
                    help="repeat each sparse dump this many times in the 30-fps movie")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    frame_dir = os.path.join(args.dest, "frames")
    source_frame_dir = os.path.join(args.dest, "source_frames")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(source_frame_dir, exist_ok=True)
    # A reused destination must not retain frames from an older/longer run.
    for directory in (frame_dir, source_frame_dir):
        for name in os.listdir(directory):
            if name.endswith(".png"):
                os.remove(os.path.join(directory, name))

    prim_files = get_athdf_file_paths(args.output_dir, "out2")
    uov_files = get_athdf_file_paths(args.output_dir, "out3")
    if len(prim_files) != len(uov_files):
        raise RuntimeError("out2 and out3 dump counts differ")
    hst = read_hst(os.path.join(args.output_dir, "nrfld_conv_star.hst"))
    make_history_plot(hst, os.path.join(args.dest, "history.png"))

    diagnostics = []
    frame_number = 0
    for n, (pf, uf) in enumerate(zip(prim_files, uov_files)):
        prim, x, y, time = center_xy(pf, "prim")
        uov, _, _, utime = center_xy(uf, "user_out_var")
        if abs(time-utime) > 2.0e-6:
            raise RuntimeError(f"dump time mismatch at frame {n}: {time}, {utime}")
        rho = prim["rho"]*RHO_UNIT
        press = prim["press"]*EGAS_UNIT
        vx, vy, vz = prim["vel1"], prim["vel2"], prim["vel3"]
        speed2 = vx*vx+vy*vy+vz*vz
        mach = np.sqrt(speed2/np.maximum(GAMMA*prim["press"]/prim["rho"], 1e-99))
        xx, yy = np.meshgrid(x, y)
        vr = (vx*xx+vy*yy)/np.maximum(np.sqrt(xx*xx+yy*yy), 1e-10)*VEL_UNIT/1e5
        mismatch = np.abs(uov["Tgas"]-uov["Trad"])/np.maximum(uov["Tgas"], 1.0)
        diagnostics.append((time, np.nanmin(rho), np.nanmax(rho),
                            np.nanmax(mach[np.sqrt(xx*xx+yy*yy)<0.9]),
                            np.nanmax(mismatch[np.sqrt(xx*xx+yy*yy)<1.0])))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        fields = [(np.log10(rho), "magma", -12, -4, r"$\log_{10}\rho$ [g cm$^{-3}$]"),
                  (vr, "RdBu_r", -30, 30, r"$v_r$ [km s$^{-1}$]"),
                  (np.log10(np.maximum(mach, 1.0e-2)), "viridis", -2, 1,
                   r"$\log_{10}\mathcal{M}$"),
                  (np.log10(np.maximum(mismatch, 1e-8)), "cividis", -8, 0,
                   r"$\log_{10}|T_g-T_r|/T_g$")]
        for ax, (field, cmap, lo, hi, title) in zip(axes.flat, fields):
            im=ax.pcolormesh(x,y,field,cmap=cmap,vmin=lo,vmax=hi,shading="auto")
            ax.add_patch(plt.Circle((0,0),1.0,fill=False,color="white",lw=0.8,alpha=0.8))
            ax.set_aspect("equal"); ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15)
            ax.set_title(title); ax.set_xlabel(r"$x/R_*$"); ax.set_ylabel(r"$y/R_*$")
            fig.colorbar(im,ax=ax,shrink=0.82)
        fig.suptitle(f"equatorial slice: t={time:.4f} ({time*TIME_UNIT_DAYS:.2f} day)")
        # Keep source frames out of frame_dir: the helper intentionally reads
        # every PNG in frame_dir, not only files named ImgForMovie*.png.
        base=os.path.join(source_frame_dir,f"frame_{n:05d}.png")
        fig.savefig(base,dpi=140); plt.close(fig)
        for _ in range(args.repeat):
            shutil.copyfile(base,os.path.join(frame_dir,f"ImgForMovie{frame_number:05d}.png"))
            frame_number += 1

    movie=os.path.join(args.dest,"equatorial_slice.mp4")
    make_movie_from_imgs_general(frame_dir,movie)
    np.savetxt(os.path.join(args.dest,"snapshot_diagnostics.txt"),diagnostics,
               header="time rho_min rho_max Mach_max_r_lt_0p9 max_T_mismatch_r_lt_1")
    print(f"wrote {movie}")


if __name__ == "__main__":
    main()
