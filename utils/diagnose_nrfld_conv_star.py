#!/usr/bin/env python3
"""Print quantitative stability diagnostics for an NR-FLD RSG run."""

import argparse
import glob
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "vis", "python"))
import athena_read  # noqa: E402

GAMMA = 5.0/3.0


def latest_pair(directory):
    prim = sorted(glob.glob(os.path.join(directory, "*.out2.*.athdf")))
    uov = sorted(glob.glob(os.path.join(directory, "*.out3.*.athdf")))
    if not prim or not uov:
        raise RuntimeError("both out2 (primitive) and out3 (UOV) dumps are required")
    pmap = {os.path.basename(x).split(".")[-2]: x for x in prim}
    umap = {os.path.basename(x).split(".")[-2]: x for x in uov}
    common = sorted(set(pmap) & set(umap))
    if not common:
        raise RuntimeError("out2 and out3 have no matching dump number")
    key = common[-1]
    return pmap[key], umap[key]


def history_summary(directory):
    files = glob.glob(os.path.join(directory, "*.hst"))
    if len(files) != 1:
        return [f"history: expected one .hst file, found {len(files)}"]
    a = np.atleast_2d(np.loadtxt(files[0], comments="#"))
    q = a[-1]
    lines = [
        f"history: t={q[0]:.8g}, cycles={len(a)-1}, dt={q[1]:.6g}",
        f"history: mass drift={q[2]/a[0,2]-1.0:+.6e}",
        f"history: all-star Mach max(current/run)={q[10]:.6g}/{np.max(a[:,10]):.6g}",
    ]
    if a.shape[1] >= 14:
        rms = np.sqrt(np.maximum(a[:, 12], 0.0)/np.maximum(a[:, 13], 1.0e-99))
        lines.append(
            "history: envelope Mach max(current/run)="
            f"{q[11]:.6g}/{np.max(a[:,11]):.6g}; "
            f"mass-weighted RMS(current/run)={rms[-1]:.6g}/{np.max(rms):.6g}"
        )
    if a.shape[1] >= 16:
        thermal = a[:, 14]+a[:, 15]
        lines.append(
            "history: Egas/Erad="
            f"{q[14]:.6g}/{q[15]:.6g}; "
            f"thermal-energy drift={thermal[-1]/thermal[0]-1.0:+.6e}"
        )
    return lines


def snapshot_summary(directory, density_cut, inner, outer):
    pfile, ufile = latest_pair(directory)
    p = athena_read.athdf(pfile)
    u = athena_read.athdf(ufile)
    if abs(float(p["Time"])-float(u["Time"])) > 2.0e-6:
        raise RuntimeError("latest matching dump numbers have different times")
    zz, yy, xx = np.meshgrid(p["x3v"], p["x2v"], p["x1v"], indexing="ij")
    rr = np.sqrt(xx*xx+yy*yy+zz*zz)
    speed2 = p["vel1"]**2+p["vel2"]**2+p["vel3"]**2
    mach = np.sqrt(speed2/np.maximum(GAMMA*p["press"]/p["rho"], 1.0e-99))
    mask = (rr > inner) & (rr < outer) & (p["rho"] >= density_cut)
    if not np.any(mask):
        raise RuntimeError("the requested envelope diagnostic mask is empty")
    dx = np.diff(p["x1f"])[None, None, :]
    dy = np.diff(p["x2f"])[None, :, None]
    dz = np.diff(p["x3f"])[:, None, None]
    mass = p["rho"]*dx*dy*dz
    env_rms = np.sqrt(np.sum(mass[mask]*mach[mask]**2)/np.sum(mass[mask]))
    radial_velocity = (
        p["vel1"]*xx+p["vel2"]*yy+p["vel3"]*zz
    )/np.maximum(rr, 1.0e-99)
    radial_mean = np.sum(mass[mask]*radial_velocity[mask])/np.sum(mass[mask])
    masked_mach = np.where(mask, mach, -np.inf)
    max_index = np.unravel_index(np.argmax(masked_mach), mach.shape)
    order = np.argsort(rr.ravel())
    cum = np.cumsum(mass.ravel()[order])
    cum /= cum[-1]
    radii = np.interp([0.5, 0.9, 0.99], cum, rr.ravel()[order])
    mismatch = np.abs(u["Tgas"]-u["Trad"])/np.maximum(u["Tgas"], 1.0)
    arrays = [p[k] for k in ("rho", "press", "vel1", "vel2", "vel3")]
    arrays += [u[k] for k in ("Tgas", "Trad", "Ptot")]
    finite = all(np.all(np.isfinite(x)) for x in arrays)
    lines = [
        f"snapshot: {os.path.basename(pfile)}, t={float(p['Time']):.8g}",
        f"snapshot: finite={finite}, rho=[{np.min(p['rho']):.6e}, {np.max(p['rho']):.6e}]",
        f"snapshot: envelope Mach max/RMS={np.max(mach[mask]):.6g}/{env_rms:.6g} "
        f"for {inner}<r/R*<{outer}, rho>={density_cut}",
        "snapshot: envelope Mach maximum at "
        f"r/R*={rr[max_index]:.6g}, rho={p['rho'][max_index]:.6g}; "
        f"mass-weighted <v_r>={radial_mean:+.6g}",
        "snapshot: mass radii r50/r90/r99="+"/".join(f"{x:.5f}" for x in radii),
        f"snapshot: max |Tgas-Trad|/Tgas in r<R*={np.max(mismatch[rr<1.0]):.6e}",
        "snapshot: max |Tgas-Trad|/Tgas in diagnostic envelope="
        f"{np.max(mismatch[mask]):.6e}",
    ]
    radial_key = next((key for key in u if key.startswith("hse_corr_radial")), None)
    if radial_key is not None:
        correction = u[radial_key]
        lines.append(
            "snapshot: radial HSE correction/gravity in envelope "
            f"mean/min/max={np.mean(correction[mask]):+.6e}/"
            f"{np.min(correction[mask]):+.6e}/{np.max(correction[mask]):+.6e}"
        )
    return lines


def warning_summary(directory):
    path = os.path.join(directory, "run.log")
    if not os.path.exists(path):
        return ["log: run.log not found"]
    pattern = re.compile(r"warning|failed|reject|nan|inf", re.IGNORECASE)
    with open(path, encoding="utf-8", errors="replace") as stream:
        hits = [line.strip() for line in stream if pattern.search(line)]
    lines = [f"log: suspicious warning/failure lines={len(hits)}"]
    lines.extend(f"  {line}" for line in hits[:10])
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--density-cut", type=float, default=1.0e-4)
    parser.add_argument("--inner", type=float, default=0.20)
    parser.add_argument("--outer", type=float, default=0.90)
    parser.add_argument("--output")
    args = parser.parse_args()
    lines = []
    lines += history_summary(args.output_dir)
    lines += snapshot_summary(args.output_dir, args.density_cut, args.inner, args.outer)
    lines += warning_summary(args.output_dir)
    report = "\n".join(lines)+"\n"
    print(report, end="")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(report)


if __name__ == "__main__":
    main()
