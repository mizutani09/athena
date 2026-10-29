#!/usr/bin/env python3
"""Verify positivity, LTE initialization, HSE residual, and mapped cube mass."""
import argparse
import json
from pathlib import Path

import numpy as np

G = 6.67430e-8
MSUN = 1.98847e33
RSUN = 6.957e10
AR = 7.5657e-15


def gravity(r, gm, r0, r1):
    b = 1.0+(r/r1)**8
    a = r0**4+r**4/np.sqrt(b)
    return gm*r**3/(b**1.5*a**1.25)


def mapped_mass(r, rho, n, half_width):
    x = (np.arange(n)+0.5)*(2*half_width/n)-half_width
    dx = 2*half_width/n
    lr, ld = np.log(r), np.log(rho)
    xy2 = x[:, None]**2+x[None, :]**2
    total = 0.0
    for z in x:
        rr = np.sqrt(xy2+z*z)
        total += np.exp(np.interp(np.log(np.maximum(rr, r[0])), lr, ld)).sum()*dx**3
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", type=Path)
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--box-half-rsun", type=float, default=1010.5)
    p.add_argument("--potential-mass-msun", type=float, default=12.0)
    p.add_argument("--radius-rsun", type=float, default=830.0)
    p.add_argument("--r0-fraction", type=float, default=0.2)
    p.add_argument("--r1-fraction", type=float, default=1.2)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    a = np.loadtxt(args.profile)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError("expected seven profile columns")
    r, rho, egas, erad, temp, pressure, mass = a[:, :7].T
    if not (np.all(np.diff(r) > 0) and np.all(a > 0) and np.all(np.isfinite(a))):
        raise ValueError("profile must be finite, positive, and ordered")
    radius = args.radius_rsun*RSUN
    gm = G*args.potential_mass_msun*MSUN
    g = gravity(r, gm, args.r0_fraction*radius, args.r1_fraction*radius)
    hse = np.gradient(pressure, r)+rho*g
    scale = np.maximum(rho*g, np.max(rho*g)*1e-12)
    interior = (r > 0.01*radius) & (r < 0.99*radius)
    result = {
        "rows": int(len(r)),
        "positive_finite": True,
        "lte_max_relative": float(np.max(np.abs(erad-AR*temp**4)/erad)),
        "hse_interior_median_relative": float(np.median(np.abs(hse[interior])/scale[interior])),
        "hse_interior_p99_relative": float(np.percentile(np.abs(hse[interior])/scale[interior], 99)),
        "spherical_mass_at_radius_msun": float(mass[np.searchsorted(r, radius)]/MSUN),
        "mapped_mass_msun": float(mapped_mass(
            r, rho, args.grid, args.box_half_rsun*RSUN)/MSUN),
        "grid": args.grid,
    }
    text = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
