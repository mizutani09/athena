#!/usr/bin/env python3
"""Check the c0 spherical-core acceleration against the analytic uniform sphere."""

import argparse
import json

import h5py
import numpy as np


AU_CM = 1.495978707e13
G_CGS = 6.67430e-8
RHO_CORE_CGS = 5.251585415881407e-18
VELOCITY_UNIT_SQUARED = 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("output_json")
    args = parser.parse_args()
    with h5py.File(args.snapshot) as h5:
        names = [name.decode() for name in h5.attrs["VariableNames"]]
        phi = h5["prim"][names.index("phi")].astype(np.float64)
        x = h5["x1v"][:].astype(np.float64)
        y = h5["x2v"][:].astype(np.float64)
        z = h5["x3v"][:].astype(np.float64)
    dx = (x[:, 1] - x[:, 0])[:, None, None, None]
    dy = (y[:, 1] - y[:, 0])[:, None, None, None]
    dz = (z[:, 1] - z[:, 0])[:, None, None, None]
    gx = (phi[:, 1:-1, 1:-1, 2:] - phi[:, 1:-1, 1:-1, :-2]) / (2 * dx)
    gy = (phi[:, 1:-1, 2:, 1:-1] - phi[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    gz = (phi[:, 2:, 1:-1, 1:-1] - phi[:, :-2, 1:-1, 1:-1]) / (2 * dz)
    xx = x[:, None, None, 1:-1]
    yy = y[:, None, 1:-1, None]
    zz = z[:, 1:-1, None, None]
    radius = np.broadcast_to(np.sqrt(xx * xx + yy * yy + zz * zz), gx.shape)
    measured = (gx * xx + gy * yy + gz * zz) / radius
    expected = (4 * np.pi / 3 * G_CGS * RHO_CORE_CGS * AU_CM**2
                / VELOCITY_UNIT_SQUARED * radius)
    ratio = measured / expected
    bands = {}
    for lo, hi in [(300, 600), (600, 1000), (1000, 1500), (1500, 2500)]:
        values = ratio[(radius >= lo) & (radius < hi)]
        bands[f"{lo}-{hi}"] = {
            "count": int(values.size),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    result = {"snapshot": args.snapshot,
              "method": "same-block centered gradient, interior cells only",
              "analytic_model": "uniform sphere, radius 3000 au, rho 5.2515854e-18 g/cm3",
              "acceleration_ratio_measured_over_analytic": bands}
    with open(args.output_json, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
