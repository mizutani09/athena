#!/usr/bin/env python3
"""Compare LP and Minerbo closure on interior cells of a collapse snapshot.

This is an instantaneous diagnostic. It does not evolve the Minerbo model.
"""

import argparse
import json

import h5py
import numpy as np


AU_CM = 1.495978707e13
RHO_UNIT = 1e-18


def summarize(values, weights):
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "gradient_weighted_mean": float(np.average(values, weights=weights)),
        "fraction_above_1pct": float(np.mean(values > 0.01)),
        "fraction_above_5pct": float(np.mean(values > 0.05)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("output_json")
    args = parser.parse_args()
    with h5py.File(args.snapshot) as h5:
        names = [name.decode() for name in h5.attrs["VariableNames"]]
        assert names.index("Er") < h5["prim"].shape[0]
        rho = h5["prim"][names.index("rho")].astype(np.float64) * RHO_UNIT
        er = h5["prim"][names.index("Er")].astype(np.float64)
        x = h5["x1v"][:].astype(np.float64)
        y = h5["x2v"][:].astype(np.float64)
        z = h5["x3v"][:].astype(np.float64)

    # The first/last cell of each block has no same-block centered stencil.
    dx = (x[:, 1] - x[:, 0])[:, None, None, None]
    dy = (y[:, 1] - y[:, 0])[:, None, None, None]
    dz = (z[:, 1] - z[:, 0])[:, None, None, None]
    gx = (er[:, 1:-1, 1:-1, 2:] - er[:, 1:-1, 1:-1, :-2]) / (2 * dx)
    gy = (er[:, 1:-1, 2:, 1:-1] - er[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    gz = (er[:, 2:, 1:-1, 1:-1] - er[:, :-2, 1:-1, 1:-1]) / (2 * dz)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    er_i = er[:, 1:-1, 1:-1, 1:-1]
    rho_i = rho[:, 1:-1, 1:-1, 1:-1]
    radius = np.sqrt(x[:, None, None, 1:-1] ** 2
                     + y[:, None, 1:-1, None] ** 2
                     + z[:, 1:-1, None, None] ** 2)
    radius = np.broadcast_to(radius, er_i.shape)
    # Kappa_R=1 cm²/g. Coordinates are au; Er's code unit cancels.
    reduced_gradient = grad / (AU_CM * rho_i * er_i)
    r = reduced_gradient
    lp = (2 + r) / (6 + 3 * r + r * r)
    minerbo = np.where(r <= 1.5, 2 / (3 + np.sqrt(9 + 12 * r * r)),
                       1 / (1 + r + np.sqrt(1 + 2 * r)))
    relative = np.abs(lp / minerbo - 1)
    valid = np.isfinite(relative) & np.isfinite(r) & (rho_i > 0) & (er_i > 0)
    result = {"snapshot": args.snapshot,
              "method": "same-block centered gradients; exclude outer cell layer",
              "kappa_R_cm2_g": 1.0, "rho_unit_g_cm3": RHO_UNIT,
              "bands_au": {}}
    for lo, hi in [(0, 7), (7, 30), (30, 300), (300, 3000)]:
        mask = valid & (radius >= lo) & (radius < hi)
        if np.any(mask):
            weights = grad[mask] + np.finfo(float).tiny
            entry = summarize(relative[mask], weights)
            entry["reduced_gradient_median"] = float(np.median(r[mask]))
            entry["reduced_gradient_p95"] = float(np.percentile(r[mask], 95))
            result["bands_au"][f"{lo}-{hi}"] = entry
    with open(args.output_json, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
