#!/usr/bin/env python3
"""Check E_total = E_gas + E_kinetic + E_magnetic in a 3D MHD+FLD dump."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from plot_simple_convection_surface3d import read_volume  # noqa: E402
from plot_simple_convection_tau import units_from_input  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prim", type=Path, required=True)
    parser.add_argument("--cons", type=Path, required=True)
    parser.add_argument("--uov", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    prim, xyz, time = read_volume(args.prim, "prim")
    bcc, xyz_b, time_b = read_volume(args.prim, "B")
    cons, xyz_c, time_c = read_volume(args.cons, "cons")
    uov, xyz_u, time_u = read_volume(args.uov, "user_out_var")
    if not (time == time_b == time_c == time_u and all(
        np.array_equal(a, b) for coords in (xyz_b, xyz_c, xyz_u)
        for a, b in zip(xyz, coords)
    )):
        parser.error("Mismatched volume times or grids")
    egas_unit = units_from_input(args.input)["egas_unit"]
    gas = uov[0]/egas_unit
    kinetic = 0.5*prim[0]*np.sum(prim[2:5]**2, axis=0)
    magnetic = 0.5*np.sum(bcc**2, axis=0)
    # Athena++ stores conserved variables as (rho, total energy, M1, M2, M3).
    residual = cons[1] - gas - kinetic - magnetic
    if not (np.isfinite(residual).all() and np.isfinite(gas).all()
            and (gas > 0).all()):
        raise ValueError("Nonfinite or nonpositive gas-energy state")
    relative = np.abs(residual)/gas
    summary = {
        "time_code": time,
        "max_abs_energy_residual_code": float(np.max(np.abs(residual))),
        "max_abs_residual_over_gas": float(np.max(relative)),
        "mean_abs_residual_over_gas": float(np.mean(relative)),
        "max_magnetic_over_gas": float(np.max(magnetic/gas)),
        "tolerance": args.tolerance,
        "pass": bool(np.max(relative) < args.tolerance),
    }
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
