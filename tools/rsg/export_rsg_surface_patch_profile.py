#!/usr/bin/env python3
"""Export an RSG 1D profile for the existing Cartesian convection-box pgen.

This is a local surface diagnostic, not a full-star model.  The box can use
the imported radial gravity and a Marshak radiation boundary.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RSUN = 6.957e10
G = 6.67430e-8
MSUN = 1.98847e33


def gravity_cgs(r, mass_msun=12.0, rcore_rsun=166.0, router_rsun=996.0):
    """Magnitude of the Freytag potential acceleration in the RSG pgen."""
    c = rcore_rsun * RSUN
    o = router_rsun * RSUN
    s = (r / o) ** 8
    a = c**4 + r**4 / np.sqrt(1.0 + s)
    return G * mass_msun * MSUN * r**3 / ((1.0 + s) ** 1.5 * a**1.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--reference-rsun", type=float, default=830.0)
    args = ap.parse_args()

    p = np.loadtxt(args.profile)
    if p.ndim != 2 or p.shape[1] < 6 or np.any(np.diff(p[:, 0]) <= 0):
        raise ValueError("expected increasing r and columns r,rho,egas,erad,T,Pgas")
    r_ref = args.reference_rsun * RSUN
    if not (p[0, 0] < r_ref < p[-1, 0]):
        raise ValueError("reference radius outside profile")
    rho_ref = np.interp(r_ref, p[:, 0], p[:, 1])
    press_ref = np.interp(r_ref, p[:, 0], p[:, 5])
    grav_ref = float(gravity_cgs(r_ref))
    if min(rho_ref, press_ref, grav_ref) <= 0:
        raise ValueError("invalid reference scale")

    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((p[:, 0] / r_ref, p[:, 1] / rho_ref,
                            p[:, 5] / press_ref,
                            gravity_cgs(p[:, 0]) / grav_ref))
    np.savetxt(out.with_suffix(".dat"), data,
               header="r/r_ref rho/rho_ref Pgas/Pgas_ref g/g_ref")
    scales = (f"r_b_cgs = {r_ref:.17e}\n"
              f"r_b_rsun = {args.reference_rsun:.17e}\n"
              f"rho_b_cgs = {rho_ref:.17e}\n"
              f"p_b_cgs = {press_ref:.17e}\n"
              f"g_b_cgs = {grav_ref:.17e}\n")
    out.with_suffix(".scales").write_text(scales)
    manifest = {"input": str(args.profile),
                "input_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                "reference_rsun": args.reference_rsun,
                "rho_ref_cgs": rho_ref, "pgas_ref_cgs": press_ref,
                "gravity_ref_cgs": grav_ref,
                "source_columns": ["r_cm", "rho_g_cm3", "Pgas_dyn_cm2"],
                "gravity": "WOR-14 fixed Freytag potential, Mpot=12 Msun, rcore=166 Rsun, router=996 Rsun"}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
