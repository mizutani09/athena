#!/usr/bin/env python3
"""Summarize an RSG primitive/user-output pair without loading the full mesh."""

import argparse
import json

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("primitive")
    parser.add_argument("user_output")
    args = parser.parse_args()

    core_mass = core_v2 = shell_mass = shell_temp = 0.0
    min_er = np.inf
    shell_mach = global_mach = 0.0
    global_location = None
    sector_v2 = np.zeros(2)
    sector_mass = np.zeros(2)
    sector_count = np.zeros(2, dtype=int)

    with h5py.File(args.primitive) as prim, h5py.File(args.user_output) as uov:
        for lo in range(0, prim["prim"].shape[1], 32):
            hi = min(lo + 32, prim["prim"].shape[1])
            state = prim["prim"][:, lo:hi].astype(float)
            temperature = uov["user_out_var"][0, lo:hi].astype(float)
            x, y, z = (prim[f"x{axis}v"][lo:hi] for axis in (1, 2, 3))
            xx, yy, zz = x[:, None, None, :], y[:, None, :, None], z[:, :, None, None]
            radius = np.sqrt(xx * xx + yy * yy + zz * zz)
            safe_radius = np.maximum(radius, 1.0e-300)
            dx, dy, dz = (np.diff(prim[f"x{axis}f"][lo:hi], axis=1)
                          for axis in (1, 2, 3))
            volume = dz[:, :, None, None] * dy[:, None, :, None] * dx[:, None, None, :]

            rho, pressure = state[0], state[1]
            mass = rho * volume
            speed2 = np.sum(state[2:5] ** 2, axis=0)
            mach = np.sqrt(speed2 / ((5.0 / 3.0) * pressure / rho))
            radial_velocity = (state[2] * xx + state[3] * yy
                               + state[4] * zz) / safe_radius
            core = radius < 166.0
            shell = (radius >= 800.0) & (radius <= 860.0)
            core_mass += mass[core].sum()
            core_v2 += (mass[core] * speed2[core]).sum()
            shell_mass += mass[shell].sum()
            shell_temp += (mass[shell] * temperature[shell]).sum()
            if shell.any():
                shell_mach = max(shell_mach, float(mach[shell].max()))
            min_er = min(min_er, float(state[5].min()))

            direction = np.maximum.reduce(np.broadcast_arrays(
                abs(xx) / safe_radius, abs(yy) / safe_radius,
                abs(zz) / safe_radius))
            for index, selector in enumerate((direction > 0.9, direction < 0.7)):
                selected = (radius >= 80.0) & (radius < 160.0) & selector
                sector_v2[index] += (mass[selected] * radial_velocity[selected] ** 2).sum()
                sector_mass[index] += mass[selected].sum()
                sector_count[index] += selected.sum()

            at_max = np.unravel_index(np.argmax(mach), mach.shape)
            if mach[at_max] > global_mach:
                global_mach = float(mach[at_max])
                block, k, j, i = at_max
                global_location = [float(x[block, i]), float(y[block, j]),
                                   float(z[block, k]), float(radius[at_max]),
                                   float(rho[at_max])]

        result = {
            "time": float(prim.attrs["Time"]),
            "core_radius_rsun": 166.0,
            "min_er": min_er,
            "core_rms_speed": float(np.sqrt(core_v2 / core_mass)),
            "surface_mean_temperature": shell_temp / shell_mass,
            "surface_max_mach_proxy": shell_mach,
            "global_max_mach_proxy": global_mach,
            "global_max_xyz_r_rho": global_location,
            "axis_diagonal_rms_radial_speed":
                np.sqrt(sector_v2 / sector_mass).tolist(),
            "axis_diagonal_cell_counts": sector_count.tolist(),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
