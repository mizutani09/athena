#!/usr/bin/env python3
"""Diagnose the resolved central structure of a protostellar-collapse snapshot.

The force terms are deliberately labelled as spherical proxies.  They use the
monopole enclosed mass and shell-averaged scalar pressures; magnetic tension
and the actual FLD limiter are not available in the HDF output.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AU = 1.495978707e13
MSUN = 1.98847e33
YEAR = 3.15576e7
G = 6.67430e-8
RHO_UNIT = 1.0e-18
EGAS_UNIT = 1.0e-9
VELOCITY_UNIT = np.sqrt(EGAS_UNIT/RHO_UNIT)


def finite_fill(values):
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if good.sum() < 2:
        return values
    out = values.copy()
    indices = np.arange(len(values))
    out[~good] = np.interp(indices[~good], indices[good], values[good])
    return out


def smooth(values):
    values = finite_fill(values)
    if len(values) < 3:
        return values
    padded = np.pad(values, 1, mode="edge")
    return (padded[:-2]+2.0*padded[1:-1]+padded[2:])/4.0


def crossing(radius, mach, lower=2.0, upper=12.0):
    """Innermost outward transition from subsonic to supersonic infall."""
    valid = np.isfinite(mach) & (radius >= lower) & (radius <= upper)
    indices = np.where(valid)[0]
    for left, right in zip(indices[:-1], indices[1:]):
        if right != left+1:
            continue
        if mach[left] < 1.0 <= mach[right]:
            fraction = (1.0-mach[left])/(mach[right]-mach[left])
            return float(radius[left]+fraction*(radius[right]-radius[left]))
    return None


def weighted_percentile(values, weights, percentile):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    order = np.argsort(values[valid])
    values = values[valid][order]
    weights = weights[valid][order]
    if not len(values):
        return None
    cumulative = np.cumsum(weights)
    return float(np.interp(percentile*cumulative[-1], cumulative, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path,
                        help="Optional matched-density lower-resolution summary")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.snapshot, "r") as hdf:
        names = [item.decode() for item in hdf.attrs["VariableNames"]]
        data = hdf["hydro"]

        def field(name):
            return np.asarray(data[names.index(name)], dtype=np.float64)

        rho = field("rho_cgs")
        density_code = field("dens")
        pressure_cgs = field("Pgas_cgs")
        pressure_code = field("press")
        temperature = field("Tgas_K")
        erad_cgs = field("Erad_cgs")
        b_gauss = field("B_G")
        velocity = np.stack([field("vel1"), field("vel2"), field("vel3")])
        peak = np.unravel_index(np.argmax(rho), rho.shape)
        center = np.array([hdf["x1v"][peak[0], peak[3]],
                           hdf["x2v"][peak[0], peak[2]],
                           hdf["x3v"][peak[0], peak[1]]], dtype=np.float64)
        x = np.asarray(hdf["x1v"][:], dtype=np.float64)[:, None, None, :]-center[0]
        y = np.asarray(hdf["x2v"][:], dtype=np.float64)[:, None, :, None]-center[1]
        z = np.asarray(hdf["x3v"][:], dtype=np.float64)[:, :, None, None]-center[2]
        radius = np.broadcast_to(np.sqrt(x*x+y*y+z*z), rho.shape)
        cylindrical = np.broadcast_to(np.sqrt(x*x+y*y), rho.shape)
        dx = np.diff(hdf["x1f"][:], axis=1)[:, None, None, :]
        dy = np.diff(hdf["x2f"][:], axis=1)[:, None, :, None]
        dz = np.diff(hdf["x3f"][:], axis=1)[:, :, None, None]
        volume = np.broadcast_to(dx*dy*dz, rho.shape)
        cycle = int(hdf.attrs["NumCycles"])
        time_code = float(hdf.attrs["Time"])
        blocks = int(hdf.attrs["NumMeshBlocks"])
        level = int(np.asarray(hdf["Levels"][:]).max())

    mass = rho*volume*AU**3
    core_bulk_mask = radius <= 5.0
    bulk_velocity = np.array([
        np.sum(mass[core_bulk_mask]*component[core_bulk_mask])
        / np.sum(mass[core_bulk_mask]) for component in velocity])
    velocity -= bulk_velocity[:, None, None, None, None]
    safe_radius = np.maximum(radius, 1.0e-30)
    radial_velocity = (velocity[0]*x+velocity[1]*y+velocity[2]*z)/safe_radius
    vphi = (-velocity[0]*y+velocity[1]*x)/np.maximum(cylindrical, 1.0e-30)
    speed2 = np.sum(velocity*velocity, axis=0)
    tangential2 = np.maximum(speed2-radial_velocity*radial_velocity, 0.0)
    sound_proxy = np.sqrt(np.maximum(pressure_code/density_code, 0.0))
    inward_mach = -radial_velocity/np.maximum(sound_proxy, 1.0e-30)
    beta = 8.0*np.pi*pressure_cgs/np.maximum(b_gauss*b_gauss, 1.0e-99)

    edges = np.geomspace(0.22, 50.0, 52)
    mid = np.sqrt(edges[:-1]*edges[1:])
    bin_index = np.searchsorted(edges, radius.ravel())-1
    valid = (bin_index >= 0) & (bin_index < len(mid))
    index = bin_index[valid]
    volume_flat = volume.ravel()[valid]
    mass_flat = mass.ravel()[valid]
    shell_volume = np.bincount(index, weights=volume_flat, minlength=len(mid))
    shell_mass = np.bincount(index, weights=mass_flat, minlength=len(mid))

    def volume_mean(values):
        total = np.bincount(index, weights=values.ravel()[valid]*volume_flat,
                            minlength=len(mid))
        return np.divide(total, shell_volume, out=np.full(len(mid), np.nan),
                         where=shell_volume > 0.0)

    def mass_mean(values):
        total = np.bincount(index, weights=values.ravel()[valid]*mass_flat,
                            minlength=len(mid))
        return np.divide(total, shell_mass, out=np.full(len(mid), np.nan),
                         where=shell_mass > 0.0)

    profile = {
        "rho": volume_mean(rho),
        "temperature": volume_mean(temperature),
        "pressure_cgs": volume_mean(pressure_cgs),
        "erad_cgs": volume_mean(erad_cgs),
        "magnetic_pressure_cgs": volume_mean(b_gauss*b_gauss/(8.0*np.pi)),
        "vr_kms": mass_mean(radial_velocity)*VELOCITY_UNIT/1.0e5,
        "vphi_kms": mass_mean(vphi)*VELOCITY_UNIT/1.0e5,
        "inward_mach_proxy_mass_weighted": mass_mean(inward_mach),
        "inward_mach_proxy_volume_weighted": volume_mean(inward_mach),
        "beta_mass_mean": mass_mean(beta),
        "vt2_code": mass_mean(tangential2),
    }
    profile["enclosed_mass_msun"] = np.cumsum(shell_mass)/MSUN

    r_cm = mid*AU
    rho_s = smooth(profile["rho"])
    gravity = G*(profile["enclosed_mass_msun"]*MSUN)/np.maximum(r_cm*r_cm, 1.0)
    gas_support = -np.gradient(smooth(profile["pressure_cgs"]), r_cm)/rho_s
    radiation_support = -np.gradient(smooth(profile["erad_cgs"])/3.0, r_cm)/rho_s
    magnetic_support = -np.gradient(smooth(profile["magnetic_pressure_cgs"]), r_cm)/rho_s
    rotation_support = profile["vt2_code"]*VELOCITY_UNIT**2/np.maximum(r_cm, 1.0)
    for name, values in (("gas", gas_support), ("radiation_thick_proxy", radiation_support),
                         ("magnetic_pressure_proxy", magnetic_support),
                         ("centrifugal", rotation_support)):
        profile[name+"_over_monopole_gravity"] = values/np.maximum(gravity, 1.0e-99)
    profile["scalar_support_sum_over_monopole_gravity"] = (
        gas_support+radiation_support+magnetic_support+rotation_support
    )/np.maximum(gravity, 1.0e-99)

    sonic_radius = crossing(mid, profile["inward_mach_proxy_volume_weighted"])
    if sonic_radius is None:
        core_radius = 7.0
    else:
        core_radius = sonic_radius
    core = radius <= core_radius
    core_mass = float(np.sum(mass[core])/MSUN)
    core_beta_mass_p10 = weighted_percentile(beta[core], mass[core], 0.10)
    core_beta_mass_median = weighted_percentile(beta[core], mass[core], 0.50)
    beta_lt_one_mass_fraction = float(np.sum(mass[core & (beta < 1.0)])/np.sum(mass[core]))
    beta_lt_one_volume_fraction = float(np.sum(volume[core & (beta < 1.0)])/np.sum(volume[core]))

    cos_theta = np.abs(z)/safe_radius
    polar = cos_theta >= 0.75
    equatorial = cos_theta <= 0.25
    in_plane = cos_theta <= 0.25
    plane_radius = np.maximum(cylindrical, 1.0e-30)
    axis_direction = in_plane & (np.maximum(np.abs(x), np.abs(y))/plane_radius >= np.cos(np.deg2rad(20.0)))
    diagonal_direction = in_plane & ((np.abs(x)+np.abs(y))/(np.sqrt(2.0)*plane_radius) >= np.cos(np.deg2rad(20.0)))

    def conditional_volume_mean(values, condition):
        selected = valid & condition.ravel()
        selected_index = bin_index[selected]
        selected_volume = volume.ravel()[selected]
        denominator = np.bincount(selected_index, weights=selected_volume,
                                  minlength=len(mid))
        numerator = np.bincount(selected_index,
            weights=values.ravel()[selected]*selected_volume, minlength=len(mid))
        return np.divide(numerator, denominator, out=np.full(len(mid), np.nan),
                         where=denominator > 0.0)

    polar_t = conditional_volume_mean(temperature, polar)
    equatorial_t = conditional_volume_mean(temperature, equatorial)
    axis_t = conditional_volume_mean(temperature, axis_direction)
    diagonal_t = conditional_volume_mean(temperature, diagonal_direction)
    polar_rho = conditional_volume_mean(rho, polar)
    equatorial_rho = conditional_volume_mean(rho, equatorial)
    polar_equatorial_t = polar_t/equatorial_t
    polar_equatorial_rho = polar_rho/equatorial_rho
    grid_t_ratio = axis_t/diagonal_t

    core_bins = (mid >= 1.0) & (mid <= core_radius)
    grid_valid = core_bins & np.isfinite(grid_t_ratio)
    physical_valid = core_bins & np.isfinite(polar_equatorial_t)
    grid_max_deviation = float(np.max(np.abs(grid_t_ratio[grid_valid]-1.0)))
    physical_max_deviation = float(np.max(np.abs(polar_equatorial_t[physical_valid]-1.0)))
    inner_bins = (mid >= 1.0) & (mid <= 4.0)
    transition_values = {}
    if sonic_radius is not None:
        for name in ("inward_mach_proxy_mass_weighted",
                     "scalar_support_sum_over_monopole_gravity",
                     "gas_over_monopole_gravity",
                     "centrifugal_over_monopole_gravity",
                     "beta_mass_mean"):
            transition_values[name] = float(np.interp(sonic_radius, mid, profile[name]))

    summary = {
        "status": "diagnostic_pass",
        "source_hdf": str(args.snapshot.resolve()),
        "cycle": cycle,
        "time_code": time_code,
        "meshblocks": blocks,
        "level_index_max": level,
        "density_peak_center_au": center.tolist(),
        "rho_peak_g_cm3": float(np.max(rho)),
        "temperature_peak_K": float(np.max(temperature)),
        "bulk_velocity_inside_5au_km_s": (bulk_velocity*VELOCITY_UNIT/1.0e5).tolist(),
        "volume_weighted_inner_sonic_transition_radius_au": sonic_radius,
        "candidate_core_mass_inside_transition_msun": core_mass,
        "median_scalar_support_over_gravity_1au_to_4au": float(np.nanmedian(
            profile["scalar_support_sum_over_monopole_gravity"][inner_bins])),
        "median_gas_pressure_support_over_gravity_1au_to_4au": float(np.nanmedian(
            profile["gas_over_monopole_gravity"][inner_bins])),
        "values_at_volume_weighted_sonic_transition": transition_values,
        "core_beta_mass_weighted_p10": core_beta_mass_p10,
        "core_beta_mass_weighted_median": core_beta_mass_median,
        "core_beta_lt_one_mass_fraction": beta_lt_one_mass_fraction,
        "core_beta_lt_one_volume_fraction": beta_lt_one_volume_fraction,
        "temperature_polar_equatorial_max_deviation_1au_to_core": physical_max_deviation,
        "temperature_equatorial_axis_diagonal_max_deviation_1au_to_core": grid_max_deviation,
        "limitations": [
            "P/rho is an acoustic speed proxy, not the tabulated adiabatic sound speed.",
            "Gravity is the spherical enclosed-mass monopole.",
            "The magnetic term is scalar magnetic-pressure support and omits tension.",
            "The radiation term assumes the optically thick E_r/3 pressure and does not use the unavailable local FLD limiter.",
            "A single snapshot does not prove time-stationary hydrostatic equilibrium."
        ]
    }
    reference_comparison = {}
    if args.reference_summary is not None:
        reference = json.loads(args.reference_summary.read_text())
        for key in ("volume_weighted_inner_sonic_transition_radius_au",
                    "candidate_core_mass_inside_transition_msun",
                    "median_scalar_support_over_gravity_1au_to_4au",
                    "median_gas_pressure_support_over_gravity_1au_to_4au",
                    "core_beta_mass_weighted_median",
                    "core_beta_lt_one_mass_fraction",
                    "temperature_polar_equatorial_max_deviation_1au_to_core",
                    "temperature_equatorial_axis_diagonal_max_deviation_1au_to_core"):
            low, high = reference[key], summary[key]
            reference_comparison[key] = {
                "reference": low, "current": high, "relative_change": high/low-1.0}
        summary["matched_density_resolution_comparison"] = reference_comparison
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    columns = ["radius_au"]+list(profile)+[
        "temperature_polar_over_equatorial", "density_polar_over_equatorial",
        "temperature_equatorial_axis_over_diagonal"]
    with (args.output/"radial_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i, r in enumerate(mid):
            writer.writerow([r]+[profile[name][i] for name in profile]
                            +[polar_equatorial_t[i], polar_equatorial_rho[i], grid_t_ratio[i]])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.loglog(mid, profile["rho"], label="density")
    ax2 = ax.twinx()
    ax2.semilogx(mid, profile["temperature"], color="#d95f02", label="temperature")
    ax.set(ylabel=r"$\langle\rho\rangle_V$ [g cm$^{-3}$]")
    ax2.set(ylabel="Gas temperature [K]")
    ax = axes[0, 1]
    ax.semilogx(mid, profile["inward_mach_proxy_volume_weighted"],
                label="volume weighted")
    ax.semilogx(mid, profile["inward_mach_proxy_mass_weighted"],
                label="mass weighted")
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set(ylabel=r"$-\langle v_r/\sqrt{P/\rho}\rangle$")
    ax.legend(frameon=False)
    ax = axes[1, 0]
    ax.loglog(mid, profile["enclosed_mass_msun"])
    ax.set(ylabel=r"Enclosed mass [$M_\odot$]")
    ax = axes[1, 1]
    ax.semilogx(mid, profile["vr_kms"], label=r"$v_r$")
    ax.semilogx(mid, profile["vphi_kms"], label=r"$v_\phi$")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set(ylabel="Mass-weighted velocity [km s$^{-1}$]")
    ax.legend(frameon=False)
    for ax in axes.ravel():
        ax.set(xlim=(0.3, 30.0), xlabel="Radius [au]")
        ax.grid(alpha=0.2)
        if sonic_radius is not None:
            ax.axvline(sonic_radius, color="#7570b3", ls="--", lw=1.0)
    fig.savefig(args.output/"core_structure.svg")
    fig.savefig(args.output/"core_structure.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for name, label in (("gas_over_monopole_gravity", "gas pressure"),
                        ("centrifugal_over_monopole_gravity", "centrifugal"),
                        ("radiation_thick_proxy_over_monopole_gravity", "radiation P proxy"),
                        ("magnetic_pressure_proxy_over_monopole_gravity", "magnetic P proxy"),
                        ("scalar_support_sum_over_monopole_gravity", "scalar sum")):
        axes[0].semilogx(mid, profile[name], label=label)
    axes[0].axhline(1.0, color="0.4", lw=0.8)
    axes[0].set(ylim=(-0.5, 3.0), ylabel="Outward support / monopole gravity")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].loglog(mid, profile["beta_mass_mean"])
    axes[1].axhline(1.0, color="0.4", lw=0.8)
    axes[1].set(ylabel=r"Mass-weighted plasma $\beta$")
    for ax in axes:
        ax.set(xlim=(0.5, 20.0), xlabel="Radius [au]")
        ax.grid(alpha=0.2)
        if sonic_radius is not None:
            ax.axvline(sonic_radius, color="#7570b3", ls="--", lw=1.0)
    fig.savefig(args.output/"support_and_beta.svg")
    fig.savefig(args.output/"support_and_beta.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    axes[0].semilogx(mid, polar_equatorial_t)
    axes[0].set(ylabel=r"$T_{polar}/T_{equatorial}$")
    axes[1].semilogx(mid, polar_equatorial_rho)
    axes[1].set(ylabel=r"$\rho_{polar}/\rho_{equatorial}$")
    axes[2].semilogx(mid, grid_t_ratio)
    axes[2].set(ylabel=r"$T_{axis}/T_{diagonal}$ (equatorial)")
    for ax in axes:
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set(xlim=(0.5, 20.0), xlabel="Radius [au]")
        ax.grid(alpha=0.2)
        if sonic_radius is not None:
            ax.axvline(sonic_radius, color="#7570b3", ls="--", lw=1.0)
    fig.savefig(args.output/"anisotropy.svg")
    fig.savefig(args.output/"anisotropy.png", dpi=180)
    plt.close(fig)

    items = "".join("<tr><th>{}</th><td>{}</td></tr>".format(
        html.escape(key), html.escape("{:.8g}".format(value) if isinstance(value, float)
                                     else str(value)))
        for key, value in summary.items()
        if key not in ("limitations", "source_hdf", "matched_density_resolution_comparison"))
    limitations = "".join("<li>{}</li>".format(html.escape(item))
                           for item in summary["limitations"])
    comparison_rows = "".join(
        "<tr><th>{}</th><td>{:.7g}</td><td>{:.7g}</td><td>{:+.3%}</td></tr>".format(
            html.escape(key), value["reference"], value["current"],
            value["relative_change"])
        for key, value in reference_comparison.items())
    comparison_section = ("<h2>Matched-density resolution comparison</h2>"
        "<table><tr><th>Metric</th><th>lower AMR</th><th>AMR+1</th><th>change</th></tr>"
        +comparison_rows+"</table>") if comparison_rows else ""
    page = """<!doctype html><meta charset='utf-8'><title>Collapse core diagnostics</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}}table{{border-collapse:collapse}}th,td{{padding:.35rem .7rem;border-bottom:1px solid #ccc;text-align:left}}img{{width:100%}}.note{{background:#fff4dd;padding:.8rem 1rem}}</style>
<h1>Additional first-core-candidate diagnostics</h1>
<p>The dashed line is the volume-weighted inner transition from supersonic to subsonic infall, measured with the explicitly labelled <code>sqrt(P/rho)</code> acoustic proxy. The separate mass-weighted curve diagnoses whether most of the mass shares that transition.</p>
<table>{}</table>
{}
<h2>Central radial structure</h2><img src='core_structure.svg' alt='Core radial structure'>
<h2>Spherical support proxies and plasma beta</h2><img src='support_and_beta.svg' alt='Support proxies and beta'>
<h2>Physical-axis and Cartesian-grid anisotropy</h2><img src='anisotropy.svg' alt='Anisotropy diagnostics'>
<div class='note'><strong>Interpretation limit:</strong><ul>{}</ul></div>
<p><a href='radial_diagnostics.csv'>radial CSV</a> · <a href='summary.json'>summary JSON</a> · <a href='../resolution_comparison/index.html'>resolution comparison</a></p>
""".format(items, comparison_section, limitations)
    (args.output/"index.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
