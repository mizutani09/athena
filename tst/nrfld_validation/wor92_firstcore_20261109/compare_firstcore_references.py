#!/usr/bin/env python3
"""Compare WOR-92 c1010 with published first-core and analytic limit references."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
G = 6.67430e-8
KB = 1.380649e-16
MH = 1.6735575e-24
AU = 1.495978707e13
MSUN = 1.98847e33
YEAR = 31557600.0
T0 = 14.0
MU = 2.381


def read_csv(name):
    with (ROOT / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def run():
    profile = read_csv("c1010_profile.csv")
    paper = read_csv("mayer_fig7_approx_read_off.csv")
    radius = (profile["r_inner_au"] + profile["r_outer_au"]) / 2
    rho = profile["mean_rho_g_cm3"]
    temp = profile["mass_weighted_temp_K"]
    vin = -profile["mass_weighted_vrad_km_s"]
    sound = np.sqrt(KB*T0/(MU*MH))
    rho_lp = 8.86*sound**2/(4*np.pi*G*(radius*AU)**2)
    rho_sis = sound**2/(2*np.pi*G*(radius*AU)**2)
    envelope = (radius >= 20) & (radius <= 300)
    ratio = rho[envelope]/rho_lp[envelope]
    slope = np.polyfit(np.log10(radius[envelope]), np.log10(rho[envelope]), 1)[0]
    tff = np.sqrt(3*np.pi/(32*G*(3*MSUN/(4*np.pi*(3000*AU)**3))))/YEAR

    def at(r, values, logarithmic=False):
        if logarithmic:
            return float(10**np.interp(np.log10(r), np.log10(radius), np.log10(values)))
        return float(np.interp(r, radius, values))

    mayer_points = []
    for i, r in enumerate(paper["r_au"]):
        model_rho = at(r, rho, True)
        model_t = at(r, temp)
        model_v = at(r, vin)
        mayer_points.append({"radius_au": float(r),
                             "paper_rho_est_g_cm3": float(paper["rho_est_g_cm3"][i]),
                             "model_rho_g_cm3": model_rho,
                             "density_ratio": model_rho/paper["rho_est_g_cm3"][i],
                             "paper_T_est_K": float(paper["Tgas_est_K"][i]),
                             "model_T_K": model_t,
                             "temperature_ratio": model_t/paper["Tgas_est_K"][i],
                             "paper_infall_est_km_s": float(-paper["vr_est_km_s"][i]),
                             "model_infall_km_s": model_v,
                             "infall_ratio": model_v/(-paper["vr_est_km_s"][i])})
    # The final 3D shell mass is cumulative through the outer edge of each shell.
    enclosed = float(np.interp(7.8, profile["r_outer_au"],
                                profile["enclosed_mass_g"])/MSUN)
    rho_rev = np.log10(rho[::-1])
    t_at_density = {str(target): float(np.interp(np.log10(target), rho_rev, temp[::-1]))
                    for target in (1e-12, 1e-11)}
    results = {
        "snapshot_cycle": 1010,
        "snapshot_time_kyr": 29.109536454968722,
        "initial_uniform_density_g_cm3": 3*MSUN/(4*np.pi*(3000*AU)**3),
        "pressureless_uniform_sphere_tff_kyr": tff/1000,
        "snapshot_time_over_tff": 29109.536454968722/tff,
        "analytic_envelope": {"reference": "Gong and Ostriker 2009 Larson-Penston isothermal asymptote; not an exact RHD solution",
                              "sound_speed_km_s": sound/1e5,
                              "window_au": [20, 300],
                              "shells": int(np.count_nonzero(envelope)),
                              "median_rho_over_LP": float(np.median(ratio)),
                              "min_rho_over_LP": float(np.min(ratio)),
                              "max_rho_over_LP": float(np.max(ratio)),
                              "rms_log10_deviation_dex": float(np.sqrt(np.mean(np.log10(ratio)**2))),
                              "fitted_density_power_law": float(slope),
                              "T_range_K": [float(np.min(temp[envelope])),float(np.max(temp[envelope]))]},
        "mayer_2025": {"source": "https://arxiv.org/html/2510.12620#S5",
                       "figure": "Figure 7, approximate raster PDF read-off from prior Mayer5 audit",
                       "published_first_core_time_kyr": 29.09,
                       "published_shock_radius_au": [6,7],
                       "estimated_radial_points": mayer_points,
                       "published_phase_temperature_300_to_400_K_at_rho_g_cm3": [1e-12,1e-11],
                       "model_shell_temperature_at_fixed_density_K": t_at_density,
                       "limitation": "AREPO constant EOS/opacity, approximate digitization; not a pointwise numerical acceptance test"},
        "commercon_2011": {"source": "https://arxiv.org/html/1102.2921v1#S4.SS1",
                           "table": "Table 1, FLD at central rho=1e-10 g/cm3",
                           "published_first_core_radius_au": 8.1,
                           "published_first_core_mass_msun": 0.0233,
                           "published_central_temperature_K": 411,
                           "model_shock_radius_interval_au": [7.50732421875,7.87353515625],
                           "model_mass_within_7p8au_msun": enclosed,
                           "model_peak_temperature_K": 431.26325446369225,
                           "limitation": "Different initial temperature, cloud radius, and boundary conditions; qualitative cross-code reference"}}
    (ROOT / "firstcore_reference_comparison.json").write_text(json.dumps(results,indent=2)+"\n")

    fig, axes = plt.subplots(3,1,figsize=(8.4,9.5),sharex=True,layout="constrained")
    for ax in axes:
        ax.axvspan(6,7,color="#b7c9d1",alpha=.25)
        ax.axvspan(7.5073,7.8735,color="#e2b279",alpha=.18)
        ax.grid(True,which="both",alpha=.22)
        ax.set_xscale("log")
        ax.set_xlim(.6,500)
    axes[0].loglog(radius,rho,color="#126c80",label="WOR-92 3D shells")
    axes[0].loglog(radius,rho_lp,"--",color="#9a502e",label="Larson–Penston, 14 K")
    axes[0].loglog(radius,rho_sis,":",color="#715da8",label="Singular isothermal sphere")
    axes[0].scatter(paper["r_au"],paper["rho_est_g_cm3"],marker="o",facecolors="none",
                    edgecolors="#a12835",label="Mayer Fig. 7 approximate read-off")
    axes[0].set_ylabel(r"Density [g cm$^{-3}$]")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(radius,temp,color="#126c80",label="WOR-92 3D shells")
    axes[1].scatter(paper["r_au"],paper["Tgas_est_K"],marker="o",facecolors="none",edgecolors="#a12835")
    axes[1].axhline(T0,color="0.5",ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gas temperature [K]")
    axes[2].semilogx(radius,vin,color="#126c80")
    axes[2].scatter(paper["r_au"],-paper["vr_est_km_s"],marker="o",facecolors="none",edgecolors="#a12835")
    axes[2].set_ylabel("Infall speed [km/s]")
    axes[2].set_xlabel("Radius [au]")
    fig.suptitle("WOR-92 first core: analytic envelope and Mayer Section 5")
    fig.savefig(ROOT/"firstcore_reference_comparison.png",dpi=155)
    print(json.dumps({"LP_median":results["analytic_envelope"]["median_rho_over_LP"],
                      "LP_slope":slope,"tff_kyr":tff/1000,"core_mass":enclosed},indent=2))


if __name__ == "__main__":
    run()
