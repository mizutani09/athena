#!/usr/bin/env python3
"""Summarize the trial 1D efficient-MLT flux-closed RSG family."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/rsg"))
from make_fld_exterior_trial import fld_luminosity  # noqa: E402
from make_gray_rsg_hse import CLIGHT, G, LSUN, MSUN, RSUN, gravity  # noqa: E402

FOLDER = ROOT / "tst/nrfld_validation/wor122_mlt_flux_closed_20261118"
FIGURE = ROOT / "doc/figures/wor122_mlt_flux_closed_20261118.png"
ALPHAS = (0.1, 0.3, 1.0, 1.2, 1.4, 1.52, 1.537, 1.54, 1.56, 1.6, 2.0, 3.0)


def tag(alpha):
    return (str(int(alpha)) if float(alpha).is_integer() else str(alpha)).replace(
        ".", "p")


def transition(profile):
    radius = profile[:, 0]/RSUN
    luminosity = fld_luminosity(profile, 91932*LSUN)
    mask = (radius > 800) & (radius < 830)
    low, high = (float(np.interp(level, luminosity[mask], radius[mask]))
                 for level in (.1, .9))
    return radius, luminosity, high-low


def main():
    models = {}
    for alpha in ALPHAS:
        path = FOLDER / f"alpha{tag(alpha)}_base.dat.json"
        data = json.loads(path.read_text())
        models[str(alpha)] = {
            "mapped_mass_msun": data["mapped_mass_msun"],
            "central_logrho_logT": data["central_logrho_logT"],
            "max_abs_resolved_interior_hse":
                data["max_abs_resolved_interior_hse"],
        }
    fine = json.loads((FOLDER / "alpha1p537_fine_base.dat.json").read_text())
    fine_base = np.loadtxt(FOLDER / "alpha1p537_fine_base.dat")
    coarse_profile = np.loadtxt(FOLDER / "alpha1p537_fld.dat")
    fine_profile = np.loadtxt(FOLDER / "alpha1p537_fine_fld.dat")
    reference = np.loadtxt(ROOT / "data/ic/st35gm03n07_wor55_lte_fld_trial.dat")
    ref_radius, ref_lum, ref_width = transition(reference)
    coarse_radius, coarse_lum, coarse_width = transition(coarse_profile)
    fine_radius, fine_lum, fine_width = transition(fine_profile)
    models["1.537 fine"] = {
        "mapped_mass_msun": fine["mapped_mass_msun"],
        "central_logrho_logT": fine["central_logrho_logT"],
        "max_abs_resolved_interior_hse":
            fine["max_abs_resolved_interior_hse"],
    }
    eddy_leakage = {}
    for nominal, sample in fine["samples"].items():
        row = fine_base[np.argmin(abs(fine_base[:, 0]/RSUN - float(nominal)))]
        radius, density, _, _, _, pressure, _, _, kappa, flux = row[:10]
        grav = gravity(radius, G*12*MSUN, .2*830*RSUN, 1.2*830*RSUN)
        geff = grav-kappa*flux/CLIGHT
        height = pressure/(density*geff)
        speed = np.sqrt(sample["turbulent_pressure_over_gas"]
                        *pressure/density)
        optical_depth = kappa*density*1.537*height
        eddy_leakage[nominal] = {
            "mixing_length_optical_depth": float(optical_depth),
            "optical_depth_times_velocity_over_c": float(
                optical_depth*speed/CLIGHT),
        }
    summary = {
        "model_assumptions": "Efficient MLT, no eddy radiative leakage, "
                             "no turbulent pressure in HSE, diffusion interior",
        "models": models,
        "reference_surface_10_90_width_rsun": ref_width,
        "selected_coarse_surface_10_90_width_rsun": coarse_width,
        "selected_fine_surface_10_90_width_rsun": fine_width,
        "selected_fine_samples": fine["samples"],
        "selected_fine_eddy_leakage_scale": eddy_leakage,
    }
    (FOLDER / "scan_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    aa = np.array(ALPHAS)
    mm = np.array([models[str(a)]["mapped_mass_msun"] for a in ALPHAS])
    axs[0, 0].plot(aa, mm, "o-")
    axs[0, 0].axhline(3, color="black", linestyle="--", linewidth=.8)
    axs[0, 0].axvline(1.537, color="tab:red", linestyle=":", linewidth=.9)
    axs[0, 0].set(xlabel="Efficient MLT alpha", ylabel="Mapped mass [Msun]",
                  yscale="log", title="Flux closure changes the structure")

    radii = [v["radius_rsun"] for v in fine["samples"].values()]
    axs[0, 1].plot(radii, [v["radiative_luminosity_fraction"]
                            for v in fine["samples"].values()], "o-")
    axs[0, 1].set(xlabel="Radius [Rsun]", ylabel="Lrad / Lstar",
                  ylim=(0, 1.05), title="Selected 3-Msun trial")
    axs[1, 0].plot(radii, [v["turbulent_pressure_over_gas"]
                            for v in fine["samples"].values()], "o-")
    axs[1, 0].set(xlabel="Radius [Rsun]", ylabel="Estimated Pturb / Pgas",
                  ylim=(0, .4), title="Support omitted from the integrated HSE")

    axs[1, 1].plot(ref_radius, ref_lum,
                   label=f"Reference: {ref_width:.2f} Rsun")
    axs[1, 1].plot(coarse_radius, coarse_lum,
                   label=f"Trial 768 points: {coarse_width:.2f} Rsun")
    axs[1, 1].plot(fine_radius, fine_lum,
                   label=f"Trial 3072 points: {fine_width:.2f} Rsun")
    axs[1, 1].axhline(.1, color="0.6", linestyle=":", linewidth=.7)
    axs[1, 1].axhline(.9, color="0.6", linestyle=":", linewidth=.7)
    axs[1, 1].set(xlim=(810, 820), ylim=(0, 1.05),
                  xlabel="Radius [Rsun]", ylabel="FLD luminosity / Lstar",
                  title="Surface transition is below one 256-cubed cell")
    axs[1, 1].legend(fontsize=8)
    for ax in axs.ravel():
        ax.grid(alpha=.25, which="both")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print(FOLDER / "scan_summary.json")
    print(FIGURE)


if __name__ == "__main__":
    main()
