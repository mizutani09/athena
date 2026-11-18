#!/usr/bin/env python3
"""Compare the radiatively cooled Henyey MLT RSG trial with prior profiles."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/rsg"))
from make_fld_exterior_trial import fld_luminosity  # noqa: E402
from make_gray_rsg_hse import LSUN, RSUN, AthenaEos, Opacity  # noqa: E402
from make_mlt_rsg_hse_trial import local_closure  # noqa: E402

FOLDER = ROOT / "tst/nrfld_validation/wor122_henyey_mlt_20261118"
FIGURE = ROOT / "doc/figures/wor122_henyey_mlt_20261118.png"
ALPHAS = (2.0, 2.2, 2.35, 2.36, 2.366, 2.4, 2.6, 3.0)


def tag(alpha):
    return (str(int(alpha)) if float(alpha).is_integer() else str(alpha)).replace(
        ".", "p")


def width(profile):
    radius = profile[:, 0]/RSUN
    lum = fld_luminosity(profile, 91932*LSUN)
    mask = (radius > 800) & (radius < 830)
    local_radius, local_lum = radius[mask], lum[mask]
    crossings = []
    for level in (.1, .9):
        index = np.flatnonzero((local_lum[:-1] < level)
                               & (local_lum[1:] >= level))
        if index.size == 0:
            raise ValueError(f"no outward FLD luminosity crossing at {level}")
        i = index[0]
        crossings.append(float(local_radius[i] + (level-local_lum[i])
                               *(local_radius[i+1]-local_radius[i])
                               /(local_lum[i+1]-local_lum[i])))
    low, high = crossings
    return radius, lum, high-low


def main():
    masses = {}
    for alpha in ALPHAS:
        suffix = "_fine" if alpha == 2.366 else ""
        path = FOLDER / f"alpha{tag(alpha)}{suffix}_base.dat.json"
        data = json.loads(path.read_text())
        masses[str(alpha)] = data["mapped_mass_msun"]
    trial_data = json.loads((FOLDER / "alpha2p366_fine_base.dat.json").read_text())
    base = np.loadtxt(FOLDER / "alpha2p366_fine_base.dat")
    trial = np.loadtxt(FOLDER / "alpha2p366_fine_fld.dat")
    reference = np.loadtxt(ROOT / "data/ic/st35gm03n07_wor55_lte_fld_trial.dat")
    efficient = np.loadtxt(ROOT / "tst/nrfld_validation/"
                           "wor122_mlt_flux_closed_20261118/alpha1p537_fine_fld.dat")
    rr, ll, rw = width(reference)
    er, el, ew = width(efficient)
    tr, tl, tw = width(trial)

    args = SimpleNamespace(radius_rsun=830., potential_mass_msun=12.,
                           r0_fraction=.2, r1_fraction=1.2,
                           luminosity_lsun=91932., heating_radius_fraction=.2,
                           alpha=2.366, velocity_coefficient=8.,
                           mlt_option="henyey")
    eos = AthenaEos(ROOT / "data/eos/rsg_tomida2013_material_eos.tab")
    opacity = Opacity(ROOT / "data/opacity/rsg_gray_opacity_wor55.h5")
    pressure_gradient = {}
    base_radius = base[:, 0]/RSUN
    for lower, upper in ((800, 810), (810, 813), (813, 817),
                         (815, 819), (817, 821)):
        indices = [int(np.argmin(abs(base_radius-r))) for r in (lower, upper)]
        states = [local_closure(eos, opacity, base[i, 0], base[i, 5],
                                base[i, 4], args) for i in indices]
        gradient = ((states[1]["turbulent_pressure"]
                     -states[0]["turbulent_pressure"])
                    /(base[indices[1], 0]-base[indices[0], 0]))
        mean_weight = np.mean([s["density"]*s["gravity"] for s in states])
        pressure_gradient[f"{lower}-{upper} Rsun"] = {
            "raw_rho_v2_gradient_over_rho_g": float(gradient/mean_weight),
            "one_third_isotropic_estimate_over_rho_g":
                float(gradient/(3*mean_weight)),
        }
    summary = {
        "literature": "Ludwig, Freytag & Steffen 1999, Appendix A, Henyey row",
        "corrected_henyey_coefficients": trial_data["mlt_coefficients"],
        "assumptions": "Diffusion interior; no turbulent pressure in HSE; "
                       "static 3D velocity field not initialized",
        "mapped_mass_msun_by_alpha": masses,
        "selected_mass_inside_radius_msun": trial_data[
            "mass_inside_radius_msun"],
        "selected_resolved_interior_hse_max_abs": trial_data[
            "max_abs_resolved_interior_hse"],
        "selected_local_samples": trial_data["samples"],
        "reference_fld_10_90_width_rsun": rw,
        "efficient_mlt_fld_10_90_width_rsun": ew,
        "henyey_fld_10_90_width_rsun": tw,
        "raw_mlt_turbulent_pressure_gradient": pressure_gradient,
    }
    (FOLDER / "scan_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axs[0, 0].plot(ALPHAS, [masses[str(a)] for a in ALPHAS], "o-")
    axs[0, 0].axhline(3, color="k", linestyle="--", linewidth=.8)
    axs[0, 0].axvline(2.366, color="tab:red", linestyle=":", linewidth=.8)
    axs[0, 0].set(xlabel="Henyey MLT alpha", ylabel="Mapped mass [Msun]",
                  yscale="log", title="Mass matched by alpha = 2.366")
    samples = trial_data["samples"]
    radii = [v["radius_rsun"] for v in samples.values()]
    axs[0, 1].plot(radii, [v["eddy_efficiency_gamma"]
                            for v in samples.values()], "o-")
    axs[0, 1].axhline(1, color="k", linestyle="--", linewidth=.8)
    axs[0, 1].set(xlabel="Radius [Rsun]", ylabel="Henyey Gamma",
                  yscale="log", title="Eddy cooling sharpens the surface switch")
    axs[1, 0].plot(radii, [v["turbulent_pressure_over_gas"]
                            for v in samples.values()], "o-")
    axs[1, 0].set(xlabel="Radius [Rsun]", ylabel="Raw rho v² / Pgas",
                  title="MLT pressure scale is large near the surface")
    axs[1, 1].plot(rr, ll, label=f"Current: {rw:.2f} Rsun")
    axs[1, 1].plot(er, el, label=f"Efficient MLT: {ew:.2f} Rsun")
    axs[1, 1].plot(tr, tl, label=f"Henyey MLT: {tw:.2f} Rsun")
    axs[1, 1].axhline(.1, color="0.6", linestyle=":", linewidth=.8)
    axs[1, 1].axhline(.9, color="0.6", linestyle=":", linewidth=.8)
    axs[1, 1].set(xlim=(810, 820), ylim=(0, 1.05),
                  xlabel="Radius [Rsun]", ylabel="FLD luminosity / Lstar",
                  title="The transition remains below one 256-cubed cell")
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
