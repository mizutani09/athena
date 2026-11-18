#!/usr/bin/env python3
"""Compare deep flux closure and surface FLD width for RSG 1D trial profiles."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/rsg"))
from make_fld_exterior_trial import LSUN, RSUN, fld_luminosity  # noqa: E402

TRIALS = (
    ("reference", ROOT / "data/ic/st35gm03n07_wor55_lte_fld_trial.dat",
     ROOT / "tst/nrfld_validation/wor122_flux_closure_gap_20261118.json",
     ROOT / "data/ic/st35gm03n07_wor55_t3227p8_gray_base.dat.json"),
    *[(f"width {width} Rstar",
       ROOT / f"tst/nrfld_validation/wor122_flux_closure_scan_20261118/width{tag}_fld.dat",
       ROOT / f"tst/nrfld_validation/wor122_flux_closure_scan_20261118/width{tag}_closure.json",
       ROOT / f"tst/nrfld_validation/wor122_flux_closure_scan_20261118/width{tag}_base.dat.json")
      for width, tag in ((0.2, "0p2"), (0.3, "0p3"), (0.4, "0p4"))],
)
FIGURE = ROOT / "doc/figures/wor122_flux_closure_scan_20261118.png"
SUMMARY = ROOT / "tst/nrfld_validation/wor122_flux_closure_scan_20261118/scan_summary.json"
LUMINOSITY = 91932 * LSUN


def main():
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8),
                                   constrained_layout=True)
    summary = {}
    for label, profile_path, closure_path, metadata_path in TRIALS:
        profile = np.loadtxt(profile_path)
        points = json.loads(closure_path.read_text())["points"]
        metadata = json.loads(metadata_path.read_text())
        radius = profile[:, 0] / RSUN
        lum = fld_luminosity(profile, LUMINOSITY)
        surface = (radius >= 795) & (radius <= 850)
        low, high = [float(np.interp(level, lum[surface], radius[surface]))
                     for level in (0.1, 0.9)]
        rr = [point["radius_rsun"] for point in points.values()]
        capacity = [point[
            "optimistic_convective_capacity_over_required_by_window"]["5Rsun"]
            for point in points.values()]
        line, = ax0.plot(rr, np.maximum(capacity, 1e-5), marker="o",
                         label=label)
        ax1.plot(radius, lum, color=line.get_color(),
                 label=f"{label}: {high-low:.2f} Rsun")
        summary[label] = {
            "profile": str(profile_path.relative_to(ROOT)),
            "mapped_mass_msun": metadata["mapped_mass_msun"],
            "max_abs_resolved_interior_hse": metadata["hse_relative"][
                "max_abs_resolved_interior"],
            "fld_luminosity_10_percent_radius_rsun": low,
            "fld_luminosity_90_percent_radius_rsun": high,
            "fld_transition_width_rsun": high-low,
            "optimistic_capacity_by_radius": {
                key: point[
                    "optimistic_convective_capacity_over_required_by_window"][
                        "5Rsun"] for key, point in points.items()},
        }
    ax0.axhline(1, color="black", linewidth=.8)
    ax0.set(xlim=(390, 820), ylim=(1e-5, 30), yscale="log",
            ylabel="Optimistic convective flux / required flux",
            title="Broader temperature gradient raises deep transport capacity")
    ax1.axhline(.1, color="0.5", linewidth=.7, linestyle=":")
    ax1.axhline(.9, color="0.5", linewidth=.7, linestyle=":")
    ax1.set(xlim=(805, 825), ylim=(-.05, 1.05),
            xlabel="Radius [Rsun]", ylabel="FLD luminosity / Lstar",
            title="Surface transition remains < 7.89-Rsun 256-cubed cell")
    for ax in (ax0, ax1):
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(FIGURE)
    print(SUMMARY)


if __name__ == "__main__":
    main()
