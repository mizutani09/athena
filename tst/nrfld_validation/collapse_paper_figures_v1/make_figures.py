#!/usr/bin/env python3
"""Make scoped first-core publication figure candidates from the verified run."""

import argparse
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tst/nrfld_validation/wor92_firstcore_20261109"))
from analyze_core_profile import eos_temperature  # noqa: E402

RHO_UNIT = 1e-18
PRESS_UNIT = 1e-9
A_RAD = 7.5657e-15
SHOCK_INTERVAL_AU = (7.507, 7.874)
EXPECTED = {
    "history": "0b3c78fde38f3525a7d9cedfd291e9b8caa89a67c524151816b4201c8f01d806",
    "profile": "46f4225864cadf75a49b9ad7cc62c69bb1b034f54eb4db4a8c9683be722bc9cd",
    "hdf": "13767189008f53ffb9e802b7ef66dcd97faff7045adbdd59051d19269a0354ac",
    "eos": "742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.asarray([float(row[key]) for row in rows])
            for key in rows[0] if key != "job"}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "savefig.dpi": 220,
    })


def save(fig, out, stem):
    fig.savefig(out / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (stem + ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def evolution(history, out):
    # Segment joins can repeat one output row.  Keep one sample per time.
    time, unique = np.unique(history["time_kyr"], return_index=True)
    density = history["rho_max_g_cm3"][unique]
    temperature = history["Tgas_max_K"][unique]
    if not np.all(np.diff(time) > 0) or np.any(density <= 0) or np.any(temperature <= 0):
        raise ValueError("invalid collapse history")

    fig = plt.figure(figsize=(8.2, 3.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1), hspace=0.08)
    ax_rho = fig.add_subplot(grid[0, 0])
    ax_temp = fig.add_subplot(grid[1, 0])
    ax_phase = fig.add_subplot(grid[:, 1])
    ax_rho.semilogy(time, density, color="#165b78", lw=1.7)
    ax_rho.axhline(1e-10, color="0.4", ls="--", lw=0.8)
    ax_rho.set(ylabel=r"$\rho_{\max}$ [g cm$^{-3}$]", xlim=(0, 29.2))
    ax_rho.tick_params(labelbottom=False)
    ax_temp.plot(time, temperature, color="#b35331", lw=1.7)
    ax_temp.set(xlabel="Time [kyr]", ylabel=r"$T_{\mathrm{gas},\max}$ [K]",
                xlim=(28.5, 29.12))
    ax_phase.loglog(density, temperature, color="#5a497d", lw=1.8)
    ax_phase.plot(density[0], temperature[0], "o", color="#165b78", ms=4, label="Initial")
    ax_phase.plot(density[-1], temperature[-1], "o", color="#b35331", ms=5,
                  label="First-core endpoint")
    ax_phase.set(xlabel=r"$\rho_{\max}$ [g cm$^{-3}$]",
                 ylabel=r"$T_{\mathrm{gas},\max}$ [K]")
    ax_phase.legend(frameon=False, loc="lower right")
    for label, ax in zip(("(a)", "(b)", "(c)"), (ax_rho, ax_temp, ax_phase)):
        ax.text(0.02, 0.95, label, transform=ax.transAxes, va="top", ha="left",
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    for ax in (ax_rho, ax_temp, ax_phase):
        ax.grid(alpha=0.18, which="both")
    save(fig, out, "collapse_evolution")
    return {"unique_history_samples": len(time), "last_time_kyr": float(time[-1]),
            "last_rho_max": float(density[-1]), "last_Tgas_max": float(temperature[-1])}


def final_profiles(profile, out):
    r0 = profile["r_inner_au"]
    r1 = profile["r_outer_au"]
    radius = np.sqrt(r0 * r1)
    radius[0] = r1[0] / 2.0
    valid = ((radius >= 0.25) & (radius <= 100.0)
             & (profile["cells"] > 0)
             & np.isfinite(profile["mass_weighted_temp_K"]))
    radius = radius[valid]
    density = profile["mean_rho_g_cm3"][valid]
    tgas = profile["mass_weighted_temp_K"][valid]
    trad_from_mean_er = (profile["mean_erad_erg_cm3"][valid] / A_RAD) ** 0.25
    vr = profile["mass_weighted_vrad_km_s"][valid]
    if np.any(density <= 0) or np.any(tgas <= 0) or np.any(~np.isfinite(trad_from_mean_er)):
        raise ValueError("invalid final radial profile")

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), layout="constrained")
    axes[0].loglog(radius, density, color="#165b78", lw=1.8)
    axes[0].set(ylabel=r"Shell-mean $\rho$ [g cm$^{-3}$]")
    axes[1].semilogx(radius, tgas, color="#b35331", lw=1.8,
                     label=r"Gas $T$ (mass weighted)")
    axes[1].semilogx(radius, trad_from_mean_er, color="#5a497d", lw=1.5, ls="--",
                     label=r"$(\langle E_\mathrm{r}\rangle/a_\mathrm{r})^{1/4}$")
    axes[1].set(ylabel="Temperature [K]")
    axes[1].legend(frameon=False, loc="upper right")
    axes[2].semilogx(radius, vr, color="#3c7651", lw=1.8)
    axes[2].axhline(0, color="0.4", lw=0.7)
    axes[2].set(ylabel=r"Mass-weighted $v_r$ [km s$^{-1}$]")
    for ax in axes:
        ax.axvspan(*SHOCK_INTERVAL_AU, color="#b35331", alpha=0.15, lw=0)
        ax.set(xlabel="Radius [au]", xlim=(0.25, 100))
        ax.grid(alpha=0.18, which="both")
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top", ha="left",
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    save(fig, out, "collapse_final_radial")
    return {"radial_bins_shown": int(len(radius)),
            "shock_interval_xaxis_au": list(SHOCK_INTERVAL_AU)}


def xy_slice(hdf, eos_table, out):
    # Native leaf-cell values at the cell-centered plane nearest z=0. No smoothing.
    extent_au = 32.0
    n = 512
    centers = (np.arange(n) + 0.5) * (2 * extent_au / n) - extent_au
    rho_map = np.full((n, n), np.nan)
    p_map = np.full((n, n), np.nan)
    with h5py.File(hdf) as h:
        names = [s.decode() for s in h.attrs["VariableNames"]]
        prim_names = names[:int(h.attrs["NumVariables"][0])]
        rho_idx, p_idx = prim_names.index("rho"), prim_names.index("press")
        for b in range(h["prim"].shape[1]):
            # The collapse input defines one code-length unit as one au.
            zf = np.asarray(h["x3f"][b], dtype=float)
            if not (zf[0] <= 0 < zf[-1]):
                continue
            k = int(np.searchsorted(zf, 0, side="right") - 1)
            xf = np.asarray(h["x1f"][b], dtype=float)
            yf = np.asarray(h["x2f"][b], dtype=float)
            if xf[-1] < -extent_au or xf[0] > extent_au or yf[-1] < -extent_au or yf[0] > extent_au:
                continue
            rho = np.asarray(h["prim"][rho_idx, b, k], dtype=float) * RHO_UNIT
            press = np.asarray(h["prim"][p_idx, b, k], dtype=float) * PRESS_UNIT
            for j in range(8):
                iy0, iy1 = np.searchsorted(centers, [yf[j], yf[j + 1]], side="left")
                for i in range(8):
                    ix0, ix1 = np.searchsorted(centers, [xf[i], xf[i + 1]], side="left")
                    rho_map[iy0:iy1, ix0:ix1] = rho[j, i]
                    p_map[iy0:iy1, ix0:ix1] = press[j, i]
        cycle = int(h.attrs["NumCycles"])
    if cycle != 1080 or not np.isfinite(rho_map).all() or not np.isfinite(p_map).all():
        raise ValueError("unexpected HDF cycle or uncovered slice pixels")
    temp_map, outside = eos_temperature(eos_table, rho_map.ravel(), p_map.ravel())
    if outside:
        raise ValueError("slice leaves EOS table domain")
    temp_map = temp_map.reshape(n, n)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained")
    bounds = (-extent_au, extent_au, -extent_au, extent_au)
    im0 = axes[0].imshow(rho_map, origin="lower", extent=bounds,
                         norm=LogNorm(vmin=1e-15, vmax=1.1e-10), cmap="viridis")
    im1 = axes[1].imshow(temp_map, origin="lower", extent=bounds,
                         vmin=14, vmax=440, cmap="inferno")
    fig.colorbar(im0, ax=axes[0], label=r"$\rho$ [g cm$^{-3}$]", pad=0.02)
    fig.colorbar(im1, ax=axes[1], label=r"$T_\mathrm{gas}$ [K]", pad=0.02)
    for label, ax in zip(("(a)", "(b)"), axes):
        ax.set(xlabel="x [au]", ylabel="y [au]")
        ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top", ha="left",
                color="white", fontweight="bold")
    save(fig, out, "collapse_final_xy_slice")
    return {"slice_cycle": cycle, "plane": "cell containing z=0",
            "extent_au": extent_au, "pixels_per_axis": n,
            "eos_outside_table": outside,
            "rho_min": float(rho_map.min()), "rho_max": float(rho_map.max()),
            "T_min": float(temp_map.min()), "T_max": float(temp_map.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--eos", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    paths = {"history": HERE / "data/history.csv",
             "profile": HERE / "data/final_radial_profile.csv",
             "hdf": args.hdf, "eos": args.eos}
    hashes = {name: sha256(path) for name, path in paths.items()}
    for name, digest in hashes.items():
        if digest != EXPECTED[name]:
            raise ValueError(f"{name} SHA256 mismatch: {digest}")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    result = {"source_commit": "ff60f8f07842acb00e7753932bcfe30867e3a66e",
              "solver_src_tree": "c4ab200383c7317a2feccc50aa9578d5cf71478d",
              "input_sha256": "aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6",
              "sha256": hashes}
    result["evolution"] = evolution(read_csv(paths["history"]), args.out)
    result["radial"] = final_profiles(read_csv(paths["profile"]), args.out)
    result["slice"] = xy_slice(paths["hdf"], paths["eos"], args.out)
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    review_html(args.out)
    print(json.dumps(result, indent=2))


def review_html(out):
    figures = [
        ("最終状態の半径分布（主図候補）", "collapse_final_radial",
         "Density and mass-weighted radial velocity show a compact, slowly moving inner region "
         "bounded by an accretion-shock candidate at 7.507–7.874 au (shading, inferred from "
         "the +x-axis profile). Gas temperature is mass weighted; the radiation-equivalent "
         "temperature is derived from the shell-mean radiation energy density."),
        ("最高密度と温度の進化（主図候補）", "collapse_evolution",
         "The domain maxima of gas density and temperature rise to "
         "1.044×10⁻¹⁰ g cm⁻³ and 434.524 K at 29.110 kyr. The right panel combines the "
         "two maxima at each output time; it is not a trajectory of one fluid element."),
        ("最終 xy 断面（補助図候補）", "collapse_final_xy_slice",
         "Native leaf-cell density and table-EOS gas temperature in the cell-centered plane "
         "containing z=0, rasterized without smoothing. The field of view is 64×64 au."),
    ]
    sections = []
    for title, stem, caption in figures:
        data = base64.b64encode((out / (stem + ".png")).read_bytes()).decode("ascii")
        sections.append(f'<section><h2>{title}</h2><img src="data:image/png;base64,{data}" '
                        f'alt="{title}"><p>{caption}</p><p><a href="{stem}.pdf">PDF</a></p></section>')
    html = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>Collapse paper figure candidates</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;padding:1.4rem;color:#263743;background:#f5f7f8}
section{background:#fff;border:1px solid #dfe6e9;border-radius:8px;padding:1rem;margin:1rem 0}
img{max-width:100%;height:auto}a{color:#165b78}code{word-break:break-all}
</style></head><body><h1>Table-EOS first-core collapse: figure candidates</h1>
<p>XD jobs 338469/338485/338486; source <code>ff60f8f0</code>; cycle 0–1080. These are draft figures for
showing that the specified table EOS and variable opacity configuration reached a first-core candidate.
No matched-condition agreement with Mayer et al. is claimed.</p>"""
    html += "".join(sections)
    html += """<section><h2>図を選ぶときの注意</h2><ul>
<li>主図は半径分布と進化の2枚を推奨。xy断面は3D形状を示したい場合の補助図。</li>
<li>背景ガス・放射場は固定処理があるため、全エネルギー保存を主張しない。</li>
<li>線形FGMRESはbounded-inexact cap 1e−7を使用。Newton収束、Jeans解像度、質量保存は別途確認済み。</li>
<li>第一コアまでを対象とし、H₂解離・第二コアは範囲外。</li>
</ul><p>構図の参考：<a href="https://arxiv.org/html/2510.12620v4">Mayer et al. (2025), Figs. 7–9</a>、
<a href="https://arxiv.org/html/1501.04102v1">Tomida et al. (2015), Figs. 1, 2, 4</a>。
定量比較ではなく図の診断量を参考にした。</p><p>入力データと SHA256 は <code>manifest.json</code> に記録。</p></section>
</body></html>"""
    (out / "review.html").write_text(html)


if __name__ == "__main__":
    main()
