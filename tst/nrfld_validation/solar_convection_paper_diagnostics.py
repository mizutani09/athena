#!/usr/bin/env python3
"""Paper-style convection diagnostics from a frozen prefix of a live campaign.

Only completed segments listed in state.json are read. The report is a
numerical/physical diagnostic, not an emergent-intensity synthesis or a claim
that a grey FLD photosphere reproduces the observed Sun.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from plot_simple_convection_surface3d import read_volume, surface_at  # noqa: E402
from plot_simple_convection_tau import opacity_interpolator, units_from_input  # noqa: E402

F_SUN = 6.3e10
T_SUN = 5772.0
PAPERS = [
    ("Rempel (2014)", "https://arxiv.org/abs/1405.6814",
     "inflow-entropy calibration, mean flow and resolution sensitivity"),
    ("Beeck et al. (2012)", "https://arxiv.org/abs/1201.1103",
     "mean stratification, velocity structure and optical-surface diagnostics"),
    ("Stein & Nordlund (2000)",
     "https://web.pa.msu.edu/people/steinr/papers/realistic_convection.00.pdf",
     "flux partition, solar granulation and radiative-transfer observables"),
    ("Stein & Nordlund (2002)", "https://arxiv.org/abs/astro-ph/0209510",
     "radiative transfer is needed near optical depth unity"),
]


def save_plot(path: Path) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def shell_spectrum(maps: list[np.ndarray]) -> np.ndarray:
    """Average normalized 2D Fourier power in integer radial wave-number bins."""
    ny, nx = maps[0].shape
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    shell = np.rint(np.hypot(ky[:, None], kx[None, :])).astype(int)
    spectra = []
    for field in maps:
        if field.shape != (ny, nx):
            raise ValueError("Surface map shape changed")
        power = abs(np.fft.fft2(field - np.mean(field)))**2
        binned = np.bincount(shell.ravel(), weights=power.ravel())
        spectra.append(binned / np.sum(binned[1:]))
    return np.mean(spectra, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--snapshots", type=int, default=21,
                   help="last completed segments, one terminal full volume each")
    p.add_argument("--output", type=Path)
    p.add_argument("--input", type=Path,
                   help="base Athena input (default: campaign/athinput.base.validated)")
    p.add_argument("--data-dir", type=Path,
                   help="directory containing the opacity table and initial solar profile")
    args = p.parse_args()
    if args.snapshots < 5:
        p.error("--snapshots must be at least five")
    campaign = args.campaign.resolve()
    state = json.loads((campaign / "state.json").read_text())
    chosen = state["completed"][-args.snapshots:]
    if len(chosen) < 5:
        raise ValueError("Need at least five completed snapshots")
    start = int(chosen[0]["end_time_code"])
    end = int(chosen[-1]["end_time_code"])
    out = args.output or campaign / f"paper_validation_t{start}_{end}"
    out.mkdir(parents=True, exist_ok=True)
    images = out / "imgs"
    images.mkdir(exist_ok=True)
    units = units_from_input(args.input or campaign / "athinput.base.validated")
    data_dir = args.data_dir or campaign / "data"
    if not data_dir.is_dir():
        p.error(f"Data directory not found: {data_dir}; supply --data-dir")
    opacity = opacity_interpolator(data_dir / "opacity_table_rhoT_3p0_6p3.h5")
    initial = np.loadtxt(data_dir / "solar_initial_profile.dat", comments="#")
    speed_unit = units["leng_unit"] / units["time_unit"] / 1e5
    records = []
    profiles = []
    maps_t, maps_v = [], []
    last_map = None
    for item in chosen:
        directory = campaign / item["case"]
        prim_files = sorted(directory.glob("*.prim_full.*.athdf"))
        if not prim_files:
            raise FileNotFoundError(f"No full volume: {directory}")
        prim_path = prim_files[-1]
        uov_path = directory / prim_path.name.replace(".prim_full.", ".uov_full.")
        prim, (x, y, z), time = read_volume(prim_path, "prim")
        uov, _, other_time = read_volume(uov_path, "user_out_var")
        if abs(time-other_time) > 1e-5 or abs(time-item["end_time_code"]) > .1:
            raise ValueError(f"Time mismatch: {directory}")
        rho = prim[0] * units["rho_unit"]
        temp = uov[2]
        if np.min(rho) <= 0 or np.min(temp) <= 0:
            raise ValueError(f"Nonpositive thermodynamic state: {directory}")
        sigma = (10 ** opacity(np.stack((np.log10(rho), np.log10(temp)), axis=-1))
                 * rho * units["leng_unit"])
        dz = float(z[1]-z[0])
        tau = np.cumsum((sigma*dz)[::-1], axis=0)[::-1] - .5*sigma*dz
        z23, t23 = surface_at(tau, z, temp, 2/3)
        _, v23 = surface_at(tau, z, prim[4]*speed_unit, 2/3)
        if not (np.isfinite(t23).all() and np.isfinite(v23).all()):
            raise ValueError(f"Incomplete tau=2/3 surface: {directory}")
        mid = int(np.argmin(abs(z)))
        vmid = prim[4, mid] * speed_unit
        tmid = temp[mid]
        flux = json.loads((directory / "flux3d.json").read_text())["records"][-1]
        surf = json.loads((directory / "surface3d.json").read_text())["records"][-1]
        if abs(flux["time_code"]-time) > .1 or abs(surf["time_code"]-time) > .1:
            raise ValueError(f"Precomputed diagnostic time mismatch: {directory}")
        records.append({
            "time_code": time,
            "top_flux_solar": item["rolling_top_flux_over_solar"],
            "bottom_flux_solar": item["rolling_bottom_total_flux_over_solar"],
            "entropy_cgs": item["entropy_cgs"],
            "T_tau23_K": float(np.mean(t23)),
            "T_tau23_rms_K": float(np.std(t23)),
            "vz_tau23_rms_km_s": float(np.sqrt(np.mean(v23**2))),
            "tau23_upflow_area": float(np.mean(v23 > 0)),
            "tau23_T_vz_corr": float(np.corrcoef(t23.ravel(), v23.ravel())[0, 1]),
            "mid_upflow_area": float(np.mean(vmid > 0)),
            "mid_T_vz_corr": float(np.corrcoef(tmid.ravel(), vmid.ravel())[0, 1]),
            "z_tau23_mean_Mm": float(np.mean(z23)*units["leng_unit"]/1e8),
            "z_tau23_rms_km": float(np.std(z23)*units["leng_unit"]/1e5),
        })
        profiles.append({
            "z_Mm": np.asarray(z)*units["leng_unit"]/1e8,
            "rho_cgs": np.mean(rho, axis=(1, 2)),
            "T_K": np.mean(temp, axis=(1, 2)),
            "vz_rms_km_s": np.sqrt(np.mean((prim[4]*speed_unit)**2, axis=(1, 2))),
            **{key: np.asarray(flux[key])/F_SUN for key in
               ("radiative", "convective_enthalpy", "mean_flow_enthalpy",
                "kinetic", "gravitational", "total")},
        })
        maps_t.append(t23)
        maps_v.append(v23)
        last_map = (x, y, t23, v23, time)
    ts = np.asarray([r["time_code"] for r in records])
    if not np.all(np.diff(ts) > 0):
        raise ValueError("Snapshot times are not strictly increasing")
    z_m = profiles[0]["z_Mm"]
    profile_mean = {key: np.mean([pr[key] for pr in profiles], axis=0)
                    for key in profiles[0] if key != "z_Mm"}
    profile_std = {key: np.std([pr[key] for pr in profiles], axis=0)
                   for key in profiles[0] if key != "z_Mm"}
    transport = (profile_mean["radiative"] + profile_mean["convective_enthalpy"]
                 + profile_mean["kinetic"])
    transport_scatter = np.std([
        pr["radiative"] + pr["convective_enthalpy"] + pr["kinetic"]
        for pr in profiles], axis=0)
    middle = int(np.argmin(abs(z_m)))
    interior = (z_m > -1.0) & (z_m < 0.85)
    mean = {key: float(np.mean([r[key] for r in records]))
            for key in records[0] if key != "time_code"}
    mean["top_effective_temperature_K"] = T_SUN*mean["top_flux_solar"]**.25
    mean["mean_tau23_proxy_flux_solar"] = float(np.mean(np.asarray(maps_t)**4)
                                                  * 5.670374419e-5/F_SUN)
    mean["resolution_km"] = float((x[1]-x[0])*units["leng_unit"]/1e5)
    mean["domain_horizontal_Mm"] = float((x[-1]-x[0]+x[1]-x[0])
                                          *units["leng_unit"]/1e8)
    mean["domain_vertical_Mm"] = float((z[-1]-z[0]+z[1]-z[0])
                                        *units["leng_unit"]/1e8)
    spectrum_v = shell_spectrum(maps_v)
    spectrum_t = shell_spectrum(maps_t)
    resolved_bins = np.arange(1, min(17, len(spectrum_v)))
    peak_v = int(resolved_bins[np.argmax(spectrum_v[resolved_bins])])
    mean["surface_velocity_spectrum_peak_Mm"] = (
        mean["domain_horizontal_Mm"] / peak_v)
    mean["surface_velocity_spectrum_peak_shell"] = peak_v
    mean["mid_convective_enthalpy_solar"] = float(profile_mean["convective_enthalpy"][middle])
    mean["mid_kinetic_solar"] = float(profile_mean["kinetic"][middle])
    mean["mid_radiative_solar"] = float(profile_mean["radiative"][middle])
    mean["interior_transport_median_solar"] = float(np.median(transport[interior]))
    mean["interior_transport_range_solar"] = [float(np.min(transport[interior])),
                                               float(np.max(transport[interior]))]
    mean["entropy_change_window_cgs"] = float(records[-1]["entropy_cgs"]
                                               - records[0]["entropy_cgs"])
    mean["newton_converged_window"] = int(sum(
        item.get("newton_status_counts", {}).get("converged", 0) for item in chosen))
    mean["slow_mg_warnings_window"] = int(sum(
        item.get("slow_mg_warning_count", 0) for item in chosen))
    mean["max_mass_drift_window"] = float(max(
        item.get("max_mass_drift", 0) for item in chosen))
    summary = {
        "scope": "completed segments only; one terminal 3D volume per segment",
        "campaign": str(campaign), "first_time_code": float(ts[0]),
        "last_time_code": float(ts[-1]), "snapshot_count": len(ts),
        "source_segments": [r["case"] for r in chosen],
        "means": mean, "records": records,
        "caveats": ["Grey FLD has no angle-dependent emergent intensity or spectrum.",
                    "Surface sigma*T^4 is a gas-temperature proxy, not a radiation measurement.",
                    "Cell-centered interior flux reconstruction is approximate; use HST for exact boundary flux.",
                    "Sparse 3D samples alias coherent mean-flow oscillations. The plotted interior transport excludes mean-flow enthalpy and gravity and is not an exact total-energy ledger.",
                    "Initial 1D solar profile is a starting model, not an independent observation."],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16})

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    axes[0].plot(ts, [r["top_flux_solar"] for r in records], label="top, rolling 500")
    axes[0].plot(ts, [r["bottom_flux_solar"] for r in records], label="bottom, rolling 500")
    axes[0].axhline(1, color="black", ls="--", label="solar")
    axes[0].set(ylabel=r"Flux / $F_\odot$")
    axes[0].legend()
    axes[1].plot(ts, [r["T_tau23_K"] for r in records], label="gas T at tau=2/3")
    axes[1].axhline(T_SUN, color="black", ls="--", label="solar Teff")
    axes[1].axhline(mean["top_effective_temperature_K"], color="tab:red",
                    ls=":", label="Teff from top flux")
    axes[1].set(ylabel="Temperature [K]")
    axes[1].legend()
    axes[2].plot(ts, [r["vz_tau23_rms_km_s"] for r in records],
                 label="tau=2/3 vertical RMS")
    axes[2].plot(ts, [r["tau23_upflow_area"] for r in records],
                 label="upflow area fraction")
    axes[2].set(xlabel="Code time", ylabel="km/s or fraction")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    save_plot(images / "evolution.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    z_initial = initial[:, 3]*units["leng_unit"]/1e8
    for ax, key, ylabel, reference in [
        (axes[0], "T_K", "Gas temperature [K]", initial[:, 1]),
        (axes[1], "rho_cgs", "Density [g/cm³]", initial[:, 4]),
        (axes[2], "vz_rms_km_s", "Vertical RMS [km/s]", None)]:
        ax.plot(z_m, profile_mean[key], label="3D time mean")
        ax.fill_between(z_m, profile_mean[key]-profile_std[key],
                        profile_mean[key]+profile_std[key], alpha=.2,
                        label="snapshot scatter")
        if reference is not None:
            ax.plot(z_initial, reference, "k--", label="initial 1D solar model")
        if key == "rho_cgs":
            ax.set_yscale("log")
        ax.set(xlabel="z [Mm]", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=10)
    save_plot(images / "stratification.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True,
                             constrained_layout=True)
    for key, label in [("radiative", "FLD radiation"),
                       ("convective_enthalpy", "convective enthalpy"),
                       ("kinetic", "kinetic")]:
        axes[0].plot(profile_mean[key][1:-1], z_m[1:-1], label=label)
    axes[1].plot(transport[1:-1], z_m[1:-1], label="mass-neutral transport proxy")
    axes[1].fill_betweenx(z_m[1:-1],
                          (transport-transport_scatter)[1:-1],
                          (transport+transport_scatter)[1:-1],
                          alpha=.2, label="snapshot scatter")
    axes[1].axhspan(-1, .85, alpha=.08, color="tab:green", label="interior comparison")
    for ax in axes:
        ax.axvline(1, color="black", ls="--", label="solar")
        ax.axvline(mean["top_flux_solar"], color="tab:red", ls=":",
                   label="mean top flux")
        ax.set(xlabel=r"Interior flux / $F_\odot$", ylabel="z [Mm]")
        ax.grid(alpha=.25)
        ax.legend(fontsize=11)
    fig.suptitle("Interior components; right-hand sum excludes aliased mean flow and gravity")
    save_plot(images / "flux_partition.png")

    x, y, t23, v23, last_time = last_map
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    length_to_Mm = units["leng_unit"]/1e8
    extent = [x[0]*length_to_Mm, x[-1]*length_to_Mm,
              y[0]*length_to_Mm, y[-1]*length_to_Mm]
    im = axes[0].imshow(t23, origin="lower", extent=extent, cmap="inferno")
    fig.colorbar(im, ax=axes[0], label="K")
    limit = np.percentile(abs(v23), 99)
    im = axes[1].imshow(v23, origin="lower", extent=extent,
                         cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(im, ax=axes[1], label="km/s")
    for ax, title in zip(axes, ("Gas T at tau=2/3", "Vertical speed at tau=2/3")):
        ax.set(title=title, xlabel="x [Mm]", ylabel="y [Mm]")
    fig.suptitle(f"Latest completed 3D snapshot: t={last_time:.0f}")
    save_plot(images / "surface_maps.png")

    all_t = np.asarray(maps_t).ravel()
    all_v = np.asarray(maps_v).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].hist(all_t, bins=55, density=True, color="tab:orange", alpha=.8)
    axes[0].set(xlabel="Gas T at tau=2/3 [K]", ylabel="Pixel probability density")
    axes[1].hexbin(all_v, all_t, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    axes[1].set(xlabel="Vertical speed at tau=2/3 [km/s]", ylabel="Gas T [K]")
    for ax in axes:
        ax.grid(alpha=.2)
    save_plot(images / "surface_statistics.png")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bins = np.arange(1, min(33, len(spectrum_v)))
    wavelength = mean["domain_horizontal_Mm"] / bins
    ax.loglog(wavelength, spectrum_v[bins], "o-", label="vertical speed")
    ax.loglog(wavelength, spectrum_t[bins], "o-", label="gas temperature")
    ax.axvline(mean["surface_velocity_spectrum_peak_Mm"], ls="--",
               color="tab:blue", label="velocity spectral peak")
    ax.invert_xaxis()
    ax.set(xlabel="Horizontal wavelength [Mm]", ylabel="Fractional shell power")
    ax.grid(alpha=.25)
    ax.legend()
    save_plot(images / "surface_spectrum.png")

    paragraphs = [
        ("Energy transport", "Interior FLD, convective enthalpy and kinetic fluxes are separated. "
         "The boundary values in the time plot come from the HST rolling integrals. "
         "The interior profiles use cell-centered reconstruction and are approximate. "
         "Sparse 3D samples alias the bulk mean-flow oscillation; the plotted sum removes "
         "that term and gravity. It is a mass-neutral transport proxy, not an exact energy ledger."),
        ("Mean stratification", "Horizontal and temporal means of T, rho and vertical RMS speed. "
         "The dashed 1D solar curve was used to initialize this run; its agreement is not an independent validation."),
        ("Optical surface", "Rosseland tau=2/3 gas temperature and vertical speed are interpolated "
         "on the full 3D grid. Positive T-v correlation and a substantial upflow area "
         "show organized convection, but they do not test emergent intensity. "
         "The horizontal Fourier peak measures a pattern scale, not a segmented granule diameter."),
    ]
    def metric(label: str, value: str) -> str:
        return f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
    rows = "".join([
        metric("Frozen completed window", f"t={ts[0]:.0f}–{ts[-1]:.0f}; {len(ts)} volumes"),
        metric("Domain / cell", f"{mean['domain_horizontal_Mm']:.3f}×{mean['domain_horizontal_Mm']:.3f}×{mean['domain_vertical_Mm']:.3f} Mm; {mean['resolution_km']:.0f} km"),
        metric("Top / bottom rolling flux", f"{mean['top_flux_solar']:.3f} / {mean['bottom_flux_solar']:.3f} F_sun"),
        metric("Numerical status in window", f"Newton {mean['newton_converged_window']:,} converged; slow-MG warnings {mean['slow_mg_warnings_window']}; max mass drift {100*mean['max_mass_drift_window']:.3f}%"),
        metric("Inflow entropy change in window", f"{mean['entropy_change_window_cgs']:+.3g} erg g⁻¹ K⁻¹ (controller still active)"),
        metric("Middle: convective / kinetic / FLD", f"{mean['mid_convective_enthalpy_solar']:+.3f} / {mean['mid_kinetic_solar']:+.3f} / {mean['mid_radiative_solar']:+.3f} F_sun"),
        metric("Mass-neutral interior transport", f"median {mean['interior_transport_median_solar']:.3f} F_sun, z=-1.0–0.85 Mm; approximate"),
        metric("Teff from upper flux", f"{mean['top_effective_temperature_K']:.0f} K (solar {T_SUN:.0f} K)"),
        metric("Gas T at tau=2/3", f"{mean['T_tau23_K']:.0f} K; spatial RMS {mean['T_tau23_rms_K']:.0f} K"),
        metric("Vertical speed at tau=2/3", f"RMS {mean['vz_tau23_rms_km_s']:.2f} km/s"),
        metric("Upflow area at tau=2/3", f"{100*mean['tau23_upflow_area']:.1f}%"),
        metric("T-v correlation at tau=2/3", f"{mean['tau23_T_vz_corr']:+.2f}"),
        metric("Velocity-pattern spectral peak", f"{mean['surface_velocity_spectrum_peak_Mm']:.2f} Mm (64² horizontal grid; unresolved convergence)"),
        metric("Gas-temperature flux proxy", f"{mean['mean_tau23_proxy_flux_solar']:.3f} F_sun (not emergent radiation)"),
    ])
    source_rows = "".join(f'<li><a href="{url}">{html.escape(name)}</a>: '
                          f'{html.escape(note)}</li>' for name, url, note in PAPERS)
    sections = "".join(f"<h2>{title}</h2><p>{description}</p>"
                       for title, description in paragraphs)
    document = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solar convection: paper-style diagnostics, t={start}–{end}</title>
<style>body{{font:20px/1.65 system-ui,sans-serif;max-width:1180px;margin:1.5rem auto;padding:0 1.2rem;color:#172b3a}}
h1{{font-size:2.1rem}}h2{{font-size:1.5rem;margin-top:2rem}}.callout{{padding:1rem 1.25rem;background:#edf4fa;border-left:6px solid #2876ad}}
.warn{{background:#fff4e7;border-color:#cd7910}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.5rem .75rem;border-bottom:1px solid #ddd}}
th{{width:42%}}img{{max-width:100%;height:auto}}figure{{margin:1.5rem 0 2rem}}figcaption{{font-size:1rem;color:#445}}
code{{font-size:.9em}}a{{color:#075a9d}}</style>
<h1>太陽対流計算：論文に沿った診断</h1>
<p>完了済み区間 <strong>t={start}–{end}</strong> の3Dスナップショット {len(ts)} 枚を固定して解析。進行中の計算は変更していません。</p>
<div class="callout"><strong>判定：</strong>持続する3D対流、質量・エネルギー収支、安定した数値解を確認できます。
一方、上端の平均流束は <strong>{mean['top_flux_solar']:.3f} F_sun</strong> で、太陽フラックスの再現には達していません。
これは現在の境界条件・制御則のもとで約0.63に近づいた可能性を示しますが、流入エントロピーは調整中です。
境界条件、解像度、放射モデルを分離する検証が残ります。</div>
<div class="callout warn"><strong>FLDについて：</strong>この計算はFLDを含む対流系が長時間安定に動く統合試験です。
光学的厚さが約1の層では角度依存の放射輸送が重要なため、灰色FLDの出力だけで太陽表面の放射・粒状斑の観測的一致は証明できません。</div>
<h2>主要な数値</h2><table>{rows}</table>
<figure><img src="imgs/evolution.png"><figcaption>上端・下端の500コード時間移動平均、光学面の温度と速度。各点は完了済みセグメントのみ。</figcaption></figure>
<h2>文献の評価項目と今回の到達点</h2>
<table><tr><th>評価項目</th><th>今回の判断</th></tr>
<tr><td>流束と下端エントロピー</td><td><strong>未達</strong>：上端は {mean['top_flux_solar']:.3f} F_sun。制御は継続中で、最終定常解と断定できない。</td></tr>
<tr><td>平均成層と対流速度</td><td><strong>測定済み</strong>：温度・密度・速度の3D平均と揺らぎを表示。初期1Dモデルは独立した観測基準ではない。</td></tr>
<tr><td>放射・エンタルピー・運動エネルギー輸送</td><td><strong>暫定測定</strong>：内部の上向き対流エンタルピーと下向き運動エネルギーを確認。内部再構成は厳密な収支検査ではない。</td></tr>
<tr><td>光学面の構造と水平スケール</td><td><strong>暫定測定</strong>：τ=2/3面の温度・速度・スペクトルを表示。96 km格子の解像度収束は未確認。</td></tr>
<tr><td>放射強度・スペクトル・周辺減光</td><td><strong>未実施</strong>：灰色FLDから角度・波長依存の観測量は直接出せない。</td></tr></table>
{sections}
<figure><img src="imgs/flux_partition.png"><figcaption>内部の流束成分。右は平均流を除いた輸送の代理指標で、厳密な全エネルギー流束ではありません。端のセルは除外。境界の正確な値はHSTで判定。</figcaption></figure>
<figure><img src="imgs/stratification.png"><figcaption>平均成層と初期1Dモデル。帯はスナップショット間の標準偏差。</figcaption></figure>
<figure><img src="imgs/surface_maps.png"><figcaption>最後の完了済み3D出力。実際の放射強度画像ではありません。</figcaption></figure>
<figure><img src="imgs/surface_statistics.png"><figcaption>全スナップショットの光学面温度分布と鉛直速度との関係。</figcaption></figure>
<figure><img src="imgs/surface_spectrum.png"><figcaption>τ=2/3 面の水平2Dフーリエスペクトル。波長ごとのパワーを表示。粒状斑の直接測定や解像度収束ではありません。</figcaption></figure>
<h2>検証の境界と次の実験</h2>
<ol><li><strong>解像度試験：</strong>同じ境界条件で48 km、可能なら24 kmに上げ、流束、平均成層、速度、光学面構造が収束するか測る。</li>
<li><strong>太陽フラックス較正：</strong>長い熱緩和後の流入エントロピーと上端流束の応答を測定する。移動平均が一定でも太陽値への一致は別条件。</li>
<li><strong>放射モデル比較：</strong>tau≈1層の観測量は多群・角度依存放射輸送または独立の放射輸送後処理で比較する。</li>
<li><strong>独立比較：</strong>高解像度の既発表計算と共通の平均成層、速度分布、流束分解を同じ光学深さで照合する。</li></ol>
<p>温度から作った σT⁴ は表面の<strong>代理指標</strong>です。計算された放射流束や観測強度ではありません。
内部流束はセル中心値から再構成した近似量です。初期1Dプロファイルは観測データとして扱いません。</p>
<h2>診断項目の出典</h2><ul>{source_rows}</ul>
<p><a href="summary.json">数値JSON</a> · <a href="../interim_t5000_21200/overview.html">先行する長時間・収支レポート</a></p>
</html>"""
    (out / "index.html").write_text(document)
    print(out / "index.html")
    print(json.dumps({"window": [start, end], "snapshot_count": len(ts),
                      "means": mean}, indent=2))


if __name__ == "__main__":
    main()
