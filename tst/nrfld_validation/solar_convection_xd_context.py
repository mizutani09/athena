#!/usr/bin/env python3
"""Place the shared XD 24 km history beside the local 96 km early history.

The histories have different inflow entropy control and are both transient.
This is context for WOR-74, not a causal resolution-convergence test.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from solar_convection_report import (  # noqa: E402
    combine_histories, input_values, read_history, sampled_energy_ledger,
    total_bottom_flux, total_top_flux,
)
from plot_simple_convection_surface3d import read_volume  # noqa: E402

FSUN = 6.3e10
WINDOWS = [(700, 884), (884, 1000), (1000, 1103)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def window(h: dict[str, np.ndarray], left: float, right: float,
           units: dict[str, float]) -> dict:
    t = h["time"]
    inner = (t > left) & (t < right)
    sampled = np.flatnonzero((t >= left) & (t <= right))
    def average(x: np.ndarray) -> float:
        xs = np.r_[left, t[inner], right]
        ys = np.interp(xs, t, x)
        return float(np.trapezoid(ys, xs) / (right - left))
    ledger = sampled_energy_ledger(h, sampled, units, FSUN)
    return {
        "start_time_code": left, "end_time_code": right,
        "samples": len(sampled),
        "top_flux_over_solar": average(total_top_flux(h)) / FSUN,
        "bottom_flux_over_solar": average(total_bottom_flux(h)) / FSUN,
        "storage_trend_over_solar": ledger["stored_energy_trend_over_solar"],
        "storage_endpoint_rate_over_solar": ledger["stored_energy_rate_over_solar"],
        "sampled_ledger_residual_over_solar": ledger["residual_equivalent_solar_flux"],
        "max_abs_mass_drift": float(np.max(np.abs(h["MassDrift"][sampled]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xd-initial", type=Path, required=True)
    parser.add_argument("--xd-continuation", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--local-input", type=Path, required=True)
    parser.add_argument("--xd-profiles", type=Path)
    parser.add_argument("--local-profile", type=Path)
    parser.add_argument("--xd-boundary-energy", type=Path)
    parser.add_argument("--local-uov", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    initial = read_history(args.xd_initial)
    continuation = read_history(args.xd_continuation, args.xd_initial)
    if initial["time"][-1] >= continuation["time"][0]:
        raise ValueError("XD initial/continuation history overlap unexpectedly")
    local_files = [args.local_root / f"feedback_{i:03d}_t{i*100}"
                   / "simple_convection_general_6Mm_box.hst" for i in range(1, 13)]
    names = set(initial) & set(continuation) & set(read_history(local_files[0]))
    xd = {key: np.r_[initial[key], continuation[key]] for key in names}
    local = combine_histories(local_files, local_files[0], columns=names)
    units = input_values(args.local_input)
    results = {
        "scope": "Early-time context only; no causal resolution conclusion",
        "xd_24km": {"grid": [256, 256, 128], "s_in_cgs": 1581644350.815519,
                    "s_in_policy": "from 1D profile, fixed",
                    "history_initial": str(args.xd_initial),
                    "history_initial_sha256": sha256(args.xd_initial),
                    "history_continuation": str(args.xd_continuation),
                    "history_continuation_sha256": sha256(args.xd_continuation),
                    "last_history_time_code": float(xd["time"][-1]),
                    "windows": [window(xd, a, b, units) for a, b in WINDOWS]},
        "local_96km": {"grid": [64, 64, 32],
                       "s_in_policy": "rolling feedback from 1D profile; changes between segments",
                       "s_in_cgs_at_t0": 1581097454.457282,
                       "s_in_cgs_at_t1100_segment": 1581157949.552845,
                       "history_paths": [str(p) for p in local_files],
                       "last_history_time_code": float(local["time"][-1]),
                       "windows": [window(local, a, b, units) for a, b in WINDOWS]},
        "dropbox_source": "/AstroResearch/FLD_solar_XD2000_20261105/wor91_24km_results_20261109/",
    }
    surface_args = (args.xd_profiles, args.local_profile,
                    args.xd_boundary_energy, args.local_uov)
    if any(path is not None for path in surface_args):
        if any(path is None for path in surface_args):
            parser.error("Provide all four surface-profile inputs together")
        xd_profile = json.loads(args.xd_profiles.read_text())["records"][-1]
        local_profile = json.loads(args.local_profile.read_text())["records"][-1]
        xd_boundary = json.loads(args.xd_boundary_energy.read_text())
        if abs(xd_profile["time_code"] - 900) > 0.1 or abs(local_profile["time_code"] - 900) > 0.1:
            raise ValueError("Surface profiles must describe t≈900")
        local_uov, _, local_uov_time = read_volume(args.local_uov, "user_out_var")
        if abs(local_uov_time - 900) > 0.1:
            raise ValueError("Local radiation energy must describe t≈900")
        local_e = float(np.mean(local_uov[1, -1]))
        xd_e = xd_boundary["at_t900"]["E_rad_code_mean"]
        flux_24 = float(np.interp(900, xd["time"], total_top_flux(xd)) / FSUN)
        flux_96 = float(np.interp(900, local["time"], total_top_flux(local)) / FSUN)
        z_common_mm = 1.45
        z_common_code = z_common_mm * 1e8 / units["leng_unit"]
        tx = float(np.interp(z_common_code, xd_profile["z_code"],
                             xd_profile["T_mean_K"]))
        tl = float(np.interp(z_common_code, local_profile["z_code"],
                             local_profile["T_mean_K"]))
        results["surface_at_t900"] = {
            "xd_profiles_sha256": sha256(args.xd_profiles),
            "local_profile_sha256": sha256(args.local_profile),
            "xd_boundary_energy_sha256": sha256(args.xd_boundary_energy),
            "local_uov_sha256": sha256(args.local_uov),
            "xd_top_cell_E_rad_code": xd_e,
            "local_top_cell_E_rad_code": local_e,
            "top_cell_E_rad_ratio_24_over_96": xd_e / local_e,
            "top_flux_24_over_solar": flux_24,
            "top_flux_96_over_solar": flux_96,
            "top_flux_ratio_24_over_96": flux_24 / flux_96,
            "top_cell_center_z_24_Mm": xd_profile["z_code"][-1]*units["leng_unit"]/1e8,
            "top_cell_center_z_96_Mm": local_profile["z_code"][-1]*units["leng_unit"]/1e8,
            "common_z_Mm": z_common_mm,
            "T_24_at_common_z_K": tx,
            "T_96_at_common_z_K": tl,
        }
        fig_profile, axis = plt.subplots(figsize=(10, 7))
        for profile, label, color in ((xd_profile, "24 km", "#d97706"),
                                      (local_profile, "96 km", "#2563eb")):
            z_mm = np.array(profile["z_code"]) * units["leng_unit"] / 1e8
            axis.plot(profile["T_mean_K"], z_mm, color=color, label=label, lw=2)
        axis.set(xlabel="Mean gas temperature [K]", ylabel="z [Mm]",
                 ylim=(0.7, 1.54), title="t=900: upper thermal structure")
        axis.grid(alpha=.3)
        axis.legend()
        fig_profile.tight_layout()
        fig_profile.savefig(args.output / "upper_temperature_t900.png", dpi=160)
        plt.close(fig_profile)
    (args.output / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 14})
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for name, h, color in (("24 km, fixed s_in", xd, "#d97706"),
                           ("96 km, feedback", local, "#2563eb")):
        axes[0].plot(h["time"], total_top_flux(h) / FSUN,
                     label=name, color=color, lw=2)
        axes[1].plot(h["time"], total_bottom_flux(h) / FSUN,
                     label=name, color=color, lw=1, alpha=.75)
    for ax in axes:
        ax.axhline(1, color="black", ls="--", lw=1, alpha=.5)
        ax.grid(alpha=.3)
    axes[0].set_xlim(0, 1103)
    axes[1].set(xlim=(200, 1103), ylim=(-0.5, 1.5))
    axes[0].set_ylabel("Top total flux / F_sun")
    axes[1].set_ylabel("Bottom total flux / F_sun")
    axes[1].set_xlabel("Time [code]")
    axes[0].legend()
    fig.suptitle("WOR-74: shared XD 24 km and local 96 km early histories", fontsize=19)
    fig.tight_layout()
    fig.savefig(args.output / "early_flux_context.png", dpi=160)
    plt.close(fig)

    rows = []
    for a, b in WINDOWS:
        for name, key in (("XD 24 km", "xd_24km"), ("local 96 km", "local_96km")):
            w = next(w for w in results[key]["windows"] if w["start_time_code"] == a)
            rows.append(f"<tr><th>{a}–{b}</th><td>{html.escape(name)}</td>"
                        f"<td>{w['top_flux_over_solar']:.4f}</td>"
                        f"<td>{w['bottom_flux_over_solar']:.4f}</td>"
                        f"<td>{w['storage_trend_over_solar']:+.4f}</td></tr>")
    surface_html = ""
    if "surface_at_t900" in results:
        s = results["surface_at_t900"]
        surface_html = f"""<h2>t≈900 の表面熱構造</h2>
<p>最上層セルの放射エネルギーは24 km/96 kmで {s['xd_top_cell_E_rad_code']:.3f}/
{s['local_top_cell_E_rad_code']:.3f}（比{s['top_cell_E_rad_ratio_24_over_96']:.3f}）。
同時刻の上端流束比は{s['top_flux_ratio_24_over_96']:.3f}で近いです。
共通の物理高さ z={s['common_z_Mm']:.2f} Mm でもガス温度は
{s['T_24_at_common_z_K']:.0f}/{s['T_96_at_common_z_K']:.0f} K。
境界セル中心の高さは24/96 kmで{s['top_cell_center_z_24_Mm']:.3f}/
{s['top_cell_center_z_96_Mm']:.3f} Mmと異なります。
上端流束の違いは表面放射エネルギーの違いに伴いますが、温度差の原因を解像度だけには帰せません。</p>
<img src="upper_temperature_t900.png" alt="t900の上部ガス温度プロファイル">"""
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>WOR-74: XD 24 kmの初期経過との比較</title><style>
body{{font:21px/1.55 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1{{font-size:2rem}}h2{{font-size:1.5rem}}strong{{color:#075985}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cbd5e1;padding:.6rem;text-align:right}}
th:first-child,td:nth-child(2){{text-align:left}}img{{max-width:100%;height:auto}}
.note{{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}}
</style><h1>WOR-74：XD 24 km の初期経過</h1>
<p class="note"><strong>解像度の因果比較ではありません。</strong> 24 km計算の共有HSTは t=1103.1 までです。
96 kmの長時間系列 t=24000とは熱緩和時間が違い、この表の96 km系列には流入エントロピーの
feedbackが入っています。24 km系列は初期profile由来の値を固定しています。</p>
<table><tr><th>t [code]</th><th>系列</th><th>上端 / F☉</th><th>下端 / F☉</th><th>蓄積傾向 / F☉</th></tr>
{''.join(rows)}</table>
<img src="early_flux_context.png" alt="初期の上下端流束">
<p>下段は起動時の大きな振動を除いて t≥200、−0.5～1.5 F☉ を表示しています。</p>
<p>24 kmの上端は t=700–1103 に約0.56 F☉で推移。後半は下端が上端を下回り、
箱内の熱エネルギーが減少中です。96 kmは同時刻で約0.63 F☉ですが、固定s<sub>in</sub>同士の
比較や長時間の24 km定常値ではありません。24 km側の±10⁶短期s<sub>in</sub>対照も下端には
約±0.3 F☉効き、上端には20 code-timeでほぼ効きません。</p>
{surface_html}
<p><a href="summary.json">数値、履歴SHA256、質量ずれ、積分台帳残差</a>。</p></html>"""
    (args.output / "index.html").write_text(page)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
