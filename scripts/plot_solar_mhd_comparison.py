#!/usr/bin/env python3
"""Summarize the equal-state zero-field and magnetic solar MHD+FLD pilots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_simple_convection_surface3d import read_volume, surface_at
from plot_simple_convection_tau import opacity_interpolator, units_from_input

FSUN = 6.3e10
CASES = {"zero_field": "B=0", "zero_net_50g": "50 G zero-net seed"}


def history(path: Path, names: list[str] | None = None) -> dict[str, np.ndarray]:
    with path.open() as stream:
        first = stream.readline()
        if first.startswith("#"):
            names = re.findall(r"\[\d+\]=([^\s]+)", stream.readline())
        elif names is None:
            raise ValueError(f"Headerless history requires column names: {path}")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if len(names) != data.shape[1]:
        raise ValueError(f"History columns do not match header: {path}")
    return dict(zip(names, data.T))


def joined_history(root: Path, case: str) -> dict[str, np.ndarray]:
    first = history(root / case / "simple_convection_general_6Mm_box.hst")
    second = history(root / (case + "_t10") /
                     "simple_convection_general_6Mm_box.hst", list(first))
    parts = [first, second]
    if parts[0].keys() != parts[1].keys():
        raise ValueError(f"History columns changed within {case}")
    out = {key: np.r_[parts[0][key], parts[1][key]] for key in parts[0]}
    order = np.argsort(out["time"], kind="stable")
    out = {key: value[order] for key, value in out.items()}
    _, unique = np.unique(out["time"], return_index=True)
    return {key: value[unique] for key, value in out.items()}


def tau1_bz(root: Path, case: str, opacity, units: dict[str, float]) -> dict:
    run = root / (case + "_t10")
    prim_files = sorted(run.glob("*.prim_full.*.athdf"))
    if not prim_files:
        raise ValueError(f"No full 3D dumps in {run}")
    b_unit = np.sqrt(4.0 * np.pi * units["egas_unit"])
    records = []
    for prim_path in prim_files:
        suffix = prim_path.name.split(".prim_full.")[-1]
        uov_path = run / prim_path.name.replace(".prim_full.", ".uov_full.")
        if not uov_path.exists():
            raise ValueError(f"Missing matching {uov_path}")
        prim, (_, _, z), time = read_volume(prim_path, "prim")
        bcc, _, btime = read_volume(prim_path, "B")
        uov, _, utime = read_volume(uov_path, "user_out_var")
        if time != btime or time != utime:
            raise ValueError(f"Volume times differ at {time}")
        rho = prim[0] * units["rho_unit"]
        temp = uov[2]
        sigma = 10.0 ** opacity(np.stack((np.log10(rho), np.log10(temp)), axis=-1))
        sigma *= rho * units["leng_unit"]
        dz = float(z[1] - z[0])
        tau = np.cumsum((sigma * dz)[::-1], axis=0)[::-1] - 0.5 * sigma * dz
        z1, bz1 = surface_at(tau, z, bcc[2] * b_unit, 1.0)
        _, pmag1 = surface_at(tau, z, 0.5*np.sum(bcc**2, axis=0), 1.0)
        _, pgas1 = surface_at(tau, z, prim[1], 1.0)
        coverage = float(np.mean(np.isfinite(z1)))
        if coverage == 0:
            raise ValueError(f"tau=1 outside volume at {time}")
        records.append({"time_code": time,
                        "tau1_coverage": coverage,
                        "mean_abs_bz_tau1_gauss": float(np.nanmean(np.abs(bz1))),
                        "mean_pmag_over_pgas_tau1": float(np.nanmean(
                            pmag1 / np.maximum(pgas1, 1e-30)))})
    return {"records": records, "last": records[-1]}


def energy_invariant(root: Path, case: str, units: dict[str, float]) -> dict:
    """Check the latest paired full dumps against Athena's total-energy layout."""
    run = root / (case + "_t10")
    cons_files = sorted(run.glob("*.cons_full.*.athdf"), reverse=True)
    prim_files = sorted(run.glob("*.prim_full.*.athdf"), reverse=True)
    for cons_path in cons_files:
        cons, _, cons_time = read_volume(cons_path, "cons")
        for prim_path in prim_files:
            prim, _, prim_time = read_volume(prim_path, "prim")
            if abs(cons_time - prim_time) > 1e-5:
                continue
            uov_path = run / prim_path.name.replace(".prim_full.", ".uov_full.")
            uov, _, uov_time = read_volume(uov_path, "user_out_var")
            bcc, _, btime = read_volume(prim_path, "B")
            if abs(uov_time - cons_time) > 1e-5 or abs(btime - cons_time) > 1e-5:
                raise ValueError(f"Energy fields have different times in {run}")
            gas = uov[0] / units["egas_unit"]
            kinetic = 0.5 * prim[0] * np.sum(prim[2:5]**2, axis=0)
            magnetic = 0.5 * np.sum(bcc**2, axis=0)
            residual = cons[1] - gas - kinetic - magnetic
            relative = np.abs(residual) / gas
            if not np.isfinite(relative).all() or np.any(gas <= 0):
                raise ValueError(f"Invalid gas energy in {run}")
            return {"time_code": cons_time,
                    "max_residual_over_gas": float(np.max(relative)),
                    "max_magnetic_over_gas": float(np.max(magnetic/gas)),
                    "pass": bool(np.max(relative) < 1e-4)}
    raise ValueError(f"No paired full energy dumps in {run}")


def analyze(root: Path, input_path: Path, opacity_path: Path) -> dict:
    units = units_from_input(input_path)
    opacity = opacity_interpolator(opacity_path)
    histories = {case: joined_history(root, case) for case in CASES}
    common_end = min(float(h["time"][-1]) for h in histories.values())
    common_start = max(2.0, common_end - 2.0)
    common_times = np.linspace(common_start, common_end, 101)
    output = {"scope": "96 km, 64x64x32, about 0 to 10 code time; transient MHD+FLD pilot",
              "source_nonmagnetic_time_code": 24000, "fsun_cgs": FSUN,
              "common_flux_window_code": [common_start, common_end],
              "cases": {}}
    for case in CASES:
        hist = histories[case]
        logs = [(root / suffix / "run.log").read_text()
                for suffix in (case, case + "_t10")]
        final_cycles = re.findall(r"^time=[^\s]+ cycle=(\d+)$", logs[-1], re.M)
        if not final_cycles:
            raise ValueError(f"No final cycle record for {case}")
        nr_converged = sum(log.count("[NR] status=converged") for log in logs)
        end_time = float(hist["time"][-1])
        record = {
            "final_time_code": end_time,
            "flux_window_start_code": common_start,
            "samples": len(hist["time"]),
            "final_cycle": int(final_cycles[-1]),
            "nr_converged": nr_converged,
            "all_nr_converged": nr_converged == int(final_cycles[-1]),
            "fatal_count": sum(log.count("FATAL ERROR") for log in logs),
            "top_flux_final_over_fsun": float(hist["FtotTop"][-1] / FSUN),
            "top_flux_final_window_over_fsun": float(np.mean(np.interp(
                common_times, hist["time"], hist["FtotTop"]) / FSUN)),
            "bottom_total_flux_final_over_fsun": float(
                (hist["FhydBottom"][-1] + hist["FradBottom"][-1]
                 + hist["FadvRadBottom"][-1]) / FSUN),
            "max_div_b_dx_relative": float(np.max(hist["DivBdxRel"])),
            "magnetic_energy_start_erg": float(hist["Emag"][0]),
            "magnetic_energy_final_erg": float(hist["Emag"][-1]),
            "top_unsigned_bz_final_gauss": float(hist["TopAbsBzG"][-1]),
            "max_abs_mass_drift": float(np.max(np.abs(hist["MassDrift"]))),
            "max_top_inflow_code": float(max(0.0, -np.min(hist["TopMdotOut"]))),
            "tau1_bz": tau1_bz(root, case, opacity, units),
            "energy_invariant": (energy_invariant(root, case, units)
                                 if case == "zero_net_50g" else None),
        }
        output["cases"][case] = record
    output["delta_top_flux_final_window_over_fsun"] = (
        output["cases"]["zero_net_50g"]["top_flux_final_window_over_fsun"]
        - output["cases"]["zero_field"]["top_flux_final_window_over_fsun"])
    hydro_path = root / "hydro_reference" / "simple_convection_general_6Mm_box.hst"
    if hydro_path.exists():
        hydro = history(hydro_path)
        hydro_log = (root / "hydro_reference" / "run.log").read_text()
        zero = joined_history(root, "zero_field")
        mhd_zero_at_hydro_end = float(np.interp(hydro["time"][-1], zero["time"],
                                                zero["FtotTop"]))
        output["hydro_reference"] = {
            "final_time_code": float(hydro["time"][-1]),
            "top_flux_final_over_fsun": float(hydro["FtotTop"][-1]/FSUN),
            "zero_b_mhd_minus_hydro_flux_over_fsun": float(
                (mhd_zero_at_hydro_end-hydro["FtotTop"][-1])/FSUN),
            "nr_converged": hydro_log.count("[NR] status=converged"),
            "fatal_count": hydro_log.count("FATAL ERROR"),
        }
    output["pass_short_solver_check"] = all(
        item["final_time_code"] >= 10 and item["fatal_count"] == 0
        and item["all_nr_converged"] and item["max_div_b_dx_relative"] < 1e-10
        and (item["energy_invariant"] is None or item["energy_invariant"]["pass"])
        and item["max_top_inflow_code"] < 1e-4
        and item["max_abs_mass_drift"] < 0.01
        for item in output["cases"].values())
    return output


def plot(root: Path) -> None:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for case, label in CASES.items():
        h = joined_history(root, case)
        axes[0, 0].plot(h["time"], h["FtotTop"] / FSUN, label=label)
        axes[0, 1].plot(h["time"], h["FhydBottom"] / FSUN, label=label)
        axes[1, 0].plot(h["time"], h["TopAbsBzG"], label=label)
        axes[1, 1].plot(h["time"], h["DivBdxRel"], label=label)
    axes[0, 0].set(ylabel=r"$F_{\rm top}/F_\odot$", title="Outgoing total flux")
    axes[0, 1].set(ylabel=r"$F_{\rm MHD,bottom}/F_\odot$",
                   title="Bottom MHD energy flux")
    axes[1, 0].set(ylabel=r"$\langle |B_z|\rangle_{\rm top}$ [G]",
                   title="Top unsigned vertical field")
    axes[1, 1].set(ylabel=r"$\max |\nabla\cdot B|\Delta x/|B|$",
                   title="Magnetic divergence")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Elapsed code time from t=24000 snapshot")
    axes[0, 0].axhline(1, color="black", ls="--", lw=1, label="Solar flux")
    hydro_path = root / "hydro_reference" / "simple_convection_general_6Mm_box.hst"
    if hydro_path.exists():
        hydro = history(hydro_path)
        axes[0, 0].plot(hydro["time"], hydro["FtotTop"]/FSUN,
                        color="black", ls=":", label="HD LHLLC-FLD")
    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[1, 1].set_yscale("symlog", linthresh=1e-17)
    fig.savefig(root / "mhd_comparison.png", dpi=150)
    plt.close(fig)


def plot_tau1_map(root: Path, input_path: Path, opacity_path: Path) -> None:
    run = root / "zero_net_50g_t10"
    prim_path = sorted(run.glob("*.prim_full.*.athdf"))[-1]
    uov_path = run / prim_path.name.replace(".prim_full.", ".uov_full.")
    units = units_from_input(input_path)
    opacity = opacity_interpolator(opacity_path)
    prim, (x, y, z), time = read_volume(prim_path, "prim")
    bcc, _, _ = read_volume(prim_path, "B")
    uov, _, _ = read_volume(uov_path, "user_out_var")
    rho = prim[0] * units["rho_unit"]
    temp = uov[2]
    sigma = 10.0 ** opacity(np.stack((np.log10(rho), np.log10(temp)), axis=-1))
    sigma *= rho * units["leng_unit"]
    dz = float(z[1] - z[0])
    tau = np.cumsum((sigma * dz)[::-1], axis=0)[::-1] - 0.5 * sigma * dz
    _, bz1 = surface_at(tau, z, bcc[2] * np.sqrt(4*np.pi*units["egas_unit"]), 1.0)
    _, t1 = surface_at(tau, z, temp, 1.0)
    extent = [x[0]*units["leng_unit"]/1e8, x[-1]*units["leng_unit"]/1e8,
              y[0]*units["leng_unit"]/1e8, y[-1]*units["leng_unit"]/1e8]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(bz1), 99))
    image = axes[0].imshow(bz1, origin="lower", extent=extent, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
    fig.colorbar(image, ax=axes[0], label=r"$B_z(\tau=1)$ [G]")
    image = axes[1].imshow(t1, origin="lower", extent=extent, cmap="inferno")
    fig.colorbar(image, ax=axes[1], label=r"$T_{\rm gas}(\tau=1)$ [K]")
    for ax in axes:
        ax.set(xlabel="x [Mm]", ylabel="y [Mm]")
    axes[0].set_title(f"Magnetogram, elapsed t={time:.2f}")
    axes[1].set_title("Gas temperature on the same surface")
    fig.savefig(root / "tau1_magnetogram.png", dpi=150)
    plt.close(fig)


def html(root: Path, summary: dict) -> None:
    rows = []
    extension_log = root / "zero_net_50g_t12" / "run.log"
    extension_link = ("<a href='zero_net_50g_t12/run.log'>t=12 まで完了</a>した。"
                      if extension_log.exists() and
                      "time=1.200000000000000e+01" in extension_log.read_text()
                      else "")
    energy_link = ("<a href='energy_check_t12.json'>t=12 のエネルギー検査</a>。"
                   if (root / "energy_check_t12.json").exists() else "")
    for case, label in CASES.items():
        s = summary["cases"][case]
        rows.append(f"<tr><th>{label}</th><td>{s['nr_converged']}</td>"
                    f"<td>{s['top_flux_final_window_over_fsun']:.4f}</td>"
                    f"<td>{s['tau1_bz']['last']['mean_abs_bz_tau1_gauss']:.2f}</td>"
                    f"<td>{s['tau1_bz']['last']['mean_pmag_over_pgas_tau1']:.2e}</td>"
                    f"<td>{s['max_div_b_dx_relative']:.2e}</td>"
                    f"<td><a href='{case}_t10/run.log'>log</a></td></tr>")
    status = "PASS" if summary["pass_short_solver_check"] else "CHECK"
    page = f"""<!doctype html><html lang='ja'><meta charset='utf-8'>
<title>WOR-111 Solar MHD+FLD pilot</title><style>
body{{font:20px/1.6 system-ui,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1{{font-size:2rem}} h2{{font-size:1.5rem}} strong{{color:#075985}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #cbd5e1;padding:.6rem;text-align:left}}
img{{max-width:100%;height:auto}} .note{{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}}
</style><h1>WOR-111: Solar MHD + FLD pilot</h1>
<p class='note'><strong>短時間の数値試験: {status}</strong>。同一の非磁化 t=24000 スナップショットから
64×64×32、96 km、CFL 0.3で比較。横方向周期、下端HD2 + 対称磁場、
上端は流出のみの流体 + 鉛直磁場。下降流はゴースト速度を反射し、流入流束を抑制した。</p>
<h2>結果</h2><p>上端流束は両ケースに共通の
<strong>t={summary['common_flux_window_code'][0]:.2f}–{summary['common_flux_window_code'][1]:.2f}</strong>
で時刻補間して平均した。</p><table><tr><th>設定</th><th>収束したNRステップ</th><th>上端流束 / F☉ (共通2コード時間)</th>
<th>τ=1の平均|Bz| [G] (最終時刻)</th><th>τ=1の磁気圧/ガス圧</th><th>最大規格化div B</th><th>詳細</th></tr>
{''.join(rows)}</table><p>磁場あり − ゼロ磁場の上端流束差（最終2コード時間）:
<strong>{summary['delta_top_flux_final_window_over_fsun']:+.4f} F☉</strong>。</p>
<p>同一初期値のHD LHLLC-FLD対照（t=2）との差:
<strong>{summary.get('hydro_reference', {}).get('zero_b_mhd_minus_hydro_flux_over_fsun', float('nan')):+.4f} F☉</strong>。
リーマンソルバーも異なるため厳密一致は要求しない。</p>
<p>上端流入の最大値（コード単位）は、B=0 が
<strong>{summary['cases']['zero_field']['max_top_inflow_code']:.2e}</strong>、
50 G が <strong>{summary['cases']['zero_net_50g']['max_top_inflow_code']:.2e}</strong>。
修正前の結果は <code>*_old_top_bc</code> に保存した。</p>
<p><strong>MHD+FLD 全エネルギー検査:</strong> 磁場あり t={summary['cases']['zero_net_50g']['energy_invariant']['time_code']:.2f} の
最大相対残差は <strong>{summary['cases']['zero_net_50g']['energy_invariant']['max_residual_over_gas']:.2e}</strong>
（出力精度の範囲）。以前の磁気エネルギー欠落を修正した。{extension_link}
{energy_link}</p>
<img src='mhd_comparison.png' alt='MHD FLD time histories'>
<h2>τ=1 面の構造</h2><img src='tau1_magnetogram.png' alt='Tau-one magnetogram and temperature'>
<h2>解釈と制約</h2><p>この窓は鉛直横断時間や熱緩和時間より短く、流束差は定常的な磁場効果ではない。
10コード時間は約3.7分で、直前の非磁化計算の鉛直横断時間の約0.09倍にすぎない。
50 Gの正弦波状・正味ゼロの鉛直種磁場は、Rempel (2014) の 10⁻³ G 微小種からの
小規模ダイナモ再現ではない。96 km格子も同論文の磁気スペクトル向け8 km以下より粗い。</p>
<p>履歴の <code>FhydBottom/FhydTop</code> はMHDビルドでは磁場を含む全流体エネルギー流束。
Poynting成分は分離していない。t=0へのスナップショット移植では境界フィードバックの内部状態を
引き継がないため、両ケース間の差のみ比較可能。</p>
<p><a href='summary.json'>summary.json</a> / <a href='data/t24000_seed.json'>共通初期値</a> /
<a href='https://arxiv.org/abs/1405.6814'>Rempel 2014</a> /
<a href='https://arxiv.org/abs/2004.01795'>Rempel 2020</a></p></html>"""
    (root / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = analyze(root, args.input.resolve(), args.opacity_table.resolve())
    plot(root)
    plot_tau1_map(root, args.input.resolve(), args.opacity_table.resolve())
    html(root, summary)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(root / "index.html")


if __name__ == "__main__":
    main()
