#!/usr/bin/env python3
"""Compare a developed-convection t=0 seed with a matched quiet startup."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_report import (input_values, parse_log, read_history,
                                     sampled_energy_ledger, total_bottom_flux,
                                     total_top_flux)


F_SUN = 6.3e10


def mean_window(t: np.ndarray, y: np.ndarray, left: float, right: float) -> float:
    if t[0] > left or t[-1] < right:
        raise ValueError(f"Missing history window {left}..{right}")
    middle = (t > left) & (t < right)
    xx = np.r_[left, t[middle], right]
    yy = np.r_[np.interp(left, t, y), y[middle], np.interp(right, t, y)]
    return float(np.trapezoid(yy, xx) / (right-left))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    labels = {"quiet_t0_40": "静止1D",
              "seeded_t0_40": "3D対流シード",
              "tapered_t0_40": "3Dシード・上下端速度減衰"}
    plot_labels = {"quiet_t0_40": "quiet 1D start",
                   "seeded_t0_40": "3D convection seed",
                   "tapered_t0_40": "3D seed, tapered velocity"}
    cases = {name: root / name for name in labels
             if (root/name/"manifest.json").is_file()}
    if not {"quiet_t0_40", "seeded_t0_40"}.issubset(cases):
        raise ValueError("Missing quiet or seeded matched case")
    manifests = {name: json.loads((run/"manifest.json").read_text())
                 for name, run in cases.items()}
    for key in ("binary_sha256", "base_input_sha256", "data_sha256"):
        if len({json.dumps(m[key], sort_keys=True) for m in manifests.values()}) != 1:
            raise ValueError(f"Matched cases differ in {key}")
    controls = {name: {item for item in m["problem_overrides"]
                       if not item.startswith("initial_seed_file=")}
                for name, m in manifests.items()}
    if len({tuple(sorted(control)) for control in controls.values()}) != 1:
        raise ValueError("Matched cases differ outside the initial seed")
    histories = {name: read_history(next(run.glob("*.hst")))
                 for name, run in cases.items()}
    logs = {name: parse_log(run/"run.log") for name, run in cases.items()}
    for name, log in logs.items():
        if (not log["completed"] or log["fatal_count"] or
                log["slow_mg_warning_count"] or
                set(log["nr_status_counts"])-{"converged", "dt_zero_initialization"}):
            raise ValueError(f"Solver did not pass in {name}")
    fluxes = {name: json.loads((run/"flux3d.json").read_text())["records"]
              for name, run in cases.items()}
    surfaces = {name: json.loads((run/"surface3d.json").read_text())["records"]
                for name, run in cases.items()}
    summary = {
        "reference_flux_cgs": F_SUN,
        "design": "Same binary, input, tables, boundaries and s_in; only 3D t=0 seed differs",
        "seed_builders": [str(path) for path in sorted(root.glob("seed*.json"))],
        "cases": {},
    }
    for name, run in cases.items():
        h = histories[name]
        top, bottom = total_top_flux(h)/F_SUN, total_bottom_flux(h)/F_SUN
        t = h["time"]
        ledger = sampled_energy_ledger(h, np.flatnonzero((t >= 30) & (t <= 40)),
                                       input_values(run/"athinput.generated"), F_SUN)
        if ledger is None:
            raise ValueError(f"Missing t=30..40 energy ledger in {name}")
        profile = min(fluxes[name], key=lambda row: abs(row["time_code"]))
        z = np.asarray(profile["z_code"])
        conv = np.asarray(profile["convective_enthalpy"])/F_SUN
        rad = np.asarray(profile["radiative"])/F_SUN
        summary["cases"][name] = {
            "run": str(run), "solver": logs[name],
            "initial_top_over_solar": float(top[0]),
            "initial_bottom_over_solar": float(bottom[0]),
            "initial_convective_z0_over_solar": float(np.interp(0, z, conv)),
            "initial_radiative_z0_over_solar": float(np.interp(0, z, rad)),
            "surface_t0_K": float(surfaces[name][0]["T_tau23_mean_K"]),
            "surface_t40_K": float(surfaces[name][-1]["T_tau23_mean_K"]),
            "mass_drift_t40": float(h["MassDrift"][-1]),
            "energy_ledger_t30_40": ledger,
            "windows": {f"{a}_{b}": {
                "top_over_solar": mean_window(t, top, a, b),
                "bottom_over_solar": mean_window(t, bottom, a, b)}
                for a, b in ((0, 2), (2, 10), (10, 20), (30, 40))},
            "initial_profile": {"z_code": z.tolist(),
                                "convective_over_solar": conv.tolist(),
                                "radiative_over_solar": rad.tolist()},
        }
    summary["followups"] = {}
    for name, interval in (("tapered_t40_100", (80, 100)),
                           ("tapered_t100_200", (180, 200)),
                           ("relaxed_restart_t9400_9440", (9430, 9440))):
        run = root/name
        if not (run/"manifest.json").is_file():
            continue
        manifest = json.loads((run/"manifest.json").read_text())
        if manifest.get("exit_code") != 0:
            continue
        log = parse_log(run/"run.log")
        if not log["completed"] or log["fatal_count"]:
            raise ValueError(f"Follow-up solver did not pass in {name}")
        h = read_history(next(run.glob("*.hst")),
                         next(cases["quiet_t0_40"].glob("*.hst")))
        top = total_top_flux(h)/F_SUN
        bottom = total_bottom_flux(h)/F_SUN
        summary["followups"][name] = {
            "run": str(run), "solver": log,
            "window": interval,
            "top_mean_over_solar": mean_window(h["time"], top, *interval),
            "bottom_mean_over_solar": mean_window(h["time"], bottom, *interval),
            "top_final_over_solar": float(top[-1]),
            "bottom_final_over_solar": float(bottom[-1]),
        }
    out = root/"comparison"
    out.mkdir(exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plt.rcParams.update({"font.size": 16})
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                             constrained_layout=True)
    for name in cases:
        label = plot_labels[name]
        h = histories[name]
        axes[0].plot(h["time"], total_top_flux(h)/F_SUN, label=label)
        axes[1].plot(h["time"], total_bottom_flux(h)/F_SUN, label=label)
        rows = surfaces[name]
        axes[2].plot([r["time_code"] for r in rows],
                     [r["T_tau23_mean_K"] for r in rows], "o-", label=label)
    axes[0].axhline(1, ls="--", color="black", lw=1, label="solar")
    axes[0].set_ylabel("Upper flux / solar")
    axes[1].set_ylabel("Lower flux / solar")
    axes[2].set(xlabel="Code time", ylabel="Mean T at τ=2/3 [K]")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(out/"startup.png", dpi=150)
    plt.close(fig)

    longer_html = ""
    if "tapered_t40_100" in summary["followups"]:
        segments = ["tapered_t0_40", "tapered_t40_100"]
        if "tapered_t100_200" in summary["followups"]:
            segments.append("tapered_t100_200")
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                 constrained_layout=True)
        quiet_h = histories["quiet_t0_40"]
        axes[0].plot(quiet_h["time"], total_top_flux(quiet_h)/F_SUN,
                     label="quiet 1D, t=0-40")
        axes[1].plot(quiet_h["time"], total_bottom_flux(quiet_h)/F_SUN,
                     label="quiet 1D, t=0-40")
        for n, segment in enumerate(segments):
            h = (histories[segment] if segment in histories else read_history(
                next((root/segment).glob("*.hst")),
                next(cases["quiet_t0_40"].glob("*.hst"))))
            axes[0].plot(h["time"], total_top_flux(h)/F_SUN,
                         color="tab:orange", label="tapered 3D seed" if n == 0 else None)
            axes[1].plot(h["time"], total_bottom_flux(h)/F_SUN,
                         color="tab:orange", label="tapered 3D seed" if n == 0 else None)
        axes[0].axhline(1, ls="--", lw=1, color="black", label="solar")
        if "relaxed_restart_t9400_9440" in summary["followups"]:
            axes[0].axhline(summary["followups"]["relaxed_restart_t9400_9440"]
                            ["top_mean_over_solar"], ls=":", lw=2,
                            color="tab:green", label="relaxed restart, t=9430-9440")
        axes[0].set_ylabel("Upper flux / solar")
        axes[1].set(xlabel="Code time since initialization",
                    ylabel="Lower flux / solar")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend()
        fig.savefig(out/"longer.png", dpi=150)
        plt.close(fig)
        longer_html = "<img src='longer.png' alt='3Dシードの長い流束経過'>"

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for name in cases:
        label = plot_labels[name]
        profile = summary["cases"][name]["initial_profile"]
        ax.plot(profile["z_code"], profile["convective_over_solar"], label=label)
    ax.set(xlabel="Height [code]", ylabel="Convective enthalpy / solar")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(out/"initial_transport.png", dpi=150)
    plt.close(fig)

    quiet = summary["cases"]["quiet_t0_40"]
    seeded = summary["cases"]["seeded_t0_40"]
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td>"
        f"<td>{case['initial_convective_z0_over_solar']:.3f}</td>"
        f"<td>{case['windows']['2_10']['top_over_solar']:.3f}</td>"
        f"<td>{case['windows']['30_40']['top_over_solar']:.3f}</td>"
        f"<td>{case['windows']['30_40']['bottom_over_solar']:.3f}</td>"
        f"<td>{case['energy_ledger_t30_40']['stored_energy_trend_over_solar']:+.3f}</td>"
        f"<td>{case['surface_t40_K']:.0f}</td></tr>"
        for name, case in summary["cases"].items()
        for label in (labels[name],))
    followup_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{row['window'][0]}–{row['window'][1]}</td>"
        f"<td>{row['top_mean_over_solar']:.3f}</td>"
        f"<td>{row['bottom_mean_over_solar']:.3f}</td></tr>"
        for name, row in summary["followups"].items())
    followup_html = (
        "<h2>継続計算</h2><table><tr><th>ケース</th><th>評価窓</th>"
        "<th>上端/Fsun</th><th>下端/Fsun</th></tr>"+followup_rows+"</table>"
        if followup_rows else "")
    final = summary["followups"].get("tapered_t100_200")
    relaxed = summary["followups"].get("relaxed_restart_t9400_9440")
    verdict_html = (
        f"<p><strong>t=180–200 の移植シード上端平均は "
        f"{final['top_mean_over_solar']:.3f} Fsun。</strong> "
        + (f"加工しない緩和済み3D状態の短時間平均 "
           f"{relaxed['top_mean_over_solar']:.3f} Fsun とほぼ同じです。"
           if relaxed else "")
        + "初期対流は起動直後の冷却を弱めますが、"
          "長期の太陽流束較正は達成していません。</p>"
        if final else "")
    (out/"index.html").write_text(
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>太陽対流の3D初期化試験</title><style>"
        "body{font:22px/1.55 system-ui;max-width:1250px;margin:2rem auto;"
        "padding:0 1rem;color:#17324a}h1{font-size:2rem}"
        "strong{color:#184c76}table{border-collapse:collapse}"
        "th,td{padding:.55rem .75rem;border-bottom:1px solid #ccc;text-align:right}"
        "th:first-child,td:first-child{text-align:left}img{max-width:100%}"
        "</style><h1>太陽対流の3D初期化試験</h1>"
        "<p>緩和済みt=9400の対流揺らぎと速度をt=0の1D平均構造へ移しました。"
        "同一のバイナリ、境界、下端エントロピー、EOS・opacity、解像度で比較しています。"
        "速度は各高さの質量流束をゼロにし、上端で滑らかに減衰させています。"
        "追加対照では下端でも減衰させました。</p>"
        f"<p><strong>初期の z≈0 対流エンタルピー流束は "
        f"静止 {quiet['initial_convective_z0_over_solar']:.3f}、"
        f"シード {seeded['initial_convective_z0_over_solar']:.3f} Fsun。</strong> "
        "これはセル中心再構成値です。物理境界流束にはHSTを使います。</p>"
        "<p><strong>下端流束も同時に評価してください。</strong> "
        "移植シードで下端流束が大きく偏る場合、上端の短時間改善を"
        "物理的に釣り合った初期状態の成功とは判定できません。</p>"
        + verdict_html +
        "<table><tr><th>初期化</th><th>t=0 z≈0対流/Fsun</th>"
        "<th>t=2–10上端/Fsun</th><th>t=30–40上端/Fsun</th>"
        "<th>t=30–40下端/Fsun</th><th>同蓄積傾向/Fsun</th>"
        "<th>t=40 τ=2/3温度 [K]</th></tr>"+rows+"</table>"
        "<img src='startup.png' alt='起動後の流束と表面温度'>"
        "<img src='initial_transport.png' alt='初期の高さ別流束'>"
        + longer_html
        + followup_html
        + "<p>各計算: "
        + " / ".join(f"<a href='../{name}/index.html'>{html.escape(labels[name])}</a>"
                     for name in cases)
        + (" / " if summary["followups"] else "")
        + " / ".join(f"<a href='../{name}/index.html'>{html.escape(name)}</a>"
                     for name in summary["followups"]) + ". "
        "<a href='summary.json'>全数値と方法</a></p></html>\n")
    (root/"index.html").write_text(
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='0; url=comparison/index.html'>"
        "<title>3D初期化試験</title><p style='font:22px system-ui'>"
        "<a href='comparison/index.html'>3D初期化試験のまとめを開く</a>"
        "</p></html>\n")
    print(out/"index.html")


if __name__ == "__main__":
    main()
