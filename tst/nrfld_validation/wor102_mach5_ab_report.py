#!/usr/bin/env python3
"""Summarize the pinned WOR-102 Mach 5 restart comparison on XD."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_mach5_convergence import reference_at
from mach5_trad_peak_audit import load
from plots.plot_radiative_shock import jump_center, load_analytic_profile


def summarize(root: Path, reference: dict, steps: int, index: int) -> tuple[dict, dict]:
    cases = {}
    for mode in ("baseline", "candidate"):
        folder = root / "output" / f"{mode}_{steps}" / "cases" / "mach5"
        profile = folder / "output" / f"radshock_mach5_amr.uov_x.{index:05d}.athdf"
        time, x, (rho, trad, tgas, limiter, width) = load(profile)
        if not all(np.all(np.isfinite(v)) for v in (rho, trad, tgas, limiter)):
            raise ValueError(f"nonfinite profile: {profile}")
        if min(rho.min(), trad.min(), tgas.min()) <= 0:
            raise ValueError(f"nonpositive profile: {profile}")
        cases[mode] = (time, x, rho, trad, tgas, width)
    tb, xb, rb, trb, tgb, wb = cases["baseline"]
    tc, xc, rc, trc, tgc, wc = cases["candidate"]
    if tb != tc or not np.array_equal(xb, xc) or not np.array_equal(wb, wc):
        raise ValueError("A/B time or mesh differs")
    center = jump_center(xb, rb)
    near = np.abs(xb - center) <= 300.0
    ref_center = jump_center(reference["x"], reference["rho"])
    shifted = xb + ref_center - center
    ref_mask = near & (shifted >= reference["x"].min()) & (
        shifted <= reference["x"].max())
    result = {"steps": steps, "time_s": tb, "shock_x_cm": center,
              "min_rho": float(min(rb.min(), rc.min())),
              "min_Tgas_K": float(min(tgb.min(), tgc.min())),
              "min_Trad_K": float(min(trb.min(), trc.min()))}
    for name, before, after in (("Tgas", tgb, tgc), ("Trad", trb, trc)):
        field = name.lower()
        ref = reference_at(shifted, reference, field, ref_center)
        result[name] = {
            "baseline_peak_K": float(before[near].max()),
            "candidate_peak_K": float(after[near].max()),
            "max_AB_difference_K": float(np.max(np.abs(after[near]-before[near]))),
            "mean_AB_difference_K": float(np.mean(np.abs(after[near]-before[near]))),
            "baseline_reference_L1_K": float(np.mean(np.abs(before[ref_mask]-ref[ref_mask]))),
            "candidate_reference_L1_K": float(np.mean(np.abs(after[ref_mask]-ref[ref_mask]))),
        }
    audit = json.loads((root / "output" / f"audit{steps}" / "summary.json").read_text())
    for case in audit["cases"]:
        mode = case["label"]
        result[mode] = {
            "max_coarse_fine_mismatch": case["max_coarse_fine_mismatch"],
            "sum_coarse_fine_balance": case["sum_coarse_fine_balance"],
            "physical_boundary_delta": case["physical_boundary_delta"],
            "sum_diffusion_delta": case["sum_diffusion_delta"],
            "max_rank_relative_balance": case["max_rank_relative_balance"],
        }
        folder = root / "output" / f"{mode}_{steps}" / "cases" / "mach5"
        log = (folder / "rank.0.log").read_text()
        newton = [(int(n), float(res)) for n, res in re.findall(
            r"\[NR\] status=converged iterations=(\d+).*?final_l2=([\deE+.-]+)", log)
            if int(n) > 0]
        fgmres = [int(n) for n in re.findall(r"\[LINEAR_FGMRES\] iterations=(\d+)", log)]
        wall = re.search(r"Elapsed \(wall clock\) time.*?: ([^\n]+)",
                         (folder / "launcher.log").read_text())
        if len(newton) != steps or not wall:
            raise ValueError(f"incomplete solver record: {folder}")
        result[mode].update(newton_max_iterations=max(n for n, _ in newton),
                            newton_max_final_l2=max(res for _, res in newton),
                            fgmres_total_iterations=sum(fgmres),
                            wall_clock=wall.group(1).strip())
    return result, {"x": xb-center, "Tgas": tgc-tgb, "Trad": trc-trb}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/work/mizutaniks/wor102"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "output" / "ab_report"
    output.mkdir(parents=True, exist_ok=True)
    reference = load_analytic_profile(Path(__file__).resolve().parent /
        "reference/radiative_shock/mach5_semianalytic.csv")
    results = []
    differences = {}
    for steps, index in ((1, 22), (10, 31)):
        row, difference = summarize(root, reference, steps, index)
        results.append(row)
        differences[steps] = difference
    for mode in ("baseline", "candidate"):
        folder = root / "output" / f"{mode}_1_rss2" / "cases" / "mach5"
        files = list(folder.glob("rss.*.txt"))
        ranks = {p.name.split(".")[1] for p in files}
        if len(files) != 32 or len(ranks) != 32:
            raise ValueError(f"incomplete rank RSS: {folder}")
        values = sorted(int(p.read_text().split("=")[1]) for p in files)
        results[0][mode]["rank_rss_kb"] = {
            "minimum": values[0], "median": (values[15]+values[16])/2,
            "maximum": values[-1], "sum": sum(values)}
    (output / "summary.json").write_text(json.dumps({"cases": results}, indent=2)+"\n")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for ax, field in zip(axes, ("Tgas", "Trad")):
        for steps in (1, 10):
            data = differences[steps]
            mask = np.abs(data["x"]) <= 300.0
            ax.plot(data["x"][mask], data[field][mask], ".-", ms=3,
                    label=f"{steps} cycle{'s' if steps > 1 else ''}")
        ax.axhline(0.0, color="black", lw=0.7)
        ax.set_ylabel(f"candidate − baseline {field} [K]")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("x − shock [cm]")
    fig.savefig(output / "ab_temperature_difference.png", dpi=180)
    plt.close(fig)
    html = ["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
            "<title>WOR-102 Mach 5 A/B</title>",
            "<style>body{font:17px/1.5 system-ui;max-width:1050px;margin:32px auto;padding:0 18px}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:7px;text-align:right}"
            "td:first-child,th:first-child{text-align:left}img{max-width:100%}</style></head><body>",
            "<h1>WOR-102: Mach 5 A/B</h1><p>同一N2048相当AMRチェックポイントから再開。"
            "温度差は同一格子・同一時刻で比較。</p><table><tr><th>cycle</th>"
            "<th>方式</th><th>粗細面最大相対不一致</th><th>Newton最終L2最大</th>"
            "<th>FGMRES反復合計</th><th>壁時計</th><th>Tgas参照L1 [K]</th>"
            "<th>Trad参照L1 [K]</th></tr>"]
    for row in results:
        for mode in ("baseline", "candidate"):
            c = row[mode]
            html.append("<tr><td>{}</td><td>{}</td><td>{:.3e}</td><td>{:.3e}</td>"
                        "<td>{}</td><td>{}</td><td>{:.5f}</td><td>{:.5f}</td></tr>".format(
                            row["steps"], escape(mode), c["max_coarse_fine_mismatch"],
                            c["newton_max_final_l2"], c["fgmres_total_iterations"],
                            escape(c["wall_clock"]),
                            row["Tgas"][f"{mode}_reference_L1_K"],
                            row["Trad"][f"{mode}_reference_L1_K"]))
    html.extend(["</table><p>1サイクルのrank別最大RSS: baseline 中央値 {:.1f} MB、"
                 "candidate 中央値 {:.1f} MB（KiB÷1024）。</p>".format(
                     results[0]["baseline"]["rank_rss_kb"]["median"]/1024,
                     results[0]["candidate"]["rank_rss_kb"]["median"]/1024),
                 "<img src='ab_temperature_difference.png' alt='A/B temperature difference'>",
                 "<p>半解析解との差は衝撃波中心から±300 cmの単純平均絶対誤差。"
                 "短い再開区間のため全長計算の精度は判定しない。</p></body></html>"])
    (output / "index.html").write_text("\n".join(html)+"\n")
    for row in results:
        print(row["steps"], "cycles", "baseline mismatch",
              row["baseline"]["max_coarse_fine_mismatch"], "candidate mismatch",
              row["candidate"]["max_coarse_fine_mismatch"])


if __name__ == "__main__":
    main()
