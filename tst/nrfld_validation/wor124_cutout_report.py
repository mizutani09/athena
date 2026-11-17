#!/usr/bin/env python3
"""Render the WOR-124 center-crop A/B diagnostics as a self-contained report."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HISTORY_FIELDS = {"rho_max": 15, "Tgas_max": 19, "Er_max": 17}
PLOT_FIELDS = {"rho_cgs": "Density", "Tgas_K": "Gas temperature",
               "Erad_cgs": "Radiation energy", "Trad_K": "Radiation temperature",
               "Bcc1": "B1"}


def history_difference(parent_path, crop_path):
    parent = np.loadtxt(parent_path)
    crop = np.loadtxt(crop_path)
    crop = crop[np.unique(crop[:, 0], return_index=True)[1]]
    cycles = np.arange(1781, 1781 + len(crop))
    indices = np.array([np.argmin(abs(parent[:, 0]-t)) for t in crop[:, 0]])
    reference = parent[indices]
    if np.max(abs(reference[:, 0]-crop[:, 0])) > 0.004:
        raise ValueError("parent and crop histories are too far apart in time")
    result = {"cycles": cycles.tolist(), "crop_time": crop[:, 0].tolist(),
              "parent_time": reference[:, 0].tolist(), "fields": {}}
    for field, column in HISTORY_FIELDS.items():
        result["fields"][field] = ((crop[:, column]/reference[:, column])-1).tolist()
    return result


def render(comparison, history, outdir, alternate=None, later=None,
           alternate_history=None):
    outdir.mkdir(parents=True, exist_ok=True)
    bins = comparison["bins"]
    x = np.array([(int(name.split("-")[0]) + int(name.split("-")[1][:-2]))/2
                  for name in bins])
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for field, label in PLOT_FIELDS.items():
        ax.semilogy(x, [b[field]["relative_l2"] for b in bins.values()], "o-", label=label)
    ax.set(xlabel="Radius bin center [au]", ylabel="Relative L2 difference",
           title="Crop vs full domain, cycle 1785")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/"radial_difference.svg")
    plt.close(fig)

    alternate_html = ""
    if alternate is not None:
        alt_bins = alternate["bins"]
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for source, label, style in ((bins, "375 au", "-"),
                                     (alt_bins, "750 au", "--")):
            radii = np.array([(int(name.split("-")[0])
                               + int(name.split("-")[1][:-2]))/2
                              for name in source])
            for field, color in (("Erad_cgs", "tab:red"),
                                 ("Trad_K", "tab:blue")):
                ax.semilogy(radii,
                            [b[field]["relative_l2"] for b in source.values()],
                            "o"+style, color=color, label=f"{field}, {label}")
        ax.set(xlabel="Radius bin center [au]", ylabel="Relative L2 difference",
               title="Radiation error vs cutout width, cycle 1785")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir/"width_comparison.svg")
        plt.close(fig)
        a = bins["150-300au"]
        b = alt_bins["150-300au"]
        alternate_html = f"""<h2>境界までの距離</h2>
<p>±750 auの短時間試験 job 341329 も正常終了。5 stepの壁時間81秒、
MaxRSS 28.8 GB。150–300 auでの輻射エネルギー相対L2差は
±375 auの {a['Erad_cgs']['relative_l2']:.2%} から
±750 auの {b['Erad_cgs']['relative_l2']:.2%} に減少した。
境界の影響は距離を増すと弱まるが、±750 auでも残る。</p>
<img src="width_comparison.svg" alt="切り出し幅と輻射場の差">
<p><a href="c1785_compare750.json">750 au比較JSON</a></p>"""
        (outdir/"c1785_compare750.json").write_text(json.dumps(alternate, indent=2)+"\n")

    later_html = ""
    if later is not None:
        fig, ax = plt.subplots(figsize=(7, 4.3))
        for data, label in ((comparison, "cycle 1785"), (later, "cycle 1800")):
            radii = np.array([(int(name.split("-")[0])
                               + int(name.split("-")[1][:-2]))/2
                              for name in data["bins"]])
            ax.semilogy(radii,
                        [b["Erad_cgs"]["relative_l2"]
                         for b in data["bins"].values()], "o-", label=label)
        ax.set(xlabel="Radius bin center [au]", ylabel="Radiation relative L2 difference",
               title="Radiation error growth, 375 au crop")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir/"time_comparison.svg")
        plt.close(fig)
        e = later["bins"]["150-300au"]["Erad_cgs"]["relative_l2"]
        later_html = f"""<h2>時間とともに広がる差</h2>
<p>375 au切り出しのc1800（20 step）では、150–300 auの輻射エネルギー差が
{e:.2%}、一致ブロックの輻射エネルギー体積積分差が
{later['volume_integrals']['Erad_cgs']['relative_signed_difference']:+.2%} に増えた。</p>
<img src="time_comparison.svg" alt="輻射場の差の時間発展">
<p><a href="c1800_compare375.json">c1800比較JSON</a></p>"""
        (outdir/"c1800_compare375.json").write_text(json.dumps(later, indent=2)+"\n")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for field, values in history["fields"].items():
        ax.semilogy(history["cycles"], np.maximum(abs(np.array(values)), 1e-12),
                    label=f"{field}, 375 au")
    if alternate_history is not None:
        for field, values in alternate_history["fields"].items():
            ax.semilogy(alternate_history["cycles"],
                        np.maximum(abs(np.array(values)), 1e-12), "--",
                        label=f"{field}, 750 au")
    ax.set(xlabel="Cycle", ylabel="Absolute relative difference",
           title="Central maxima: crop vs full-domain history")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/"history_difference.svg")
    plt.close(fig)

    rows = "\n".join(
        "<tr><th>{}</th><td>{:.3g}</td><td>{:.3g}</td><td>{:.3g}</td>"
        "<td>{:.3g}</td><td>{:.3g}</td></tr>".format(
            html.escape(name), *(b[f]["relative_l2"] for f in PLOT_FIELDS))
        for name, b in bins.items())
    last = {k: v[-1] for k, v in history["fields"].items()}
    alternate_history_html = ""
    if alternate_history is not None:
        alt_last = {k: v[-1] for k, v in alternate_history["fields"].items()}
        alternate_history_html = f"""<p>±750 au切り出しのc{alternate_history['cycles'][-1]}では、
最大密度 {alt_last['rho_max']:+.3g}、最高ガス温度
{alt_last['Tgas_max']:+.3g}、最大輻射エネルギー
{alt_last['Er_max']:+.3g}。<a href="history_compare750.json">750 au履歴JSON</a></p>"""
    mass = comparison["volume_integrals"]["rho_cgs"]
    rad = comparison["volume_integrals"]["Erad_cgs"]
    title = "WOR-124: central restart cutout A/B"
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>{title}</title><style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;
line-height:1.6;color:#18212b}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse}}
th,td{{border:1px solid #cbd5e1;padding:.35rem .65rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}
.note{{background:#fff4db;padding:1rem;border-left:4px solid #df9b18}}
code{{background:#edf2f7;padding:.1rem .2rem}}</style>
<h1>{title}</h1>
<p>Commit <code>4f886c5d1ff99c91794f5bdcf52198e0a25d0cad</code> の同一バイナリ・
EOS/opacity tableで、c1780の全領域restartから中心±375 auを面磁場を含めて
MeshBlock単位で切り出した。XD M-test-a、64 MPI rank、c1780→1785。
全領域 job 341272 と切り出し job 341264 は正常終了。</p>
<p>切り出し2528→2535 blocks、全領域4096→4124 blocks。
同じ1 nodeで、5 stepの壁時間は70秒対126秒、MaxRSSは25.7 GB対37.7 GB。
FGMRES真相対残差最大は双方とも約8.6×10⁻⁹、Newton診断は全てconverged。
opacity表域外アクセスは0。</p>
<div class="note"><strong>判定:</strong> 中心10 auは短時間ほぼ一致するが、
外側の輻射場は5 stepで大きく変わる。周期境界のままsecond coreまでの継続には使えない。
有効な非周期FLD境界と外部流入・重力の扱いが必要。</div>
<h2>空間分布（c1785）</h2><p>同一座標・levelの2532 blocks（1296384 cells）を比較。
数値は各半径範囲の相対L2差。</p>
<table><thead><tr><th>半径</th><th>密度</th><th>ガス温度</th><th>輻射エネルギー</th>
<th>輻射温度</th><th>B1</th></tr></thead><tbody>{rows}</tbody></table>
<p>一致ブロックでの体積積分差：密度 {mass['relative_signed_difference']:+.3%}、
輻射エネルギー {rad['relative_signed_difference']:+.3%}。</p>
<img src="radial_difference.svg" alt="半径ごとの相対L2差">
{alternate_html}
{later_html}
<h2>中心最大値の履歴</h2>
<p>c{history['cycles'][-1]}での全領域に対する差：
最大密度 {last['rho_max']:+.3g}、最高ガス温度 {last['Tgas_max']:+.3g}、
最大輻射エネルギー {last['Er_max']:+.3g}。</p>
{alternate_history_html}
<img src="history_difference.svg" alt="中心最大値の履歴差">
<p>生データ：<a href="c1785_compare.json">空間比較JSON</a>、
<a href="history_compare.json">履歴比較JSON</a>。</p>
<p>元計算との比較は短い重なり区間だけを検証する。周期境界は物理的な
流入・輻射条件を再現せず、長時間の妥当性を示さない。</p>
</html>"""
    (outdir/"index.html").write_text(page)
    (outdir/"history_compare.json").write_text(json.dumps(history, indent=2) + "\n")
    if alternate_history is not None:
        (outdir/"history_compare750.json").write_text(
            json.dumps(alternate_history, indent=2) + "\n")
    (outdir/"c1785_compare.json").write_text(json.dumps(comparison, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("parent_history", type=Path)
    parser.add_argument("crop_history", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--alternate-comparison", type=Path)
    parser.add_argument("--later-comparison", type=Path)
    parser.add_argument("--alternate-history", type=Path)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    history = history_difference(args.parent_history, args.crop_history)
    alternate = (json.loads(args.alternate_comparison.read_text())
                 if args.alternate_comparison else None)
    later = (json.loads(args.later_comparison.read_text())
             if args.later_comparison else None)
    alternate_history = (history_difference(args.parent_history, args.alternate_history)
                         if args.alternate_history else None)
    render(comparison, history, args.outdir, alternate, later, alternate_history)
    print(args.outdir/"index.html")


if __name__ == "__main__":
    main()
