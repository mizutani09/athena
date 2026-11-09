#!/usr/bin/env python3
"""Compare short, matched restarts for the WOR-74 lower/upper boundary diagnosis.

These branches measure a transient response. They are not new equilibria.
History face fluxes are sampled after accepted steps, so the energy ledger is
diagnostic rather than a strict conservation test.
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
from solar_convection_report import (  # noqa: E402
    input_values, parse_log, physical_time_unit, read_history,
    sampled_energy_ledger, total_bottom_flux, total_top_flux,
)

FSUN = 6.3e10
CASES = {
    "baseline": {"alpha": 0.5, "delta_s": 0.0, "label": "baseline: alpha=0.5"},
    "alpha_025": {"alpha": 0.25, "delta_s": 0.0, "label": "top: alpha=0.25"},
    "alpha_100": {"alpha": 1.0, "delta_s": 0.0, "label": "top: alpha=1.0"},
    "splus_1e6": {"alpha": 0.5, "delta_s": 1e6, "label": "bottom: ds_in=+1e6"},
    "sminus_1e6": {"alpha": 0.5, "delta_s": -1e6, "label": "bottom: ds_in=-1e6"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_over(time: np.ndarray, values: np.ndarray, left: float,
              right: float) -> float:
    if time[0] > left or time[-1] < right:
        raise ValueError(f"History does not cover [{left}, {right}]")
    inner = (time > left) & (time < right)
    ts = np.r_[left, time[inner], right]
    ys = np.interp(ts, time, values)
    return float(np.trapezoid(ys, ts) / (right - left))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    histories = {}
    log_summaries = {}
    for name in CASES:
        directory = args.root / name
        hst = directory / "simple_convection_general_6Mm_box.hst"
        log = directory / "run.log"
        histories[name] = read_history(hst, args.header_from)
        log_summaries[name] = parse_log(log)
        if not log_summaries[name]["completed"] or log_summaries[name]["fatal_count"]:
            raise ValueError(f"Incomplete or fatal run: {name}")
        if abs(log_summaries[name]["final_time"] - 24010) > 1e-6:
            raise ValueError(f"Unexpected final time: {name}")
        h = histories[name]
        directional_error = np.max(np.abs(
            h["FhydUpBottom"] + h["FhydDownBottom"] - h["FhydBottom"]))
        if directional_error > 1e-9 * FSUN:
            raise ValueError(f"Directional lower-face fluxes do not sum: {name}")
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if abs(manifest["rad_top_alpha"] - CASES[name]["alpha"]) > 1e-12:
                raise ValueError(f"Unexpected upper boundary coefficient: {name}")
            if abs(manifest["s_in"] - (1.583570672066638e9
                                         + CASES[name]["delta_s"])) > 1e-4:
                raise ValueError(f"Unexpected inflow entropy: {name}")
            if manifest["binary_sha256"] != sha256(args.binary):
                raise ValueError(f"Binary hash differs: {name}")
            if manifest["restart_sha256"] != sha256(args.restart):
                raise ValueError(f"Restart hash differs: {name}")

    first_times = [h["time"][0] for h in histories.values()]
    if max(first_times) - min(first_times) > 1e-7:
        raise ValueError("The branches are not aligned on their first sample")
    reference_bottom = total_bottom_flux(histories["baseline"])[0]
    for name in ("alpha_025", "alpha_100"):
        if abs(total_bottom_flux(histories[name])[0] - reference_bottom) > 1e-8*FSUN:
            raise ValueError(f"Upper-boundary branch did not start identically: {name}")

    start = max(h["time"][0] for h in histories.values())
    end = min(h["time"][-1] for h in histories.values())
    if end - start < 5:
        raise ValueError("Common interval too short for this diagnostic")
    units = input_values(args.input)
    velocity_unit = np.sqrt(units["egas_unit"] / units["rho_unit"])
    time_unit = physical_time_unit(units, velocity_unit)
    height = (units["x3max"] - units["x3min"]) * units["leng_unit"]
    baseline_vz = np.sqrt(np.maximum(histories["baseline"]["VzRms2Mean"], 0))
    crossing_code = height / (float(np.mean(baseline_vz)) * velocity_unit) / time_unit

    summary = {
        "method": "Matched restart at t=24000; short transient, no feedback controller",
        "common_window_code": [start, end],
        "window_fraction_of_vertical_crossing": (end - start) / crossing_code,
        "vertical_crossing_code": crossing_code,
        "solar_flux_cgs": FSUN,
        "restart": str(args.restart), "restart_sha256": sha256(args.restart),
        "binary": str(args.binary), "binary_sha256": sha256(args.binary),
        "input": str(args.input), "input_sha256": sha256(args.input),
        "header_from": str(args.header_from),
        "cases": {},
    }
    for name, spec in CASES.items():
        h = histories[name]
        indices = np.flatnonzero((h["time"] >= start) & (h["time"] <= end))
        if len(indices) < 5:
            raise ValueError(f"Too few common samples: {name}")
        fluxes = {
            "top_total": total_top_flux(h),
            "bottom_total": total_bottom_flux(h),
            "bottom_up_hydro": h["FhydUpBottom"],
            "bottom_down_hydro": h["FhydDownBottom"],
            "bottom_radiation": h["FradBottom"] + h.get("FadvRadBottom", 0),
        }
        means = {key + "_over_solar": mean_over(h["time"], values, start, end) / FSUN
                 for key, values in fluxes.items()}
        summary["cases"][name] = {
            **spec, "history": str(args.root / name / "simple_convection_general_6Mm_box.hst"),
            "history_sha256": sha256(args.root / name / "simple_convection_general_6Mm_box.hst"),
            "n_samples": len(h["time"]), "first_time_code": float(h["time"][0]),
            "final_time_code": float(h["time"][-1]),
            "mean_flux": means,
            "sampled_energy_ledger": sampled_energy_ledger(h, indices, units, FSUN),
            "log": log_summaries[name],
        }

    ref = summary["cases"]["baseline"]["mean_flux"]
    for name, result in summary["cases"].items():
        result["difference_from_baseline"] = {
            key: value - ref[key] for key, value in result["mean_flux"].items()
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    quantities = [
        (total_top_flux, "Top total flux / F_sun"),
        (total_bottom_flux, "Bottom total flux / F_sun"),
        (lambda h: h["FhydUpBottom"], "Bottom upflow hydro flux / F_sun"),
        (lambda h: h["FhydDownBottom"], "Bottom downflow hydro flux / F_sun"),
    ]
    for ax, (getter, ylabel) in zip(axes.flat, quantities):
        for name, spec in CASES.items():
            h = histories[name]
            values = getter(h)
            ax.plot(h["time"] - 24000, values / FSUN, label=spec["label"], lw=2)
        ax.set(xlabel="t − 24000 [code]", ylabel=ylabel, xlim=(0, end - 24000))
        ax.grid(alpha=.3)
    axes[0, 0].axhline(1, color="black", ls="--", lw=1, alpha=.7)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("WOR-74: matched-restart boundary perturbations", fontsize=20)
    fig.tight_layout()
    fig.savefig(args.output / "boundary_sweep.png", dpi=160)
    plt.close(fig)

    rows = []
    for name, result in summary["cases"].items():
        flux = result["mean_flux"]
        delta = result["difference_from_baseline"]
        ledger = result["sampled_energy_ledger"]
        rows.append(
            f"<tr><th>{html.escape(result['label'])}</th>"
            f"<td>{flux['top_total_over_solar']:.3f}</td>"
            f"<td>{flux['bottom_total_over_solar']:.3f}</td>"
            f"<td>{delta['top_total_over_solar']:+.3f}</td>"
            f"<td>{delta['bottom_total_over_solar']:+.3f}</td>"
            f"<td>{ledger['stored_energy_trend_over_solar']:+.3f}</td></tr>"
        )
    report = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>WOR-74 粗格子境界試験</title><style>
body{{font:21px/1.55 system-ui,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1{{font-size:2rem}}h2{{font-size:1.5rem}}strong{{color:#075985}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #cbd5e1;padding:.6rem;text-align:right}}
th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}
.note{{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}}
</style><h1>WOR-74：粗格子の境界摂動試験</h1>
<p class="note"><strong>これは過渡応答の試験です。</strong> 同じ t=24000 restart と同じ実行バイナリを使い、
上端放射境界係数 α または流入比エントロピー s<sub>in</sub> だけを変えました。
比較区間 t={start:.2f}–{end:.2f} は鉛直横断時間の <strong>{(end-start)/crossing_code:.2f} 倍</strong>です。
したがって新しい定常値は判断できません。</p>
<table><tr><th>条件</th><th>上端 / F☉</th><th>下端 / F☉</th><th>上端差</th><th>下端差</th><th>蓄積率 / F☉</th></tr>
{''.join(rows)}</table>
<img src="boundary_sweep.png" alt="上端と下端の流束の時系列">
<h2>読めること</h2>
<p>αを変えると上端の放射流束と下端のhydro流束がともに変わります。
一方、s<sub>in</sub>を ±10⁶ erg g⁻¹ K⁻¹ 変えると下端の流束が先に応答し、
上端流束の変化はこの短い区間では小さいです。
下端の上昇・下降流は大きな逆向き流束を持ち、正味流束はその差です。</p>
<p><strong>制限：</strong>境界条件を急に変えた直後なので、流束には音波や対流の揺らぎも含まれます。
蓄積率は HST の EtotGrav の直線傾向から求め、面流束は各時刻の標本値です。
放射境界係数の変更は物理的な較正の提案ではありません。
<a href="summary.json">数値、入力と実行バイナリのSHA256、ログ診断</a>。</p></html>"""
    (args.output / "index.html").write_text(report)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
