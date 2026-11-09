#!/usr/bin/env python3
"""Diagnose up/down entropy and transport in completed solar snapshots.

The cell-centered fluxes are interior proxies, not the Godunov boundary
fluxes or a Lagrangian history of any particular downdraft.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_solar_initial_profile import AthenaEosTable  # noqa: E402
from plot_simple_convection_surface3d import read_volume  # noqa: E402
from plot_simple_convection_tau import units_from_input  # noqa: E402

FSUN = 6.3e10


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def paired_terminal(campaign: Path, row: dict) -> tuple[Path, Path]:
    run = campaign / row["case"]
    files = sorted(run.glob("*.prim_full.*.athdf"))
    if not files:
        raise FileNotFoundError(f"No full primitive volume in {run}")
    prim = files[-1]
    uov = run / prim.name.replace(".prim_full.", ".uov_full.")
    if not uov.exists():
        raise FileNotFoundError(uov)
    return prim, uov


def profile(prim_path: Path, uov_path: Path, eos: AthenaEosTable,
            units: dict[str, float]) -> tuple[dict, np.ndarray]:
    prim, (_, _, z), time = read_volume(prim_path, "prim")
    uov, _, uov_time = read_volume(uov_path, "user_out_var")
    if abs(time-uov_time) > 1e-5:
        raise ValueError(f"Unpaired times: {prim_path} and {uov_path}")
    rho = prim[0] * units["rho_unit"]
    gas = uov[0]
    if np.any(rho <= 0) or np.any(gas <= 0):
        raise ValueError(f"Nonpositive EOS state: {prim_path}")
    speed = np.sqrt(units["egas_unit"] / units["rho_unit"])
    vz = prim[4] * speed
    mdot = rho*vz
    entropy = 10.0**eos.interp[4](np.stack(
        (np.log10(gas/rho), np.log10(rho)), axis=-1))
    enthalpy = (gas + prim[1]*units["egas_unit"])/rho
    up = mdot > 0
    down = mdot < 0
    mup = np.where(up, mdot, 0.0)
    mdown = np.where(down, -mdot, 0.0)
    sum_up = np.sum(mup, axis=(1, 2))
    sum_down = np.sum(mdown, axis=(1, 2))
    def weighted(field: np.ndarray, weights: np.ndarray,
                 total: np.ndarray) -> np.ndarray:
        return np.divide(np.sum(weights*field, axis=(1, 2)), total,
                         out=np.full_like(total, np.nan), where=total > 0)

    s_up = weighted(entropy, mup, sum_up)
    s_down = weighted(entropy, mdown, sum_down)
    h_up = weighted(enthalpy, mup, sum_up)
    h_down = weighted(enthalpy, mdown, sum_down)
    hmean = np.mean(enthalpy, axis=(1, 2))
    neutral_flux = np.mean(mdot*(enthalpy-hmean[:, None, None]), axis=(1, 2))
    qrad = uov[4]  # positive means gas cools and radiation gains energy
    return {
        "time_code": time,
        "up_area_fraction": np.mean(up, axis=(1, 2)),
        "up_mass_flux_cgs": np.mean(mup, axis=(1, 2)),
        "down_mass_flux_cgs": np.mean(mdown, axis=(1, 2)),
        "up_entropy_cgs": s_up,
        "down_entropy_cgs": s_down,
        "entropy_contrast_cgs": s_up-s_down,
        "up_enthalpy_cgs": h_up,
        "down_enthalpy_cgs": h_down,
        "enthalpy_contrast_cgs": h_up-h_down,
        "neutral_enthalpy_flux_over_solar": neutral_flux/FSUN,
        "mean_qrad_cgs": np.mean(qrad, axis=(1, 2)),
        "up_qrad_cgs": np.sum(np.where(up, qrad, 0), axis=(1, 2)) /
                         np.maximum(np.sum(up, axis=(1, 2)), 1),
        "down_qrad_cgs": np.sum(np.where(down, qrad, 0), axis=(1, 2)) /
                           np.maximum(np.sum(down, axis=(1, 2)), 1),
        "mean_temperature_K": np.mean(uov[2], axis=(1, 2)),
    }, z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-state", required=True, type=Path)
    parser.add_argument("--eos-table", required=True, type=Path)
    parser.add_argument("--early-window", required=True, type=float, nargs=2)
    parser.add_argument("--late-window", required=True, type=float, nargs=2)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    state_path = args.feedback_state.resolve()
    campaign = state_path.parent
    state = json.loads(state_path.read_text())
    windows = {"early": args.early_window, "late": args.late_window}
    if not (windows["early"][0] < windows["early"][1] <=
            windows["late"][0] < windows["late"][1]):
        parser.error("Expected nonoverlapping, increasing windows")
    chosen = {label: [row for row in state["completed"]
                      if bounds[0] <= row["end_time_code"] <= bounds[1]]
              for label, bounds in windows.items()}
    if any(len(rows) < 3 for rows in chosen.values()):
        parser.error("At least three completed snapshots required per window")
    eos_path = args.eos_table.resolve()
    eos = AthenaEosTable(eos_path)
    if eos.nvar < 5:
        parser.error("EOS table lacks specific entropy")
    input_path = campaign / "athinput.base.validated"
    units = units_from_input(input_path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    sources = []
    common_z = None
    for label, rows in chosen.items():
        values = []
        for row in rows:
            prim, uov = paired_terminal(campaign, row)
            item, z = profile(prim, uov, eos, units)
            if abs(item["time_code"]-row["end_time_code"]) > 0.1:
                raise ValueError(f"Unexpected snapshot time: {prim}")
            if common_z is None:
                common_z = z
            elif not np.allclose(common_z, z):
                raise ValueError("Grid changed between snapshots")
            values.append(item)
            sources.append({"time_code": item["time_code"],
                            "prim": str(prim), "uov": str(uov),
                            "prim_sha256": sha256(prim), "uov_sha256": sha256(uov)})
        means = {}
        for key in values[0]:
            if key == "time_code":
                continue
            stacked = np.stack([entry[key] for entry in values])
            valid = np.isfinite(stacked)
            count = np.sum(valid, axis=0)
            avg = np.divide(np.nansum(stacked, axis=0), count,
                            out=np.full(stacked.shape[1:], np.nan), where=count > 0)
            means[key] = [float(value) if np.isfinite(value) else None for value in avg]
        results[label] = {
            "window_code": windows[label], "snapshot_count": len(values),
            "mean": means,
            "time_codes": [entry["time_code"] for entry in values],
        }
    z_mm = common_z*units["leng_unit"]/1e8
    report = {
        "method": "completed terminal 3D dumps; EOS-table entropy; cell-centered "
                  "mass-flux-weighted conditional profiles; equal-weight time means",
        "limits": "Eulerian snapshots, not plume trajectories; cell-centered flux "
                  "proxy, not the Godunov lower-face energy ledger",
        "state": str(state_path), "state_sha256": sha256(state_path),
        "eos_table": str(eos_path), "eos_sha256": sha256(eos_path),
        "input": str(input_path), "input_sha256": sha256(input_path),
        "z_Mm": z_mm.tolist(), "windows": results, "sources": sources,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 17})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"early": "tab:blue", "late": "tab:orange"}
    for label in ("early", "late"):
        mean = results[label]["mean"]
        axes[0, 0].plot(np.array(mean["entropy_contrast_cgs"], dtype=float)/1e6,
                        z_mm, label=label, color=colors[label])
        axes[0, 1].plot(mean["up_mass_flux_cgs"], z_mm,
                        label=f"{label} up", color=colors[label])
        axes[0, 1].plot(mean["down_mass_flux_cgs"], z_mm,
                        label=f"{label} down", color=colors[label], ls="--")
        axes[1, 0].plot(mean["neutral_enthalpy_flux_over_solar"], z_mm,
                        label=label, color=colors[label])
        axes[1, 1].plot(np.array(mean["enthalpy_contrast_cgs"], dtype=float)/1e10,
                        z_mm, label=label, color=colors[label])
    axes[0, 0].set(xlabel="Up − down entropy [10⁶ erg g⁻¹ K⁻¹]",
                   ylabel="z [Mm]", title="Mass-weighted entropy contrast")
    axes[0, 1].set(xlabel="|mass flux| [g cm⁻² s⁻¹]",
                   title="Interior circulation")
    axes[1, 0].set(xlabel="Mass-neutral enthalpy proxy / F☉",
                   ylabel="z [Mm]", title="Cell-centered transport")
    axes[1, 1].set(xlabel="Up − down specific enthalpy [10¹⁰ erg g⁻¹]",
                   title="Thermal contrast")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output / "vertical_profiles.png", dpi=150)
    plt.close(fig)
    rows = []
    for label in ("early", "late"):
        item = results[label]
        bottom = {key: value[0] for key, value in item["mean"].items()}
        rows.append(f"<tr><th>{html.escape(label)}</th>"
                    f"<td>{item['snapshot_count']}</td>"
                    f"<td>{bottom['up_entropy_cgs']:.8g}</td>"
                    f"<td>{bottom['down_entropy_cgs']:.8g}</td>"
                    f"<td>{bottom['entropy_contrast_cgs']:.5g}</td>"
                    f"<td>{bottom['up_mass_flux_cgs']:.5g}</td>"
                    f"<td>{bottom['down_mass_flux_cgs']:.5g}</td></tr>")
    page = """<!doctype html><html lang='ja'><meta charset='utf-8'>
<title>WOR-74 coarse downflow diagnosis</title><style>
body{font:21px/1.55 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172033}
h1{font-size:2rem}h2{font-size:1.5rem}strong{color:#075985}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #cbd5e1;padding:.6rem;text-align:right}
th:first-child{text-align:left}img{max-width:100%;height:auto}
.note{background:#eff6ff;border-left:5px solid #0284c7;padding:1rem}
</style><h1>WOR-74: 下降流の鉛直診断</h1>
<p class='note'><strong>完了済みの96 km計算だけを解析。</strong>
EOS表による比エントロピーをセル中心で計算し、鉛直質量流束の符号で上下流を分けた。</p>
<table><tr><th>窓</th><th>3D枚数</th><th>下端上昇流s</th><th>下端下降流s</th>
<th>s差 [erg g⁻¹ K⁻¹]</th><th>上昇質量流束</th><th>下降質量流束</th></tr>"""
    page += "".join(rows) + "</table>"
    page += "<img src='vertical_profiles.png' alt='Vertical profiles'>"
    page += ("<p><strong>解釈上の制限:</strong> 同じ下降プルームを追跡していない。"
             "下端面の厳密なGodunov流束は履歴の診断を使用する。"
             "瞬間のQradは強結合域で大きな交換量になるため、"
             "時間積分した放射冷却の代用にはしない。"
             "<a href='summary.json'>数値・入力hash・出力一覧</a>。</p></html>")
    (output / "index.html").write_text(page + "\n")
    print(output / "index.html")


if __name__ == "__main__":
    main()
