#!/usr/bin/env python3
"""Combine matched WOR-20 HD/MHD/FLD/gravity suites into HTML/JSON/PNG."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np


CASES = ("hd", "hd_gravity", "mhd", "mhd_gravity", "hd_fld",
         "hd_fld_gravity", "mhd_fld", "mhd_fld_gravity")


def load(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if summary["status"] != "passed" or manifest["status"] != "passed":
        raise ValueError(f"Suite failed: {path}")
    return {"summary": summary, "manifest": manifest, "root": str(path.resolve())}


def row_for(suite: dict, case: str, ranks: int) -> dict:
    aggregate = next(item for item in suite["summary"]["aggregates"]
                     if item["case"] == case and item["ranks"] == ranks)
    runs = [run for run in suite["summary"]["runs"]
            if run["case"] == case and run["ranks"] == ranks
            and run["status"] == "passed"]
    parts = {}
    for name in runs[0]["metrics"]["parts"]:
        parts[name] = float(np.median([
            run["metrics"]["parts"][name]["mean"] / run["measured_steps"]
            for run in runs]))
    return {**aggregate, "parts_seconds_per_step_median": parts,
            "max_rank_rss_kib_median": float(np.median([
                run["metrics"]["max_rank_rss_kib"] for run in runs])),
            "newton_iterations_total": sorted({
                run["metrics"]["newton_iterations_total"] for run in runs}),
            "mass_relative_drift_max_abs": max(
                abs(run["metrics"]["mass_relative_drift"]) for run in runs)}


def make_plot(rows: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {case: case.replace("_", "+") for case in CASES}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(CASES))
    width = 0.38
    for index, ranks in enumerate((1, 4)):
        selected = [next(row for row in rows if row["case"] == case
                         and row["ranks"] == ranks) for case in CASES]
        axes[0].bar(x+(index-.5)*width,
                    [row["seconds_per_step_median"] for row in selected], width,
                    label=f"{ranks} rank")
        axes[2].bar(x+(index-.5)*width,
                    [row["max_rank_rss_kib_median"]/1024 for row in selected], width,
                    label=f"{ranks} rank")
    selected = [next(row for row in rows if row["case"] == case
                     and row["ranks"] == 4) for case in CASES]
    bottom = np.zeros(len(CASES))
    phase_groups = {
        "explicit": ("explicit_integrator",),
        "gravity": ("gravity_solve",),
        "implicit FLD": ("fld_prepare", "fld_solve", "fld_sync", "fld_finish"),
        "other": ("amr", "output", "other"),
    }
    for label, names in phase_groups.items():
        values = np.array([sum(row["parts_seconds_per_step_median"].get(name, 0.0)
                               for name in names) for row in selected])
        axes[1].bar(x, values, bottom=bottom, label=label)
        bottom += values
    for axis in axes:
        axis.set_xticks(x, [labels[case] for case in CASES], rotation=45, ha="right")
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("median seconds / step")
    axes[0].legend()
    axes[1].set_ylabel("4-rank seconds / step by phase")
    axes[1].legend(fontsize=8)
    axes[2].set_ylabel("median maximum-rank RSS [MiB]")
    axes[2].legend()
    fig.suptitle("WOR-20 frozen-source physics cost matrix")
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=170)
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-fld", type=Path, required=True)
    parser.add_argument("--fld", type=Path, required=True)
    parser.add_argument("--hd-long", type=Path, required=True,
                        help="Long-window HD suite used in place of the short matrix HD run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suites = {name: load(getattr(args, name)) for name in ("non_fld", "fld", "hd_long")}
    rows = []
    for case in CASES:
        suite = suites["hd_long"] if case == "hd" else suites[
            "fld" if "fld" in case else "non_fld"]
        for ranks in (1, 4):
            rows.append({"case": case, "ranks": ranks, **row_for(suite, case, ranks)})
    if not all(row["all_windows_at_least_10s"] for row in rows):
        raise ValueError("A selected aggregate has a measurement window below ten seconds")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "passed", "source_tree":
              "c4ab200383c7317a2feccc50aa9578d5cf71478d",
              "suites": {name: suite["root"] for name, suite in suites.items()},
              "rows": rows,
              "limitations": [
                  "Controlled uniform benchmark, not a physical stellar model.",
                  "Local single-node WSL strong scaling only.",
                  "Implicit FLD timing excludes shared explicit radiation work.",
                  "Per-rank RSS peaks are not simultaneous node RSS.",
              ]}
    (output / "physics_matrix.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    image = make_plot(rows)
    (output / "physics_matrix.png").write_bytes(base64.b64decode(image))

    table = "\n".join("<tr>"+"".join(f"<td>{value}</td>" for value in (
        row["case"], row["ranks"], f'{row["seconds_per_step_median"]:.6f}',
        f'{row["efficiency"]:.3f}',
        f'{row["implicit_fld_fraction_mean"]:.3f}',
        f'{row["max_rank_rss_kib_median"]/1024:.1f}'))+"</tr>" for row in rows)
    html = f"""<!doctype html><meta charset='utf-8'><title>WOR-20 physics cost matrix</title>
<style>body{{font-family:sans-serif;max-width:1300px;margin:2rem auto;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eee;padding:.1rem .25rem}}</style>
<h1>WOR-20 frozen-source physics cost matrix</h1>
<p>Uniform 32³ benchmark on source tree <code>{result['source_tree']}</code>.
All 48 selected runs passed; every selected aggregate uses three repeats and
measurement windows of at least ten seconds.</p>
<img alt='physics timing matrix' src='data:image/png;base64,{image}'>
<table><thead><tr><th>case</th><th>ranks</th><th>median s/step</th>
<th>strong efficiency</th><th>implicit FLD fraction</th><th>max-rank RSS MiB</th></tr></thead>
<tbody>{table}</tbody></table>
<h2>Interpretation</h2><p>Implicit FLD dominates all FLD-enabled cases.
Adding ideal MHD or self-gravity increases total cost much less than enabling FLD.
The gravity and MHD+gravity controls scale poorly at this small fixed grid because
global/coarse work is large relative to local work. Strong-field collapse difficulty
must not be inferred from the inexpensive smooth ideal-MHD control.</p>
<h2>Limitations</h2><ul>{''.join(f'<li>{item}</li>' for item in result['limitations'])}</ul>
<p>Exact paths and phase metrics are in <code>physics_matrix.json</code>.</p>"""
    (output / "index.html").write_text(html)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
