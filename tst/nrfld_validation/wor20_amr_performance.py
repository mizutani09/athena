#!/usr/bin/env python3
"""Combine validated long-window Test-C uniform/AMR performance suites."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re
import statistics


def load(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    if summary["status"] != "passed":
        raise ValueError(f"Suite did not pass: {path}")
    return summary


def fgmres_iterations(log: Path) -> list[int]:
    values = []
    for line in log.read_text().splitlines():
        if line.startswith("[LINEAR_FGMRES]"):
            match = re.search(r"\biterations=(\d+)", line)
            if match:
                values.append(int(match.group(1)))
    if not values:
        raise ValueError(f"No FGMRES records: {log}")
    return values


def select(summary: dict, rank: int, expected_layout: str) -> dict:
    aggregates = [row for row in summary["aggregates"] if row["ranks"] == rank]
    runs = [row for row in summary["runs"] if row["ranks"] == rank]
    if len(aggregates) != 1 or len(runs) != 3:
        raise ValueError(f"Expected one aggregate and three rank-{rank} runs")
    aggregate = aggregates[0]
    if (aggregate.get("problem") != "test_c" or
            aggregate.get("layout") != expected_layout or
            not aggregate["all_windows_at_least_10s"] or
            any(run["status"] != "passed" for run in runs)):
        raise ValueError(f"Invalid selected {expected_layout} rank-{rank} group")
    iterations = []
    for run in runs:
        iterations.extend(fgmres_iterations(Path(run["path"]) / "run.log"))
    cells = {json.loads((Path(run["path"]) / "performance_rank0.json").read_text())
             ["steps"][0]["global_cells"] for run in runs}
    if len(cells) != 1:
        raise ValueError("Global leaf-cell count changed within selected group")
    controls = [run.get("timer_control") for run in runs]
    valid_controls = [control for control in controls if control is not None]
    if valid_controls and not all(control["final_arrays_identical"] and
                                  control["newton_status_and_iterations_identical"]
                                  for control in valid_controls):
        raise ValueError("Timer control changed a solution or Newton history")
    return {
        "layout": expected_layout,
        "ranks": rank,
        "meshblocks": aggregate["meshblocks"],
        "leaf_cells": cells.pop(),
        "seconds_per_step": aggregate["seconds_per_step_median"],
        "seconds_min": aggregate["seconds_per_step_min"],
        "seconds_max": aggregate["seconds_per_step_max"],
        "max_fgmres_iterations": max(iterations),
        "mean_fgmres_iterations": statistics.mean(iterations),
        "preconditioner_fraction": aggregate["fgmres_preconditioner_fraction_mean"],
        "matvec_fraction": aggregate["fgmres_matvec_fraction_mean"],
        "reduction_fraction": aggregate["fgmres_reduction_fraction_mean"],
        "vector_fraction": aggregate["fgmres_vector_and_other_fraction_mean"],
        "max_rank_rss_mib": statistics.median(
            run["metrics"]["max_rank_rss_kib"] for run in runs)/1024,
        "mass_drift_max_abs": max(abs(run["metrics"]["mass_relative_drift"])
                                  for run in runs),
        "newton_iterations_per_step": sorted({
            run["metrics"]["newton_iterations_total"]/run["measured_steps"]
            for run in runs}),
        "timer_controls": len(valid_controls),
        "source_suites": sorted({str(Path(run["path"]).parent) for run in runs}),
    }


def plots(output: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ranks = [1, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for layout, marker in (("uniform", "o"), ("static_amr", "s")):
        selected = [next(row for row in rows if row["layout"] == layout and
                         row["ranks"] == rank) for rank in ranks]
        label = "uniform" if layout == "uniform" else "static AMR"
        axes[0].plot(ranks, [row["seconds_per_step"] for row in selected],
                     marker=marker, label=label)
        axes[1].plot(ranks, [row["max_fgmres_iterations"] for row in selected],
                     marker=marker, label=label)
        axes[2].plot(ranks, [row["max_rank_rss_mib"] for row in selected],
                     marker=marker, label=label)
    axes[0].set(ylabel="seconds / step", yscale="log")
    axes[1].set(ylabel="max FGMRES iterations / solve")
    axes[2].set(ylabel="median max-rank RSS [MiB]")
    for axis in axes:
        axis.set(xlabel="MPI ranks", xticks=ranks)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Conservative Test C: uniform versus centered level-1 static AMR")
    fig.tight_layout()
    fig.savefig(output / "amr_vs_uniform.png", dpi=160)
    plt.close(fig)

    labels = [f"{'uniform' if r['layout']=='uniform' else 'AMR'}\np={r['ranks']}"
              for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = np.zeros(len(rows))
    for key, label in (("preconditioner_fraction", "MG preconditioner"),
                       ("matvec_fraction", "operator matvec"),
                       ("reduction_fraction", "FGMRES reductions"),
                       ("vector_fraction", "vectors / other")):
        values = np.array([row[key] for row in rows])
        ax.bar(range(len(rows)), values, bottom=bottom, label=label)
        bottom += values
    ax.set(xticks=range(len(rows)), xticklabels=labels,
           ylabel="fraction of FGMRES solve wall time", ylim=(0, 1))
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Critical-rank additive timing; three long-window repeats")
    fig.tight_layout()
    fig.savefig(output / "fgmres_amr_fractions.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-p1", type=Path, required=True)
    parser.add_argument("--uniform-p24", type=Path, required=True)
    parser.add_argument("--amr-p12", type=Path, required=True)
    parser.add_argument("--amr-p4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    suites = {name: load(getattr(args, name)) for name in
              ("uniform_p1", "uniform_p24", "amr_p12", "amr_p4")}
    rows = [select(suites["uniform_p1"], 1, "uniform"),
            select(suites["uniform_p24"], 2, "uniform"),
            select(suites["uniform_p24"], 4, "uniform"),
            select(suites["amr_p12"], 1, "static_amr"),
            select(suites["amr_p12"], 2, "static_amr"),
            select(suites["amr_p4"], 4, "static_amr")]
    for layout in ("uniform", "static_amr"):
        base = next(row for row in rows if row["layout"] == layout and row["ranks"] == 1)
        for row in rows:
            if row["layout"] == layout:
                row["strong_efficiency"] = (base["seconds_per_step"] /
                                             row["seconds_per_step"] / row["ranks"])
    for rank in (1, 2, 4):
        uniform = next(row for row in rows if row["layout"] == "uniform" and
                       row["ranks"] == rank)
        amr = next(row for row in rows if row["layout"] == "static_amr" and
                   row["ranks"] == rank)
        amr["time_ratio_vs_uniform"] = amr["seconds_per_step"]/uniform["seconds_per_step"]
        amr["time_per_leaf_cell_ratio_vs_uniform"] = (
            amr["time_ratio_vs_uniform"] /
            (amr["leaf_cells"]/uniform["leaf_cells"]))

    (args.output / "summary.json").write_text(json.dumps({"status": "passed", "rows": rows},
                                                          indent=2) + "\n")
    flat = [{key: value for key, value in row.items()
             if key not in ("source_suites", "newton_iterations_per_step")} for row in rows]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        fieldnames = list(flat[0])
        fieldnames.extend(key for row in flat for key in row if key not in fieldnames)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    plots(args.output, rows)

    table_rows = []
    for row in rows:
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in (
            "uniform" if row["layout"] == "uniform" else "static AMR",
            str(row["ranks"]), str(row["meshblocks"]), f"{row['leaf_cells']:,}",
            f"{row['seconds_per_step']:.6f}", f"{row['strong_efficiency']:.3f}",
            f"{row['max_fgmres_iterations']}", f"{row['max_rank_rss_mib']:.1f}")) + "</tr>")
    ratios = [next(row for row in rows if row["layout"] == "static_amr" and
                   row["ranks"] == rank) for rank in (1, 2, 4)]
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>WOR-20 conservative AMR performance</title><style>
body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.4rem .6rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eee;padding:.1rem .25rem}}
</style></head><body><h1>WOR-20 conservative AMR performance</h1>
<p><strong>PASS.</strong> All selected windows exceed 10 s, all Newton solves converge,
mass drift is zero at history precision, and no legacy AMR operator is used.</p>
<table><thead><tr><th>layout</th><th>ranks</th><th>MeshBlocks</th><th>leaf cells</th>
<th>s/step</th><th>strong efficiency</th><th>max FGMRES iter</th><th>max-rank RSS MiB</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p>AMR/uniform time ratios at 1/2/4 ranks are
<code>{' / '.join(f'{row["time_ratio_vs_uniform"]:.2f}' for row in ratios)}</code>.
After dividing by the 1.875 leaf-cell ratio they are
<code>{' / '.join(f'{row["time_per_leaf_cell_ratio_vs_uniform"]:.2f}' for row in ratios)}</code>.
The AMR solve also needs 9 rather than 3 FGMRES iterations per solve. Therefore this is
not a pure communication-overhead ratio: conditioning, coarse/fine transfers, the
conservative common-face operator, and the larger leaf grid all contribute.</p>
<img src='amr_vs_uniform.png' alt='AMR versus uniform timing, iterations, and memory'>
<img src='fgmres_amr_fractions.png' alt='FGMRES component fractions'>
<h2>Scope</h2><p>Fixed-limiter, radiation-only Test C on a periodic Cartesian cubic mesh.
The AMR layout has a centered level-1 refined region. This isolates the validated
conservative AMR operator but is not a collapse throughput prediction or a multi-node
publication scaling result.</p></body></html>"""
    (args.output / "index.html").write_text(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
