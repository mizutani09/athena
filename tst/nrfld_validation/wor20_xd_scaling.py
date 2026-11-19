#!/usr/bin/env python3
"""Validate and summarize XD conservative-AMR strong-scaling runs for WOR-20."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def load_run(path):
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text())
    required = {
        "status", "nodes", "ranks", "repeat", "meshblocks", "leaf_cells",
        "measured_steps", "seconds_per_step", "max_rank_rss_kib",
        "sum_rank_peak_rss_kib", "newton_iterations", "fgmres_iterations_max",
        "fgmres_iterations_total", "mass_relative_drift", "linear_solves", "fractions",
    }
    missing = required - summary.keys()
    if missing:
        raise ValueError(f"{path}: missing summary keys {sorted(missing)}")
    if summary["status"] != "passed" or (path / "result.txt").read_text().strip() != "PASS":
        raise ValueError(f"{path}: run did not pass")
    if summary["ranks"] != 32 * summary["nodes"]:
        raise ValueError(f"{path}: expected 32 ranks per node")
    if summary["seconds_per_step"] <= 0 or summary["fgmres_iterations_max"] <= 0:
        raise ValueError(f"{path}: invalid timing or iteration count")
    if abs(summary["mass_relative_drift"]) > 1.0e-12:
        raise ValueError(f"{path}: mass drift exceeds tolerance")
    fraction_sum = sum(summary["fractions"].values())
    if abs(fraction_sum - 1.0) > 5.0e-6:
        raise ValueError(f"{path}: FGMRES fractions sum to {fraction_sum}")
    summary["path"] = str(path.resolve())
    summary["zone_steps_per_second"] = summary["leaf_cells"] / summary["seconds_per_step"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = sorted((load_run(path) for path in args.runs),
                  key=lambda row: (row["nodes"], row["repeat"]))
    pairs = [(row["nodes"], row["repeat"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate (nodes, repeat) pair")
    invariant = ("meshblocks", "leaf_cells", "measured_steps", "newton_iterations")
    for key in invariant:
        if len({json.dumps(row[key], sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"runs disagree on {key}")

    groups = {}
    for row in rows:
        groups.setdefault(row["nodes"], []).append(row)
    if 1 not in groups:
        raise ValueError("a one-node baseline is required")

    aggregate = []
    baseline = statistics.median(row["seconds_per_step"] for row in groups[1])
    for nodes, selected in sorted(groups.items()):
        seconds = [row["seconds_per_step"] for row in selected]
        median = statistics.median(seconds)
        fgmres_max = {row["fgmres_iterations_max"] for row in selected}
        fgmres_total = {row["fgmres_iterations_total"] for row in selected}
        linear_solves = {row["linear_solves"] for row in selected}
        aggregate.append({
            "nodes": nodes,
            "ranks": 32 * nodes,
            "repeats": len(selected),
            "seconds_per_step_median": median,
            "seconds_per_step_min": min(seconds),
            "seconds_per_step_max": max(seconds),
            "strong_efficiency": baseline / (nodes * median),
            "speedup": baseline / median,
            "zone_steps_per_second_median": statistics.median(
                row["zone_steps_per_second"] for row in selected),
            "max_rank_rss_mib_median": statistics.median(
                row["max_rank_rss_kib"] / 1024.0 for row in selected),
            "fgmres_iterations_max": sorted(fgmres_max),
            "fgmres_iterations_total": sorted(fgmres_total),
            "linear_solves": sorted(linear_solves),
            "fractions": {name: statistics.median(row["fractions"][name] for row in selected)
                          for name in selected[0]["fractions"]},
        })

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"status": "passed", "runs": rows, "aggregate": aggregate}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "nodes", "ranks", "repeats", "seconds_per_step_median",
            "seconds_per_step_min", "seconds_per_step_max", "speedup",
            "strong_efficiency", "zone_steps_per_second_median",
            "max_rank_rss_mib_median", "fgmres_iterations_max",
            "fgmres_iterations_total", "linear_solves",
        ])
        writer.writeheader()
        for row in aggregate:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    nodes = [row["nodes"] for row in aggregate]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(nodes, [row["speedup"] for row in aggregate], "o-", label="measured")
    axes[0].plot(nodes, nodes, "--", color="0.5", label="ideal")
    axes[0].set(xlabel="nodes (32 MPI ranks/node)", ylabel="strong-scaling speedup",
                xticks=nodes)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    bottom = [0.0] * len(nodes)
    labels = [("preconditioner_seconds", "MG preconditioner"),
              ("matvec_seconds", "operator matvec"),
              ("reduction_seconds", "FGMRES reductions"),
              ("vector_and_other_seconds", "vector / other")]
    for key, label in labels:
        values = [row["fractions"][key] for row in aggregate]
        axes[1].bar(nodes, values, bottom=bottom, label=label)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[1].set(xlabel="nodes", ylabel="fraction of FGMRES wall time", xticks=nodes, ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "xd_amr_scaling.png", dpi=170)
    plt.close(fig)

    table_rows = "\n".join(
        "<tr>" + "".join((
            f"<td>{row['nodes']}</td>", f"<td>{row['ranks']}</td>",
            f"<td>{row['repeats']}</td>", f"<td>{row['seconds_per_step_median']:.6f}</td>",
            f"<td>{row['speedup']:.3f}</td>", f"<td>{row['strong_efficiency']:.3f}</td>",
            f"<td>{row['zone_steps_per_second_median'] / 1e6:.3f}</td>",
            f"<td>{row['max_rank_rss_mib_median']:.1f}</td>",
            f"<td>{row['fgmres_iterations_max']}</td>",
        )) + "</tr>" for row in aggregate)
    html = f"""<!doctype html><meta charset=\"utf-8\"><title>WOR-20 XD AMR scaling</title>
<style>body{{font-family:sans-serif;max-width:1050px;margin:2em auto;line-height:1.45}}
table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35em .55em;text-align:right}}
th:first-child,td:first-child{{text-align:center}}img{{max-width:100%}}code{{font-size:.9em}}</style>
<h1>WOR-20: XD conservative-AMR strong scaling</h1>
<p>Status: <strong>PASS</strong>. Fixed 128³ root grid with centered level-1 static AMR,
{rows[0]['meshblocks']:,} MeshBlocks and {rows[0]['leaf_cells']:,} active leaf cells.
All runs used 32 MPI ranks/node and {rows[0]['measured_steps']} measured steps.</p>
<table><thead><tr><th>nodes</th><th>ranks</th><th>repeats</th><th>s/step</th>
<th>speedup</th><th>efficiency</th><th>M leaf-cell steps/s</th><th>max-rank RSS MiB</th>
<th>max FGMRES iter</th></tr></thead><tbody>{table_rows}</tbody></table>
<p><img src=\"xd_amr_scaling.png\" alt=\"XD scaling and FGMRES timing fractions\"></p>
<p>Efficiency is relative to the one-node median. Rank peak RSS is a per-process high-water mark;
the sum of rank peaks is not simultaneous node memory. Raw paths and all numeric fields are in
<code>summary.json</code>.</p>"""
    (args.output / "index.html").write_text(html + "\n")


if __name__ == "__main__":
    main()
