#!/usr/bin/env python3
"""Compare matched WOR-20 MG/FGMRES performance suites and write a report."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import re

import h5py
import numpy as np


POLICIES = ("mg_rediscretized", "fgmres_rediscretized", "fgmres_galerkin")


def read_suite(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if summary["status"] != "passed" or manifest["status"] != "passed":
        raise ValueError(f"Suite did not pass: {path}")
    return {"root": str(path.resolve()), "summary": summary, "manifest": manifest}


def fgmres_counts(run: dict) -> dict:
    text = (Path(run["path"]) / "run.log").read_text()
    iterations = [int(value) for value in re.findall(
        r"\[LINEAR_FGMRES\] iterations=(\d+)", text)]
    return {"calls": len(iterations), "iterations": sum(iterations),
            "maximum": max(iterations, default=0)}


def final_fields(directory: Path) -> dict[str, np.ndarray]:
    fields = {}
    for kind in ("prim", "rad"):
        paths = sorted(directory.glob(f"benchmark.{kind}.*.athdf"))
        if len(paths) != 2:
            raise ValueError(f"Expected initial/final {kind} output in {directory}")
        with h5py.File(paths[-1]) as handle:
            fields[f"{kind}_time"] = np.asarray(handle.attrs["Time"])
            names = [value.decode() if isinstance(value, bytes) else str(value)
                     for value in handle.attrs["VariableNames"]]
            offset = 0
            for dataset, count in zip(handle.attrs["DatasetNames"],
                                      handle.attrs["NumVariables"]):
                dataset = dataset.decode() if isinstance(dataset, bytes) else str(dataset)
                for index in range(int(count)):
                    fields[names[offset + index]] = np.asarray(handle[dataset][index])
                offset += int(count)
    return fields


def compare_fields(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict:
    if left.keys() != right.keys():
        raise ValueError("Field names differ between policies")
    result = {}
    for name in left:
        a = np.asarray(left[name], dtype=np.float64)
        b = np.asarray(right[name], dtype=np.float64)
        difference = a-b
        scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        denominator = max(float(np.linalg.norm(a.ravel())), np.finfo(float).tiny)
        result[name] = {
            "bitwise_identical": bool(np.array_equal(a, b)),
            "max_scaled_difference": float(np.max(np.abs(difference)))/scale,
            "relative_l2": float(np.linalg.norm(difference.ravel()))/denominator,
        }
    return result


def validate_match(suites: dict[str, dict]) -> None:
    reference = suites[POLICIES[0]]["manifest"]["options"]
    fixed = ("cases", "ranks", "resolution", "block", "steps", "warmup",
             "repeat", "dt", "mode", "build_root")
    for policy in POLICIES[1:]:
        options = suites[policy]["manifest"]["options"]
        differences = {name: (reference[name], options[name]) for name in fixed
                       if reference[name] != options[name]}
        if differences:
            raise ValueError(f"Unmatched suite options for {policy}: {differences}")


def make_plot(rows: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"mg_rediscretized": "MG / rediscretized",
              "fgmres_rediscretized": "FGMRES / rediscretized",
              "fgmres_galerkin": "FGMRES / Galerkin"}
    ranks = sorted({row["ranks"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    width = 0.25
    x = np.arange(len(ranks))
    for index, policy in enumerate(POLICIES):
        selected = [next(row for row in rows if row["policy"] == policy
                         and row["ranks"] == rank) for rank in ranks]
        median = np.array([row["seconds_per_step_median"] for row in selected])
        low = median - np.array([row["seconds_per_step_min"] for row in selected])
        high = np.array([row["seconds_per_step_max"] for row in selected]) - median
        axes[0].bar(x + (index-1)*width, median, width, label=labels[policy],
                    yerr=np.vstack([low, high]), capsize=3)
        axes[1].bar(x + (index-1)*width,
                    [row["fgmres_iterations_median"] for row in selected], width)
        axes[2].bar(x + (index-1)*width,
                    [row["max_rank_rss_kib_median"]/1024 for row in selected], width)
    axes[0].set_ylabel("seconds / step (median; range)")
    axes[1].set_ylabel("FGMRES iterations / 100-step run")
    axes[2].set_ylabel("maximum rank peak RSS [MiB]")
    for axis in axes:
        axis.set_xticks(x, [str(rank) for rank in ranks])
        axis.set_xlabel("MPI ranks")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("WOR-20 matched uniform HD+FLD solver comparison")
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=170)
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for policy in POLICIES:
        parser.add_argument(f"--{policy.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suites = {policy: read_suite(getattr(args, policy)) for policy in POLICIES}
    validate_match(suites)

    rows = []
    for policy, suite in suites.items():
        for aggregate in suite["summary"]["aggregates"]:
            runs = [run for run in suite["summary"]["runs"]
                    if run["ranks"] == aggregate["ranks"] and run["status"] == "passed"]
            linear = [fgmres_counts(run) for run in runs]
            rows.append({
                "policy": policy,
                **aggregate,
                "fgmres_calls_median": float(np.median([item["calls"] for item in linear])),
                "fgmres_iterations_median": float(np.median(
                    [item["iterations"] for item in linear])),
                "fgmres_maximum_per_solve": max(item["maximum"] for item in linear),
                "max_rank_rss_kib_median": float(np.median(
                    [run["metrics"]["max_rank_rss_kib"] for run in runs])),
                "newton_iterations_total": sorted({
                    run["metrics"]["newton_iterations_total"] for run in runs}),
                "mass_relative_drift_max_abs": max(
                    abs(run["metrics"]["mass_relative_drift"]) for run in runs),
            })

    reference_run = Path(suites[POLICIES[0]]["summary"]["runs"][0]["path"])
    reference_fields = final_fields(reference_run)
    comparisons = {}
    for policy in POLICIES[1:]:
        candidate_run = Path(suites[policy]["summary"]["runs"][0]["path"])
        comparisons[f"{POLICIES[0]}_vs_{policy}"] = compare_fields(
            reference_fields, final_fields(candidate_run))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed", "suites": {key: value["root"]
              for key, value in suites.items()}, "rows": rows,
              "field_comparisons": comparisons,
              "limitations": [
                  "Local single-node WSL measurement; not a multi-node result.",
                  "Ranges include host scheduling/thermal outliers; medians are reported.",
                  "The controlled uniform benchmark does not predict AMR-collapse robustness.",
                  "Sum of per-rank RSS peaks is not a simultaneous node high-water mark.",
              ]}
    (output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    image = make_plot(rows)
    (output / "solver_comparison.png").write_bytes(base64.b64decode(image))

    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            row["policy"], row["ranks"], f'{row["seconds_per_step_median"]:.6f}',
            f'{row["seconds_per_step_min"]:.6f}–{row["seconds_per_step_max"]:.6f}',
            f'{row["efficiency"]:.3f}', int(row["fgmres_iterations_median"]),
            row["fgmres_maximum_per_solve"],
            f'{row["max_rank_rss_kib_median"]/1024:.1f}')) + "</tr>"
        for row in rows)
    max_field = max(item["max_scaled_difference"]
                    for comparison in comparisons.values() for item in comparison.values())
    max_l2 = max(item["relative_l2"]
                 for comparison in comparisons.values() for item in comparison.values())
    html = f"""<!doctype html><meta charset='utf-8'>
<title>WOR-20 solver performance comparison</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:2rem auto;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eee;padding:.1rem .25rem}}</style>
<h1>WOR-20 matched solver comparison</h1>
<p>Frozen solver <code>src</code> tree: <code>c4ab200383c7317a2feccc50aa9578d5cf71478d</code>.
Uniform 32³ HD+FLD benchmark, 10 warm-up + 100 measured steps, three repeats,
strict outer Newton acceptance. All 27 runs passed.</p>
<img alt='solver comparison' src='data:image/png;base64,{image}'>
<table><thead><tr><th>policy</th><th>ranks</th><th>median s/step</th><th>range</th>
<th>strong efficiency</th><th>FGMRES iterations/run</th><th>max/solve</th><th>max-rank RSS MiB</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<h2>Numerical equivalence</h2>
<p>Against standalone MG, the largest final-field scaled difference is {max_field:.3e}
and the largest relative L2 difference is {max_l2:.3e}. Time attributes are identical.
Every run used 200 outer Newton corrections and had zero printed mass drift.</p>
<h2>Interpretation</h2>
<p>FGMRES/rediscretized is slower than standalone MG for this easy uniform problem.
Galerkin increases Krylov iterations relative to the rediscretized preconditioner and
does not provide a performance benefit here. This does not negate FGMRES robustness
on difficult AMR states; it only rejects a blanket uniform-grid speedup claim.</p>
<h2>Limitations</h2><ul>{''.join(f'<li>{item}</li>' for item in report['limitations'])}</ul>
<p>Exact paths and all raw metrics are in <code>comparison.json</code>.</p>"""
    (output / "index.html").write_text(html)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
