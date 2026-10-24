#!/usr/bin/env python3
"""Compare summaries and final history rows from the collapse solver A/B/C job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # The summary and history comparison still work without HDF5.
    h5py = None


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def final_history_row(run_dir: Path) -> tuple[Path | None, np.ndarray | None]:
    files = sorted(run_dir.glob("*.hst"))
    if not files:
        return None, None
    rows = []
    for raw in files[-1].read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append([float(item) for item in line.split()])
        except ValueError:
            continue
    return files[-1], (np.asarray(rows[-1]) if rows else None)


def final_xline(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("*.xline.*.athdf"))
    return files[-1] if files else None


def compare_hdf5(reference: Path, candidate: Path) -> dict:
    result = {"reference": str(reference), "candidate": str(candidate),
              "datasets": {}, "all_bitwise_equal": True}
    with h5py.File(reference, "r") as ref, h5py.File(candidate, "r") as cand:
        for name in sorted(set(ref.keys()) | set(cand.keys())):
            if name not in ref or name not in cand:
                result["datasets"][name] = {"missing": True}
                result["all_bitwise_equal"] = False
                continue
            left, right = ref[name][...], cand[name][...]
            equal = (left.shape == right.shape and left.dtype == right.dtype
                     and np.array_equal(left, right))
            item = {"shape": list(left.shape), "bitwise_equal": bool(equal)}
            if left.shape == right.shape and np.issubdtype(left.dtype, np.number):
                if equal:
                    max_abs = max_relative = 0.0
                else:
                    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                        difference = np.where(left == right, 0.0, np.abs(right - left))
                        scale = np.maximum(
                            np.maximum(np.abs(left), np.abs(right)), 1e-300)
                        relative = difference / scale
                    max_abs = float(np.nanmax(difference))
                    max_relative = float(np.nanmax(relative))
                item["max_abs_difference"] = max_abs
                item["max_relative_difference"] = max_relative
            result["datasets"][name] = item
            result["all_bitwise_equal"] = result["all_bitwise_equal"] and equal
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("array_job_id", help="SLURM_ARRAY_JOB_ID from sbatch")
    parser.add_argument(
        "--root", type=Path,
        default=Path("/work/mizutaniks/output/fld/collapse/test11"))
    parser.add_argument("--json", type=Path, help="also write the comparison as JSON")
    args = parser.parse_args()

    records = {}
    for index, tag in enumerate(("rb", "patience2", "double")):
        run_dir = args.root / f"smoother_ab_{tag}_{args.array_job_id}_{index}"
        summary_path = run_dir / "numerical_summary.txt"
        if not summary_path.is_file():
            raise SystemExit(f"missing summary: {summary_path}")
        summary = read_key_values(summary_path)
        history_path, history = final_history_row(run_dir)
        xline_path = final_xline(run_dir)
        records[tag] = {
            "run_dir": str(run_dir),
            "summary": summary,
            "history_path": str(history_path) if history_path else None,
            "history_final": history.tolist() if history is not None else None,
            "xline_path": str(xline_path) if xline_path else None,
        }

    rb = records["rb"]["history_final"]
    comparisons = {}
    xline_comparisons = {}
    if rb is not None:
        rb_array = np.asarray(rb)
        for tag in ("patience2", "double"):
            candidate = records[tag]["history_final"]
            if candidate is None:
                continue
            candidate_array = np.asarray(candidate)
            if rb_array.shape != candidate_array.shape:
                continue
            difference = candidate_array - rb_array
            scale = np.maximum(
                np.maximum(np.abs(rb_array), np.abs(candidate_array)), 1e-300)
            comparisons[tag] = {
                "columns": int(rb_array.size),
                "max_abs_difference": float(np.max(np.abs(difference))),
                "max_relative_difference": float(np.max(np.abs(difference) / scale)),
                "l1_difference": float(np.mean(np.abs(difference))),
            }
    if h5py is not None:
        ref_xline = records["rb"]["xline_path"]
        for tag in ("patience2", "double"):
            candidate_xline = records[tag]["xline_path"]
            if ref_xline and candidate_xline:
                xline_comparisons[tag] = compare_hdf5(
                    Path(ref_xline), Path(candidate_xline))

    output = {"array_job_id": args.array_job_id, "runs": records,
              "history_comparison_vs_rb": comparisons,
              "xline_comparison_vs_rb": xline_comparisons}
    print(json.dumps(output, indent=2))
    if args.json:
        args.json.write_text(json.dumps(output, indent=2) + "\n")

    results = [records[tag]["summary"].get("numerical_result") for tag in records]
    return 0 if results == ["PASS", "PASS", "PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
