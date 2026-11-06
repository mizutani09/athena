#!/usr/bin/env python3
"""Collect bounded Test C operator runs without reinterpreting failed solves as passes."""
import argparse
import json
import math
from pathlib import Path
import re


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def records(lines, prefix):
    return [fields(line) for line in lines if line.startswith(prefix)]


def numeric_fields(row, keys):
    try:
        return all(math.isfinite(float(row[key])) for key in keys)
    except (KeyError, ValueError):
        return False


def first_jv(rows, region):
    matches = [r for r in rows if r.get("direction") == "newton"
               and r.get("closure") == "frozen" and r.get("region") == region]
    return next((r for r in matches if math.isclose(float(r["epsilon"]), 1e-4,
                                                   rel_tol=1e-10)), None)


def first_mode_mismatch(rows):
    """Group the first V-cycle's actual MeshBlock corrections by local level."""
    groups = {}
    seen = set()
    for row in rows:
        if row.get("vcycle") != "0":
            continue
        key = (row["driver_coarse_level"], row["local_coarse_level"])
        gid = row["block_gid"]
        if (key, gid) in seen:
            continue
        seen.add((key, gid))
        groups.setdefault("/".join(key), []).append(row)
    result = {}
    for level, group in groups.items():
        mismatch2 = 0.0
        action2 = 0.0
        for row in group:
            a = float(row["rahp_l2"])
            b = float(row["scaled_ahe_l2"])
            action2 += max(a*a, b*b)
            mismatch2 += float(row["mismatch_relative_l2"])**2 * max(a*a, b*b)
        result[level] = {"rank0_blocks": len(group),
                         "relative_l2": math.sqrt(mismatch2/action2) if action2 else None}
    return result


def face_balance(rows, x):
    left = sum(float(r["xplus_delta"]) for r in rows
               if math.isclose(float(r["x1max"]), x, abs_tol=1e-12))
    right = sum(float(r["xminus_delta"]) for r in rows
                if math.isclose(float(r["x1min"]), x, abs_tol=1e-12))
    return {"left": left, "right": right, "sum": left + right,
            "relative": abs(left+right)/max(abs(left), abs(right), 1e-300)}


def summarize_collection(root):
    manifest = json.loads((root / "manifest.json").read_text())
    output = []
    for sample in manifest["samples"]:
        case = root / sample["tag"]
        lines = (case / "run.log").read_text(errors="replace").splitlines()
        raw_energy = records(lines, "[NR_ENERGY_LEDGER]")
        energy_keys = ("gas_delta", "radiation_delta", "diffusion_delta", "balance")
        energy = [r for r in raw_energy if numeric_fields(r, energy_keys)]
        energy_complete = len(energy) == manifest["arguments"]["mpi_ranks"]
        raw_faces = records(lines, "[NR_INTERFACE_LEDGER]")
        face_keys = ("gid", "x1min", "x1max", "xminus_delta", "xplus_delta")
        faces = [r for r in raw_faces if numeric_fields(r, face_keys)]
        faces_complete = len(faces) == sample.get("meshblocks", len(faces))
        root_modes = records(lines, "[NRFLD AMR Galerkin]")
        block_modes = records(lines, "[NRFLD Galerkin]")
        trials = records(lines, "[NR_TRIAL]")
        mode = records(lines, "[NRFLD_INTERFACE_MODE]")
        edge = [float(sample["settings"][f"refinement1/x1{side}"])
                for side in ("min", "max")] if sample["layout"].startswith("amr") else []
        face_results = ({str(x): face_balance(faces, x) for x in edge}
                        if faces and faces_complete else None)
        balance_sum = (sum(float(r["balance"]) for r in energy)
                       if energy and energy_complete else None)
        conservation_pass = (abs(balance_sum) <= 1e-10
                             and (not edge or all(v["relative"] <= 1e-8
                                                  for v in face_results.values()))) \
            if balance_sum is not None and (not edge or face_results is not None) else None
        output.append({
            "collection": str(root), "tag": sample["tag"],
            "git_head": manifest["provenance"]["analysis"]["git"]["commit"],
            "build_source_sha256": manifest["provenance"]["calculation"].get(
                "source_sha256"),
            "binary_sha256": manifest["provenance"]["calculation"]["binary_sha256"],
            "input_sha256": sample["input_sha256"], "input_path": str(case / "input.used"),
            "source_input_sha256": manifest["input"]["sha256"],
            "settings": sample["settings"], "command": sample["command"],
            "log_path": str(case / "run.log"), "mpi_ranks": manifest["arguments"]["mpi_ranks"],
            "layout": sample["layout"], "contrast": sample["contrast"],
            "q": sample["diffusion_number_fine_low"], "variable_limiter":
                sample["settings"]["fld/fixed_flux_limiter"] == "false",
            "operator_mode": mode[0] if mode else None,
            "pass": sample["pass"], "conservation_pass": conservation_pass,
            "raw_status": sample["raw_status"],
            "elapsed_seconds": sample["elapsed_seconds"],
            "max_rss_kib": sample.get("max_rss_kib"),
            "newton": sample["nr"], "newton_trials": trials,
            "fgmres": sample["fgmres"],
            "fgmres_iterations_total": sample["fgmres_iterations_total"],
            "fgmres_linearity": sample["fgmres_linearity"],
            "radiation_energy_relative_change":
                sample.get("radiation_energy_relative_change"),
            "jv_h1e-4": {region: first_jv(sample["global_jv"], region)
                         for region in ("all", "interior", "same_level_face",
                                        "coarse_fine_face")},
            "handoff_solve0": [r for r in sample["handoff"]
                               if r.get("solve") == "0" and r.get("region") in
                               ("all", "coarse_fine_face")],
            "mg_transfer_first": next((r for r in records(lines, "[NRFLD MG transfer]")
                                       if r.get("vcycle") == "0"), None),
            "root_octet_first": next((r for r in root_modes
                                      if r.get("vcycle") == "0"), None),
            "meshblock_mode_first": first_mode_mismatch(block_modes),
            "energy_ledger_complete": energy_complete if raw_energy else None,
            "energy_ledger_sum": {key: sum(float(r[key]) for r in energy)
                                  for key in energy_keys}
                                 if energy and energy_complete else None,
            "face_ledger_complete": faces_complete if raw_faces else None,
            "face_balance": face_results,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("collections", type=Path, nargs="+")
    args = parser.parse_args()
    result = [row for root in args.collections for row in summarize_collection(root)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result)} case records to {args.output}")


if __name__ == "__main__":
    main()
