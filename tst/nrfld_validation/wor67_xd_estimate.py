#!/usr/bin/env python3
"""Project N4096 Mach-5 wall times from XD 100-cycle scaling measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


BASELINES = {
    "uniform": ("wor67_hpc_uniform_n2048", "wor67_hpc_uniform_n2048_pilot100b",
                "wor67_hpc_uniform_n4096_pilot100"),
    "amr": ("wor67_hpc_amr_base1024", "wor67_hpc_amr_base1024_pilot100_r32",
            "wor67_hpc_amr_base2048_pilot100"),
    "smr": ("wor67_hpc_smr_base512",
            "wor67_hpc_smr_base512_pilot100_outputcadence",
            "wor67_hpc_smr_base1024_pilot100"),
}


def tail_text(path: Path, size: int = 16384) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size-size))
        return handle.read().decode(errors="replace")


def cpu_seconds(run_dir: Path) -> float:
    case_dir = run_dir/"cases/mach5" if (run_dir/"cases/mach5").is_dir() else run_dir
    text = tail_text(case_dir/"run.log")
    match = re.search(r"cpu time used\s*=\s*([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"no completed Athena timing: {run_dir}")
    return float(match[1])


def cycle_time(run_dir: Path, cycle: int) -> float:
    case_dir = run_dir/"cases/mach5" if (run_dir/"cases/mach5").is_dir() else run_dir
    text = tail_text(case_dir/"run.log")
    match = re.search(r"time=([0-9.eE+-]+) cycle="+str(cycle)+r"\s+tlim=", text)
    if not match:
        raise ValueError(f"no cycle {cycle} termination: {run_dir}")
    return float(match[1])


def key_values(path: Path) -> dict[str, str]:
    return dict(line.strip().split("=", 1) for line in path.read_text().splitlines()
                if "=" in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/work/mizutaniks/wor67/output"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    rows = []
    for layout, (full_name, pilot_name, high_name) in BASELINES.items():
        full_dir = args.root/full_name
        pilot_dir = args.root/pilot_name
        high_dir = args.root/high_name
        full = json.loads((full_dir/"summary.json").read_text())
        pilot = json.loads((high_dir/"summary.json").read_text())
        full_cycles = full["cases"]["mach5"]["execution"]["cycles"]
        if full["status"] != "passed" or pilot["status"] != "exploratory_cycle_limit":
            raise ValueError(f"incomplete reference or N4096 pilot for {layout}")
        first_time = cycle_time(pilot_dir, 100)
        high_time = pilot["cases"]["mach5"]["execution"]["time"]
        cycle_ratio = first_time/high_time
        predicted_cycles = full_cycles*cycle_ratio
        full_per_cycle = cpu_seconds(full_dir)/full_cycles
        pilot_per_cycle = cpu_seconds(pilot_dir)/100
        historical_factor = full_per_cycle/pilot_per_cycle
        cases[layout] = {"N2048_full_cycles": full_cycles,
                         "N2048_pilot_time_100_s": first_time,
                         "N4096_pilot_time_100_s": high_time,
                         "predicted_N4096_cycles": predicted_cycles,
                         "N2048_early_to_full_factor": historical_factor,
                         "N2048_full_solver_seconds_per_cycle": full_per_cycle,
                         "N2048_pilot_solver_seconds_per_cycle": pilot_per_cycle}
        for run_dir in sorted(args.root.glob(f"wor67_hpc_{layout}_n4096_scale_n*r*")):
            summary_path = run_dir/"cases/mach5/summary.json"
            provenance_path = run_dir/"cases/mach5/provenance.txt"
            if not summary_path.is_file() or not provenance_path.is_file():
                continue
            summary = json.loads(summary_path.read_text())
            if summary["cycles"] != 100 or not summary["solver"]["good"] or summary[
                    "solver"]["relaxed_accepted"]:
                raise ValueError(f"invalid scale run: {run_dir}")
            provenance = key_values(provenance_path)
            nodes = int(provenance["nodes"])
            ranks = int(provenance["ranks"])
            solver_seconds = cpu_seconds(run_dir)
            raw_hours = solver_seconds/100*predicted_cycles/3600
            adjusted_hours = raw_hours*historical_factor
            rows.append({"layout": layout, "nodes": nodes, "ranks": ranks,
                         "job_id": provenance["job_id"],
                         "meshblocks": summary["meshblocks_final"],
                         "pilot_solver_seconds": solver_seconds,
                         "pilot_launcher_seconds": summary["wall_seconds"],
                         "launch_overhead_seconds": summary["wall_seconds"]-solver_seconds,
                         "raw_extrapolated_hours": raw_hours,
                         "historically_adjusted_hours": adjusted_hours,
                         "adjusted_node_hours": nodes*adjusted_hours,
                         "run_dir": str(run_dir.resolve())})
    rows.sort(key=lambda row: (row["layout"], row["nodes"], row["ranks"]))
    output = {"model": "N4096 100-cycle Athena CPU time × predicted cycles; historical correction from the N2048 full/early-cycle ratio at the baseline rank count. Startup and queue time excluded. Estimates are scenarios, not confidence intervals.",
              "cases": cases, "scale_runs": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"estimate.json").write_text(json.dumps(output, indent=2)+"\n")
    lines = ["# WOR-67 N4096 scaling estimate", "",
             "Athena 100-cycle time excludes MPI startup. The projected cycle count uses the ratio of N2048 and N4096 100-cycle end times. The adjusted scenario applies the observed N2048 full/early-cycle change for each layout; its transfer to N4096 or higher rank counts is uncertain.", "",
             "| Layout | Nodes | Ranks | Blocks | 100-cycle solver [s] | Raw full [h] | N2048-adjusted full [h] | Adjusted node-hours |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| {layout} | {nodes} | {ranks} | {meshblocks} | {pilot_solver_seconds:.1f} | {raw_extrapolated_hours:.1f} | {historically_adjusted_hours:.1f} | {adjusted_node_hours:.1f} |".format(**row))
    lines += ["", "N2048 early/full correction factors: " + ", ".join(
        f"{layout} {data['N2048_early_to_full_factor']:.3f}"
        for layout, data in cases.items()) + ".", "",
        "A full run must fit the selected partition's wall limit (M-large-a: 24 h) or use a validated restart plan. Performance beyond the measured configurations is not inferred from ideal linear scaling."]
    (args.output/"estimate.md").write_text("\n".join(lines)+"\n")
    print(args.output/"estimate.md", flush=True)


if __name__ == "__main__":
    main()
