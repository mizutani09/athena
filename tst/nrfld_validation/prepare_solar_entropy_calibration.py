#!/usr/bin/env python3
"""Prepare matched HD2 inflow-entropy restarts from one solar-box checkpoint.

This writes a reviewable plan and an executable campaign script.  It does not
start the long calculations.  Each case uses the same restart and numerical
settings; only the inflow entropy differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(delta: float) -> str:
    return ("plus" if delta > 0 else "minus") + (
        f"{abs(delta) / 1e6:g}".replace(".", "p") + "e6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--header-from", type=Path, required=True)
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-s-in", type=float, required=True,
                        help="Effective HD2 entropy in erg g^-1 K^-1 from the baseline log")
    parser.add_argument("--delta-s-in", type=float, action="append",
                        help="Offset in erg g^-1 K^-1; repeat for each variant")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--tlim", type=float, required=True,
                        help="Common absolute end time in code units")
    parser.add_argument("--cpu-mask", help="Optional taskset CPU list")
    parser.add_argument("--checkpoint-dt", type=float, default=10.0)
    args = parser.parse_args()

    repo = args.repo.resolve()
    restart = args.restart.resolve()
    header = args.header_from.resolve()
    base_input = args.base_input.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    binary = repo / "bin/athena"
    source_manifest = restart.parent / "manifest.json"
    required = (restart, header, base_input, binary, source_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(missing))
    if args.ranks < 1 or args.ranks > 16:
        parser.error("Set a rank count between 1 and 16 for the 64x64x32 local box")
    if args.tlim <= 0 or args.checkpoint_dt <= 0:
        parser.error("End time and checkpoint cadence must be positive")
    if args.baseline_s_in <= 0:
        parser.error("Baseline entropy must be positive")
    offsets = args.delta_s_in if args.delta_s_in is not None else [-1e6, 1e6, 2e6]
    if any(delta == 0 or args.baseline_s_in + delta <= 0 for delta in offsets):
        parser.error("Offsets must be nonzero and leave positive entropy")
    if len(set(offsets)) != len(offsets):
        parser.error("Duplicate entropy offsets")

    source = json.loads(source_manifest.read_text())
    if source.get("binary_sha256") != sha256(binary):
        parser.error("Current binary differs from the restart campaign binary")
    if source.get("base_input_sha256") != sha256(base_input):
        parser.error("Base input differs from the restart campaign input")
    data_hashes = {}
    for name, expected in source.get("data_sha256", {}).items():
        path = data_dir / name
        if not path.is_file():
            parser.error(f"Missing input data: {path}")
        actual = sha256(path)
        if actual != expected:
            parser.error(f"Input data changed: {path}")
        data_hashes[name] = actual
    if len(data_hashes) != 3:
        parser.error("Expected EOS, opacity, and initial-profile hashes")
    if not re.search(r"(?m)^# \[1\]=time", header.read_text(errors="replace")):
        parser.error("History header file lacks the named-column header")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory must be empty: {output_dir}")

    bind = "core" if args.ranks <= 8 else "hwthread"
    map_by = bind
    common = ["python3", str(repo / "tst/nrfld_validation/run_solar_convection.py"),
              "--repo", str(repo), "--binary", str(binary),
              "--base-input", str(base_input), "--data-dir", str(data_dir),
              "--output-root", str(output_dir), "--solver", "fgmres",
              "--cfl", "0.3", "--nx1", "64", "--nx2", "64", "--nx3", "32",
              "--ranks", str(args.ranks), "--mpi-bind-to", bind,
              "--mpi-map-by", map_by, "--tlim", str(args.tlim),
              "--checkpoint-dt", str(args.checkpoint_dt),
              "--history-dt", "0.1", "--history-precision", "16",
              "--restart", str(restart), "--history-header-from", str(header),
              "--xy-slice-z", "2.0",
              "--problem-set", "mass_balance_relax_time=2.0",
              "--problem-set", "mass_balance_rate_gain=40"]
    prefix = ["taskset", "-c", args.cpu_mask] if args.cpu_mask else []
    cases = [("baseline", None)] + [(label(delta), delta) for delta in offsets]
    commands = []
    for name, delta in cases:
        command = prefix + common + ["--case", name]
        if delta is not None:
            command += ["--problem-set", "s_in_from_profile=false",
                        "--problem-set", f"s_in={args.baseline_s_in + delta:.15e}"]
        commands.append(command)
    comparison = ["python3", str(repo / "tst/nrfld_validation/solar_convection_entropy_sweep.py"),
                  "--baseline", str(output_dir / "baseline"),
                  "--baseline-s-in", f"{args.baseline_s_in:.15e}",
                  "--header-from", str(header),
                  "--output-dir", str(output_dir / "comparison")]
    for name, delta in cases[1:]:
        comparison += ["--variant", name, str(output_dir / name)]
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "purpose": "Matched transient sensitivity, followed by long-window solar-flux calibration",
        "source_restart": str(restart), "source_restart_sha256": sha256(restart),
        "source_manifest": str(source_manifest),
        "binary_sha256": source["binary_sha256"],
        "base_input_sha256": source["base_input_sha256"],
        "data_sha256": data_hashes,
        "baseline_s_in_cgs": args.baseline_s_in,
        "cases": [{"name": name, "delta_s_in_cgs": delta,
                   "effective_s_in_cgs": args.baseline_s_in + (delta or 0)}
                  for name, delta in cases],
        "ranks": args.ranks, "mpi_bind_to": bind, "mpi_map_by": map_by,
        "cpu_mask": args.cpu_mask, "tlim_code": args.tlim,
        "checkpoint_dt_code": args.checkpoint_dt,
        "commands": commands, "comparison_command": comparison,
        "interpretation": "Do not calibrate from a transient top-flux sample. Compare >=1 crossing-time late windows and stored-energy trends.",
    }
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    script = "#!/usr/bin/env bash\nset -euo pipefail\n"
    script += "\n".join(shlex.join(command) for command in [*commands, comparison]) + "\n"
    command_file = output_dir / "run_all.sh"
    command_file.write_text(script)
    command_file.chmod(0o755)
    print(f"Prepared {command_file} with {len(cases)} matched runs and comparison")


if __name__ == "__main__":
    main()
