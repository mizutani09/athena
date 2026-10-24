#!/usr/bin/env python3
"""WOR-23/24: bounded Test C operator audits, including instrumentation neutrality.

Expected NR failures are observations, not successes of the physical solver.
The exit code checks execution and diagnostic neutrality, not that all solves
converged. See audit.json and each collection's manifest.json for NR outcomes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from nrfld_validation.manifest import sha256_file, write_manifest


def compare(left, right):
    a = json.loads((left / "manifest.json").read_text())
    b = json.loads((right / "manifest.json").read_text())
    checks = []
    for x, y in zip(a["samples"], b["samples"]):
        assert x["tag"] == y["tag"]
        tag = x["tag"]
        ca, cb = left / tag, right / tag
        files_a = sorted(p.name for p in (ca / "output").glob("*.athdf"))
        files_b = sorted(p.name for p in (cb / "output").glob("*.athdf"))
        fields_equal = files_a == files_b and bool(files_a)
        for name in files_a:
            if name not in files_b:
                continue
            with h5py.File(ca / "output" / name) as f, h5py.File(cb / "output" / name) as g:
                fields_equal &= set(f.keys()) == set(g.keys())
                fields_equal &= all(k in g and np.array_equal(f[k][...], g[k][...]) for k in f)
        histories_a = sorted((ca / "output").glob("*.hst"))
        histories_b = sorted((cb / "output").glob("*.hst"))
        history_equal = len(histories_a) == len(histories_b) and all(
            p.read_bytes() == q.read_bytes() for p, q in zip(histories_a, histories_b))
        checks.append(dict(tag=tag, nr_equal=x["nr"] == y["nr"],
                           raw_status_equal=x["raw_status"] == y["raw_status"],
                           fields_equal=bool(fields_equal), history_equal=history_equal))
    assert len(a["samples"]) == len(b["samples"]) == len(checks)
    return checks


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("test_c_controlled.py").resolve()
    binary = args.binary.resolve() if args.binary else None
    common = ["--contrasts", "1", "--nr-threshold", "1e-7", "--base-nx1", "64",
              "--shifted-left", ".125", "--shifted-right", ".625",
              "--pulse-center", ".5", "--pulse-width", ".04",
              "--layouts", "uniform_fine", "amr_aligned", "amr_shifted"]
    collections = [
        ("fixed_off", ["--diffusion-numbers", "1e4", "1e7"]),
        ("fixed_on", ["--diffusion-numbers", "1e4", "1e7", "--global-jv", "--handoff"]),
        ("fixed_direction1", ["--diffusion-numbers", "1e7", "--global-jv",
                              "--handoff", "--jv-iteration", "1"]),
        ("variable_off", ["--diffusion-numbers", "5e5", "7e5", "--variable-limiter"]),
        ("variable_on", ["--diffusion-numbers", "5e5", "7e5", "--variable-limiter",
                          "--global-jv", "--handoff"]),
    ]
    report = {"commands": [], "neutrality": [], "binary": None,
              "scope": "single MPI rank, periodic, uncoupled radiation-only Test C"}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for name, extra in collections:
        out = root / name
        command = [sys.executable, str(script), "--output", str(out)] + common + extra
        if binary:
            command += ["--binary", str(binary)]
        # Each simulation has its own 30-s/4-GiB/32-MiB limits. Bound the outer
        # collection as well, allowing the first isolated build up to 600 s.
        result = subprocess.run(command, env=env, timeout=900, check=False)
        manifest = json.loads((out / "manifest.json").read_text())
        if result.returncode not in (0, 1) or manifest.get("status") != "finished":
            raise RuntimeError(f"Incomplete collection: {name}")
        for sample in manifest["samples"]:
            if sample["timeout"] or len(sample["nr"]) != 1:
                raise RuntimeError(f"Missing NR result or timeout: {name}/{sample['tag']}")
            if name.endswith("_on") or name == "fixed_direction1":
                stages = {r["stage"] for r in sample["handoff"]}
                if not sample["global_jv"] or stages != {
                        "mg_end", "retrieved", "boundary", "fine_sweeps"}:
                    raise RuntimeError(f"Missing diagnostic rows: {name}/{sample['tag']}")
        if binary is None:
            binary = out / "build_source/bin/athena"
        report["commands"].append({"name": name, "command": command,
                                    "status": result.returncode})
        report["binary"] = {"path": str(binary), "sha256": sha256_file(binary)}
        write_manifest(root / "audit.json", report)
    for family in ("fixed", "variable"):
        checks = compare(root / (family + "_off"), root / (family + "_on"))
        report["neutrality"].append({"family": family, "checks": checks})
    report["instrumentation_neutral"] = all(
        all(c[k] for k in ("nr_equal", "raw_status_equal", "fields_equal", "history_equal"))
        for group in report["neutrality"] for c in group["checks"])
    report["note"] = "This result does not assert physical solver convergence."
    write_manifest(root / "audit.json", report)
    print("INSTRUMENTATION_NEUTRAL=", report["instrumentation_neutral"], flush=True)
    return 0 if report["instrumentation_neutral"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    sys.exit(run(parser.parse_args()))
