"""Independent Tomida-table roots for the mixed-frame AddDifference probe."""

import csv
import hashlib
import importlib.util
import json
import math
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIM = Path("/home/kosuke/simulation")
HERE = Path(__file__).resolve().parent
SRC = SIM / "test/rad_nr/wor92_local_general_cell_20261107"
OUT = HERE / "wor92_local_general_cell_results_20261107"
EOS = SIM / "test/rad_nr/wor92_20261107/bundle/data/eos/protostellar_tomida2015_material.tab"
MODULE = HERE.parents[1] / "tools/diagnose_rsg_lte_exchange.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_root(func, old):
    lower, upper = 0.5*old, 1.5*old
    assert func(lower) < 0 < func(upper)
    for _ in range(200):
        middle = 0.5*(lower+upper)
        if middle == lower or middle == upper:
            break
        if func(middle) < 0:
            lower = middle
        else:
            upper = middle
    return 0.5*(lower+upper)


def main():
    OUT.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("eos_reference", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.TableTemperature(EOS, 1e-18, 1e-9, 1)
    raw = (SRC / "mixed_valid.log").read_text(errors="replace")
    measured = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            index, gas, rad, temperature, residual = map(float, parts)
            measured[int(index)] = (gas, rad, temperature, residual)
    assert "mixed_active=1" in raw
    assert len(measured) == 11
    rows = []
    for line in (SRC / "cases_mixed_valid.txt").read_text().splitlines():
        index, fixed, old, old_rad, new_rad, coupling = map(float, line.split())
        index = int(index)
        assert fixed == 0
        def residual(gas):
            temperature = table.temperature_and_slope(5, gas)[0]
            return (gas-old + coupling*(7.5657e-20*temperature**4-new_rad)
                    + 0.1*(new_rad-old_rad))
        reference = reference_root(residual, old)
        gas, rad, temperature, cpp_residual = measured[index]
        expected_temperature = table.temperature_and_slope(5, gas)[0]
        assert math.isclose(rad, new_rad, rel_tol=1e-15)
        assert math.isclose(temperature, expected_temperature, rel_tol=1e-12)
        rows.append({
            "case": index, "rho": 5, "old_gas": old,
            "old_temperature": table.temperature_and_slope(5, old)[0],
            "old_rad": old_rad, "new_rad": new_rad,
            "coupling": coupling, "mixed_drift_delta_factor": -0.1,
            "reference_gas": reference, "production_gas": gas,
            "relative_root_error": abs(reference-gas)/old,
            "production_raw_residual": cpp_residual,
            "independent_raw_residual": residual(gas),
        })
    assert max(r["relative_root_error"] for r in rows) < 1e-14
    columns = list(rows[0])
    with (OUT / "cell_probe.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "cases": len(rows),
        "max_relative_root_error": max(r["relative_root_error"] for r in rows),
        "max_normalized_residual": max(
            abs(r["independent_raw_residual"])/r["old_gas"] for r in rows),
        "mixed_active": True,
        "eos_sha256": sha(EOS),
        "harness_sha256": sha(SRC / "local_probe.cpp"),
        "probe_binary_sha256": sha(SRC / "local_probe_mixed"),
        "cases_sha256": sha(SRC / "cases_mixed_valid.txt"),
        "log_sha256": sha(SRC / "mixed_valid.log"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    for temperature in sorted(set(round(r["old_temperature"]) for r in rows)):
        group = [r for r in rows if round(r["old_temperature"]) == temperature]
        ax.semilogx([r["coupling"] for r in group],
                    [r["relative_root_error"] for r in group],
                    "o", label=f"{temperature} K")
    ax.set_yscale("log")
    ax.set_xlabel("dt c sigma_P")
    ax.set_ylabel("gas root error / old gas")
    ax.set_title("Tomida table, mixed-frame local root")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cell_probe.png", dpi=170)
    plt.close(fig)
    with tarfile.open(OUT / "evidence.tar.gz", "w:gz") as archive:
        for name in ("local_probe.cpp", "cases_mixed_valid.txt", "mixed_valid.log"):
            archive.add(SRC / name, arcname=name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
