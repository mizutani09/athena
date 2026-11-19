#!/usr/bin/env python3
"""Build the WOR-116 solar-application evidence table and standalone HTML."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def closed_row(label: str, resolution_km: int, path: Path, window: str) -> dict:
    data = load(path)
    values = data["windows"][window]
    return {
        "label": label,
        "resolution_km": resolution_km,
        "lower_boundary": "closed + volumetric heating",
        "window_code_time": int(window),
        "top_radiation_over_fsun": values["radiative_top_over_solar"],
        "top_material_gravity_over_fsun": (
            values["hydro_top_over_solar"]
            + values["advective_radiation_top_over_solar"]
            + values["gravitational_top_over_solar"]),
        "bottom_total_over_fsun": values["bottom_over_solar"],
        "storage_over_fsun": values["stored_over_solar"],
        "closure_or_input_error_over_fsun": (
            values["inferred_heating_over_solar"] - 1.0),
        "source": str(path),
        "source_sha256": sha256(path),
    }


def hd2_row(label: str, resolution_km: int, path: Path, window: str) -> dict:
    data = load(path)
    values = data["windows"][window]
    if values is None:
        raise ValueError(f"Missing {window}-code-time window in {path}")
    return {
        "label": label,
        "resolution_km": resolution_km,
        "lower_boundary": "open HD2 inflow entropy",
        "window_code_time": int(window),
        "top_radiation_over_fsun": values["top_radiation"],
        "top_material_gravity_over_fsun": values["top_gas"] + values["top_gravity"],
        "bottom_total_over_fsun": values["bottom_total"],
        "storage_over_fsun": values["storage"],
        "closure_or_input_error_over_fsun": values["closure_error"],
        "mass_drift_fraction": values["mass_drift"],
        "source": str(path),
        "source_sha256": sha256(path),
    }


def validate(rows: list[dict]) -> None:
    required = ("top_radiation_over_fsun", "top_material_gravity_over_fsun",
                "bottom_total_over_fsun", "storage_over_fsun",
                "closure_or_input_error_over_fsun")
    for row in rows:
        if not all(math.isfinite(row[key]) for key in required):
            raise ValueError(f"Nonfinite evidence row: {row['label']}")
        if row["window_code_time"] < 500:
            raise ValueError(f"Window too short for gate table: {row['label']}")
    solar_96 = [row for row in rows if row["resolution_km"] == 96]
    if not solar_96 or not all(0.95 <= row["top_radiation_over_fsun"] <= 1.05
                               for row in solar_96):
        raise ValueError("The selected 96-km evidence does not bracket/meet the limited gate")


def make_plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(rows))
    top = np.array([row["top_radiation_over_fsun"] for row in rows])
    storage = np.array([row["storage_over_fsun"] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, top, width=0.36, label="top radiation / Fsun")
    ax.bar(x + 0.18, storage, width=0.36, label="storage rate / Fsun")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="solar flux")
    ax.axhspan(0.95, 1.05, color="tab:green", alpha=0.09)
    ax.set_xticks(x, [row["label"] for row in rows], rotation=25, ha="right")
    ax.set_ylabel("Normalized energy rate")
    ax.set_title("WOR-116 evidence: 96-km limited gate and unresolved grid dependence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_html(summary: dict, output: Path) -> None:
    rows = summary["rows"]
    table = []
    for row in rows:
        table.append("<tr>" + "".join((
            f"<td>{html.escape(row['label'])}</td>",
            f"<td>{row['resolution_km']}</td>",
            f"<td>{html.escape(row['lower_boundary'])}</td>",
            f"<td>{row['window_code_time']}</td>",
            f"<td>{row['top_radiation_over_fsun']:.4f}</td>",
            f"<td>{row['top_material_gravity_over_fsun']:.4f}</td>",
            f"<td>{row['bottom_total_over_fsun']:.4f}</td>",
            f"<td>{row['storage_over_fsun']:+.4f}</td>",
            f"<td>{row['closure_or_input_error_over_fsun']:+.4f}</td>",
        )) + "</tr>")
    output.write_text(f"""<!doctype html><html lang='en'><meta charset='utf-8'>
<title>WOR-116 solar application gate</title><style>
body{{font:18px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172b3a}}
table{{border-collapse:collapse;font-size:15px}}th,td{{padding:.45rem .6rem;border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child,th:nth-child(3),td:nth-child(3){{text-align:left}}img{{max-width:100%}}
.pass{{color:#117347;font-weight:700}}.limit{{color:#9a5b00;font-weight:700}}code{{background:#f3f6f8;padding:.1rem .3rem}}
</style><h1>WOR-116: solar local-convection application gate</h1>
<p class='pass'>Decision: INCLUDE AS A LIMITED NUMERICAL APPLICATION.</p>
<p>The supported claim is stable 3D non-magnetic convection and radiative transport with grey FLD,
including a quasi-steady near-solar-flux state at 96 km. This is not validation of the observed solar
photosphere, a non-grey atmosphere, or grid-converged radiative flux.</p>
<p class='limit'>The 48 km HD2 continuation emits about 0.87 Fsun, so 96-to-48 km flux convergence is not established.
The currently running 24 km job is excluded from this decision.</p>
<p><a href='summary.json'>Machine-readable summary</a> · <a href='flux_gate.png'>Figure</a></p>
<img src='flux_gate.png' alt='Flux and storage comparison'>
<h2>Fixed evidence windows</h2><table><tr><th>case</th><th>km</th><th>lower boundary</th><th>window</th>
<th>top radiation</th><th>top material+gravity</th><th>bottom total</th><th>storage</th><th>closure/input error</th></tr>
{''.join(table)}</table>
<h2>Publication boundary</h2><ul>
<li>Allowed: solver stability, sustained 3D convection, radiative/convective energy transport, and a controlled 96-km near-solar-flux state.</li>
<li>Not allowed: observational solar agreement, spectral/limb-darkening validation, resolution convergence, or a unique asymptotic open-boundary equilibrium.</li>
<li>The Optab/LTE snapshot post-processing is independent evidence about the saved atmosphere; it is not validation of the grey-FLD evolution equation.</li>
<li>The top-face opacity reconstruction is application-boundary logic and must remain opt-in until its source integration and regression scope are reviewed.</li>
</ul><h2>Provenance</h2><pre>{html.escape(json.dumps(summary['provenance'], indent=2))}</pre></html>""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--hd2-48-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    rows = [
        closed_row("96 km / 2-cell", 96,
                   root / "heatdepth/final_summary_topface_alpha056_60000.json", "1000"),
        closed_row("96 km / 3-cell", 96,
                   root / "heatdepth/final_summary_topface_quadratic_55000.json", "1000"),
        hd2_row("96 km HD2 / low s", 96,
                root / "hd2_warm96_20261119/audit_342612.json", "2000"),
        hd2_row("96 km HD2 / high s", 96,
                root / "hd2_warm96_20261119/audit_342616.json", "2000"),
        closed_row("48 km refined", 48,
                   root / "heatdepth/final_summary_amr48_alpha056_57500.json", "1000"),
        hd2_row("48 km HD2", 48, args.hd2_48_audit.resolve(), "500"),
    ]
    validate(rows)
    summary = {
        "decision": "include_as_limited_numerical_application",
        "supports_observational_solar_validation": False,
        "supports_resolution_convergence": False,
        "supports_stable_3d_grey_fld_transport": True,
        "supports_96km_quasi_steady_near_solar_flux": True,
        "rows": rows,
        "provenance": {
            "results_root": str(root),
            "hd2_48_audit": str(args.hd2_48_audit.resolve()),
            "generator": str(Path(__file__).resolve()),
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plot(rows, output / "flux_gate.png")
    write_html(summary, output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
