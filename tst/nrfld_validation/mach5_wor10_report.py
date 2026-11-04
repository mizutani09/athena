#!/usr/bin/env python3
"""Collect reproducible WOR-10 Mach-5 audit reports into one HTML index."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import shutil


def copy_report(source: Path, destination: Path) -> dict:
    if not (source / "index.html").is_file() or not (source / "summary.json").is_file():
        raise ValueError(f"missing report files in {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return json.loads((destination / "summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="append", required=True,
                        help="SLUG=REPORT_DIR (mach5_trad_peak_audit output)")
    parser.add_argument("--stage", action="append", required=True,
                        help="SLUG=REPORT_DIR (mach5_split_stage_audit output)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audits, stages = {}, {}
    for kind, specs, target in (("audits", args.audit, audits),
                                ("stages", args.stage, stages)):
        for spec in specs:
            slug, sep, path = spec.partition("=")
            if not sep or not slug.replace("_", "").isalnum() or slug in target:
                parser.error(f"bad or duplicate {kind} entry: {spec}")
            target[slug] = copy_report(Path(path), args.output/kind/slug)

    css = ("body{font:20px/1.5 system-ui;max-width:1250px;margin:32px auto;"
           "padding:0 20px;color:#1d2530}h1,h2{line-height:1.2}"
           "table{border-collapse:collapse;width:100%;margin:18px 0}"
           "th,td{padding:9px 12px;border:1px solid #b5bdc8;text-align:right}"
           "th:first-child,td:first-child{text-align:left}"
           "th{background:#eef3fa}strong{color:#154a70}"
           "img{max-width:100%;height:auto}a{color:#155a91}"
           ".scroll{overflow-x:auto}.note{background:#f1f6fc;padding:12px 18px}")
    html = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<title>WOR-10 Mach 5 shock audit</title>",
            f"<style>{css}</style></head><body>",
            "<h1>WOR-10: Mach 5 shock temperature audit</h1>",
            "<p class='note'><strong>Output timing:</strong> Athena++ writes the normal HDF5 "
            "profile after the implicit FLD solve. Stage reports below measure both the "
            "preceding explicit RK2 state and the post-implicit state.</p>",
            "<p><strong>Interpretation:</strong> The small spatial T_rad wiggle is "
            "sensitive to mesh, boundary, and reconstruction. It is distinct from "
            "the positive residual against the semi-analytic reference. The narrow "
            "gas-temperature spike remains under-resolved; WOR-67 tracks its "
            "high-resolution verification.</p>"]
    for slug, report in audits.items():
        html.extend([f"<h2>{escape(slug.replace('_', ' ').title())}</h2>",
                     f"<p><a href='audits/{escape(slug)}/index.html'>Detailed plots, "
                     "profiles, and metrics</a></p>",
                     "<div class='scroll'><table><tr><th>Case</th><th>dx [cm]</th>"
                     "<th>T_rad wiggle [K]</th><th>Reference residual [K]</th>"
                     "<th>T_gas peak [K]</th><th>Raw shock x [cm]</th></tr>"])
        for row in report["cases"]:
            html.append("<tr><td>{}</td><td>{:.3f}</td><td>{:.6g}</td>"
                        "<td>{:.6g}</td><td>{:.3f}</td><td>{:.3f}</td></tr>".format(
                            escape(row["label"]), row["shock_cell_width_cm"],
                            row["spatial_wiggle"]["height_K"],
                            row["residual_bump_near_shock_K"],
                            row["gas_temperature_peak_near_shock_K"],
                            row["shock_raw_x_cm"]))
        html.append("</table></div>")
    for slug, report in stages.items():
        row = report["final"]
        html.extend([f"<h2>Stage audit: {escape(slug)}</h2>",
                     f"<p><a href='stages/{escape(slug)}/index.html'>Stage plots "
                     "and conservation checks</a></p>",
                     "<div class='scroll'><table><tr><th>Last step</th>"
                     "<th>After explicit RK2</th><th>After implicit FLD</th></tr>",
                     "<tr><th>T_gas peak [K]</th><td>{:.3f}</td><td>{:.3f}</td></tr>".format(
                         row["Tgas_peak_explicit_K"], row["Tgas_peak_implicit_K"]),
                     "<tr><th>T_rad peak [K]</th><td>{:.3f}</td><td>{:.3f}</td></tr>".format(
                         row["Trad_peak_explicit_K"], row["Trad_peak_implicit_K"]),
                     "<tr><th>T_rad spatial wiggle [K]</th><td>{:.4f}</td>"
                     "<td>{:.4f}</td></tr>".format(row["Trad_wiggle_explicit_K"],
                                                   row["Trad_wiggle_implicit_K"]),
                     "</table></div>"])
    manifest = {"audits": {k: v for k, v in audits.items()},
                "stages": {k: v for k, v in stages.items()}}
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    html.append("<p><a href='summary.json'>Full machine-readable audit</a></p></body></html>")
    (args.output / "index.html").write_text("\n".join(html) + "\n")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
