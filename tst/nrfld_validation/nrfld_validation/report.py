"""Portable HTML summary for an NR-FLD validation run directory."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote
from typing import Any

from .report_style import REPORT_TYPOGRAPHY


def _relative_url(root: Path, raw_path: str | Path) -> str | None:
    path = Path(raw_path).resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError:
        return None
    return quote(relative.as_posix(), safe="/") if path.is_file() else None


def _link(url: str | None, label: str) -> str:
    if url is None:
        return ""
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'


def _metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_html_summary(root: Path, summary: dict[str, Any]) -> Path:
    """Link only files inside root, so copying the whole directory keeps images usable."""
    sections = []
    index_rows = []
    for case, entry in summary["cases"].items():
        case_dir = root / case
        manifest_path = case_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = entry.get("metrics", {})
        status = entry.get("status", "unknown")
        status_class = "pass" if status == "passed" else "fail"
        index_rows.append(
            f'<tr><td><a href="#{html.escape(case, quote=True)}">'
            f'{html.escape(case)}</a></td><td>'
            f'<span class="badge {status_class}">{html.escape(status.upper())}</span></td></tr>'
        )
        metric_rows = "".join(
            "<tr><th>" + html.escape(key) + "</th><td>"
            + html.escape(_metric(value)) + "</td></tr>"
            for key, value in metrics.items()
        )
        settings = manifest.get("input", {}).get("effective_settings", {})
        resources = manifest.get("resources", {})
        run_records = [record for record in manifest.get("commands", {}).get("records", [])
                       if record.get("phase") == "run"]
        run_seconds = (run_records[-1].get("elapsed_seconds") if run_records else None)
        conditions = {
            "solver": settings.get("nrfld/linear_solver", "input default"),
            "initial gas energy": settings.get("problem/eg_0", "n/a"),
            "initial radiation energy": settings.get("problem/Er_0", "n/a"),
            "initial timestep [s]": settings.get("problem/dt0", "n/a"),
            "timestep growth": settings.get("problem/dt_growth", "n/a"),
            "cycle limit": settings.get("time/nlim", "n/a"),
            "MPI ranks": resources.get("mpi_ranks", "n/a"),
            "OMP threads": resources.get("omp_threads", "n/a"),
            "peak RSS [KiB]": resources.get("max_rss_kib", "n/a"),
            "run time [s]": run_seconds if run_seconds is not None else "n/a",
        }
        condition_rows = "".join(
            "<tr><th>" + html.escape(key) + "</th><td>"
            + html.escape(_metric(value)) + "</td></tr>"
            for key, value in conditions.items()
        )
        artifact_links = []
        for name in ("manifest.json", "input.used", "run.log", "resources.txt",
                     "configure.log", "build.log"):
            url = _relative_url(root, case_dir / name)
            if url:
                artifact_links.append(_link(url, name))
        plot_url = _relative_url(root, manifest.get("artifacts", {}).get("plot", ""))
        figure = (
            f'<figure><img src="{html.escape(plot_url, quote=True)}" '
            f'alt="{html.escape(case, quote=True)} validation plot" loading="lazy">'
            f'<figcaption>{html.escape(case)}: numerical result and reference</figcaption>'
            "</figure>"
            if plot_url else "<p>No plot was produced for this case.</p>"
        )
        git = manifest.get("git", {})
        commit = html.escape(str(git.get("commit") or "unknown"))
        source = "dirty source snapshot" if git.get("dirty") else "clean source"
        sections.append(
            f'<section id="{html.escape(case, quote=True)}"><h2>{html.escape(case)} '
            f'<span class="badge {status_class}">{html.escape(status.upper())}</span></h2>'
            f'<p>Commit: <code>{commit}</code> · {source}</p>'
            f'<h3>Conditions and resources</h3><table class="metrics">{condition_rows}</table>'
            f'<h3>Validation metrics</h3><table class="metrics">{metric_rows}</table>{figure}'
            f'<p class="links">{" · ".join(artifact_links)}</p></section>'
        )
    title = "NR-FLD validation: " + str(summary.get("run", "run"))
    passed_count = sum(entry.get("status") == "passed"
                       for entry in summary["cases"].values())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{font: 18px/1.6 system-ui,sans-serif; color: #17212b; background: #f7f9fb}}
body {{max-width: 1200px; margin: 2rem auto; padding: 0 1rem}}
h1,h2 {{line-height: 1.25}} h1 {{font-size: 2rem}} h2 {{font-size: 1.5rem}}
section {{background: white; margin: 1.5rem 0; padding: 1.4rem; border: 1px solid #dce4eb; border-radius: 10px}}
table {{border-collapse: collapse; width: 100%}} th,td {{padding: .55rem .7rem; border-bottom: 1px solid #e4eaf0; text-align: left}}
.metrics th {{width: 48%; font-weight: 700}} .badge {{display: inline-block; border-radius: 5px; padding: .1rem .5rem; font-weight: 750}}
.pass {{color: #0b683e; background: #e4f5eb}} .fail {{color: #ad261f; background: #fdebea}}
.overview {{background: #eaf3fb; border-left: 5px solid #1769aa; padding: .8rem 1rem; font-size: 1.1rem}}
figure {{margin: 1.4rem 0}} img {{max-width: 100%; height: auto; border: 1px solid #dce4eb; background: white}}
figcaption {{font-size: .9rem; color: #536170}} .links {{overflow-wrap: anywhere}} a {{color: #075ca8}} code {{overflow-wrap: anywhere}}
{REPORT_TYPOGRAPHY}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="overview"><strong>{passed_count}/{len(summary['cases'])} cases passed</strong></p>
<p>Each result is linked to its plot and reproducibility files. Copy this directory as a unit to retain the links.</p>
<table><thead><tr><th>Case</th><th>Status</th></tr></thead><tbody>{''.join(index_rows)}</tbody></table>
{''.join(sections)}
<p>{_link('summary.json', 'Machine-readable summary')}</p>
</body></html>
"""
    destination = root / "index.html"
    destination.write_text(document, encoding="utf-8")
    return destination
