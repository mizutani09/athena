#!/usr/bin/env python3
"""Run NR-FLD validation sequentially with a concise index beside case folders."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nrfld_validation.io import find_repo_root  # noqa: E402
from nrfld_validation.manifest import (base_manifest, sha256_file,
                                       write_manifest)  # noqa: E402
from nrfld_validation.memory_monitor import run_monitored  # noqa: E402
ORDER = ("couple", "diffusion", "marshak", "streaming", "shock_tube",
         "radiative_shock", "advection_pulse", "static_equilibrium",
         "thermal_wave", "closed_box", "fgmres_amr_restart", "mixed_frame")
SCRIPTS = {"couple": "coupling_sweep.py", "diffusion": "diffusion_validation.py",
           "marshak": "marshak_validation.py", "streaming": "streaming_validation.py",
           "shock_tube": "shock_tube_validation.py",
           "radiative_shock": "radiative_shock_validation.py",
           "advection_pulse": "advection_pulse_validation.py",
           "static_equilibrium": "static_equilibrium_validation.py",
           "thermal_wave": "run_thermal_wave_matrix.py",
           "closed_box": "closed_box_validation.py",
           "fgmres_amr_restart": "fgmres_amr_restart_validation.py",
           "mixed_frame": "run_mixed_frame_matrix.py"}
SUCCESS = {"passed", "diagnostic_complete"}
CAVEAT = {"completed_with_solver_caveat"}
ACCURACY_GAP = {"accuracy_not_met"}
COMPLETED = SUCCESS | CAVEAT | ACCURACY_GAP
COMMENTS = {
    "couple": "小さい dt と大きい dt の両方を確認",
    "diffusion": "時間・空間の収束性を測定",
    "marshak": "解析解との比較",
    "streaming": "前線位置の遅れは WOR-59 で継続調査",
    "shock_tube": "Rad EoS の高解像度 hydro 解と比較",
    "radiative_shock": "Mach 5 の温度スパイクは解像度不足なら精度未達",
    "advection_pulse": "移動距離だけシフトして静止解と比較",
    "static_equilibrium": "初期平衡からのずれを測定",
    "thermal_wave": "3D 一段 AMR と解析解を比較",
    "closed_box": "質量・全エネルギーの保存性を測定",
    "fgmres_amr_restart": "再格子化・再起動・MPI rank 変更を比較",
    "mixed_frame": "陰的・陽的配置と Jacobian を比較",
}


class CaseExecutionError(RuntimeError):
    def __init__(self, message: str, commands: list[dict], kind: str):
        super().__init__(message)
        self.commands = commands
        self.kind = kind


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/output/rad"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path, help="Continue an incomplete suite directory.")
    parser.add_argument("--reassess-suite", type=Path,
                        help="Apply current physical accuracy criteria to three saved reports.")
    parser.add_argument("--only", action="append", choices=ORDER,
                        help="Run only these named problems in their normal order.")
    parser.add_argument("--import", dest="imports", action="append", default=[],
                        metavar="NAME=DIR", help="Copy a completed case report into the suite.")
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--tier", choices=("fast", "full", "hpc"), default="full",
                        help="fast uses smaller grids; hpc selects larger supported presets")
    parser.add_argument("--case-timeout", type=float, default=86400)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--refresh-imported", action="store_true",
                        help="On resume, replace selected completed imported cases and archive old reports.")
    parser.add_argument("--skip-marshak-spatial", action="store_true")
    parser.add_argument("--skip-radiative-amr", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print the case order and exit.")
    args = parser.parse_args()
    if args.max_cores < 1 or args.case_timeout <= 0:
        parser.error("cores and case timeout must be positive")
    args.selected = tuple(case for case in ORDER if not args.only or case in args.only)
    imports = {}
    for item in args.imports:
        name, separator, raw_path = item.partition("=")
        if not separator or name not in ORDER or name in imports:
            parser.error(f"invalid or duplicate --import: {item}")
        imports[name] = Path(raw_path).expanduser().resolve()
        if not (imports[name]/"index.html").is_file() or not (imports[name]/"summary.json").is_file():
            parser.error(f"not a complete report directory: {imports[name]}")
    if set(imports)-set(args.selected):
        parser.error("imported reports must be among the selected cases")
    args.imports = imports
    if args.resume and args.run_name:
        parser.error("--resume cannot be combined with --run-name")
    if args.reassess_suite and (args.resume or args.run_name or args.imports or args.only):
        parser.error("--reassess-suite takes only an existing suite directory")
    if args.refresh_imported and (not args.resume or not imports):
        parser.error("--refresh-imported requires --resume and at least one --import")
    args.layout = "direct"
    return args


def case_base(root: Path, layout: str) -> Path:
    """New suites keep case folders beside index.html; old suites retain cases/."""
    return root if layout == "direct" else root/"cases"


def report_status(path: Path) -> tuple[str, dict]:
    summary = json.loads((path/"summary.json").read_text())
    status = summary.get("status", "missing_status")
    if status not in COMPLETED:
        raise ValueError(f"child report is not complete: {path}: {status}")
    manifests = list(path.rglob("manifest.json"))
    if not (path/"index.html").is_file() or not manifests:
        raise ValueError(f"child report lacks HTML or manifest: {path}")
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.refs = []
        def handle_starttag(self, tag, attrs):
            self.refs.extend(value for key, value in attrs
                             if key in ("href", "src") and value)
    links = Links()
    links.feed((path/"index.html").read_text())
    missing = []
    for raw in links.refs:
        ref = urlsplit(raw)
        if ref.scheme or ref.netloc or not ref.path:
            continue
        if not (path/unquote(ref.path)).exists():
            missing.append(raw)
    if missing:
        raise ValueError(f"child report has missing local references: {path}: {missing}")
    return status, summary


def normalize_imported_html(path: Path) -> int:
    """Apply the suite's readable font size to copies of older reports."""
    override = ("<style id=\"validation-suite-type\">"
                ":root{font-size:21px!important}"
                "@media(max-width:700px){:root{font-size:19px!important}}"
                "</style>")
    count = 0
    for html in path.rglob("index.html"):
        source = html.read_text()
        if "validation-suite-type" in source:
            continue
        if "</head>" not in source:
            raise ValueError(f"cannot normalize typography in {html}")
        html.write_text(source.replace("</head>", override+"</head>", 1))
        count += 1
    return count


def overview(name: str, summary: dict) -> str:
    try:
        if name == "couple":
            errors = [case["metrics"]["tail_3_max_relative_analytic_error"]
                      for case in summary["cases"].values()]
            return f"simple {len(summary['simple'])} 件、sweep {len(errors)} 件。後半誤差 最大 {max(errors):.2g}"
        if name == "diffusion":
            spatial = summary["metrics"]["spatial_extrapolated_order"][-1]
            temporal = summary["metrics"]["temporal_difference_order"]
            return f"空間 {spatial:.2f} 次、時間 {temporal:.2f} 次"
        if name == "marshak":
            finest = max(summary["cases"], key=lambda tag: int(tag[1:]))
            error = summary["cases"][finest]["stages"]["final"]["u"]["l1_relative"]
            return f"{finest.upper()} の最終 L1 {100*error:.3g}%"
        if name == "streaming":
            lag = min(case["final"]["x50_minus_expected_cm"]
                      for case in summary["cases"].values())
            return f"50% 前線の最大位置差 {lag:.2f} cm"
        if name == "shock_tube":
            strict = sum(case["strict_converged"] for case in summary["solver"].values())
            steps = sum(case["nr_status_count"] for case in summary["solver"].values())
            error = summary["comparison"]["n128"]["temperature"]["relative_l1"]
            return f"Newton 厳密収束 {strict}/{steps} step、Tgas L1 {100*error:.3g}%"
        if name == "radiative_shock":
            errors = [summary["cases"][f"mach{mach}"]["analysis"]["final"]
                      ["agreement"]["trad"]["global_relative_L1_percent"]
                      for mach in (2, 5)]
            spike = summary["cases"]["mach5"]["analysis"]["gas_temperature_peak"]
            return (f"Mach 2/5 Trad L1 {errors[0]:.3g}% / {errors[1]:.3g}%; "
                    f"Mach 5 スパイク再現 {100*spike['spike_amplitude_recovery']:.1f}%")
        if name == "advection_pulse":
            metric = summary["analysis"]["metrics"]["Tgas"]["relative_L1"]
            return f"移流あり/なし Tgas L1 {100*metric:.3g}%"
        if name == "static_equilibrium":
            metric = summary["analysis"]["final"]["Tgas"]["max_pointwise_relative_drift"]
            return f"Tgas 最大相対ずれ {100*metric:.3g}%"
        if name == "thermal_wave":
            metric = summary["cases"][summary["high_cases"][1]]["error"]["final"]["Tgas_contrast_L1"]
            return f"高解像度 AMR Tgas L1 {100*metric:.3g}%"
        if name == "closed_box":
            drift = max(case["energy_max_relative_drift"]
                        for case in summary["cases"].values())
            return f"全エネルギー最大相対ずれ {drift:.2g}"
        if name == "fgmres_amr_restart":
            error = max(case["relative_L1"] for case in summary["comparison"].values())
            return f"再起動後の最大相対 L1 {error:.2g}"
        if name == "mixed_frame":
            audit = summary["diagnostic"]["status"]
            return f"{len(summary['cases'])} 比較、Jacobian 診断 {audit}"
    except (KeyError, TypeError, ValueError, IndexError):
        pass
    return "詳細レポート参照"


def write_html(root: Path, state: dict):
    rows = []
    base = case_base(root, state.get("layout", "legacy"))
    labels = {
        "passed": ("PASS", "ok"),
        "diagnostic_complete": ("診断完了", "diagnostic"),
        "accuracy_not_met": ("精度未達", "caveat"),
        "completed_with_solver_caveat": ("要確認", "caveat"),
        "failed": ("FAIL", "failed"),
        "running": ("実行中", "pending"),
        "pending": ("未実行", "pending"),
    }
    for name in state["selected"]:
        case = state["cases"].get(name, {})
        status = case.get("status", "pending")
        label, css = labels.get(status, (status, "pending"))
        destination = base/name
        report = (destination/"index.html").relative_to(root)
        link = (f'<a href="{escape(report.as_posix())}">{escape(name.replace("_", " "))}</a>'
                if (destination/"index.html").is_file() else
                escape(name.replace("_", " ")))
        summary_path = destination/"summary.json"
        result = case.get("overview", "—")
        if status in COMPLETED and summary_path.is_file():
            try:
                result = overview(name, json.loads(summary_path.read_text()))
            except (ValueError, OSError):
                pass
        if status == "failed":
            result = "—"
            comment = case.get("failure_kind", "計算・検証に失敗")
        elif status == "running":
            comment = "計算中"
        elif status == "pending":
            comment = "実行待ち"
        else:
            comment = COMMENTS[name]
            if any("imported_from" in item for item in case.get("commands", [])):
                comment += "・既存出力を取込"
        rows.append(f'<tr><th scope="row">{link}</th><td class="{css}"><strong>{escape(label)}</strong></td>'
                    f'<td>{escape(result)}</td><td>{escape(comment)}</td></tr>')
    counts = {status: sum(state["cases"].get(name, {}).get("status", "pending") == status
                          for name in state["selected"]) for status in labels}
    progress = (f'PASS {counts["passed"]} 件、精度未達 {counts["accuracy_not_met"]} 件、'
                f'診断完了 {counts["diagnostic_complete"]} 件、'
                f'要確認 {counts["completed_with_solver_caveat"]} 件、'
                f'FAIL {counts["failed"]} 件、未完了 {counts["running"]+counts["pending"]} 件')
    root.joinpath("index.html").write_text(f'''<!doctype html><html lang="ja"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLD テスト計算のまとめ</title><style>
:root{{font-family:system-ui,sans-serif;font-size:21px;line-height:1.6}}
body{{max-width:1350px;margin:1.5rem auto;padding:0 1rem;color:#17212b;background:#f8fafc}}
h1{{font-size:2rem;margin-bottom:.2rem}}.lead{{margin:.3rem 0 1rem}}
.progress{{font-weight:700;color:#16324f}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#fff}}
th,td{{padding:.7rem .8rem;border-bottom:1px solid #cbd5e1;text-align:left;vertical-align:top}}
thead th{{background:#eaf2f8}}tbody th{{font-weight:600}}
.ok{{color:#08764b}}.diagnostic{{color:#075985}}.caveat{{color:#9a4f00}}
.failed{{color:#b42318}}.pending{{color:#526170}}
a{{color:#075ca8}}small{{font-size:.85rem;color:#475569}}
@media(max-width:700px){{:root{{font-size:19px}}}}
</style></head><body><h1>FLD テスト計算のまとめ</h1>
<p class="lead">各テスト名から結果の HTML を開けます。実行層: {escape(state.get("tier", "full"))}。</p>
<p class="progress">{escape(progress)}</p>
<div class="scroll"><table><thead><tr><th>テスト</th><th>判定</th><th>簡単な結果</th><th>コメント</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</div><p><small>「精度未達」は計算が正常終了し、定めた物理精度基準を満たさなかった状態です。
「診断完了」は物理的な合格閾値をまだ設けていない計算を示します。
詳細・図・動画・ログは各テストの HTML から参照できます。</small></p>
<p><small><a href="suite.json">実行状態 JSON</a> ·
<a href="manifest.json">manifest</a></small></p></body></html>''')


def save(root: Path, state: dict):
    (root/"suite.json").write_text(json.dumps(state, indent=2)+"\n")
    path = root/"manifest.json"
    manifest = (json.loads(path.read_text()) if path.exists()
                else base_manifest(find_repo_root(HERE), "nrfld_validation_suite",
                                   state.get("max_cores", 1)))
    manifest["status"] = state["status"]
    manifest["metrics"] = state["cases"]
    manifest["selected"] = state["selected"]
    manifest["tier"] = state.get("tier", "full")
    manifest["import_sources"] = state.get("import_sources", {})
    manifest["analysis_script_sha256"] = sha256_file(Path(__file__))
    write_manifest(path, manifest)
    write_html(root, state)


def suite_status(state: dict) -> str:
    statuses = [state["cases"].get(name, {}).get("status") for name in state["selected"]]
    if not all(status in COMPLETED for status in statuses):
        return "incomplete"
    if any(status in CAVEAT for status in statuses):
        return "complete_with_solver_caveat"
    if any(status in ACCURACY_GAP for status in statuses):
        return "complete_with_accuracy_gap"
    return "complete"


def reassess_suite(root: Path) -> int:
    state = json.loads((root/"suite.json").read_text())
    base = case_base(root, state.get("layout", "legacy"))
    for name in ("radiative_shock", "advection_pulse", "static_equilibrium"):
        if name not in state["selected"]:
            continue
        destination = base/name
        if name == "radiative_shock" and (destination/"amr/summary.json").is_file():
            subprocess.run([sys.executable, str(HERE/SCRIPTS[name]),
                            "--reassess", str(destination/"amr")], check=True)
        subprocess.run([sys.executable, str(HERE/SCRIPTS[name]),
                        "--reassess", str(destination)], check=True)
        status, summary = report_status(destination)
        state["cases"][name].update(status=status, overview=overview(name, summary))
        save(root, state)
    state["status"] = suite_status(state)
    state["accuracy_reassessed_at"] = datetime.now(timezone.utc).isoformat()
    save(root, state)
    print(root, flush=True)
    return 0


def command_for(name: str, root: Path, args) -> list[str]:
    base = case_base(root, args.layout)
    command = [sys.executable, str(HERE/SCRIPTS[name]),
               "--output-root", str(base), "--run-name", name]
    if name in ("thermal_wave", "mixed_frame"):
        command += ["--mpi-ranks", str(args.max_cores),
                    "--run-timeout", str(args.case_timeout)]
        command += ["--preset", args.tier]
        return command
    if name == "fgmres_amr_restart":
        return command + ["--max-cores", str(args.max_cores),
                          "--timeout", str(args.case_timeout)]
    command += ["--run-timeout", str(args.case_timeout)]
    if name == "shock_tube":
        command += ["--strict-backward-error"]
    if name in ("couple", "diffusion", "marshak", "streaming", "shock_tube", "closed_box"):
        command += ["--max-cores", str(args.max_cores)]
    else:
        command += ["--mpi-ranks", str(args.max_cores)]
    if args.tier == "fast":
        if name == "couple":
            command += ["--steps", "4", "--dt-values", "1e-5", "1e-4"]
        elif name == "diffusion":
            command += ["--resolutions", "16", "32", "64", "--base-steps", "40"]
        elif name == "marshak":
            command += ["--resolutions", "32", "64", "--steps", "170"]
        elif name == "radiative_shock":
            command += ["--nx1-mach2", "64", "--nx1-mach5", "64"]
        elif name == "advection_pulse":
            command += ["--nx1", "32"]
        elif name == "static_equilibrium":
            command += ["--nx", "32"]
    elif args.tier == "hpc":
        if name == "diffusion":
            command += ["--resolutions", "64", "128", "256", "--base-steps", "160"]
        elif name == "marshak":
            command += ["--resolutions", "64", "128", "256", "512"]
        elif name == "radiative_shock":
            command += ["--preset", "hpc"]
        elif name == "advection_pulse":
            command += ["--nx1", "512"]
        elif name == "static_equilibrium":
            command += ["--nx", "128"]
    if name == "static_equilibrium":
        source = base/"advection_pulse/build/build_source"
        if source.is_dir():
            command += ["--reuse-build", str(source)]
    return command


def execute(command: list[str], root: Path, log_name: str, timeout: float) -> dict:
    log = root/"logs"/f"{log_name}.log"
    log.parent.mkdir(exist_ok=True)
    resource = root/"resources"/f"{log_name}.json"
    result = run_monitored(command, HERE.parent.parent, log, timeout, resource)
    return {"command": command, "log": str(log.relative_to(root)),
            "resource": str(resource.relative_to(root)),
            "returncode": result["returncode"],
            "peak_process_tree_rss_kib": result["peak_process_tree_rss_kib"],
            "peak_athena_rss_sum_kib": result["peak_athena_rss_sum_kib"],
            "mpi_rank_hwm_kib": {rank: row["hwm_kib"] for rank, row in result["mpi_ranks"].items()},
            "oom_classification": result["oom_classification"],
            **({"error": f"timeout after {timeout:g} seconds"} if result["timeout"] else {})}


def command_failure_kind(result: dict) -> str:
    if result.get("error", "").startswith("timeout"):
        return "timeout"
    if result.get("oom_classification") == "cgroup_oom_kill":
        return "oom_kill"
    if result.get("oom_classification") == "possible_sigkill":
        return "possible_sigkill"
    return "command_failure"


def refine_failure_kind(root: Path, name: str, kind: str, layout: str) -> str:
    if kind not in {"command_failure", "report_validation_or_filesystem"}:
        return kind
    summary = case_base(root, layout) / name / "summary.json"
    if summary.is_file():
        try:
            if json.loads(summary.read_text()).get("status") == "failed":
                return "validation_failure"
        except (ValueError, OSError):
            pass
    log = root / "logs" / f"{name}.log"
    if log.is_file():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size-100000))
            tail = handle.read().decode(errors="replace")
        if any(marker in tail for marker in ("FATAL ERROR", "Newton-Raphson solve failed",
                                            "status=backtracking_exhausted", "status=max_iterations")):
            return "solver_failure"
    return kind


def run_one(name: str, root: Path, args) -> list[dict]:
    commands = []
    base = case_base(root, args.layout)
    main = command_for(name, root, args)
    # Child runners bound each calculation; a problem can contain several runs.
    command_timeout = args.case_timeout*8+3600
    first = execute(main, root, name, command_timeout)
    commands.append(first)
    if first["returncode"] != 0:
        raise CaseExecutionError(f"main command failed: {first}", commands,
                                 command_failure_kind(first))
    if name == "marshak" and args.tier != "fast" and not args.skip_marshak_spatial:
        command = [sys.executable, str(HERE/"marshak_spatial.py"),
                   "--report-root", str(base/name),
                   "--max-cores", str(args.max_cores)]
        result = execute(command, root, "marshak_spatial", command_timeout)
        commands.append(result)
        if result["returncode"] != 0:
            raise CaseExecutionError(f"Marshak spatial extension failed: {result}",
                                     commands, command_failure_kind(result))
    if name == "radiative_shock" and args.tier != "fast" and not args.skip_radiative_amr:
        amr_parent = base/name if args.layout == "direct" else base
        amr_name = "amr_run" if args.layout == "direct" else "radiative_shock_amr"
        amr_run = amr_parent/amr_name
        command = [sys.executable, str(HERE/SCRIPTS[name]),
                   "--output-root", str(amr_parent),
                   "--run-name", amr_name, "--layout", "amr",
                   "--reuse-build", str(base/name/"build/build_source"),
                   "--mpi-ranks", str(args.max_cores),
                   "--run-timeout", str(args.case_timeout)]
        if args.tier == "hpc":
            command += ["--preset", "hpc"]
        result = execute(command, root, "radiative_shock_amr", command_timeout)
        commands.append(result)
        if result["returncode"] != 0:
            raise CaseExecutionError(f"radiative shock AMR failed: {result}",
                                     commands, command_failure_kind(result))
        command = [sys.executable, str(HERE/SCRIPTS[name]),
                   "--postprocess", str(base/name),
                   "--attach-amr-run", str(amr_run)]
        result = execute(command, root, "radiative_shock_attach_amr", command_timeout)
        commands.append(result)
        if result["returncode"] != 0:
            raise CaseExecutionError(f"radiative shock AMR attachment failed: {result}",
                                     commands, command_failure_kind(result))
    return commands


def main() -> int:
    args = arguments()
    if args.reassess_suite:
        return reassess_suite(args.reassess_suite.expanduser().resolve())
    if args.plan:
        print("tier", args.tier)
        for name in args.selected:
            print(name, "import" if name in args.imports else "run",
                  " ".join(command_for(name, Path("<suite>"), args)))
        return 0
    os.environ["OMP_NUM_THREADS"] = "1"
    root = (args.resume.expanduser().resolve() if args.resume else
            args.output_root.expanduser().resolve() /
            (args.run_name or f"validation_suite_{datetime.now(timezone(timedelta(hours=9))):%Y%m%d_%H%M%S}_JST"))
    if args.resume:
        state = json.loads((root/"suite.json").read_text())
        args.layout = state.get("layout", "legacy")
        if not args.only:
            args.selected = tuple(state["selected"])
        if tuple(state["selected"]) != args.selected:
            raise ValueError("--resume selection differs from the original suite")
        if state.get("tier", "full") != args.tier:
            raise ValueError("--resume tier differs from the original suite")
        state.setdefault("import_sources", {}).update(
            {name: str(path) for name, path in args.imports.items()})
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root/"logs").mkdir()
        state = {"status": "running", "selected": args.selected, "cases": {},
                 "tier": args.tier, "layout": args.layout,
                 "started_at": datetime.now(timezone.utc).isoformat(),
                 "max_concurrent_heavy_cases": 1, "max_cores": args.max_cores,
                 "import_sources": {name: str(path) for name, path in args.imports.items()}}
    save(root, state)
    base = case_base(root, args.layout)
    for name in args.selected:
        prior = state["cases"].get(name, {})
        refresh = args.refresh_imported and name in args.imports
        if prior.get("status") in COMPLETED and not refresh:
            continue
        destination = base/name
        if refresh and destination.exists():
            archive = root/"replaced_imports"/name
            archive.parent.mkdir(exist_ok=True)
            serial = 1
            while archive.exists():
                archive = root/"replaced_imports"/f"{name}_{serial}"
                serial += 1
            destination.rename(archive)
            state.setdefault("replaced_imports", []).append({
                "case": name, "archived_to": str(archive),
                "replacement": str(args.imports[name])})
        recoverable = (prior.get("status") == "running" or
                       (prior.get("status") == "failed" and args.retry_failed))
        if recoverable and destination.exists():
            try:
                status, summary = report_status(destination)
            except (ValueError, OSError, KeyError):
                pass
            else:
                state["cases"][name] = {
                    "status": status, "overview": overview(name, summary),
                    "commands": prior.get("commands", []) + [{
                        "recovered_completed_child_report": True,
                        "prior_status": prior["status"],
                        "resource_monitor_incomplete": prior["status"] == "running"}],
                    "report": str((destination/"index.html").relative_to(root))}
                save(root, state)
                continue
        if prior.get("status") == "failed" and not args.retry_failed:
            continue
        if (prior.get("status") in {"failed", "running"}
                and destination.exists()):
            archive = root/"failed_attempts"/name
            archive.parent.mkdir(exist_ok=True)
            serial = 1
            while archive.exists():
                archive = root/"failed_attempts"/f"{name}_{serial}"
                serial += 1
            destination.rename(archive)
            if prior.get("status") == "running":
                state.setdefault("interrupted_attempts", []).append(
                    {"case": name, "archived_to": str(archive),
                     "classification": "external_interruption_or_process_stop"})
        state["cases"][name] = {"status": "running"}
        save(root, state)
        try:
            if name in state["import_sources"]:
                source = Path(state["import_sources"][name])
                shutil.copytree(source, destination)
                normalized = normalize_imported_html(destination)
                commands = [{"imported_from": str(source),
                             "typography_normalized_html_files": normalized}]
                monitor_source = next((candidate for candidate in (
                    source/"suite_resource.json",
                    source.parent.parent/"resources"/f"{name}.json")
                    if candidate.is_file()), None)
                if monitor_source is not None:
                    monitor_target = root/"resources"/f"{name}.json"
                    monitor_target.parent.mkdir(exist_ok=True)
                    shutil.copy2(monitor_source, monitor_target)
                    measured = json.loads(monitor_target.read_text())
                    commands[0].update({
                        "resource": str(monitor_target.relative_to(root)),
                        "peak_process_tree_rss_kib": measured["peak_process_tree_rss_kib"],
                        "peak_athena_rss_sum_kib": measured["peak_athena_rss_sum_kib"],
                        "mpi_rank_hwm_kib": {rank: row["hwm_kib"] for rank, row
                                             in measured["mpi_ranks"].items()},
                        "oom_classification": measured["oom_classification"]})
            else:
                if destination.exists():
                    raise FileExistsError(f"case directory exists; preserve it and choose a new suite: {destination}")
                commands = run_one(name, root, args)
            status, summary = report_status(destination)
            state["cases"][name] = {"status": status,
                                    "overview": overview(name, summary),
                                    "commands": commands,
                                    "report": str((destination/"index.html").relative_to(root))}
        except CaseExecutionError as error:
            state["cases"][name] = {"status": "failed", "error": str(error),
                                    "failure_kind": refine_failure_kind(root, name, error.kind, args.layout),
                                    "commands": error.commands}
        except Exception as error:
            state["cases"][name] = {"status": "failed", "error": str(error),
                                    "failure_kind": refine_failure_kind(
                                        root, name, "report_validation_or_filesystem", args.layout)}
        save(root, state)
    state["status"] = suite_status(state)
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(root, state)
    print(root, flush=True)
    return 0 if state["status"] in {"complete", "complete_with_solver_caveat"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
