#!/usr/bin/env python3
"""Regenerate a portable solar-convection campaign HTML and its movies.

Run this from any working directory with the same Git checkout used for the
simulation. The feedback state and completed segment directories may have
been copied to a different filesystem prefix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feedback-state", type=Path,
                        help="state.json from a segmented entropy-feedback run")
    source.add_argument("--run", type=Path, action="append",
                        help="Consecutive run directories, in time order; repeat")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <feedback-state directory>/combined")
    parser.add_argument("--header-from", type=Path,
                        help="HST with column headers, if absent from first run")
    parser.add_argument("--allow-s-in-variation", action="store_true",
                        help="For manually listed runs, permit controlled s_in changes")
    parser.add_argument("--opacity-table", type=Path,
                        help="Rosseland table for the tau=1 contour in xz movie")
    parser.add_argument("--skip-movies", action="store_true",
                        help="Generate only the joined history report")
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir is None and args.feedback_state is None:
        parser.error("--output-dir is required with manually listed --run directories")
    if not args.skip_movies and args.opacity_table is None:
        parser.error("--opacity-table is required to render the tau=1 movie; "
                     "pass --skip-movies for a history-only report")
    if args.run is not None and len(args.run) < 2:
        parser.error("Pass at least two consecutive --run directories")

    here = Path(__file__).resolve().parent
    output = (args.output_dir or args.feedback_state.parent / "combined").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = [sys.executable, str(here / "solar_convection_continuation_report.py")]
    if args.feedback_state is not None:
        report += ["--feedback-state", str(args.feedback_state.resolve())]
    else:
        for run in args.run:
            report += ["--run", str(run.resolve())]
        if args.allow_s_in_variation:
            report += ["--allow-s-in-variation"]
    if args.header_from is not None:
        report += ["--header-from", str(args.header_from.resolve())]
    report += ["--output-dir", str(output)]
    subprocess.run(report, check=True)

    if not args.skip_movies:
        movie = [sys.executable, str(here / "solar_convection_campaign_movies.py"),
                 "--source-summary", str(output / "summary.json"),
                 "--output-root", str(output),
                 "--opacity-table", str(args.opacity_table.resolve()),
                 "--max-frames", str(args.max_frames), "--fps", str(args.fps)]
        subprocess.run(movie, check=True)
        subprocess.run(report, check=True)  # Embed the completed movies in HTML.
    if args.feedback_state is not None and output.parent == args.feedback_state.resolve().parent:
        from run_solar_entropy_feedback_long import render
        state_path = args.feedback_state.resolve()
        render(state_path.parent, json.loads(state_path.read_text()))
    print(output / "index.html")


if __name__ == "__main__":
    main()
