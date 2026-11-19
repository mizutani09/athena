#!/usr/bin/env python3
"""Unit tests for the WOR-20 XD scaling result reader."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wor20_xd_scaling as scaling  # noqa: E402


def valid_summary() -> dict:
    return {
        "status": "passed",
        "nodes": 2,
        "ranks": 64,
        "repeat": 0,
        "meshblocks": 7680,
        "leaf_cells": 3932160,
        "measured_steps": 10,
        "seconds_per_step": 2.0,
        "max_rank_rss_kib": 1024,
        "sum_rank_peak_rss_kib": 65536,
        "newton_iterations": [1],
        "fgmres_iterations_max": 9,
        "fgmres_iterations_total": 108,
        "mass_relative_drift": 0.0,
        "linear_solves": 12,
        "fractions": {
            "preconditioner_seconds": 0.4,
            "matvec_seconds": 0.5,
            "reduction_seconds": 0.05,
            "vector_and_other_seconds": 0.05,
        },
    }


class Wor20XdScalingTests(unittest.TestCase):
    def make_run(self, root: Path, summary: dict) -> None:
        (root / "summary.json").write_text(json.dumps(summary))
        (root / "result.txt").write_text("PASS\n")

    def test_load_run_checks_and_derives_throughput(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root, valid_summary())
            result = scaling.load_run(root)
            self.assertEqual(result["path"], str(root.resolve()))
            self.assertAlmostEqual(result["zone_steps_per_second"], 1966080.0)

    def test_load_run_rejects_mass_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = valid_summary()
            summary["mass_relative_drift"] = 2.0e-12
            self.make_run(root, summary)
            with self.assertRaisesRegex(ValueError, "mass drift"):
                scaling.load_run(root)

    def test_load_run_rejects_nonadditive_fractions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = valid_summary()
            summary["fractions"]["vector_and_other_seconds"] = 0.1
            self.make_run(root, summary)
            with self.assertRaisesRegex(ValueError, "fractions sum"):
                scaling.load_run(root)


if __name__ == "__main__":
    unittest.main()
