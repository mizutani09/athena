"""Checks for misleading-success paths in the performance result reader."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from performance import analyse, compare_fields, input_text


class PerformanceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(warmup=1, steps=2, dt=1e-4, mode="strong",
                                       resolution=32, block=16)
        (self.root / "run.log").write_text("Terminating on cycle limit\n")
        (self.root / "benchmark.hst").write_text("0 0.0001 1\n0.0003 0.0001 1\n")
        for kind, names in (("prim", ["rho", "press"]), ("rad", ["Er"])):
            for index in (0, 1):
                with h5py.File(self.root / f"benchmark.{kind}.{index:05}.athdf", "w") as h:
                    h.attrs["Time"] = 0 if index == 0 else 3e-4
                    h.attrs["VariableNames"] = np.array(names, dtype="S")
                    h.attrs["DatasetNames"] = np.array([kind], dtype="S")
                    h.attrs["NumVariables"] = [len(names)]
                    h[kind] = np.ones((len(names), 1, 1, 1, 1))
        self.raw = []
        for rank in (0, 1):
            self.raw.append({"rank": rank, "wall_seconds": 1.0,
                "seconds": {"explicit_integrator": 0.9 if rank == 0 else 0.1,
                            "fld_solve": 0.1 if rank == 0 else 0.9},
                "steps": [{"cycle": i, "dt": 1e-4, "global_cells": 32768,
                           "newton_iterations": 2, "newton_l2": 1e-10,
                           "newton_max": 1e-9, "newton_reason": "converged"}
                          for i in (1, 2)]})
        self.save()

    def save(self):
        for rank, raw in enumerate(self.raw):
            (self.root / f"performance_rank{rank}.json").write_text(json.dumps(raw))

    def read(self):
        return analyse(self.root, self.args, 2, "hd_fld")

    def test_rank_means_not_sum_of_component_maxima(self):
        result = self.read()
        self.assertEqual(result["implicit_fld_fraction_rank_mean"], 0.5)
        self.assertEqual(result["zone_cycles_per_second"], 65536)

    def test_fatal_even_if_launcher_returned_zero(self):
        (self.root / "run.log").write_text("### FATAL ERROR\n")
        with self.assertRaises(ValueError):
            self.read()

    def test_relaxed_acceptance_is_not_a_pass(self):
        self.raw[1]["steps"][0]["newton_reason"] = "accepted_relaxed"
        self.save()
        with self.assertRaises(ValueError):
            self.read()

    def test_truncated_window(self):
        self.raw[0]["steps"].pop()
        self.save()
        with self.assertRaises(ValueError):
            self.read()

    def test_nonfinite_fields(self):
        with h5py.File(self.root / "benchmark.prim.00001.athdf", "r+") as h:
            h["prim"][0, 0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.read()

    def test_cfl_reduced_step_rejected(self):
        self.raw[0]["steps"][0]["dt"] *= 0.5
        self.save()
        with self.assertRaises(ValueError):
            self.read()

    def test_weak_scaling_grows_domain_and_grid(self):
        self.args.mode = "weak"
        text = input_text(self.args, 4, True)
        self.assertIn("nx1 = 64", text)
        self.assertIn("nx2 = 64", text)
        self.assertIn("nx3 = 32", text)
        self.assertIn("x1max = 2", text)

    def test_roundoff_is_distinguished_from_changed_solution(self):
        a = {"Bcc1": np.array([0.1])}
        b = {"Bcc1": np.nextafter(a["Bcc1"], np.inf)}
        result = compare_fields(a, b)
        json.dumps(result)  # Must persist even when numpy comparisons return scalar bools.
        self.assertFalse(result["final_arrays_identical"])
        self.assertTrue(result["within_roundoff"])
        self.assertFalse(compare_fields(a, {"Bcc1": a["Bcc1"] + 1e-10})["within_roundoff"])

    def test_output_endpoint_matches_repeated_time_addition(self):
        self.args.warmup, self.args.steps = 5, 20
        end = 0.0
        for _ in range(25):
            end += self.args.dt
        self.assertNotEqual(end, 25 * self.args.dt)
        self.assertIn(f"tlim = {end}", input_text(self.args, 1, True))


if __name__ == "__main__":
    unittest.main()
