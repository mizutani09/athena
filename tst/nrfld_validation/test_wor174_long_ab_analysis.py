#!/usr/bin/env python3
"""Unit tests for the clean-source WOR-174 long A/B audit."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wor174_long_ab_analysis as analysis  # noqa: E402


class Wor174LongAbAnalysisTests(unittest.TestCase):
    def test_window_uses_clean_36_column_layout(self):
        history = np.zeros((11, 36), dtype=float)
        history[:, analysis.COL["time"]] = np.linspace(0.0, 100.0, 11)
        history[:, analysis.COL["dt"]] = 10.0
        history[:, analysis.COL["frad_top"]] = 0.9 * analysis.FSUN
        history[:, analysis.COL["fhyd_top"]] = 0.05 * analysis.FSUN
        history[:, analysis.COL["top_mdot_out"]] = 0.01 / analysis.GRAVITY_FACTOR
        history[:, analysis.COL["fhyd_bottom"]] = 0.8 * analysis.FSUN
        history[:, analysis.COL["frad_bottom"]] = 0.1 * analysis.FSUN
        history[:, analysis.COL["bottom_mdot_up"]] = -0.02 / analysis.GRAVITY_FACTOR
        history[:, analysis.COL["bottom_mdot_down"]] = 0.0
        history[:, analysis.COL["etot_grav"]] = (
            history[:, analysis.COL["time"]]
            * 0.005 * analysis.AREA * analysis.FSUN * analysis.TIME_UNIT)
        history[:, analysis.COL["mass_drift"]] = np.linspace(0.0, 1.0e-3, 11)
        history[:, analysis.COL["pbnd_scale"]] = 1.0
        history[:, analysis.COL["vz_rms2_mean"]] = 4.0

        result = analysis.audit_window(history, 100)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["top_radiation"], 0.9)
        self.assertAlmostEqual(result["top_total"], 0.96)
        self.assertAlmostEqual(result["bottom_total"], 0.92)
        self.assertAlmostEqual(result["storage"], 0.005)
        self.assertAlmostEqual(result["closure_error"], -0.045)
        self.assertAlmostEqual(result["mass_drift_final"], 1.0e-3)
        self.assertAlmostEqual(result["vz_rms_time_mean"], 2.0)

    def test_short_window_is_not_reported(self):
        history = np.zeros((11, 36), dtype=float)
        history[:, analysis.COL["time"]] = np.linspace(0.0, 90.0, 11)
        self.assertIsNone(analysis.audit_window(history, 100))


if __name__ == "__main__":
    unittest.main()
