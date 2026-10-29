import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "make_hse_model", ROOT/"tools/rsg/make_hse_model.py")
RSG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RSG)


class FreytagGravityTest(unittest.TestCase):
    def test_analytic_derivative(self):
        radius = 830*RSG.RSUN
        gm = RSG.G*12*RSG.MSUN
        r0, r1 = 0.2*radius, 1.2*radius
        for r in np.geomspace(2e-2*radius, 4*radius, 80):
            h = r*1e-4
            numerical = -(RSG.potential(r+h, gm, r0, r1)
                          - RSG.potential(r-h, gm, r0, r1))/(2*h)
            analytic = -RSG.gravity(r, gm, r0, r1)
            self.assertAlmostEqual(numerical/analytic, 1.0, places=6)

    def test_limits(self):
        radius = 830*RSG.RSUN
        gm = RSG.G*12*RSG.MSUN
        r0, r1 = 0.2*radius, 1.2*radius
        self.assertEqual(RSG.gravity(0.0, gm, r0, r1), 0.0)
        middle = 0.6*radius
        self.assertLess(abs(RSG.gravity(middle, gm, r0, r1)/(gm/middle**2)-1), 0.02)
        self.assertLess(RSG.gravity(10*r1, gm, r0, r1), RSG.gravity(r1, gm, r0, r1)*1e-3)


if __name__ == "__main__":
    unittest.main()
