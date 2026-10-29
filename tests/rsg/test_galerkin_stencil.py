"""Algebraic support checks for the uniform RSG Galerkin transfer pair."""

from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FACE_OFFSETS = (
    (0, 0, 0),
    (0, 0, -1), (0, 0, 1),
    (0, -1, 0), (0, 1, 0),
    (-1, 0, 0), (1, 0, 0),
)


def constant_transfer_rap(row_coefficients):
    """Aggregate one 2^3 fine-cell row block using R=1/8 and constant P."""
    coarse = {}
    for ok in range(2):
        for oj in range(2):
            for oi in range(2):
                for coefficient, (dk, dj, di) in zip(
                        row_coefficients[ok, oj, oi], FACE_OFFSETS):
                    fine_column = (ok + dk, oj + dj, oi + di)
                    coarse_column = tuple(value // 2 for value in fine_column)
                    coarse[coarse_column] = (
                        coarse.get(coarse_column, 0.0) + coefficient/8.0)
    return coarse


class GalerkinStencilTest(unittest.TestCase):
    def test_random_variable_coefficients_have_only_seven_point_support(self):
        rng = np.random.default_rng(5317)
        rows = rng.normal(size=(2, 2, 2, len(FACE_OFFSETS)))
        rap = constant_transfer_rap(rows)
        expected = set(FACE_OFFSETS)
        self.assertEqual(set(rap), expected)
        self.assertFalse(any(sum(value != 0 for value in offset) > 1
                             for offset in rap))

    def test_production_and_experimental_inputs_are_separate(self):
        source = (ROOT/"src/linear_solver/linearMG/linearMG.cpp").read_text()
        self.assertIn('coarse_operator == "galerkin_27_experimental"', source)
        self.assertIn("index >= linearSolver::NMATRIX", source)
        rsg_input = (ROOT/"inputs/rsg/athinput.st35gm03n07_fld").read_text()
        self.assertIn("coarse_operator = galerkin", rsg_input)
        self.assertNotIn("galerkin_27_experimental", rsg_input)


if __name__ == "__main__":
    unittest.main()
