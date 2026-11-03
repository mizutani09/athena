#!/usr/bin/env python3
"""Measure an AMR interface R A_h P mismatch in the validated x-only Test C model.

This is a composite-grid reference, not Athena's octet/root MG hierarchy.  The
fine operator uses the same trilinear coarse/fine ghost rule as the existing
Test C direct-solve reference.  P interpolates a uniform root-cell correction
onto refined leaves; R volume-averages leaf values back to root cells.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from test_c_composite_reference import CompositeLine


ROOT_N = 64
FINE_N = 128


def transfer(line):
    """Return volume-average R and root-to-composite trilinear P matrices."""
    h = 1.0 / ROOT_N
    coarse_x = (np.arange(ROOT_N) + 0.5) * h
    prolong = np.zeros((line.n, ROOT_N))
    restrict = np.zeros((ROOT_N, line.n))
    for leaf in range(line.n):
        cell = int(np.floor(line.x[leaf] / h)) % ROOT_N
        if np.isclose(line.dx[leaf], h):
            prolong[leaf, cell] = 1.0
        else:
            offset = (line.x[leaf] - coarse_x[cell]) / h
            neighbor = (cell + (1 if offset > 0 else -1)) % ROOT_N
            prolong[leaf, cell] = 1.0 - abs(offset)
            prolong[leaf, neighbor] = abs(offset)
        left = int(np.floor((line.edges[leaf] + 1e-12) / h)) % ROOT_N
        restrict[left, leaf] = line.dx[leaf] / h
    assert np.allclose(prolong.sum(axis=1), 1.0)
    assert np.allclose(restrict.sum(axis=1), 1.0)
    assert np.allclose(restrict @ prolong @ np.ones(ROOT_N), 1.0)
    return restrict, prolong


def root_laplacian(v):
    h = 1.0 / ROOT_N
    return (np.roll(v, -1) - 2.0*v + np.roll(v, 1)) / h**2


def mode_result(v, coarse_action, rap_action, interface):
    ah = coarse_action(v)
    rahp = rap_action(v)
    difference = ah - rahp
    scale = max(np.linalg.norm(ah), np.linalg.norm(rahp), 1e-300)
    return {
        "relative_l2": float(np.linalg.norm(difference) / scale),
        "relative_interface_l2": float(np.linalg.norm(difference[interface])
                                       / max(np.linalg.norm(ah[interface]),
                                             np.linalg.norm(rahp[interface]), 1e-300)),
        "interface_mismatch_fraction": float(np.linalg.norm(difference[interface])
                                              / max(np.linalg.norm(difference), 1e-300)),
        "cosine": float(np.dot(ah, rahp) / max(np.linalg.norm(ah)
                                                * np.linalg.norm(rahp), 1e-300)),
    }


def evaluate(left, right, q):
    line = CompositeLine(left, right, ROOT_N)
    restrict, prolong = transfer(line)
    tau = q / FINE_N**2
    root_x = (np.arange(ROOT_N) + 0.5) / ROOT_N
    initial = 1.0 + 0.1*np.exp(-0.5*((line.x - 0.5)/0.04)**2)
    rhs = tau*line.laplacian(initial, "nr")
    coarse_action = lambda v: v - tau*root_laplacian(v)
    rap_action = lambda v: restrict @ (prolong @ v
                                       - tau*line.laplacian(prolong @ v, "mg"))
    coarse_rhs = restrict @ rhs
    coarse_matrix = np.column_stack(
        [coarse_action(np.eye(ROOT_N)[:, j]) for j in range(ROOT_N)])
    coarse_correction = np.linalg.solve(coarse_matrix, coarse_rhs)
    actual_direction = np.linalg.solve(line.eye - tau*line.matrix_mg(), rhs)
    interface = np.zeros(ROOT_N, dtype=bool)
    for edge in (left, right):
        cell = int(round(edge*ROOT_N)) % ROOT_N
        interface[[(cell-2) % ROOT_N, (cell-1) % ROOT_N,
                   cell, (cell+1) % ROOT_N]] = True
    modes = {
        "constant": np.ones(ROOT_N),
        "smooth_sine": np.sin(2*np.pi*root_x),
        "pulse": np.exp(-0.5*((root_x-0.5)/0.04)**2),
        "restricted_rhs": coarse_rhs,
        "coarse_correction": coarse_correction,
        "restricted_exact_direction": restrict @ actual_direction,
    }
    checks = {name: mode_result(v, coarse_action, rap_action, interface)
              for name, v in modes.items()}
    return {
        "refined_interval": [left, right], "q": q,
        "leaf_cells": line.n, "root_cells": ROOT_N,
        "operator": "A_h=I-tau*L_mg on composite leaves; A_H=I-tau*L_root",
        "transfer": "P root-cell trilinear to leaves; R leaf-volume average",
        "tau": tau, "interface_root_cells": np.flatnonzero(interface).tolist(),
        "fine_constant_row_error": float(np.max(np.abs(
            line.matrix_mg() @ np.ones(line.n)))),
        "root_constant_row_error": float(np.max(np.abs(root_laplacian(
            np.ones(ROOT_N))))),
        "modes": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [evaluate(lo, hi, q) for q in (1e4, 1e7)
            for lo, hi in ((0.25, 0.75), (0.125, 0.625))]
    report = {"scope": "independent x-only Test C composite AMR model",
              "warning": "Not production Athena octet/root RAhP or collapse evidence",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print("q={:g} refine={} leaf={} constant={:.2e} pulse={:.3g} "
              "actual={:.3g}".format(row["q"], row["refined_interval"],
                                      row["leaf_cells"],
                                      row["modes"]["constant"]["relative_l2"],
                                      row["modes"]["pulse"]["relative_l2"],
                                      row["modes"]["restricted_exact_direction"]
                                      ["relative_l2"]))
    assert all(row["fine_constant_row_error"] < 1e-10
               and row["root_constant_row_error"] < 1e-10
               and row["modes"]["constant"]["relative_l2"] < 1e-10
               for row in rows)


if __name__ == "__main__":
    main()
