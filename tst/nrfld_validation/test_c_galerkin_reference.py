#!/usr/bin/env python3
"""Compare rediscretized and Galerkin MG hierarchies for uniform Test C.

SciPy is used only in this offline reference.  No production solver path is
changed.  The fine operator and R/P are identical between the two runs; only
the coarse matrices differ.
"""
import argparse
import json
from pathlib import Path

import numpy as np
try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("test_c_galerkin_reference.py requires scipy") from exc

from test_c_coarse_operator_reference import (
    FINE_SHAPE, fine_coefficients, initial_profile, restrict_coefficients)


def flat_index(i, j, k, shape):
    return (i*shape[1] + j)*shape[2] + k


def stencil_matrix(coeff):
    shape = coeff[0].shape
    ii, jj, kk = np.indices(shape)
    center = flat_index(ii, jj, kk, shape).ravel()
    rows, columns, values = [], [], []
    diagonal = np.ones(shape)
    for axis, qface in enumerate(coeff):
        diagonal += qface + np.roll(qface, 1, axis)
        for sign, qside in ((1, qface), (-1, np.roll(qface, 1, axis))):
            neighbor = [ii, jj, kk]
            neighbor[axis] = (neighbor[axis] + sign) % shape[axis]
            rows.append(center)
            columns.append(flat_index(*neighbor, shape).ravel())
            values.append((-qside).ravel())
    rows.append(center)
    columns.append(center)
    values.append(diagonal.ravel())
    return coo_matrix((np.concatenate(values),
                       (np.concatenate(rows), np.concatenate(columns))),
                      shape=(center.size, center.size)).tocsr()


def transfer_matrices(coarse_shape):
    fine_shape = tuple(2*n for n in coarse_shape)
    n_fine, n_coarse = np.prod(fine_shape), np.prod(coarse_shape)
    rows, columns, values = [], [], []
    for fi in range(fine_shape[0]):
        oi = ((fi//2, 0.75),
              ((fi//2 + (-1 if fi % 2 == 0 else 1)) % coarse_shape[0], 0.25))
        for fj in range(fine_shape[1]):
            oj = ((fj//2, 0.75),
                  ((fj//2 + (-1 if fj % 2 == 0 else 1)) % coarse_shape[1], 0.25))
            for fk in range(fine_shape[2]):
                ok = ((fk//2, 0.75),
                      ((fk//2 + (-1 if fk % 2 == 0 else 1)) % coarse_shape[2],
                       0.25))
                row = flat_index(fi, fj, fk, fine_shape)
                for ci, wi in oi:
                    for cj, wj in oj:
                        for ck, wk in ok:
                            rows.append(row)
                            columns.append(flat_index(ci, cj, ck, coarse_shape))
                            values.append(wi*wj*wk)
    prolong = coo_matrix((values, (rows, columns)),
                         shape=(n_fine, n_coarse)).tocsr()

    rows, columns, values = [], [], []
    for ci in range(coarse_shape[0]):
        for cj in range(coarse_shape[1]):
            for ck in range(coarse_shape[2]):
                row = flat_index(ci, cj, ck, coarse_shape)
                for di in (0, 1):
                    for dj in (0, 1):
                        for dk in (0, 1):
                            rows.append(row)
                            columns.append(flat_index(
                                2*ci+di, 2*cj+dj, 2*ck+dk, fine_shape))
                            values.append(0.125)
    restrict = coo_matrix((values, (rows, columns)),
                          shape=(n_coarse, n_fine)).tocsr()
    return restrict, prolong


def rb_smooth(u, rhs, matrix, shape, pairs=2):
    parity = (sum(np.indices(shape)) & 1).ravel()
    diagonal = matrix.diagonal()
    for _ in range(pairs):
        for color in (0, 1):
            correction = (rhs - matrix @ u)/diagonal
            mask = parity == color
            u[mask] += correction[mask]
    return u


def spectral_summary(vector, shape):
    """Compact description of the residual frequencies on one MG level."""
    field = vector.reshape(shape)
    spectrum = np.abs(np.fft.fftn(field))**2
    frequency = [np.abs(np.fft.fftfreq(n)) for n in shape]
    high_mask = np.zeros(shape, dtype=bool)
    for axis, component in enumerate(frequency):
        axis_shape = [1, 1, 1]
        axis_shape[axis] = shape[axis]
        high_mask |= component.reshape(axis_shape) >= 0.25
    total_power = float(np.sum(spectrum))
    checker = (-1.0)**sum(np.indices(shape))
    denominator = np.linalg.norm(field)*np.linalg.norm(checker)
    return {
        "high_frequency_power_fraction": (
            float(np.sum(spectrum[high_mask])/total_power)
            if total_power > 0.0 else 0.0),
        "checkerboard_correlation": (
            float(abs(np.vdot(field, checker))/denominator)
            if denominator > 0.0 else 0.0),
    }


def vcycle(u, rhs, matrices, shapes, restrict, prolong, level=0, metrics=None):
    matrix = matrices[level]
    if level == len(matrices)-1:
        solution = spsolve(matrix, rhs)
        if metrics is not None:
            residual = rhs - matrix @ solution
            metrics["levels"].append({
                "level": level,
                "shape": list(shapes[level]),
                "kind": "direct",
                "rhs_norm": float(np.linalg.norm(rhs)),
                "residual_norm": float(np.linalg.norm(residual)),
                "rhs_spectrum": spectral_summary(rhs, shapes[level]),
            })
        return solution
    before = np.linalg.norm(rhs - matrix @ u)
    rb_smooth(u, rhs, matrix, shapes[level])
    after_pre = np.linalg.norm(rhs - matrix @ u)
    coarse_rhs = restrict[level] @ (rhs - matrix @ u)
    correction = vcycle(np.zeros_like(coarse_rhs), coarse_rhs, matrices,
                        shapes, restrict, prolong, level+1, metrics)
    coarse_after = np.linalg.norm(coarse_rhs - matrices[level+1] @ correction)
    u += prolong[level] @ correction
    after_prolong = np.linalg.norm(rhs - matrix @ u)
    rb_smooth(u, rhs, matrix, shapes[level])
    after_post = np.linalg.norm(rhs - matrix @ u)
    if metrics is not None:
        metrics["levels"].append({
            "level": level,
            "shape": list(shapes[level]),
            "kind": "vcycle",
            "before": float(before),
            "after_pre": float(after_pre),
            "pre_factor": float(after_pre/before),
            "coarse_rhs": float(np.linalg.norm(coarse_rhs)),
            "coarse_after": float(coarse_after),
            "coarse_solve_factor": float(
                coarse_after/max(np.linalg.norm(coarse_rhs), 1.0e-300)),
            "after_prolong": float(after_prolong),
            "prolong_factor": float(after_prolong/max(after_pre, 1.0e-300)),
            "after_post": float(after_post),
            "post_factor": float(after_post/max(after_prolong, 1.0e-300)),
            "full_factor": float(after_post/before),
            "fine_residual_spectrum": spectral_summary(
                rhs - matrix @ u, shapes[level]),
            "coarse_rhs_spectrum": spectral_summary(
                coarse_rhs, shapes[level+1]),
        })
    if level == 0:
        metrics.update(before_pre=float(before), after_pre=float(after_pre),
                       coarse_rhs=float(np.linalg.norm(coarse_rhs)),
                       coarse_after=float(coarse_after),
                       after_prolong=float(after_prolong),
                       after_post=float(after_post),
                       full_factor=float(after_post/before))
    return u


def evaluate(diffusion_number):
    coefficients = [fine_coefficients(diffusion_number)]
    while coefficients[-1][0].shape[0] > 8:
        coefficients.append(restrict_coefficients(coefficients[-1]))
    shapes = [entry[0].shape for entry in coefficients]
    rediscretized = [stencil_matrix(entry) for entry in coefficients]
    restrict, prolong = [], []
    for shape in shapes[1:]:
        r, p = transfer_matrices(shape)
        restrict.append(r)
        prolong.append(p)
    galerkin = [rediscretized[0]]
    for r, p in zip(restrict, prolong):
        galerkin.append((r @ galerkin[-1] @ p).tocsr())

    profile = initial_profile()
    initial = np.broadcast_to(profile[:, None, None], FINE_SHAPE).copy().ravel()
    rhs = initial - rediscretized[0] @ initial
    output = {}
    for name, hierarchy in (("rediscretized", rediscretized),
                            ("galerkin", galerkin)):
        metrics = {"levels": []}
        correction = vcycle(np.zeros_like(rhs), rhs, hierarchy, shapes,
                            restrict, prolong, metrics=metrics)
        # Both are judged against the same physical fine operator.
        metrics["fine_residual_factor"] = float(
            np.linalg.norm(rhs - rediscretized[0] @ correction)/np.linalg.norm(rhs))
        metrics["nonzeros_by_level"] = [matrix.nnz for matrix in hierarchy]
        output[name] = metrics
    prefixes = []
    for deepest_galerkin_level in range(len(rediscretized)):
        hierarchy = [rediscretized[0]] + [
            galerkin[level] if level <= deepest_galerkin_level
            else rediscretized[level]
            for level in range(1, len(rediscretized))]
        metrics = {"levels": []}
        correction = vcycle(np.zeros_like(rhs), rhs, hierarchy, shapes,
                            restrict, prolong, metrics=metrics)
        prefixes.append({
            "deepest_galerkin_level": deepest_galerkin_level,
            "fine_residual_factor": float(
                np.linalg.norm(rhs-rediscretized[0] @ correction)/np.linalg.norm(rhs)),
            "top_coarse_residual_factor": metrics["coarse_after"]/metrics["coarse_rhs"],
        })
    return {"diffusion_number": diffusion_number, "level_shapes": shapes,
            "hierarchies": output, "galerkin_prefix_controls": prefixes}


def run(args):
    results = [evaluate(q) for q in (5.0e5, 7.0e5)]
    report = {
        "scope": "offline uniform Test C, identical fine operator/R/P/smoother",
        "results": results,
        "warning": "SciPy reference only; Galerkin matrices are not a production change",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        red = row["hierarchies"]["rediscretized"]["fine_residual_factor"]
        gal = row["hierarchies"]["galerkin"]["fine_residual_factor"]
        print(f"q={row['diffusion_number']:g} rediscretized={red:.9g} "
              f"galerkin={gal:.9g}")
    # Guard the reproducer and the qualitative Galerkin control.
    return 0 if (results[0]["hierarchies"]["rediscretized"]["fine_residual_factor"] < 1
                 and results[1]["hierarchies"]["rediscretized"]["fine_residual_factor"] > 1
                 and results[1]["hierarchies"]["galerkin"]["fine_residual_factor"] < 1) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
