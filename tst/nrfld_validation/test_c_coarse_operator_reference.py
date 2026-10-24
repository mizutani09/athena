#!/usr/bin/env python3
"""Independent uniform-grid MG reference for variable-limiter Test C.

This reproduces the first linear V-cycle without importing Athena++ code.  It
also compares the rediscretized coarse operator A_H with R A_h P.  The model is
deliberately limited to the periodic, radiation-only, uniform-fine Test C
geometry; it is a diagnostic and not a replacement solver.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


C_CODE = 2.99792458e3
DT = 2.0e-4
FINE_SHAPE = (128, 16, 16)
DX = 1.0 / FINE_SHAPE[0]


def limiter(r):
    return (2.0 + r) / (6.0 + 3.0*r + r*r)


def initial_profile():
    x = (np.arange(FINE_SHAPE[0]) + 0.5) * DX
    return 1.0 + 0.1*np.exp(-0.5*((x - 0.5)/0.04)**2)


def fine_coefficients(diffusion_number):
    """Return dimensionless dt*D_face/dx^2 coefficients in x/y/z."""
    e = initial_profile()
    sigma = DT*C_CODE/(3.0*DX*DX*diffusion_number)
    eface = 0.5*(e + np.roll(e, -1))
    grad_face = np.abs(np.roll(e, -1) - e)/DX
    qx = DT*C_CODE*limiter(grad_face/(sigma*eface))/(sigma*DX*DX)

    # The y/z face closure still contains the x component of |grad E|.  Since
    # E is y/z independent, this is the centered x gradient at the cell.
    grad_transverse_face = np.abs(np.roll(e, -1) - np.roll(e, 1))/(2.0*DX)
    qt = DT*C_CODE*limiter(grad_transverse_face/(sigma*e))/(sigma*DX*DX)
    transverse = np.ones((1, FINE_SHAPE[1], FINE_SHAPE[2]))
    return (qx[:, None, None]*transverse,
            qt[:, None, None]*transverse,
            qt[:, None, None]*transverse)


def apply_operator(u, coeff):
    qx, qy, qz = coeff
    return ((1.0 + qx + np.roll(qx, 1, 0)
            + qy + np.roll(qy, 1, 1) + qz + np.roll(qz, 1, 2))*u
            - qx*np.roll(u, -1, 0) - np.roll(qx, 1, 0)*np.roll(u, 1, 0)
            - qy*np.roll(u, -1, 1) - np.roll(qy, 1, 1)*np.roll(u, 1, 1)
            - qz*np.roll(u, -1, 2) - np.roll(qz, 1, 2)*np.roll(u, 1, 2))


def l2(a):
    return float(np.sqrt(np.mean(a*a)))


def restrict(a):
    return sum(a[di::2, dj::2, dk::2]
               for di in (0, 1) for dj in (0, 1) for dk in (0, 1))/8.0


def restrict_coefficients(coeff):
    qx, qy, qz = coeff
    # Athena++ area-averages the external fine faces.  The additional /4 is
    # the change in dt/dx^2 when dx doubles.
    return (sum(qx[1::2, dj::2, dk::2]
                for dj in (0, 1) for dk in (0, 1))/16.0,
            sum(qy[di::2, 1::2, dk::2]
                for di in (0, 1) for dk in (0, 1))/16.0,
            sum(qz[di::2, dj::2, 1::2]
                for di in (0, 1) for dj in (0, 1))/16.0)


def prolong(a):
    """Periodic cell-centred trilinear interpolation (27/9/3/1 rule)."""
    for axis in range(3):
        shape = list(a.shape)
        shape[axis] *= 2
        out = np.empty(shape)
        sl = [slice(None)]*3
        sl[axis] = slice(0, None, 2)
        out[tuple(sl)] = 0.75*a + 0.25*np.roll(a, 1, axis=axis)
        sl[axis] = slice(1, None, 2)
        out[tuple(sl)] = 0.75*a + 0.25*np.roll(a, -1, axis=axis)
        a = out
    return a


def prolong_minmod(a):
    """Limited prolongation applied directly to a coarse-grid correction.

    This mirrors the uniform-grid geometry of MeshRefinement's cell-centred
    minmod reconstruction.  It is a diagnostic control, not a linear transfer
    operator: the selected slope depends on ``a`` itself.
    """
    slopes = []
    for axis in range(3):
        left = a - np.roll(a, 1, axis=axis)
        right = np.roll(a, -1, axis=axis) - a
        same_sign = left*right > 0.0
        slopes.append(np.where(
            same_sign,
            np.sign(left)*np.minimum(np.abs(left), np.abs(right)),
            0.0))

    out = np.repeat(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1), 2, axis=2)
    for axis, slope in enumerate(slopes):
        expanded = np.repeat(np.repeat(np.repeat(slope, 2, axis=0), 2, axis=1),
                             2, axis=2)
        signs = np.where(np.arange(out.shape[axis]) % 2 == 0, -0.25, 0.25)
        shape = [1, 1, 1]
        shape[axis] = out.shape[axis]
        out += expanded*signs.reshape(shape)
    return out


def rb_smooth(u, rhs, coeff, pairs=2, first_color=0):
    indices = np.indices(u.shape)
    parity = (indices[0] + indices[1] + indices[2]) & 1
    diag = 1.0 + sum(q + np.roll(q, 1, axis)
                     for axis, q in enumerate(coeff))
    for _ in range(pairs):
        for color in (first_color, 1-first_color):
            correction = (rhs - apply_operator(u, coeff))/diag
            mask = parity == color
            u[mask] += correction[mask]
    return u


def direct_solve(rhs, coeff):
    shape = rhs.shape
    matrix = np.empty((rhs.size, rhs.size))
    for column in range(rhs.size):
        basis = np.zeros(shape)
        basis.flat[column] = 1.0
        matrix[:, column] = apply_operator(basis, coeff).ravel()
    return np.linalg.solve(matrix, rhs.ravel()).reshape(shape)


def vcycle(u, rhs, hierarchy, level=0, top_metrics=None, operator_levels=None,
           prolongation=prolong, pre_first_color=0, post_first_color=0):
    if level == len(hierarchy)-1:
        u[:] = direct_solve(rhs, hierarchy[level])
        return u

    before = l2(rhs - apply_operator(u, hierarchy[level]))
    rb_smooth(u, rhs, hierarchy[level], first_color=pre_first_color)
    fine_residual = rhs - apply_operator(u, hierarchy[level])
    after_pre = l2(fine_residual)
    coarse_rhs = restrict(fine_residual)
    if operator_levels is not None:
        operator_levels.append({
            "fine_shape": list(hierarchy[level][0].shape),
            "coarse_shape": list(hierarchy[level+1][0].shape),
            "path": ("meshblock-local" if level < len(hierarchy)-2 else "root"),
            "modes": operator_comparison(
                hierarchy[level], hierarchy[level+1], coarse_rhs),
        })
    coarse_correction = np.zeros_like(coarse_rhs)
    vcycle(coarse_correction, coarse_rhs, hierarchy, level+1,
           operator_levels=operator_levels, prolongation=prolongation,
           pre_first_color=pre_first_color, post_first_color=post_first_color)
    coarse_before_prolong = l2(
        coarse_rhs - apply_operator(coarse_correction, hierarchy[level+1]))
    u += prolongation(coarse_correction)
    after_prolong = l2(rhs - apply_operator(u, hierarchy[level]))
    rb_smooth(u, rhs, hierarchy[level], first_color=post_first_color)
    after_post = l2(rhs - apply_operator(u, hierarchy[level]))
    if level == 0:
        top_metrics.update(before_pre=before,
                           after_pre_before_restrict=after_pre,
                           coarse_after_restrict=l2(coarse_rhs),
                           coarse_before_prolong=coarse_before_prolong,
                           after_prolong_before_post=after_prolong,
                           after_post=after_post,
                           full_factor=after_post/before)
    return u


def operator_comparison(fine_coeff, coarse_coeff, restricted_residual):
    shape = coarse_coeff[0].shape
    ix = np.arange(shape[0])
    modes = {
        "constant": np.ones(shape),
        "smooth_x": np.broadcast_to(
            np.sin(2.0*np.pi*(ix + 0.5)/shape[0])[:, None, None], shape).copy(),
        "pulse": np.broadcast_to(
            np.exp(-0.5*(((ix + 0.5)/shape[0] - 0.5)/0.04)**2)[:, None, None],
            shape).copy(),
        "checkerboard": (-1.0)**sum(np.indices(shape)),
        "restricted_residual": restricted_residual,
    }
    result = {}
    for name, vector in modes.items():
        ah = apply_operator(vector, coarse_coeff)
        rahp = restrict(apply_operator(prolong(vector), fine_coeff))
        denominator = max(np.linalg.norm(ah), np.linalg.norm(rahp), 1e-300)
        result[name] = {
            "relative_difference": float(np.linalg.norm(ah-rahp)/denominator),
            "ah_l2": l2(ah),
            "rahp_l2": l2(rahp),
        }
    return result


TRANSFER_RE = re.compile(r"^\[NRFLD MG transfer\].*?before_pre=(\S+)"
                         r".*?after_pre_before_restrict=(\S+)"
                         r".*?coarse_after_restrict=(\S+)"
                         r".*?coarse_before_prolong=(\S+)"
                         r".*?after_prolong_before_post=(\S+)"
                         r".*?after_post=(\S+).*?full_factor=(\S+)")


def measured_transfer(audit, diffusion_number):
    tag = f"q{diffusion_number:g}_contrast1_uniform_fine"
    log = audit / "variable_off" / tag / "run.log"
    for line in log.read_text().splitlines():
        match = TRANSFER_RE.match(line)
        if match:
            keys = ("before_pre", "after_pre_before_restrict",
                    "coarse_after_restrict", "coarse_before_prolong",
                    "after_prolong_before_post", "after_post", "full_factor")
            return dict(zip(keys, map(float, match.groups())))
    raise RuntimeError(f"first MG transfer record not found in {log}")


def evaluate(diffusion_number, audit):
    hierarchy = [fine_coefficients(diffusion_number)]
    while hierarchy[-1][0].shape[0] > 8:
        hierarchy.append(restrict_coefficients(hierarchy[-1]))
    profile = initial_profile()
    initial = np.broadcast_to(profile[:, None, None], FINE_SHAPE).copy()
    rhs = initial - apply_operator(initial, hierarchy[0])

    metrics = {}
    operator_levels = []
    vcycle(np.zeros_like(rhs), rhs, hierarchy, top_metrics=metrics,
           operator_levels=operator_levels)
    limited_metrics = {}
    vcycle(np.zeros_like(rhs), rhs, hierarchy, top_metrics=limited_metrics,
           prolongation=prolong_minmod)
    reversed_post_metrics = {}
    vcycle(np.zeros_like(rhs), rhs, hierarchy, top_metrics=reversed_post_metrics,
           pre_first_color=0, post_first_color=1)
    measured = measured_transfer(audit, diffusion_number) if audit else None
    differences = ({key: abs(metrics[key]-measured[key]) for key in metrics}
                   if measured else {})
    matched = (not measured or all(np.isclose(metrics[key], measured[key],
                                               rtol=2e-10, atol=2e-10)
                                  for key in metrics))
    return {
        "diffusion_number": diffusion_number,
        "level_shapes": [list(level[0].shape) for level in hierarchy],
        "fine_face_coefficient_min": float(hierarchy[0][0].min()),
        "fine_face_coefficient_max": float(hierarchy[0][0].max()),
        "reference_transfer": metrics,
        "minmod_correction_transfer": limited_metrics,
        "reversed_post_rb_transfer": reversed_post_metrics,
        "athena_transfer": measured,
        "absolute_differences": differences,
        "athena_match": bool(matched),
        "ah_vs_rahp_by_coarsening": operator_levels,
    }


def run(args):
    results = [evaluate(q, args.audit) for q in (5.0e5, 7.0e5)]
    report = {
        "scope": "variable-limiter periodic uniform-fine Test C first V-cycle",
        "results": results,
        "interpretation": (
            "The uniform reference contains no AMR or MeshBlock boundaries.  "
            "A reproduced divergent q=7e5 V-cycle therefore cannot be caused "
            "by an AMR transfer boundary.  A_H-versus-RAP is evaluated, not "
            "silently assumed equivalent."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        transfer = row["reference_transfer"]
        limited = row["minmod_correction_transfer"]
        reversed_post = row["reversed_post_rb_transfer"]
        relevant = row["ah_vs_rahp_by_coarsening"][0]["modes"]["restricted_residual"]
        print(f"q={row['diffusion_number']:g} "
              f"pre_factor={transfer['after_pre_before_restrict']/transfer['before_pre']:.9g} "
              f"vcycle_factor={transfer['full_factor']:.9g} "
              f"minmod_correction_factor={limited['full_factor']:.9g} "
              f"reversed_post_factor={reversed_post['full_factor']:.9g} "
              f"A_H_vs_RAP(actual)={relevant['relative_difference']:.9g} "
              f"athena_match={row['athena_match']}")
    return 0 if all(row["athena_match"] for row in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    raise SystemExit(run(parser.parse_args()))
