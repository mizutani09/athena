#!/usr/bin/env python3
"""Independent 1D composite-operator reference for the x-only fixed-lambda Test C.

No MG cycles/coarsening, Newton globalization, hydro or thermal exchange.
This does not replace Athena's solver. Agreement with its measured first
direction must be checked before extending the interpretation to 3D collapse.
"""
import argparse
import json
from pathlib import Path

import numpy as np


class CompositeLine:
    def __init__(self, left, right, root_n=64):
        edges = [0.0]
        for i in range(root_n):
            if left <= i / root_n < right:
                edges.extend([(i + .5) / root_n, (i + 1) / root_n])
            else:
                edges.append((i + 1) / root_n)
        self.edges = np.asarray(edges)
        self.dx = np.diff(self.edges)
        self.x = .5 * (self.edges[1:] + self.edges[:-1])
        self.n = len(self.dx)
        self.eye = np.eye(self.n)

    def average_weights(self, a, b):
        return sum(np.maximum(0, np.minimum(self.edges[1:] + shift, b)
                              - np.maximum(self.edges[:-1] + shift, a))
                   for shift in (-1, 0, 1)) / (b - a)

    def ghost(self, u, i, sign, kind):
        j = (i + sign) % self.n
        if self.dx[j] <= self.dx[i]:
            a = self.edges[i + 1] if sign == 1 else self.edges[i] - self.dx[i]
            return self.average_weights(a, a + self.dx[i]) @ u
        xc, h = self.x[j], self.dx[j]
        ul = self.average_weights(xc - 1.5*h, xc - .5*h) @ u
        ur = self.average_weights(xc + .5*h, xc + 1.5*h) @ u
        if kind == "mg":
            # x-only reduction of MGBoundaryValues' 27/9/3/1 trilinear rule:
            # .75 coarse center + .25 coarse neighbor toward the fine region.
            return .75*u[j] + .25*(ul if sign == 1 else ur)
        dl, dr = u[j] - ul, ur - u[j]
        slope = 0.0 if dl*dr <= 0 else np.sign(dl)*min(abs(dl), abs(dr))
        return u[j] - sign*.25*slope

    def ghost_tangent(self, base, direction, i, sign):
        """Directional derivative of the NR minmod ghost reconstruction.

        Branch ties are assigned to the right slope.  The diagnostic profiles
        avoid relying on derivatives exactly at a minmod switching point.
        """
        j = (i + sign) % self.n
        if self.dx[j] <= self.dx[i]:
            a = self.edges[i + 1] if sign == 1 else self.edges[i] - self.dx[i]
            return self.average_weights(a, a + self.dx[i]) @ direction
        xc, h = self.x[j], self.dx[j]
        wl = self.average_weights(xc - 1.5*h, xc - .5*h)
        wr = self.average_weights(xc + .5*h, xc + 1.5*h)
        ul, ur = wl @ base, wr @ base
        dl, dr = base[j] - ul, ur - base[j]
        if dl*dr <= 0:
            dslope = 0.0
        elif abs(dl) < abs(dr):
            dslope = direction[j] - wl @ direction
        else:
            dslope = wr @ direction - direction[j]
        return direction[j] - sign*.25*dslope

    def laplacian(self, u, kind):
        # Difference form avoids cancellation of 2*u for nearly constant data.
        return np.asarray([((self.ghost(u, i, -1, kind)-u[i])
                            +(self.ghost(u, i, 1, kind)-u[i])) / self.dx[i]**2
                           for i in range(self.n)])

    def matrix_mg(self):
        return np.column_stack([self.laplacian(v, "mg") for v in self.eye])

    def laplacian_tangent(self, base, direction):
        return np.asarray([
            ((self.ghost_tangent(base, direction, i, -1)-direction[i])
             +(self.ghost_tangent(base, direction, i, 1)-direction[i]))
            / self.dx[i]**2 for i in range(self.n)])

    def matrix_tangent(self, base):
        return np.column_stack([self.laplacian_tangent(base, v)
                                for v in self.eye])

    def matrix_conservative(self):
        lap = np.zeros((self.n, self.n))
        for i in range(self.n):
            j = (i + 1) % self.n
            conductance = 2 / (self.dx[i] + self.dx[j])
            # Same face flux, opposite sign, divided by each cell volume.
            lap[i, j] += conductance/self.dx[i]
            lap[i, i] -= conductance/self.dx[i]
            lap[j, i] += conductance/self.dx[j]
            lap[j, j] -= conductance/self.dx[j]
        return lap


def evaluate(left, right, q):
    line = CompositeLine(left, right)
    u = 1 + .1*np.exp(-.5*((line.x - .5)/.04)**2)
    tau = q / 128**2  # dt*D, q defined on the finest cells
    lmg, lfv = line.matrix_mg(), line.matrix_conservative()
    amg, afv = line.eye - tau*lmg, line.eye - tau*lfv
    rhs = tau*line.laplacian(u, "nr")
    # Athena's first correction: NR residual uses minmod state ghosts, MG's
    # linear solve uses its trilinear ghosts. This is the mixed-operator test.
    delta = np.linalg.solve(amg, rhs)
    # The consistent Newton linearization uses the tangent of the state
    # reconstruction at u.  This differs from applying minmod directly to the
    # correction, because in general G(delta) != G'(u) delta.
    atangent = line.eye - tau*line.matrix_tangent(u)
    tangent_delta = np.linalg.solve(atangent, rhs)

    # Diagnostic only: solve the piecewise-linear equation obtained by applying
    # minmod to the correction itself.  This is not a fixed linear MG operator.
    limited_delta = delta.copy()
    limited_active_set_converged = False
    for _ in range(20):
        alimited = line.eye - tau*line.matrix_tangent(limited_delta)
        next_delta = np.linalg.solve(alimited, rhs)
        limited_residual = (next_delta
                            - tau*line.laplacian(next_delta, "nr") - rhs)
        limited_delta = next_delta
        if np.linalg.norm(limited_residual) <= 1e-10*max(np.linalg.norm(rhs), 1.0):
            limited_active_set_converged = True
            break
    # Two controls: consistent MG operator (not necessarily volume conservative)
    # and a conservative nonuniform finite-volume operator.
    consistent = np.linalg.solve(amg, u)
    conservative = np.linalg.solve(afv, u)
    average = lambda v: float(line.dx @ v)
    result = dict(left=left, right=right, q=q, cells=line.n,
                  rhs_mean=average(rhs), delta_mean=average(delta),
                  delta_min=float(delta.min()), delta_max=float(delta.max()),
                  full_step_min=float((u+delta).min()),
                  full_step_max=float((u+delta).max()),
                  nonpositive=int(np.sum(u+delta <= 0)),
                  direct_linear_rel=float(np.linalg.norm(rhs-amg@delta)/np.linalg.norm(rhs)),
                  tangent_delta_min=float(tangent_delta.min()),
                  tangent_delta_max=float(tangent_delta.max()),
                  tangent_full_step_min=float((u+tangent_delta).min()),
                  tangent_linear_rel=float(
                      np.linalg.norm(rhs-atangent@tangent_delta)/np.linalg.norm(rhs)),
                  limited_delta_min=float(limited_delta.min()),
                  limited_delta_max=float(limited_delta.max()),
                  limited_full_step_min=float((u+limited_delta).min()),
                  limited_nonlinear_rel=float(
                      np.linalg.norm(limited_delta
                                     - tau*line.laplacian(limited_delta, "nr") - rhs)
                      / np.linalg.norm(rhs)),
                  limited_active_set_converged=limited_active_set_converged,
                  mg_constant_error=float(np.max(abs(lmg @ np.ones(line.n)))),
                  mg_volume_left_null_error=float(np.max(abs(line.dx @ lmg))),
                  fv_volume_left_null_error=float(np.max(abs(line.dx @ lfv))),
                  consistent_mg_min=float(consistent.min()),
                  consistent_mg_max=float(consistent.max()),
                  consistent_mg_energy_change=average(consistent-u)/average(u),
                  conservative_min=float(conservative.min()),
                  conservative_max=float(conservative.max()),
                  conservative_energy_change=average(conservative-u)/average(u))
    result["reference_controls_pass"] = bool(
        result["mg_constant_error"] < 1e-10
        and result["fv_volume_left_null_error"] < 1e-10
        and abs(result["conservative_energy_change"]) < 1e-8
        and consistent.min() >= u.min()-1e-8 and consistent.max() <= u.max()+1e-8
        and conservative.min() >= u.min()-1e-8 and conservative.max() <= u.max()+1e-8)
    return result


def run(args):
    results = [evaluate(lo, hi, q) for q in (1e4, 1e7)
               for lo, hi in ((0., 1.), (.25, .75), (.125, .625))]
    comparison = []
    if args.audit:
        measured = json.loads((args.audit / "fixed_on/manifest.json").read_text())
        for result, sample in zip(results, measured["samples"]):
            assert result["q"] == sample["diffusion_number_fine_low"]
            row = next(x for x in sample["handoff"] if x["solve"] == "0"
                       and x["stage"] == "mg_end" and x["region"] == "all")
            checks = {}
            for key in ("rhs_mean", "delta_mean", "delta_min", "delta_max", "full_step_min"):
                checks[key] = abs(result[key]-float(row[key]))
            # The direct dense solve has O(q) conditioning; use a mixed
            # absolute/relative comparison, not a roundoff-level assertion.
            passed = all(np.isclose(result[k], float(row[k]), atol=2e-7, rtol=2e-7)
                         for k in checks)
            comparison.append(dict(tag=sample["tag"], errors=checks, passed=bool(passed)))
    report = dict(scope="fixed-lambda, contrast=1, periodic x-only Test C first solve",
                  results=results, athena_comparison=comparison,
                  warning="A reference calculation, not a production solver change")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    for row in results:
        print(f"q={row['q']:g} refine=[{row['left']},{row['right']}] "
              f"rhs_mean={row['rhs_mean']:.9g} delta_mean={row['delta_mean']:.9g} "
              f"updated_min={row['full_step_min']:.9g}")
    if comparison:
        print("ATHENA_FIRST_DIRECTION_MATCH=", all(x["passed"] for x in comparison))
    return 0 if (all(x["passed"] for x in comparison)
                 and all(x["reference_controls_pass"] for x in results)) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    raise SystemExit(run(parser.parse_args()))
