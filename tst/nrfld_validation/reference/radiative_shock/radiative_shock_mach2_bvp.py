#!/usr/bin/env python3
"""Experimental Mach 2 semi-analytic radiative-shock profile by BVP relaxation.

This is a Mach-2-specific alternative to the generic shooting script.  The
shooting formulation is fragile because bad trial states can drive the ODE into
a nearly singular region before both far states are matched.  Here the two
smooth branches are solved simultaneously with solve_bvp, while the momentum
and total energy fluxes are allowed to relax by small amounts from the rounded
tabulated far states.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from tqdm.auto import tqdm


@dataclass(frozen=True)
class State:
    rho: float
    temp: float
    vel: float


LEFT = State(rho=5.45887e-13, temp=100.0, vel=2.35435e5)
RIGHT = State(rho=1.24794e-12, temp=207.757, vel=1.02987e5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an experimental Mach 2 radiative-shock BVP profile."
    )
    parser.add_argument("--output", type=Path, default=Path("mach2_semianalytic_bvp.csv"))
    parser.add_argument("--plot", type=Path, default=Path("mach2_semianalytic_bvp.png"))
    parser.add_argument("--x-min", type=float, default=-1000.0)
    parser.add_argument("--x-max", type=float, default=500.0)
    parser.add_argument("--gamma", type=float, default=5.0/3.0)
    parser.add_argument("--rgas", type=float, default=8.31451e7)
    parser.add_argument("--a-rad", type=float, default=7.5657e-15)
    parser.add_argument("--c-light", type=float, default=2.99792458e10)
    parser.add_argument("--kappa-p", type=float, default=3.92664e-5)
    parser.add_argument("--chi-r", type=float, default=0.848902)
    parser.add_argument("--tol", type=float, default=3.0e-3)
    parser.add_argument("--bc-tol", type=float, default=3.0e-5)
    parser.add_argument("--nodes", type=int, default=240)
    parser.add_argument("--max-nodes", type=int, default=30000)
    parser.add_argument("--max-tries", type=int, default=8)
    parser.add_argument("--accept-bc", type=float, default=1.0e-4)
    parser.add_argument(
        "--shock-epsilon",
        type=float,
        default=1.0e-6,
        help="Small x offset used to write pre/post-shock states instead of duplicate x=0 rows.",
    )
    parser.add_argument(
        "--shock-refine-width",
        type=float,
        default=30.0,
        help="Width in cm around the embedded shock sampled with extra output points.",
    )
    parser.add_argument(
        "--shock-refine-points",
        type=int,
        default=600,
        help="Number of extra output points per branch near the embedded shock.",
    )
    parser.add_argument(
        "--flux-relax",
        type=float,
        default=3.0e-3,
        help="Relative scale used for BVP momentum/energy flux corrections.",
    )
    parser.add_argument("--allow-bad-profile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def equilibrium_fluxes(state: State, gamma: float, rgas: float, a_rad: float):
    rho, temp, vel = state.rho, state.temp, state.vel
    pgas = rho*rgas*temp
    erad = a_rad*temp**4
    mflux = rho*vel
    momflux = rho*vel*vel + pgas + erad/3.0
    eflux = mflux*(0.5*vel*vel + gamma/(gamma - 1.0)*pgas/rho)
    eflux += (4.0/3.0)*vel*erad
    return mflux, momflux, eflux


class Mach2BVP:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gm1 = args.gamma - 1.0
        lf = equilibrium_fluxes(LEFT, args.gamma, args.rgas, args.a_rad)
        rf = equilibrium_fluxes(RIGHT, args.gamma, args.rgas, args.a_rad)
        self.mflux = 0.5*(lf[0] + rf[0])
        self.mom0 = 0.5*(lf[1] + rf[1])
        self.ef0 = 0.5*(lf[2] + rf[2])
        print("Far-state flux consistency check:")
        for name, lv, rv in zip(("mass", "momentum", "energy"), lf, rf):
            rel = abs(lv - rv)/max(abs(lv), abs(rv), 1.0e-300)
            print(f"  {name:8s}: left={lv:.16e} right={rv:.16e} rel_diff={rel:.3e}")
        print(f"Using initial averaged fluxes: m={self.mflux:.16e}, "
              f"mom={self.mom0:.16e}, energy={self.ef0:.16e}")

    def primitive(self, u: np.ndarray, erad: np.ndarray, momflux: float):
        rho = self.mflux/u
        pgas = momflux - self.mflux*u - erad/3.0
        temp = pgas/(rho*self.args.rgas)
        return rho, pgas, temp

    def rhs_single(self, u: np.ndarray, erad: np.ndarray, momflux: float, eflux: float):
        rho, pgas, temp = self.primitive(u, erad, momflux)
        bad = (u <= 0.0) | (erad <= 0.0) | (rho <= 0.0) | (pgas <= 0.0) | (temp <= 0.0)
        temp = np.where(bad, 1.0, temp)
        pgas = np.where(bad, 1.0e100, pgas)
        gas_enthalpy_flux = 0.5*self.mflux*u*u + self.args.gamma/self.gm1*pgas*u
        frad = eflux - gas_enthalpy_flux - (4.0/3.0)*u*erad
        derdx = -(3.0*self.args.chi_r/self.args.c_light)*frad
        dgdu = self.mflux*u + self.args.gamma/self.gm1*(pgas - self.mflux*u)
        dgder = -self.args.gamma*u/(3.0*self.gm1)
        exchange = -self.args.c_light*self.args.kappa_p*(self.args.a_rad*temp**4 - erad)
        dudx = (exchange - (dgder + u/3.0)*derdx)/dgdu
        return np.where(bad, 1.0e50, dudx), np.where(bad, 1.0e50, derdx)

    def frad(self, u: float, erad: float, momflux: float, eflux: float) -> float:
        _, pgas, _ = self.primitive(np.asarray(u), np.asarray(erad), momflux)
        gas_enthalpy_flux = 0.5*self.mflux*u*u + self.args.gamma/self.gm1*float(pgas)*u
        return eflux - gas_enthalpy_flux - (4.0/3.0)*u*erad

    def unpack_flux_params(self, p: np.ndarray) -> tuple[float, float]:
        momflux = self.mom0*(1.0 + self.args.flux_relax*p[0])
        eflux = self.ef0*(1.0 + self.args.flux_relax*p[1])
        return momflux, eflux

    def fun(self, s: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        momflux, eflux = self.unpack_flux_params(p)
        out = np.zeros_like(y)
        du, de = self.rhs_single(y[0], y[1], momflux, eflux)
        out[0] = du*(-self.args.x_min)
        out[1] = de*(-self.args.x_min)
        du, de = self.rhs_single(y[2], y[3], momflux, eflux)
        out[2] = du*self.args.x_max
        out[3] = de*self.args.x_max
        return out

    def bc(self, ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        momflux, eflux = self.unpack_flux_params(p)
        left_er = self.args.a_rad*LEFT.temp**4
        right_er = self.args.a_rad*RIGHT.temp**4
        return np.asarray([
            (ya[0] - LEFT.vel)/LEFT.vel,
            math.log(max(ya[1], 1.0e-300)/left_er),
            (yb[2] - RIGHT.vel)/RIGHT.vel,
            math.log(max(yb[3], 1.0e-300)/right_er),
            math.log(max(yb[1], 1.0e-300)/max(ya[3], 1.0e-300)),
            (self.frad(float(yb[0]), float(yb[1]), momflux, eflux)
             - self.frad(float(ya[2]), float(ya[3]), momflux, eflux))
            / max(abs(eflux), 1.0),
        ])

    def initial_guess(self, factor: float):
        s = np.linspace(0.0, 1.0, self.args.nodes)
        left_er = self.args.a_rad*LEFT.temp**4
        right_er = self.args.a_rad*RIGHT.temp**4
        shock_er = left_er*(right_er/left_er)**factor
        u_minus = LEFT.vel*(1.0 - 0.08*factor)
        u_plus = RIGHT.vel*(1.0 + 0.18*factor)
        y = np.vstack([
            np.linspace(LEFT.vel, u_minus, s.size),
            np.exp(np.linspace(math.log(left_er), math.log(shock_er), s.size)),
            np.linspace(u_plus, RIGHT.vel, s.size),
            np.exp(np.linspace(math.log(shock_er), math.log(right_er), s.size)),
        ])
        return s, y, np.asarray([0.0, 0.0])

    def solve(self):
        factors = np.linspace(0.15, 0.95, self.args.max_tries)
        best = None
        for factor in tqdm(factors, desc="BVP tries", unit="try"):
            s, y, p0 = self.initial_guess(float(factor))
            sol = solve_bvp(self.fun, self.bc, s, y, p=p0, tol=self.args.tol,
                            bc_tol=self.args.bc_tol, max_nodes=self.args.max_nodes,
                            verbose=2 if self.args.verbose else 0)
            bc = self.bc(sol.y[:, 0], sol.y[:, -1], sol.p)
            score = float(np.linalg.norm(bc))
            rms = float(np.max(sol.rms_residuals)) if sol.rms_residuals.size else math.inf
            print(f"try factor={factor:.3f}: status={sol.status} bc_norm={score:.6e} "
                  f"rms_max={rms:.6e} nodes={sol.x.size}")
            item = (score, rms, sol, factor)
            if best is None or item[:2] < best[:2]:
                best = item
            if sol.status == 0 and score < self.args.accept_bc:
                break
        assert best is not None
        return best

    def rows(self, sol):
        momflux, eflux = self.unpack_flux_params(sol.p)
        s_left = self.output_mesh(sol.x, side="left")
        s_right = self.output_mesh(sol.x, side="right")
        y_left = sol.sol(s_left)
        y_right = sol.sol(s_right)
        x_left = self.args.x_min*(1.0 - s_left)
        x_right = self.args.x_max*s_right
        rows = []
        for x, u, er in zip(x_left, y_left[0], y_left[1]):
            rows.append(self.row(float(x), float(u), float(er), "upstream", momflux, eflux))
        for x, u, er in zip(x_right, y_right[2], y_right[3]):
            rows.append(self.row(float(x), float(u), float(er), "downstream", momflux, eflux))
        self.pin_far_edges(rows, momflux, eflux)
        return rows

    def far_state_row(self, x, state, branch, momflux, eflux):
        er = self.args.a_rad*state.temp**4
        pgas = state.rho*self.args.rgas*state.temp
        frad = self.frad(state.vel, er, momflux, eflux)
        return {
            "x_cm": x,
            "rho_g_cm3": state.rho,
            "u_cm_s": state.vel,
            "p_dyn_cm2": pgas,
            "Tgas_K": state.temp,
            "Er_erg_cm3": er,
            "Trad_K": state.temp,
            "Fr_erg_cm2_s": frad,
            "branch": branch,
        }

    def pin_far_edges(self, rows, momflux, eflux):
        upstream = [i for i, row in enumerate(rows) if row["branch"] == "upstream"]
        downstream = [i for i, row in enumerate(rows) if row["branch"] == "downstream"]
        if upstream:
            i = min(upstream, key=lambda idx: rows[idx]["x_cm"])
            rows[i] = self.far_state_row(self.args.x_min, LEFT, "upstream", momflux, eflux)
        if downstream:
            i = max(downstream, key=lambda idx: rows[idx]["x_cm"])
            rows[i] = self.far_state_row(self.args.x_max, RIGHT, "downstream", momflux, eflux)

    def output_mesh(self, solver_mesh: np.ndarray, side: str) -> np.ndarray:
        if side == "left":
            scale = abs(self.args.x_min)
            shock_s = 1.0
            base = solver_mesh[solver_mesh < 1.0]
            distances = self.shock_distances(scale)
            refined = shock_s - distances/scale
        else:
            scale = abs(self.args.x_max)
            shock_s = 0.0
            base = solver_mesh[solver_mesh > 0.0]
            distances = self.shock_distances(scale)
            refined = shock_s + distances/scale
        mesh = np.concatenate([base, refined])
        mesh = mesh[(mesh >= 0.0) & (mesh <= 1.0)]
        return np.unique(np.round(mesh, decimals=15))

    def shock_distances(self, scale: float) -> np.ndarray:
        eps = max(self.args.shock_epsilon, np.finfo(float).eps*scale)
        width = min(max(self.args.shock_refine_width, eps), scale)
        n = max(2, self.args.shock_refine_points)
        return np.geomspace(eps, width, n)

    def row(self, x, u, er, branch, momflux, eflux):
        rho, pgas, temp = self.primitive(np.asarray(u), np.asarray(er), momflux)
        frad = self.frad(u, er, momflux, eflux)
        return {
            "x_cm": x,
            "rho_g_cm3": float(rho),
            "u_cm_s": u,
            "p_dyn_cm2": float(pgas),
            "Tgas_K": float(temp),
            "Er_erg_cm3": er,
            "Trad_K": (er/self.args.a_rad)**0.25,
            "Fr_erg_cm2_s": frad,
            "branch": branch,
        }


def write_csv(path: Path, rows) -> None:
    fields = ["x_cm", "rho_g_cm3", "u_cm_s", "p_dyn_cm2", "Tgas_K",
              "Er_erg_cm3", "Trad_K", "Fr_erg_cm2_s", "branch"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, rows) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True)
    for branch in ("upstream", "downstream"):
        branch_rows = [row for row in rows if row["branch"] == branch]
        branch_rows = sorted(branch_rows, key=lambda row: row["x_cm"])
        x = np.asarray([row["x_cm"] for row in branch_rows])
        rho = np.asarray([row["rho_g_cm3"] for row in branch_rows])
        tgas = np.asarray([row["Tgas_K"] for row in branch_rows])
        trad = np.asarray([row["Trad_K"] for row in branch_rows])
        label_suffix = "pre" if branch == "upstream" else "post"
        axes[0].plot(x, rho, lw=1.6, color="tab:blue")
        axes[1].plot(x, tgas, lw=1.6, color="tab:blue",
                     label=rf"$T_\mathrm{{gas}}$ {label_suffix}")
        axes[1].plot(x, trad, lw=1.6, color="tab:orange",
                     label=rf"$T_\mathrm{{rad}}$ {label_suffix}")
    axes[0].set_ylabel(r"$\rho$ [g cm$^{-3}$]")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_xlabel("x [cm]")
    axes[1].set_ylabel("T [K]")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axvline(0.0, color="0.3", ls=":", lw=1.0)
        ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    problem = Mach2BVP(args)
    score, rms, sol, factor = problem.solve()
    print("\nSelected BVP solution:")
    print(f"  factor={factor:.6g} status={sol.status} message={sol.message}")
    print(f"  bc_norm={score:.6e} rms_max={rms:.6e}")
    momflux, eflux = problem.unpack_flux_params(sol.p)
    print(f"  relaxed momflux={momflux:.16e} eflux={eflux:.16e}")
    print(f"  relative corrections: dmom={args.flux_relax*sol.p[0]:.6e} "
          f"deflux={args.flux_relax*sol.p[1]:.6e}")
    if score > args.accept_bc and not args.allow_bad_profile:
        print("No accepted profile written. Re-run with --allow-bad-profile for diagnostics.")
        raise SystemExit(2)
    rows = problem.rows(sol)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.plot:
        make_plot(args.plot, rows)
        print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
