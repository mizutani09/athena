#!/usr/bin/env python3
"""Semi-analytic steady gray radiative-shock profiles.

This script computes the steady shock-frame non-equilibrium radiation
diffusion solutions used for Zhang et al. (2011), Section 6.6 style
comparisons.  It solves the ODEs on both sides of an embedded hydrodynamic
shock and shoots on the immediate pre/post-shock gas states.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from tqdm.auto import tqdm


K_BOLTZ = 1.380649e-16
M_H = 1.6735575e-24


@dataclass(frozen=True)
class FarState:
    rho: float
    temp: float
    vel: float


@dataclass(frozen=True)
class CaseConfig:
    name: str
    left: FarState
    right: FarState
    x_min: float
    x_max: float


@dataclass(frozen=True)
class ShockParams:
    gamma: float
    mu: float
    a_rad: float
    c_light: float
    kappa_p: float
    chi_r: float
    rgas: float
    gamma_over_gm1: float
    mflux: float
    momflux: float
    eflux: float
    left: FarState
    right: FarState
    x_min: float
    x_max: float


CASES = {
    "mach2": CaseConfig(
        name="mach2",
        left=FarState(rho=5.45887e-13, temp=100.0, vel=2.35435e5),
        right=FarState(rho=1.24794e-12, temp=207.757, vel=1.02987e5),
        x_min=-1000.0,
        x_max=500.0,
    ),
    "mach5": CaseConfig(
        name="mach5",
        left=FarState(rho=5.45887e-13, temp=100.0, vel=5.88588e5),
        right=FarState(rho=1.96405e-12, temp=855.720, vel=1.63592e5),
        x_min=-4000.0,
        x_max=2000.0,
    ),
}


class BadState(ValueError):
    """Raised when a trial state leaves the physical domain."""


def equilibrium_fluxes(
    state: FarState,
    *,
    gamma: float,
    rgas: float,
    a_rad: float,
) -> tuple[float, float, float]:
    rho = state.rho
    u = state.vel
    p = rho * rgas * state.temp
    er = a_rad * state.temp**4
    mflux = rho * u
    momflux = rho * u * u + p + er / 3.0
    eflux = mflux * (0.5 * u * u + gamma / (gamma - 1.0) * p / rho)
    eflux += (4.0 / 3.0) * u * er
    return mflux, momflux, eflux


def build_params(args: argparse.Namespace) -> ShockParams:
    case = CASES[args.case]
    rgas = K_BOLTZ / (args.mu * M_H)
    gamma_over_gm1 = args.gamma / (args.gamma - 1.0)

    left_flux = equilibrium_fluxes(
        case.left, gamma=args.gamma, rgas=rgas, a_rad=args.a_rad
    )
    right_flux = equilibrium_fluxes(
        case.right, gamma=args.gamma, rgas=rgas, a_rad=args.a_rad
    )
    names = ("mass", "momentum", "energy")
    print("Far-state flux consistency check:")
    for name, left_val, right_val in zip(names, left_flux, right_flux):
        denom = max(abs(left_val), abs(right_val), 1.0e-300)
        rel = abs(left_val - right_val) / denom
        print(
            f"  {name:8s}: left={left_val:.16e}  right={right_val:.16e}  "
            f"rel_diff={rel:.3e}"
        )
        if rel > args.flux_warning_rtol:
            print(
                f"  WARNING: {name} flux differs by more than "
                f"{args.flux_warning_rtol:g}; continuing with averaged flux."
            )

    if args.flux_source == "left":
        mflux, momflux, eflux = left_flux
    elif args.flux_source == "right":
        mflux, momflux, eflux = right_flux
    else:
        mflux = 0.5 * (left_flux[0] + right_flux[0])
        momflux = 0.5 * (left_flux[1] + right_flux[1])
        eflux = 0.5 * (left_flux[2] + right_flux[2])
    print(f"Using {args.flux_source} conserved fluxes for the ODE.")
    return ShockParams(
        gamma=args.gamma,
        mu=args.mu,
        a_rad=args.a_rad,
        c_light=args.c_light,
        kappa_p=args.kappa_p,
        chi_r=args.chi_r,
        rgas=rgas,
        gamma_over_gm1=gamma_over_gm1,
        mflux=mflux,
        momflux=momflux,
        eflux=eflux,
        left=case.left,
        right=case.right,
        x_min=args.x_min if args.x_min is not None else case.x_min,
        x_max=args.x_max if args.x_max is not None else case.x_max,
    )


def primitive_from_u_Er(u: float, er: float, params: ShockParams) -> dict[str, float]:
    if u <= 0.0 or er <= 0.0:
        raise BadState("non-positive u or Er")
    rho = params.mflux / u
    p = params.momflux - params.mflux * u - er / 3.0
    if rho <= 0.0 or p <= 0.0:
        raise BadState("non-positive rho or p")
    tgas = p / (rho * params.rgas)
    if tgas <= 0.0:
        raise BadState("non-positive gas temperature")
    return {"rho": rho, "u": u, "p": p, "Tgas": tgas, "Er": er}


def radiation_flux_from_u_Er(u: float, er: float, params: ShockParams) -> float:
    prim = primitive_from_u_Er(u, er, params)
    p = prim["p"]
    gas_enthalpy_flux = 0.5 * params.mflux * u * u + params.gamma_over_gm1 * p * u
    return params.eflux - gas_enthalpy_flux - (4.0 / 3.0) * u * er


def shock_jump_residual(
    vars: np.ndarray, er_s: float, params: ShockParams
) -> np.ndarray:
    u_minus, u_plus = float(vars[0]), float(vars[1])
    try:
        f_minus = radiation_flux_from_u_Er(u_minus, er_s, params)
        f_plus = radiation_flux_from_u_Er(u_plus, er_s, params)
        primitive_from_u_Er(u_minus, er_s, params)
        primitive_from_u_Er(u_plus, er_s, params)
    except BadState:
        return np.full(3, 1.0e6)
    shock_order = max((u_plus - u_minus) / max(u_minus, 1.0), 0.0)
    return np.array(
        [
            (f_minus - f_plus) / max(abs(params.eflux), 1.0),
            shock_order,
            0.0,
        ],
        dtype=float,
    )


def rhs(x: float, y: np.ndarray, params: ShockParams) -> list[float]:
    """ODE system for y = [u, Er] in the steady shock frame."""
    del x
    u = float(y[0])
    er = float(y[1])
    if u <= 0.0 or er <= 0.0:
        return [1.0e100, 1.0e100]

    mflux = params.mflux
    momflux = params.momflux
    gamma_over_gm1 = params.gamma_over_gm1
    gamma = params.gamma
    c_light = params.c_light

    rho = mflux / u
    p = momflux - mflux * u - er / 3.0
    if rho <= 0.0 or p <= 0.0:
        return [1.0e100, 1.0e100]

    tgas = p / (rho * params.rgas)
    gas_enthalpy_flux = 0.5 * mflux * u * u + gamma_over_gm1 * p * u
    frad = params.eflux - gas_enthalpy_flux - (4.0 / 3.0) * u * er
    der_dx = -(3.0 * params.chi_r / c_light) * frad

    dg_du = mflux * u + gamma_over_gm1 * (p - mflux * u)
    dg_der = -gamma * u / (3.0 * (gamma - 1.0))
    exchange = -c_light * params.kappa_p * (params.a_rad * tgas**4 - er)
    du_dx = (exchange - (dg_der + u / 3.0) * der_dx) / dg_du
    return [du_dx, der_dx]


def physical_domain_margin(x: float, y: np.ndarray, params: ShockParams) -> float:
    del x
    u = float(y[0])
    er = float(y[1])
    if u <= 0.0 or er <= 0.0:
        return -1.0
    rho = params.mflux / u
    p = params.momflux - params.mflux * u - er / 3.0
    return min(u, er, rho, p)


def integrate_branch(
    y0: tuple[float, float],
    x_end: float,
    params: ShockParams,
    args: argparse.Namespace,
    n_eval: int | None,
):
    t_eval = None
    if n_eval is not None and n_eval > 1:
        t_eval = np.linspace(0.0, x_end, n_eval)
    def event(x: float, y: np.ndarray) -> float:
        return physical_domain_margin(x, y, params)

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1.0  # type: ignore[attr-defined]
    return solve_ivp(
        lambda x, y: rhs(x, y, params),
        (0.0, x_end),
        np.asarray(y0, dtype=float),
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        t_eval=t_eval,
        events=event,
    )


def end_state_residual(
    sol, target: FarState, params: ShockParams, x_end: float
) -> np.ndarray:
    if sol.y.shape[1] == 0:
        return np.array([1.0e6, 1.0e6], dtype=float)
    u_end = float(sol.y[0, -1])
    er_end = float(sol.y[1, -1])
    er_target = params.a_rad * target.temp**4
    reached = abs(float(sol.t[-1]) / float(x_end)) if x_end != 0.0 else 1.0
    path_penalty = max(0.0, 1.0 - min(reached, 1.0))
    if (not sol.success) or (sol.t_events and len(sol.t_events[0]) > 0):
        path_penalty += 1.0
    try:
        primitive_from_u_Er(u_end, er_end, params)
    except BadState:
        u_res = (u_end - target.vel) / target.vel if math.isfinite(u_end) else 1.0e6
        er_safe = er_end if math.isfinite(er_end) and er_end > 0.0 else 1.0e-300
        er_res = math.log(er_safe / er_target)
        return np.array(
            [
                max(min(u_res, 1.0e6), -1.0e6) + 10.0 * path_penalty,
                max(min(er_res, 1.0e6), -1.0e6) + 10.0 * path_penalty,
            ],
            dtype=float,
        )
    return np.array(
        [
            (u_end - target.vel) / target.vel + 10.0 * path_penalty,
            math.log(er_end / er_target) + 10.0 * path_penalty,
        ],
        dtype=float,
    )


def unpack_trial(z: np.ndarray) -> tuple[float, float, float]:
    return float(math.exp(z[0])), float(math.exp(z[1])), float(math.exp(z[2]))




class EvaluationTimeout(RuntimeError):
    pass


class timeout_after:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.old_handler = None

    def __enter__(self):
        if self.seconds <= 0.0:
            return self
        if threading.current_thread() is not threading.main_thread():
            return self
        self.old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0.0 and threading.current_thread() is threading.main_thread():
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if self.old_handler is not None:
                signal.signal(signal.SIGALRM, self.old_handler)
        return False

    def _handle_timeout(self, signum, frame):
        del signum, frame
        raise EvaluationTimeout(f"shooting residual evaluation exceeded {self.seconds:g} s")


def shooting_residual(z: np.ndarray, params: ShockParams, args: argparse.Namespace) -> np.ndarray:
    er_s, u_minus, u_plus = unpack_trial(z)
    try:
        primitive_from_u_Er(u_minus, er_s, params)
        primitive_from_u_Er(u_plus, er_s, params)
    except BadState:
        return np.full(6, 1.0e6)

    try:
        with timeout_after(args.eval_timeout):
            left_sol = integrate_branch((u_minus, er_s), params.x_min, params, args, None)
            right_sol = integrate_branch((u_plus, er_s), params.x_max, params, args, None)
            left_res = end_state_residual(left_sol, params.left, params, params.x_min)
            right_res = end_state_residual(right_sol, params.right, params, params.x_max)
            jump_res = shock_jump_residual(np.array([u_minus, u_plus]), er_s, params)
    except EvaluationTimeout as exc:
        if args.progress or args.verbose:
            print(f"\n  timed out: {exc}", flush=True)
        return np.full(6, args.timeout_penalty, dtype=float)
    return np.array(
        [
            25.0 * jump_res[0],
            25.0 * jump_res[1],
            left_res[0],
            left_res[1],
            right_res[0],
            right_res[1],
        ],
        dtype=float,
    )




class EvaluationProgress:
    def __init__(self, enabled: bool, desc: str, total: int | None):
        self.enabled = enabled
        self.desc = desc
        self.total = total
        self.count = 0
        self.last_norm = math.nan
        self.current_start: float | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.pbar = None
        self.thread = None

    def __enter__(self):
        if self.enabled:
            self.pbar = tqdm(total=self.total, desc=self.desc, unit="eval", dynamic_ncols=True)
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.pbar is not None:
            self.pbar.close()

    def _heartbeat(self):
        while not self.stop_event.wait(1.0):
            with self.lock:
                if self.current_start is None:
                    elapsed = 0.0
                else:
                    elapsed = time.monotonic() - self.current_start
                count = self.count
                norm = self.last_norm
            if self.pbar is not None:
                postfix = {"eval": count, "running_s": f"{elapsed:.0f}"}
                if math.isfinite(norm):
                    postfix["norm"] = f"{norm:.3e}"
                self.pbar.set_postfix(postfix, refresh=True)

    def begin(self):
        if not self.enabled:
            return
        with self.lock:
            self.current_start = time.monotonic()

    def end(self, residual: np.ndarray):
        if not self.enabled:
            return
        norm = float(np.linalg.norm(residual))
        with self.lock:
            self.count += 1
            self.last_norm = norm
            self.current_start = None
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"eval": self.count, "norm": f"{norm:.3e}"}, refresh=True)


def shooting_residual_with_progress(
    z: np.ndarray, params: ShockParams, args: argparse.Namespace, progress: EvaluationProgress
) -> np.ndarray:
    progress.begin()
    residual = shooting_residual(z, params, args)
    progress.end(residual)
    return residual


def solve_shock_candidates(params: ShockParams, args: argparse.Namespace) -> list[tuple[float, object]]:
    left_er = params.a_rad * params.left.temp**4
    right_er = params.a_rad * params.right.temp**4
    guesses = [
        (math.sqrt(left_er * right_er), params.left.vel, params.right.vel),
        (0.5 * (left_er + right_er), params.left.vel * 0.98, params.right.vel * 1.02),
        (left_er, params.left.vel, params.right.vel),
    ]
    if args.include_stiff_guess:
        guesses.append((right_er, params.left.vel, params.right.vel))
    if args.max_guesses > 0:
        guesses = guesses[:args.max_guesses]
    if args.er_s is not None:
        guesses.insert(
            0,
            (
                args.er_s,
                args.u_minus if args.u_minus is not None else params.left.vel,
                args.u_plus if args.u_plus is not None else params.right.vel,
            ),
        )

    candidates: list[tuple[float, object]] = []
    for guess_index, guess in enumerate(guesses, start=1):
        z0 = np.log(np.asarray(guess, dtype=float))
        guess_desc = f"guess {guess_index}/{len(guesses)}"
        print(
            f"\nStarting {guess_desc}: "
            f"Er_s={guess[0]:.6e}, u_minus={guess[1]:.6e}, u_plus={guess[2]:.6e}",
            flush=True,
        )
        workers = args.workers if args.workers != 1 else None
        progress_enabled = args.progress and workers is None
        if args.progress and workers is not None:
            print(
                "Progress heartbeat is disabled during scipy workers parallel finite differences; "
                "use --workers 1 for detailed per-evaluation progress.",
                flush=True,
            )
        with EvaluationProgress(progress_enabled, guess_desc, args.max_nfev) as progress:
            if progress_enabled:
                result = least_squares(
                    shooting_residual_with_progress,
                    z0,
                    args=(params, args, progress),
                    xtol=args.optimize_xtol,
                    ftol=args.optimize_ftol,
                    gtol=args.optimize_gtol,
                    max_nfev=args.max_nfev,
                    verbose=2 if args.verbose else 0,
                )
            else:
                result = least_squares(
                    shooting_residual,
                    z0,
                    args=(params, args),
                    xtol=args.optimize_xtol,
                    ftol=args.optimize_ftol,
                    gtol=args.optimize_gtol,
                    max_nfev=args.max_nfev,
                    verbose=2 if args.verbose else 0,
                    workers=workers,
                )
        norm = float(np.linalg.norm(result.fun))
        candidates.append((norm, result))
        print(f"Finished {guess_desc}: success={result.success} norm={norm:.6e}", flush=True)
        if result.success and norm < args.accept_norm:
            break

    candidates.sort(key=lambda item: item[0])
    print("\nShooting candidates:")
    for n, (norm, result) in enumerate(candidates, start=1):
        er_s, u_minus, u_plus = unpack_trial(result.x)
        print(
            f"  {n}: success={result.success} norm={norm:.6e} nfev={result.nfev} "
            f"Er_s={er_s:.6e} u_minus={u_minus:.6e} u_plus={u_plus:.6e}"
        )
    return candidates


def print_shooting_result(norm: float, result) -> tuple[float, float, float]:
    er_s, u_minus, u_plus = unpack_trial(result.x)
    print("\nSelected shooting result:")
    print(f"  success={result.success}  norm={norm:.6e}  nfev={result.nfev}")
    print(f"  Er_s={er_s:.16e}")
    print(f"  u_minus={u_minus:.16e}  u_plus={u_plus:.16e}")
    print("  residual=", np.array2string(result.fun, precision=4, suppress_small=False))
    return er_s, u_minus, u_plus


def profile_rows(x: np.ndarray, u: np.ndarray, er: np.ndarray, branch: str, params: ShockParams):
    rows = []
    for xv, uv, erv in zip(x, u, er):
        prim = primitive_from_u_Er(float(uv), float(erv), params)
        trad = (prim["Er"] / params.a_rad) ** 0.25
        frad = radiation_flux_from_u_Er(prim["u"], prim["Er"], params)
        rows.append(
            {
                "x_cm": float(xv),
                "rho_g_cm3": prim["rho"],
                "u_cm_s": prim["u"],
                "p_dyn_cm2": prim["p"],
                "Tgas_K": prim["Tgas"],
                "Er_erg_cm3": prim["Er"],
                "Trad_K": trad,
                "Fr_erg_cm2_s": frad,
                "branch": branch,
            }
        )
    return rows


def far_state_row(xv: float, state: FarState, branch: str, params: ShockParams) -> dict[str, float | str]:
    er = params.a_rad * state.temp**4
    p = state.rho * params.rgas * state.temp
    frad = radiation_flux_from_u_Er(state.vel, er, params)
    return {
        "x_cm": float(xv),
        "rho_g_cm3": state.rho,
        "u_cm_s": state.vel,
        "p_dyn_cm2": p,
        "Tgas_K": state.temp,
        "Er_erg_cm3": er,
        "Trad_K": state.temp,
        "Fr_erg_cm2_s": frad,
        "branch": branch,
    }


def pin_far_edges(rows: list[dict[str, float | str]], params: ShockParams) -> None:
    if not rows:
        return
    upstream = [i for i, row in enumerate(rows) if row["branch"] == "upstream"]
    downstream = [i for i, row in enumerate(rows) if row["branch"] == "downstream"]
    if upstream:
        i = min(upstream, key=lambda idx: float(rows[idx]["x_cm"]))
        rows[i] = far_state_row(params.x_min, params.left, "upstream", params)
    if downstream:
        i = max(downstream, key=lambda idx: float(rows[idx]["x_cm"]))
        rows[i] = far_state_row(params.x_max, params.right, "downstream", params)


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "x_cm",
        "rho_g_cm3",
        "u_cm_s",
        "p_dyn_cm2",
        "Tgas_K",
        "Er_erg_cm3",
        "Trad_K",
        "Fr_erg_cm2_s",
        "branch",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, rows: list[dict[str, float | str]], title: str) -> None:
    x = np.asarray([float(row["x_cm"]) for row in rows])
    rho = np.asarray([float(row["rho_g_cm3"]) for row in rows])
    tgas = np.asarray([float(row["Tgas_K"]) for row in rows])
    trad = np.asarray([float(row["Trad_K"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(x, rho, color="tab:blue", lw=1.8)
    axes[0].set_ylabel(r"$\rho\ [\mathrm{g\ cm^{-3}}]$")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].plot(x, tgas, color="tab:red", lw=1.8, label=r"$T_\mathrm{gas}$")
    axes[1].plot(x, trad, color="tab:green", lw=1.8, label=r"$T_\mathrm{rad}$")
    axes[1].set_ylabel("T [K]")
    axes[1].set_xlabel("x [cm]")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axvline(0.0, color="0.35", lw=1.0, ls=":")
        ax.grid(alpha=0.22)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_output_profile(
    er_s: float,
    u_minus: float,
    u_plus: float,
    params: ShockParams,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    n_left = max(2, args.n_points // 2)
    n_right = max(2, args.n_points - n_left + 1)
    left_sol = integrate_branch((u_minus, er_s), params.x_min, params, args, n_left)
    right_sol = integrate_branch((u_plus, er_s), params.x_max, params, args, n_right)
    if not left_sol.success:
        raise RuntimeError(f"upstream integration failed: {left_sol.message}")
    if not right_sol.success:
        raise RuntimeError(f"downstream integration failed: {right_sol.message}")

    left_x = left_sol.t[::-1]
    left_u = left_sol.y[0, ::-1]
    left_er = left_sol.y[1, ::-1]
    right_x = right_sol.t
    right_u = right_sol.y[0]
    right_er = right_sol.y[1]
    rows = profile_rows(left_x, left_u, left_er, "upstream", params)
    rows.extend(profile_rows(right_x, right_u, right_er, "downstream", params))
    if args.pin_far_edges:
        pin_far_edges(rows, params)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a semi-analytic steady 1D gray radiative-shock profile."
    )
    parser.add_argument("--case", choices=sorted(CASES), default="mach2")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG plot path.")
    parser.add_argument(
        "--method",
        choices=("BDF", "Radau", "LSODA", "DOP853"),
        default="Radau",
        help="solve_ivp method.",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--fast", action="store_true", help="Use looser tolerances and fewer points.")
    parser.add_argument("--accurate", action="store_true", help="Use tighter solver tolerances.")
    parser.add_argument("--n-points", type=int, default=1200)
    parser.add_argument("--max-nfev", type=int, default=160)
    parser.add_argument("--accept-norm", type=float, default=2.0e-5)
    parser.add_argument("--warn-norm", type=float, default=2.0e-3)
    parser.add_argument("--optimize-xtol", type=float, default=1.0e-8)
    parser.add_argument("--optimize-ftol", type=float, default=1.0e-8)
    parser.add_argument("--optimize-gtol", type=float, default=1.0e-8)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--a-rad", type=float, default=7.5646e-15)
    parser.add_argument("--c-light", type=float, default=2.99792458e10)
    parser.add_argument("--kappa-p", type=float, default=3.92664e-5)
    parser.add_argument("--chi-r", type=float, default=0.848902)
    parser.add_argument(
        "--flux-source",
        choices=("left", "right", "average"),
        default="left",
        help="Which rounded far state supplies the conserved fluxes.",
    )
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--er-s", type=float, default=None, help="Initial guess for shock Er.")
    parser.add_argument("--u-minus", type=float, default=None, help="Initial guess for pre-shock u.")
    parser.add_argument("--u-plus", type=float, default=None, help="Initial guess for post-shock u.")
    parser.add_argument(
        "--pin-far-edges",
        dest="pin_far_edges",
        action="store_true",
        default=True,
        help="Force the first upstream and last downstream CSV rows to the pgen far states.",
    )
    parser.add_argument(
        "--no-pin-far-edges",
        dest="pin_far_edges",
        action="store_false",
        help="Write the finite-domain integration values at the CSV edges.",
    )
    parser.add_argument("--flux-warning-rtol", type=float, default=5.0e-3)
    parser.add_argument("--max-guesses", type=int, default=0, help="Maximum number of default initial guesses to try; 0 means all enabled guesses.")
    parser.add_argument("--include-stiff-guess", action="store_true", help="Also try the pure downstream-radiation-energy initial guess, which can be very slow for Mach 5.")
    parser.add_argument("--allow-bad-profile", action="store_true", help="Attempt to write a profile even when all shooting residual norms exceed --warn-norm.")
    parser.add_argument("--eval-timeout", type=float, default=120.0, help="Maximum seconds allowed for one shooting residual evaluation. Use 0 to disable. This prevents stiff bad trial states from appearing to hang.")
    parser.add_argument("--timeout-penalty", type=float, default=1.0e4, help="Residual value returned when --eval-timeout is exceeded.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers passed to scipy.optimize.least_squares for finite-difference Jacobians. Use -1 for all cores; progress heartbeat is most informative with --workers 1.")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True, help="Show tqdm progress/heartbeat during shooting evaluations.")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress/heartbeat.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.fast:
        args.rtol = 1.0e-6
        args.atol = 1.0e-10
        args.n_points = min(args.n_points, 500)
        args.max_nfev = min(args.max_nfev, 80)
    if args.accurate:
        args.rtol = 1.0e-9
        args.atol = 1.0e-13
        args.n_points = max(args.n_points, 1800)
    if args.fast and args.accurate:
        raise SystemExit("--fast and --accurate are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    params = build_params(args)
    script_dir = Path(__file__).resolve().parent
    output_path = args.output
    if output_path is None:
        output_path = script_dir / f"{args.case}_semianalytic_radiative_shock.csv"
    plot_path = args.plot
    if plot_path is None:
        plot_path = script_dir / "imgs" / f"{args.case}_semianalytic_radiative_shock.png"

    candidates = solve_shock_candidates(params, args)
    rows = None
    selected = None
    failures = []
    for norm, result in candidates:
        if norm > args.warn_norm and not args.allow_bad_profile:
            failures.append(
                f"skipped norm={norm:.6e} because it exceeds --warn-norm={args.warn_norm:g}"
            )
            continue
        er_s, u_minus, u_plus = unpack_trial(result.x)
        try:
            rows = build_output_profile(er_s, u_minus, u_plus, params, args)
            selected = (norm, result)
            break
        except RuntimeError as exc:
            failures.append(f"norm={norm:.6e}: {exc}")

    if rows is None or selected is None:
        print("\nNo usable semi-analytic profile was produced.")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "Try a shorter x range, provide --er-s/--u-minus/--u-plus from a better "
            "previous run, or use --allow-bad-profile only for debugging."
        )
        raise SystemExit(2)

    norm, result = selected
    print_shooting_result(norm, result)
    if norm > args.warn_norm:
        print(
            f"  WARNING: residual norm is larger than {args.warn_norm:g}; "
            "inspect the CSV/plot before using as a reference."
        )
    write_csv(output_path, rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")
    if args.plot:
        make_plot(plot_path, rows, f"{args.case} semi-analytic radiative shock")
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
