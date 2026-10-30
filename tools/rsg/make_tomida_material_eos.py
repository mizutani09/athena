#!/usr/bin/env python3
"""Build and audit a material-only Tomida et al. (2013) H/He EOS table.

The implementation follows Appendix 1 of Tomida et al. 2013 (ApJ 763, 6):
LTE/chemical equilibrium, fixed H2 ortho:para=3:1, seven species, and an
ideal mixture with no photon contribution.  Radiation energy and pressure
must therefore be evolved separately by FLD.

The output is Athena++'s seven-field ASCII general-EOS schema:
P/e(rho,e), e/P(rho,P), Gamma1(rho,P), T(rho,e), S(rho,e), S(rho,P),
and nabla_ad(rho,e).  A JSON sidecar records the audit and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

KB = 1.380649e-16
HPLANCK = 6.62607015e-27
ME = 9.1093837015e-28
MH = 1.6735575e-24
MHE = 4.0 * MH
EV = 1.602176634e-12
THETA_ROT = 170.64
THETA_VIB_2013 = 5984.48
THETA_VIB_2015 = 6330.0
CHI_DISS = 4.478 * EV
CHI_H = 13.598 * EV
CHI_HE1 = 24.587 * EV
CHI_HE2 = 54.418 * EV
SPECIES = ("H2", "H", "Hp", "He", "Hep", "Hepp", "e")


def log_translational(mass: float, temperature: float) -> float:
    return 1.5 * np.log(2.0 * np.pi * mass * KB * temperature) - 3.0 * np.log(HPLANCK)


def log_vibration_2013(temperature: float) -> float:
    x = THETA_VIB_2013 / (2.0 * temperature)
    return -x - np.log1p(-np.exp(-2.0 * x))


def log_vibration_2015(temperature: float) -> float:
    x = THETA_VIB_2015 / temperature
    return -np.log1p(-np.exp(-x))


def log_rotation(temperature: float, jmax: int = 200) -> float:
    even = np.arange(0, jmax + 1, 2, dtype=float)
    odd = np.arange(1, jmax + 1, 2, dtype=float)
    ze = np.sum((2.0 * even + 1.0)
                * np.exp(-even * (even + 1.0) * THETA_ROT / (2.0 * temperature)))
    zo = np.sum((2.0 * odd + 1.0)
                * np.exp(-odd * (odd + 1.0) * THETA_ROT / (2.0 * temperature)))
    return 0.25 * np.log(ze) + 0.75 * (
        np.log(3.0 * zo) + THETA_ROT / temperature)


def log_partition(temperature: float, vibration: str) -> dict[str, float]:
    lvib = (log_vibration_2013(temperature) if vibration == "tomida2013"
            else log_vibration_2015(temperature))
    return {
        "H2": log_translational(2.0 * MH, temperature)
              + log_rotation(temperature) + lvib + np.log(8.0),
        "H": log_translational(MH, temperature) + np.log(4.0)
             - CHI_DISS / (2.0 * KB * temperature),
        "Hp": log_translational(MH, temperature) + np.log(4.0)
              - (0.5 * CHI_DISS + CHI_H) / (KB * temperature),
        "He": log_translational(MHE, temperature),
        "Hep": log_translational(MHE, temperature)
               - CHI_HE1 / (KB * temperature),
        "Hepp": log_translational(MHE, temperature)
                - (CHI_HE1 + CHI_HE2) / (KB * temperature),
        "e": log_translational(ME, temperature) + np.log(2.0),
    }


def partition_slopes(temperature: float, vibration: str) -> dict[str, float]:
    step = 2.0e-5
    lo = log_partition(temperature * np.exp(-step), vibration)
    hi = log_partition(temperature * np.exp(step), vibration)
    return {name: (hi[name] - lo[name]) / (2.0 * step) for name in SPECIES}


@dataclass
class State:
    pressure: float
    energy: float
    entropy: float
    number: np.ndarray
    hydrogen_error: float
    helium_error: float
    charge_error: float


class TomidaMaterialEOS:
    def __init__(self, hydrogen: float = 0.70, helium: float = 0.28,
                 vibration: str = "tomida2013"):
        if hydrogen <= 0.0 or helium <= 0.0 or hydrogen + helium > 1.0:
            raise ValueError("require 0<X,Y and X+Y<=1")
        self.x = hydrogen
        self.y = helium
        self.vibration = vibration

    def state(self, density: float, temperature: float) -> State:
        if density <= 0.0 or temperature <= 0.0:
            raise ValueError("density and temperature must be positive")
        lz = log_partition(temperature, self.vibration)
        log_kd = 2.0 * lz["H"] - lz["H2"]
        log_kh = lz["Hp"] + lz["e"] - lz["H"]
        log_khe1 = lz["Hep"] + lz["e"] - lz["He"]
        log_khe2 = lz["Hepp"] + lz["e"] - lz["Hep"]
        kd = np.exp(np.clip(log_kd, -700.0, 700.0))
        kh = np.exp(np.clip(log_kh, -700.0, 700.0))
        khe1 = np.exp(np.clip(log_khe1, -700.0, 700.0))
        khe2 = np.exp(np.clip(log_khe2, -700.0, 700.0))
        nhn = self.x * density / MH
        nhen = self.y * density / MHE

        def populations(ne: float):
            logne = np.log(ne)
            logah = log_kh - logne
            logb = np.logaddexp(0.0, logah)
            loga = np.log(2.0) - log_kd
            logroot = 0.5 * np.logaddexp(2.0 * logb,
                                        np.log(4.0) + loga + np.log(nhn))
            logden = np.logaddexp(logb, logroot)
            nh = np.exp(np.log(2.0 * nhn) - logden)
            nhp = np.exp(np.log(max(nh, np.finfo(float).tiny)) + logah)
            nh2 = 0.5 * max(nhn - nh - nhp, 0.0)
            loga1, loga2 = log_khe1 - logne, log_khe2 - logne
            weights = np.array([0.0, loga1, loga1 + loga2])
            weights = np.exp(weights - np.max(weights))
            weights /= weights.sum()
            nhe, nhep, nhepp = nhen * weights
            return nh2, nh, nhp, nhe, nhep, nhepp

        ntop = max(nhn + 2.0 * nhen, np.finfo(float).tiny)

        def residual(logne: float) -> float:
            ne = np.exp(logne)
            _, _, nhp, _, nhep, nhepp = populations(ne)
            rhs = nhp + nhep + 2.0 * nhepp
            return (ne - rhs) / max(ne + rhs, np.finfo(float).tiny)

        # Twice the fully-ionized charge is a conservative numerical upper
        # bracket; the physical root remains below nhn+2*nhen.
        log_hi = np.log(2.0*ntop)
        # In cold molecular gas the formal Saha electron abundance can be
        # many hundreds of decades below the nuclei density.  Such electrons
        # have no thermodynamic effect, but a density-relative bracket would
        # spuriously fail before reaching that root.
        log_lo = -690.0
        flo, fhi = residual(log_lo), residual(log_hi)
        if fhi < 0.0:
            raise RuntimeError(
                f"electron bracket failed at rho={density:.6e}, T={temperature:.6e}")
        logne = (log_lo if flo >= 0.0 else
                 brentq(residual, log_lo, log_hi, xtol=2e-13, rtol=2e-13))
        ne = np.exp(logne)
        pop = populations(ne)
        number = np.array((*pop, ne))
        slopes = partition_slopes(temperature, self.vibration)
        slope = np.array([slopes[name] for name in SPECIES])
        pressure = number.sum() * KB * temperature
        energy = np.dot(number, slope) * KB * temperature
        chemical = np.log(np.maximum(number, np.finfo(float).tiny)) \
                   - np.array([lz[name] for name in SPECIES])
        entropy = KB * np.dot(number, 1.0 + slope - chemical) / density
        nh2, nh, nhp, nhe, nhep, nhepp = pop
        herr = abs(2.0 * nh2 + nh + nhp - nhn) / nhn
        heerr = abs(nhe + nhep + nhepp - nhen) / nhen
        qerr = abs(nhp + nhep + 2.0 * nhepp - ne) / max(ne, ntop * 1e-30)
        return State(pressure, energy, entropy, number, herr, heerr, qerr)


def build_forward(eos: TomidaMaterialEOS, logrho: np.ndarray, logt: np.ndarray):
    shape = (logrho.size, logt.size)
    p = np.empty(shape)
    e = np.empty(shape)
    s = np.empty(shape)
    species = np.empty((len(SPECIES), *shape))
    conservation = np.zeros(3)
    for ir, lr in enumerate(logrho):
        density = 10.0 ** lr
        for it, lt in enumerate(logt):
            state = eos.state(density, 10.0 ** lt)
            p[ir, it], e[ir, it], s[ir, it] = (
                state.pressure, state.energy, state.entropy)
            species[:, ir, it] = state.number
            conservation = np.maximum(conservation, (
                state.hydrogen_error, state.helium_error, state.charge_error))
    lnrho, lnt = logrho * np.log(10.0), logt * np.log(10.0)
    dlnp_dlnrho = np.gradient(np.log(p), lnrho, axis=0, edge_order=2)
    dlnp_dlnt = np.gradient(np.log(p), lnt, axis=1, edge_order=2)
    ds_dlnrho = np.gradient(s, lnrho, axis=0, edge_order=2)
    ds_dlnt = np.gradient(s, lnt, axis=1, edge_order=2)
    cv = ds_dlnt
    dln_t_dln_rho_s = -ds_dlnrho / ds_dlnt
    gamma1 = dlnp_dlnrho + dlnp_dlnt * dln_t_dln_rho_s
    nabla_ad = dln_t_dln_rho_s / gamma1
    return {"p": p, "e": e, "s": s, "species": species, "cv": cv,
            "gamma1": gamma1, "nabla_ad": nabla_ad,
            "conservation": conservation}


def select_common_axis(logrho, logt, forward, requested_min, requested_max):
    u = forward["e"] / (10.0 ** logrho[:, None])
    q = forward["p"] / (10.0 ** logrho[:, None])
    lu, lq = np.log10(u), np.log10(q)
    ulim = [float(np.max(lu[:, 0])), float(np.min(lu[:, -1]))]
    qlim = [float(np.max(lq[:, 0])), float(np.min(lq[:, -1]))]
    lo = max(ulim[0], requested_min) if requested_min is not None else ulim[0]
    hi = min(ulim[1], requested_max) if requested_max is not None else ulim[1]
    width = min(hi - lo, qlim[1] - qlim[0])
    if width <= 0.0:
        raise RuntimeError(f"no common inverse domain: u={ulim}, q={qlim}")
    hi = lo + width
    qcenter = 0.5 * (qlim[0] + qlim[1])
    ucenter = 0.5 * (lo + hi)
    ratio_q = 10.0 ** (ucenter - qcenter)
    return np.array([lo, hi]), ratio_q, lu, lq


def export_table(path: Path, logrho, logt, forward, nx2, requested_min, requested_max):
    xlim, ratio_q, lu, lq = select_common_axis(
        logrho, logt, forward, requested_min, requested_max)
    xaxis = np.linspace(*xlim, nx2)
    qaxis = xaxis - np.log10(ratio_q)
    out = np.empty((7, nx2, logrho.size))
    for ir in range(logrho.size):
        p, e, s = forward["p"][ir], forward["e"][ir], forward["s"][ir]
        gamma, nabla = forward["gamma1"][ir], forward["nabla_ad"][ir]
        out[0, :, ir] = np.interp(xaxis, lu[ir], np.log10(p / e))
        out[3, :, ir] = np.interp(xaxis, lu[ir], logt)
        out[4, :, ir] = np.interp(xaxis, lu[ir], np.log10(s))
        out[6, :, ir] = np.interp(xaxis, lu[ir], np.log10(nabla))
        out[1, :, ir] = np.interp(qaxis, lq[ir], np.log10(e / p))
        out[2, :, ir] = np.interp(qaxis, lq[ir], np.log10(gamma))
        out[5, :, ir] = np.interp(qaxis, lq[ir], np.log10(s))
    if not np.isfinite(out).all():
        raise RuntimeError("Athena table contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ("log10(P/e) on (rho,e/rho)", "log10(e/P) on (rho,P/rho)",
             "log10(Gamma1) on (rho,P/rho)", "log10(T/K) on (rho,e/rho)",
             "log10(S/cgs) on (rho,e/rho)", "log10(S/cgs) on (rho,P/rho)",
             "log10(nabla_ad) on (rho,e/rho)")
    ratios = (1.0, ratio_q, ratio_q, 1.0, 1.0, ratio_q, 1.0)
    with path.open("w") as stream:
        stream.write("# Tomida et al. 2013 material-only EOS; no photon terms\n")
        stream.write("# n_var n_x2 n_rho\n")
        stream.write(f"7 {nx2} {logrho.size}\n# log10 x2 limits\n")
        stream.write(f"{xlim[0]:.16e} {xlim[1]:.16e}\n# log10 rho limits\n")
        stream.write(f"{logrho[0]:.16e} {logrho[-1]:.16e}\n# coordinate ratios\n")
        stream.write(" ".join(f"{v:.16e}" for v in ratios) + "\n")
        for name, values in zip(names, out):
            stream.write(f"# {name}\n")
            np.savetxt(stream, values, fmt="%.12e")
    return out, xlim, ratios


def audit(forward, out):
    checks = {
        "all_finite": bool(all(np.isfinite(forward[k]).all()
                               for k in ("p", "e", "s", "cv", "gamma1", "nabla_ad"))),
        "positive_pressure": bool(np.all(forward["p"] > 0.0)),
        "positive_energy": bool(np.all(forward["e"] > 0.0)),
        "positive_entropy": bool(np.all(forward["s"] > 0.0)),
        "positive_cv": bool(np.all(forward["cv"] > 0.0)),
        "positive_gamma1": bool(np.all(forward["gamma1"] > 0.0)),
        "positive_nabla_ad": bool(np.all(forward["nabla_ad"] > 0.0)),
        "energy_monotone_T": bool(np.all(np.diff(forward["e"], axis=1) > 0.0)),
        "pressure_monotone_T": bool(np.all(np.diff(forward["p"], axis=1) > 0.0)),
        "entropy_monotone_T": bool(np.all(np.diff(forward["s"], axis=1) > 0.0)),
        "athena_fields_finite": bool(np.isfinite(out).all()),
    }
    checks["pass"] = bool(all(checks.values()))
    return checks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--npz", type=Path)
    p.add_argument("--logrho-min", type=float, default=-18.0)
    p.add_argument("--logrho-max", type=float, default=-3.0)
    p.add_argument("--logt-min", type=float, default=2.5)
    p.add_argument("--logt-max", type=float, default=6.3)
    p.add_argument("--nrho", type=int, default=128)
    p.add_argument("--ntemp", type=int, default=512)
    p.add_argument("--nx2", type=int, default=512)
    p.add_argument("--logx2-min", type=float)
    p.add_argument("--logx2-max", type=float)
    p.add_argument("--hydrogen", type=float, default=0.70)
    p.add_argument("--helium", type=float, default=0.28)
    p.add_argument("--vibration", choices=("tomida2013", "tomida2015"),
                   default="tomida2013")
    return p.parse_args()


def main():
    args = parse_args()
    if min(args.nrho, args.ntemp, args.nx2) < 4:
        raise ValueError("all grid dimensions must be at least four")
    logrho = np.linspace(args.logrho_min, args.logrho_max, args.nrho)
    logt = np.linspace(args.logt_min, args.logt_max, args.ntemp)
    eos = TomidaMaterialEOS(args.hydrogen, args.helium, args.vibration)
    forward = build_forward(eos, logrho, logt)
    out, xlim, ratios = export_table(
        args.output, logrho, logt, forward, args.nx2,
        args.logx2_min, args.logx2_max)
    checks = audit(forward, out)
    metadata = {
        "schema": "athena-general-eos-v7",
        "material_only": True,
        "photon_pressure_energy_entropy": False,
        "reference": "Tomida et al. 2013, ApJ 763, 6, Appendix 1",
        "reference_url": "https://arxiv.org/abs/1206.3567",
        "assumptions": ["LTE and chemical equilibrium", "ideal mixture",
                        "seven H/He species", "fixed H2 ortho:para=3:1",
                        "ground electronic states", "metals omitted"],
        "composition": {"X": args.hydrogen, "Y": args.helium,
                        "unmodelled_mass_fraction": 1.0-args.hydrogen-args.helium},
        "h2_vibration": args.vibration,
        "grid": {"logrho": [args.logrho_min, args.logrho_max, args.nrho],
                 "logT": [args.logt_min, args.logt_max, args.ntemp],
                 "logx2": [float(xlim[0]), float(xlim[1]), args.nx2]},
        "ratios": list(map(float, ratios)),
        "conservation_max_relative": {
            "hydrogen": float(forward["conservation"][0]),
            "helium": float(forward["conservation"][1]),
            "charge": float(forward["conservation"][2])},
        "ranges": {key: [float(np.min(forward[key])), float(np.max(forward[key]))]
                   for key in ("p", "e", "s", "cv", "gamma1", "nabla_ad")},
        "checks": checks,
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    meta = args.metadata or args.output.with_suffix(args.output.suffix + ".json")
    meta.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.npz:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz, logrho=logrho, logT=logt, **forward)
    print(json.dumps(metadata, indent=2))
    if not checks["pass"]:
        raise SystemExit("EOS audit failed")


if __name__ == "__main__":
    main()
