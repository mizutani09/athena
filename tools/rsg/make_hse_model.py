#!/usr/bin/env python3
"""Build a table-EOS, fixed-Freytag-potential RSG profile for Athena++.

The current initialization prescribes a smooth potential-based temperature
profile between configurable central and surface temperatures and integrates
hydrostatic balance with the production EOS.  It is a controlled HSE starting
model, not yet the final adiabatic-interior plus gray-atmosphere construction.
When ``--grid`` is given, the 1-D shooting target is iterated until the mass
after mapping onto the actual Cartesian cube matches ``--mass-msun``.  This
avoids a post-hoc multiplication of the 3-D density field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 6.67430e-8
MSUN = 1.98847e33
RSUN = 6.957e10
AR = 7.5657e-15


def numeric_tokens(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield from line.split()


class AthenaEos:
    """Reader/interpolator for the repository's ASCII general-EOS format."""

    def __init__(self, path):
        values = iter(numeric_tokens(path))
        self.nvar, self.nq, self.nrho = (int(next(values)) for _ in range(3))
        self.q_lim = np.array([float(next(values)), float(next(values))])
        self.rho_lim = np.array([float(next(values)), float(next(values))])
        self.ratios = np.array([float(next(values)) for _ in range(self.nvar)])
        count = self.nvar*self.nq*self.nrho
        self.fields = np.fromiter((float(next(values)) for _ in range(count)),
                                  dtype=float, count=count).reshape(
                                      self.nvar, self.nq, self.nrho)
        # Some legacy local files append a separate S(P/rho,rho) table after
        # the standard Athena fields.  Current NATA converters instead store
        # entropy as fields 4 and 5 in the main table, and four-field tables
        # contain no entropy at all.  The temperature-prescribed HSE builder
        # only needs fields 0 and 3, so accept all three schemas.
        self.entropy = None
        self.entropy_is_log10 = False
        trailing = list(values)
        if trailing:
            values = iter(trailing)
            try:
                nent, nq, nrho = (int(next(values)) for _ in range(3))
            except (StopIteration, RuntimeError) as exc:
                raise ValueError("truncated appended entropy-table header") from exc
            if (nent, nq, nrho) != (1, self.nq, self.nrho):
                raise ValueError("unexpected appended entropy-table shape")
            ent_q_lim = np.array([float(next(values)), float(next(values))])
            ent_rho_lim = np.array([float(next(values)), float(next(values))])
            _ = float(next(values))
            if not (np.allclose(ent_q_lim, self.q_lim)
                    and np.allclose(ent_rho_lim, self.rho_lim)):
                raise ValueError("EOS and entropy axes differ")
            self.entropy = np.fromiter(map(float, values), dtype=float,
                                       count=self.nq*self.nrho).reshape(
                                           self.nq, self.nrho)
        self.q_axis = np.linspace(*self.q_lim, self.nq)
        self.rho_axis = np.linspace(*self.rho_lim, self.nrho)

    @staticmethod
    def _axis_index(axis, x):
        if x < axis[0] or x > axis[-1]:
            raise ValueError(f"EOS lookup {x:.6g} outside [{axis[0]}, {axis[-1]}]")
        u = (x-axis[0])/(axis[-1]-axis[0])*(axis.size-1)
        i = min(int(np.floor(u)), axis.size-2)
        return i, u-i

    def interp(self, field, logq, logrho):
        iq, fq = self._axis_index(self.q_axis, logq)
        ir, fr = self._axis_index(self.rho_axis, logrho)
        a = field[iq:iq+2, ir:ir+2]
        return ((1-fq)*(1-fr)*a[0, 0] + fq*(1-fr)*a[1, 0]
                + (1-fq)*fr*a[0, 1] + fq*fr*a[1, 1])

    def state_from_rho_entropy(self, rho, entropy):
        if self.entropy is None:
            if self.nvar < 6:
                raise ValueError("EOS table has no entropy field")
            entropy_field = self.fields[5]
            entropy_is_log10 = True
        else:
            entropy_field = self.entropy
            entropy_is_log10 = self.entropy_is_log10
        lr = np.log10(rho)
        ir, fr = self._axis_index(self.rho_axis, lr)
        logs = ((1-fr)*entropy_field[:, ir]
                + fr*entropy_field[:, ir+1])
        s = 10.0**logs if entropy_is_log10 else logs
        valid = np.isfinite(s) & (s > 0.0)
        q, s = self.q_axis[valid], s[valid]
        order = np.argsort(s)
        s, q = s[order], q[order]
        keep = np.r_[True, np.diff(s) > 0.0]
        s, q = s[keep], q[keep]
        if entropy < s[0] or entropy > s[-1]:
            raise ValueError("requested isentrope leaves EOS entropy domain")
        logq = np.interp(entropy, s, q)
        qphys = 10.0**logq
        log_e_over_p = self.interp(self.fields[1], logq, lr)
        espec = qphys*10.0**log_e_over_p
        log_espec = np.log10(espec)
        temp = 10.0**self.interp(self.fields[3], log_espec, lr)
        gamma1 = 10.0**self.interp(self.fields[2], logq, lr)
        return rho*qphys, rho*espec, temp, gamma1

    def state_from_rho_temp(self, rho, temperature):
        lr = np.log10(rho)
        ir, fr = self._axis_index(self.rho_axis, lr)
        logt = ((1-fr)*self.fields[3, :, ir]
                + fr*self.fields[3, :, ir+1])
        valid = np.isfinite(logt) & (logt > -50.0)
        x, y = logt[valid], self.q_axis[valid]
        order = np.argsort(x)
        x, y = x[order], y[order]
        keep = np.r_[True, np.diff(x) > 0.0]
        x, y = x[keep], y[keep]
        target = np.log10(temperature)
        if target < x[0] or target > x[-1]:
            raise ValueError("requested temperature leaves EOS domain")
        log_espec = np.interp(target, x, y)
        espec = 10.0**log_espec
        pressure = rho*espec*10.0**self.interp(self.fields[0], log_espec, lr)
        return pressure, rho*espec


def potential(r, gm, r0, r1):
    a = r0**4 + r**4/np.sqrt(1.0+(r/r1)**8)
    return -gm/a**0.25


def gravity(r, gm, r0, r1):
    b = 1.0+(r/r1)**8
    a = r0**4+r**4/np.sqrt(b)
    return gm*r**3/(b**1.5*a**1.25)


def integrate_envelope(eos, logrho0, radius, outer_radius, mass_target,
                       gm, r0, r1, points, atmosphere_points,
                       surface_temperature, central_temperature, density_floor):
    rho0 = 10.0**logrho0
    rmin = radius*1.0e-9
    m0 = 4.0*np.pi*rmin**3*rho0/3.0
    phi0, phis = potential(0.0, gm, r0, r1), potential(radius, gm, r0, r1)

    def temperature(r):
        if r >= radius:
            return surface_temperature
        f = (potential(r, gm, r0, r1)-phis)/(phi0-phis)
        return surface_temperature+(central_temperature-surface_temperature)*f

    def rhs(r, y):
        rho = np.exp(y[0])
        if rho <= density_floor:
            return [0.0, 4.0*np.pi*r*r*density_floor]
        temp = temperature(r)
        press, _ = eos.state_from_rho_temp(rho, temp)
        eps = 2.0e-4
        pp, _ = eos.state_from_rho_temp(rho*np.exp(eps), temp)
        pm, _ = eos.state_from_rho_temp(rho*np.exp(-eps), temp)
        dp_dlnrho = (pp-pm)/(2.0*eps)
        if r < radius:
            tp = temperature(min(radius, r*(1.0+eps)))
            tm = temperature(max(rmin, r*(1.0-eps)))
            p_tp, _ = eos.state_from_rho_temp(rho, tp)
            p_tm, _ = eos.state_from_rho_temp(rho, tm)
            dp_dr_t = (p_tp-p_tm)/(min(radius, r*(1.0+eps))-max(rmin, r*(1.0-eps)))
        else:
            dp_dr_t = 0.0
        return [(-rho*gravity(r, gm, r0, r1)-dp_dr_t)/dp_dlnrho,
                4.0*np.pi*r*r*rho]

    inner = np.geomspace(rmin, radius, points)
    outer = (np.geomspace(radius*(1.0+1e-9), outer_radius, atmosphere_points)
             if outer_radius > radius else np.empty(0))
    radii = np.r_[inner, outer]
    sol = solve_ivp(rhs, (rmin, outer_radius), [np.log(rho0), m0], t_eval=radii,
                    rtol=2e-7, atol=[2e-8, mass_target*1e-9], method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    rho = np.maximum(np.exp(sol.y[0]), density_floor)
    temp = np.array([temperature(x) for x in radii])
    states = np.array([eos.state_from_rho_temp(d, t) for d, t in zip(rho, temp)])
    return radii, rho, states, temp, sol.y[1]


def mapped_mass(radii, rho, grid, half_width):
    x = (np.arange(grid)+0.5)*(2.0*half_width/grid)-half_width
    dx = 2.0*half_width/grid
    logr, logd = np.log(radii), np.log(rho)
    total = 0.0
    xy2 = x[:, None]**2+x[None, :]**2
    for z in x:
        r = np.sqrt(xy2+z*z)
        d = np.exp(np.interp(np.log(np.maximum(r, radii[0])), logr, logd))
        total += d.sum()*dx**3
    return total


def build(args, eos, mass_inside_target):
    radius = args.radius_rsun*RSUN
    gm = G*args.potential_mass_msun*MSUN
    r0, r1 = args.r0_fraction*radius, args.r1_fraction*radius

    corner = np.sqrt(3.0)*args.box_half_rsun*RSUN

    def residual(logrho):
        try:
            r, _, _, _, mass = integrate_envelope(
                eos, logrho, radius, corner, mass_inside_target, gm, r0, r1,
                args.points, args.atmosphere_points, args.surface_temperature,
                args.central_temperature, args.density_floor)
            at_surface = np.searchsorted(r, radius)
            return np.log10(mass[at_surface]/mass_inside_target)
        except (ValueError, RuntimeError, FloatingPointError):
            return np.nan

    # A coarse nine-point scan can step over the valid root when a nearby
    # trial trajectory briefly leaves a finite EOS table.  The denser scan is
    # cheap compared with the final Cartesian mass integration and preserves
    # the requirement that both endpoints themselves be valid.
    candidates = np.linspace(args.initial_logrho-2.0,
                             args.initial_logrho+2.0, 33)
    samples = [(x, residual(x)) for x in candidates]
    bracket = next(((x0, x1) for (x0, f0), (x1, f1) in zip(samples, samples[1:])
                    if np.isfinite(f0) and np.isfinite(f1) and f0*f1 <= 0.0), None)
    if bracket is None:
        raise RuntimeError(f"could not bracket HSE central density: {samples}")
    logrho = brentq(residual, *bracket, xtol=2e-8, rtol=2e-8)
    if abs(residual(logrho)) > 2e-4:
        raise RuntimeError(f"HSE shooting residual remained {residual(logrho)}")
    r, rho, states, temp, mass = integrate_envelope(
        eos, logrho, radius, corner, mass_inside_target, gm, r0, r1,
        args.points, args.atmosphere_points, args.surface_temperature,
        args.central_temperature, args.density_floor)
    press, egas = states[:, 0], states[:, 1]
    erad = AR*temp**4
    return r, rho, press, egas, erad, temp, mass, logrho


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eos", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--potential-mass-msun", type=float, default=12.0)
    p.add_argument("--mass-msun", type=float, default=3.0)
    p.add_argument("--radius-rsun", type=float, default=830.0)
    p.add_argument("--surface-temperature", type=float, default=3487.0)
    p.add_argument("--r0-fraction", type=float, default=0.2)
    p.add_argument("--r1-fraction", type=float, default=1.2)
    p.add_argument("--box-half-rsun", type=float, default=1010.5)
    p.add_argument("--grid", type=int, default=0)
    p.add_argument("--grid-mass-tolerance", type=float, default=5e-3)
    p.add_argument("--grid-iterations", type=int, default=5)
    p.add_argument("--points", type=int, default=2048)
    p.add_argument("--atmosphere-points", type=int, default=512)
    p.add_argument("--central-temperature", type=float, default=1.0e5)
    p.add_argument("--density-floor", type=float, default=2.0e-18)
    p.add_argument("--initial-logrho", type=float, default=-6.0)
    return p.parse_args()


def main():
    args = parse_args()
    eos = AthenaEos(args.eos)
    desired = args.mass_msun*MSUN
    inside_target = desired
    mapped = None
    for _ in range(args.grid_iterations if args.grid else 1):
        result = build(args, eos, inside_target)
        if not args.grid:
            break
        mapped = mapped_mass(result[0], result[1], args.grid,
                             args.box_half_rsun*RSUN)
        rel = mapped/desired-1.0
        if abs(rel) <= args.grid_mass_tolerance:
            break
        inside_target *= desired/mapped
    if args.grid and abs(mapped/desired-1.0) > args.grid_mass_tolerance:
        raise RuntimeError("Cartesian mapped mass did not converge")
    r, rho, press, egas, erad, temp, mass, logrho = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = ("r_cm rho_g_cm3 egas_erg_cm3 erad_erg_cm3 T_K Pgas_dyn_cm2 "
              "M_spherical_g\nGenerated with the Athena general EOS and Freytag fixed potential")
    np.savetxt(args.output, np.column_stack((r, rho, egas, erad, temp, press, mass)),
               fmt="%.16e", header=header)
    metadata = {
        "eos": str(args.eos.resolve()),
        "eos_sha256": hashlib.sha256(args.eos.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "potential_mass_msun": args.potential_mass_msun,
        "requested_grid_mass_msun": args.mass_msun,
        "radius_rsun": args.radius_rsun,
        "surface_temperature_K": args.surface_temperature,
        "r0_fraction": args.r0_fraction,
        "r1_fraction": args.r1_fraction,
        "central_temperature_K": args.central_temperature,
        "density_floor_g_cm3": args.density_floor,
        "shooting_log10_rhoc": float(logrho),
        "mass_inside_radius_msun": float(mass[np.searchsorted(r, args.radius_rsun*RSUN)]/MSUN),
        "grid": args.grid or None,
        "mapped_mass_msun": None if mapped is None else float(mapped/MSUN),
    }
    meta = args.metadata or args.output.with_suffix(args.output.suffix+".json")
    meta.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
