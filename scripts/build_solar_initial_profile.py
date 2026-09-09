#!/usr/bin/env python3
"""Build a hydrostatic, radiative-convective 1D solar surface profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


ARAD = 7.5657e-15
CLIGHT = 2.99792458e10


def flux_limiter(reduced_gradient):
    return (2.0 + reduced_gradient) / (
        6.0 + 3.0 * reduced_gradient + reduced_gradient**2
    )


class AthenaEosTable:
    """Minimal reader for the Athena general-EOS ASCII table."""

    def __init__(self, path: Path):
        tokens = []
        with path.open() as stream:
            for line in stream:
                line = line.strip()
                if line and not line.startswith("#"):
                    tokens.extend(line.split())

        values = iter(tokens)
        self.nvar = int(next(values))
        self.nx2 = int(next(values))
        self.nrho = int(next(values))
        self.logx2_min, self.logx2_max = float(next(values)), float(next(values))
        self.logrho_min, self.logrho_max = float(next(values)), float(next(values))
        self.ratios = np.array([float(next(values)) for _ in range(self.nvar)])
        raw = np.fromiter((float(value) for value in values), dtype=float)
        expected = self.nvar * self.nx2 * self.nrho
        if raw.size != expected:
            raise ValueError(f"EOS table has {raw.size} values; expected {expected}")
        self.data = raw.reshape(self.nvar, self.nx2, self.nrho)
        self.logx2 = np.linspace(self.logx2_min, self.logx2_max, self.nx2)
        self.logrho = np.linspace(self.logrho_min, self.logrho_max, self.nrho)
        self.interp = [
            RegularGridInterpolator(
                (self.logx2, self.logrho), field, bounds_error=True
            )
            for field in self.data
        ]

    def _field(self, index: int, logx2: float, logrho: float) -> float:
        return float(self.interp[index]((logx2, logrho)))

    def state_from_rho_temperature(self, rho: float, temperature: float):
        logrho = np.log10(rho)
        if not self.logrho_min <= logrho <= self.logrho_max:
            raise ValueError(f"logRho={logrho:.6g} is outside the EOS table")

        def temperature_error(logu):
            return self._field(3, logu, logrho) - np.log10(temperature)

        fmin = temperature_error(self.logx2_min)
        fmax = temperature_error(self.logx2_max)
        if fmin * fmax > 0.0:
            raise ValueError(
                f"T={temperature:.6g} K at logRho={logrho:.6g} is outside "
                "the EOS logU range"
            )
        logu = brentq(temperature_error, self.logx2_min, self.logx2_max)
        pressure = 10.0 ** self._field(0, logu, logrho) * rho * 10.0**logu
        logq = np.log10(pressure / rho)
        gamma1 = 10.0 ** self._field(2, logq, logrho)
        entropy = (
            10.0 ** self._field(4, logu, logrho)
            if self.nvar >= 5
            else np.nan
        )
        return pressure, 10.0**logu, gamma1, entropy, logu

    def state_from_pressure_temperature(self, pressure: float, temperature: float):
        def pressure_error(logrho):
            trial_pressure = self.state_from_rho_temperature(
                10.0**logrho, temperature
            )[0]
            return np.log(trial_pressure / pressure)

        fmin = pressure_error(self.logrho_min)
        fmax = pressure_error(self.logrho_max)
        if fmin * fmax > 0.0:
            raise ValueError(
                f"P={pressure:.6g}, T={temperature:.6g} cannot be inverted "
                "inside the EOS density range"
            )
        logrho = brentq(pressure_error, self.logrho_min, self.logrho_max)
        rho = 10.0**logrho
        state = self.state_from_rho_temperature(rho, temperature)
        return rho, state[1], state[2], state[3], state[4]


class OpacityTable:
    def __init__(self, path: Path):
        with h5py.File(path, "r") as handle:
            self.logt = handle["axes/log10_temperature"][...]
            self.logrho = handle["axes/log10_density"][...]
            rosseland = handle["kappa/log10_rosseland"][...]
        self.interp = RegularGridInterpolator(
            (self.logrho, self.logt), rosseland, bounds_error=True
        )

    def rosseland(self, rho: float, temperature: float) -> float:
        return 10.0 ** float(
            self.interp((np.log10(rho), np.log10(temperature)))
        )


class ProfileBuilder:
    def __init__(self, args, eos, opacity):
        self.args = args
        self.eos = eos
        self.opacity = opacity
        self.flux = args.flux
        self.top_temperature = (
            self.flux / (args.top_alpha * CLIGHT * ARAD)
        ) ** 0.25

    def derivatives(self, log_pressure, log_temperature, tau):
        pressure = np.exp(log_pressure)
        temperature = np.exp(log_temperature)
        rho, specific_u, gamma1, entropy, logu = (
            self.eos.state_from_pressure_temperature(pressure, temperature)
        )
        kappa = self.opacity.rosseland(rho, temperature)
        flux_factor = self.flux / (CLIGHT * ARAD * temperature**4)
        # Grey diffusion gives dE/dtau=3F/c.  The variable FLD limiter is
        # applied below when diagnosing the flux represented on the grid.
        reduced_gradient = 3.0 * flux_factor
        dlogp_dd = rho * self.args.gravity / pressure
        dlogt_rad_dd = 0.25 * kappa * rho * reduced_gradient
        nabla_ad = (gamma1 - 1.0) / gamma1
        nabla_rad = dlogt_rad_dd / dlogp_dd
        nabla_cap = nabla_ad + self.args.superadiabatic_excess
        blend = 0.5 * (
            1.0
            + np.tanh(
                (nabla_rad - nabla_cap) / self.args.gradient_blend_width
            )
        )
        nabla_used = (1.0 - blend) * nabla_rad + blend * nabla_cap
        dlogt_dd = nabla_used * dlogp_dd
        dtau_dd = kappa * rho
        return np.array([dlogp_dd, dlogt_dd, dtau_dd]), (
            rho,
            specific_u,
            gamma1,
            entropy,
            logu,
            kappa,
            nabla_ad,
            reduced_gradient,
        )

    def integrate(self, logrho_top, bottom_z, steps, keep_all=True):
        rho_top = 10.0**logrho_top
        pressure_top = self.eos.state_from_rho_temperature(
            rho_top, self.top_temperature
        )[0]
        state = np.array(
            [np.log(pressure_top), np.log(self.top_temperature), 0.0]
        )
        depth = (self.args.zmax - bottom_z) * self.args.length_unit
        step = depth / steps
        records = []
        for index in range(steps + 1):
            distance = index * step
            z_code = self.args.zmax - distance / self.args.length_unit
            deriv, thermo = self.derivatives(*state)
            if keep_all:
                records.append((z_code, distance, *state, *deriv, *thermo))
            if index == steps:
                break
            midpoint = state + 0.5 * step * deriv
            midpoint_deriv, _ = self.derivatives(*midpoint)
            state += step * midpoint_deriv
        return np.asarray(records) if keep_all else state

    def tau_at_target(self, logrho_top, steps):
        state = self.integrate(
            logrho_top, self.args.tau_target_z, steps, keep_all=False
        )
        return state[2]

    def find_top_density(self):
        margin = self.args.minimum_eos_margin
        lo = self.eos.logrho_min + margin
        hi = self.eos.logrho_max - margin

        def mismatch(logrho_top):
            tau = self.tau_at_target(logrho_top, self.args.tune_steps)
            return np.log(tau / self.args.tau_target)

        previous = None
        samples = []
        for trial in np.linspace(lo, hi, 33):
            try:
                value = mismatch(trial)
            except ValueError:
                continue
            samples.append((trial, value))
            if previous is not None and previous[1] * value <= 0.0:
                return brentq(mismatch, previous[0], trial, xtol=1.0e-10)
            previous = (trial, value)
        sampled = ", ".join(
            f"({logrho:.3f},{np.exp(value):.3g})" for logrho, value in samples
        )
        raise RuntimeError(
            "Cannot place the requested optical depth with the requested EOS "
            f"margin. Valid (logRho_top,tau_target_z) samples: {sampled}"
        )


def sample_profile(builder, fine, args):
    z_fine = fine[:, 0]
    z = np.linspace(args.zmin, args.zmax, args.nz)
    columns = {}
    source = {
        "pressure": np.exp(fine[:, 2]),
        "temperature": np.exp(fine[:, 3]),
        "tau": fine[:, 4],
    }
    for name, values in source.items():
        columns[name] = np.interp(z, z_fine[::-1], values[::-1])
    columns["z"] = z

    rho = np.empty(args.nz)
    specific_u = np.empty(args.nz)
    gamma1 = np.empty(args.nz)
    entropy = np.empty(args.nz)
    kappa = np.empty(args.nz)
    for index in range(args.nz):
        state = builder.eos.state_from_pressure_temperature(
            columns["pressure"][index], columns["temperature"][index]
        )
        rho[index], specific_u[index], gamma1[index], entropy[index], _ = state
        kappa[index] = builder.opacity.rosseland(rho[index], columns["temperature"][index])
    columns.update(
        rho=rho,
        specific_u=specific_u,
        gamma1=gamma1,
        entropy=entropy,
        kappa_R=kappa,
    )

    z_cm = z * args.length_unit
    pressure = columns["pressure"]
    temperature = columns["temperature"]
    erad = ARAD * temperature**4
    dlnp_dz = np.gradient(np.log(pressure), z_cm, edge_order=2)
    dlnt_dz = np.gradient(np.log(temperature), z_cm, edge_order=2)
    dE_dz = np.gradient(erad, z_cm, edge_order=2)
    sigma = kappa * rho
    reduced_gradient = np.abs(dE_dz) / np.maximum(sigma * erad, 1.0e-300)
    flux = -CLIGHT * flux_limiter(reduced_gradient) * dE_dz / sigma
    nabla = dlnt_dz / dlnp_dz
    nabla_ad = (gamma1 - 1.0) / gamma1
    hse = np.gradient(pressure, z_cm, edge_order=2) + rho * args.gravity
    hse_relative = hse / np.maximum(rho * args.gravity, 1.0e-300)
    columns.update(
        z_cm=z_cm,
        erad=erad,
        flux_rad=flux,
        flux_ratio=flux / args.flux,
        convective_flux_fraction=np.maximum(0.0, 1.0 - flux / args.flux),
        nabla=nabla,
        nabla_ad=nabla_ad,
        superadiabaticity=nabla - nabla_ad,
        hse_relative=hse_relative,
        logrho_margin=np.minimum(
            np.log10(rho) - builder.eos.logrho_min,
            builder.eos.logrho_max - np.log10(rho),
        ),
        logu_margin=np.minimum(
            np.log10(specific_u) - builder.eos.logx2_min,
            builder.eos.logx2_max - np.log10(specific_u),
        ),
        logt_opacity_margin=np.minimum(
            np.log10(temperature) - builder.opacity.logt[0],
            builder.opacity.logt[-1] - np.log10(temperature),
        ),
        logrho_opacity_margin=np.minimum(
            np.log10(rho) - builder.opacity.logrho[0],
            builder.opacity.logrho[-1] - np.log10(rho),
        ),
    )
    return columns


def write_profile(path, profile, args, logrho_top):
    names = list(profile)
    metadata = (
        f"rho_top={10**logrho_top:.16e} T_top={profile['temperature'][-1]:.16e} "
        f"length_unit={args.length_unit:.16e} flux_target={args.flux:.16e}"
    )
    array = np.column_stack([profile[name] for name in names])
    np.savetxt(
        path,
        array,
        fmt="%.16e",
        header=f"Solar 1D initial profile; {metadata}\n" + " ".join(names),
    )


def plot_profile(path, profile, args):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    z = profile["z"]
    axes[0, 0].plot(z, profile["temperature"])
    axes[0, 0].set(ylabel="T [K]")
    density_axis = axes[0, 1]
    pressure_axis = density_axis.twinx()
    density_line = density_axis.semilogy(z, profile["rho"], label="density")
    pressure_line = pressure_axis.semilogy(
        z, profile["pressure"], color="tab:orange", label="pressure"
    )
    density_axis.legend(density_line + pressure_line, ["density", "pressure"])
    density_axis.set(ylabel="rho [g cm$^{-3}$]")
    pressure_axis.set(ylabel="P [dyn cm$^{-2}$]")
    axes[1, 0].semilogy(z, np.maximum(profile["tau"], 1.0e-12))
    axes[1, 0].axhline(args.tau_target, color="k", lw=0.8)
    axes[1, 0].set(ylabel="optical depth")
    axes[1, 1].plot(z, profile["flux_ratio"])
    axes[1, 1].axhline(1.0, color="k", lw=0.8)
    axes[1, 1].set(ylabel=r"$F_{rad}/F_{sun}$")
    axes[2, 0].plot(z, profile["superadiabaticity"])
    axes[2, 0].axhline(0.0, color="k", lw=0.8)
    axes[2, 0].set(ylabel=r"$\nabla-\nabla_{ad}$")
    axes[2, 1].plot(z, profile["hse_relative"])
    axes[2, 1].axhline(0.0, color="k", lw=0.8)
    axes[2, 1].set(ylabel="relative HSE residual")
    for axis in axes.flat:
        axis.set_xlabel("z [code]")
        axis.tick_params(labelsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report(profile, args, eos, opacity, logrho_top):
    tau = profile["tau"]
    order = np.argsort(tau)
    z_tau = np.interp(args.tau_target, tau[order], profile["z"][order])
    t_tau = np.interp(args.tau_target, tau[order], profile["temperature"][order])
    flux_tau = np.interp(args.tau_target, tau[order], profile["flux_ratio"][order])
    imax_superad = np.argmax(profile["superadiabaticity"])
    interior = slice(1, -1)
    print("### Solar 1D profile diagnostics")
    print(f"rho_top = {10**logrho_top:.15e} g cm^-3")
    print(
        f"tau=1: z = {z_tau:.8f}, T = {t_tau:.3f} K, "
        f"Frad/Fsun = {flux_tau:.6e}"
    )
    print(
        "max(nabla-nabla_ad) = "
        f"{profile['superadiabaticity'][imax_superad]:.6e} at "
        f"z = {profile['z'][imax_superad]:.8f}"
    )
    print(
        "Frad/Fsun min/max = "
        f"{np.min(profile['flux_ratio'][interior]):.6e} / "
        f"{np.max(profile['flux_ratio'][interior]):.6e}"
    )
    photosphere = profile["tau"] <= 3.0
    print(
        "Frad/Fsun for tau<=3 min/max = "
        f"{np.min(profile['flux_ratio'][photosphere]):.6e} / "
        f"{np.max(profile['flux_ratio'][photosphere]):.6e}"
    )
    print(
        "HSE relative max/rms = "
        f"{np.max(np.abs(profile['hse_relative'][interior])):.6e} / "
        f"{np.sqrt(np.mean(profile['hse_relative'][interior]**2)):.6e}"
    )
    print(
        f"EOS logRho used = [{np.log10(profile['rho']).min():.6f}, "
        f"{np.log10(profile['rho']).max():.6f}], table = "
        f"[{eos.logrho_min:.6f}, {eos.logrho_max:.6f}], "
        f"minimum margin = {profile['logrho_margin'].min():.3f} dex"
    )
    print(
        f"EOS logU used = [{np.log10(profile['specific_u']).min():.6f}, "
        f"{np.log10(profile['specific_u']).max():.6f}], table = "
        f"[{eos.logx2_min:.6f}, {eos.logx2_max:.6f}], "
        f"minimum margin = {profile['logu_margin'].min():.3f} dex"
    )
    print(
        f"opacity logRho/logT minimum margins = "
        f"{profile['logrho_opacity_margin'].min():.3f} / "
        f"{profile['logt_opacity_margin'].min():.3f} dex"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eos-table", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=3.072)
    parser.add_argument("--nz", type=int, default=1025)
    parser.add_argument("--length-unit", type=float, default=5.0e7)
    parser.add_argument("--gravity", type=float, default=2.74e4)
    parser.add_argument("--flux", type=float, default=6.28e10)
    parser.add_argument("--top-alpha", type=float, default=0.5)
    parser.add_argument("--tau-target-z", type=float, default=2.0)
    parser.add_argument("--tau-target", type=float, default=1.0)
    parser.add_argument("--superadiabatic-excess", type=float, default=1.0e-2)
    parser.add_argument("--gradient-blend-width", type=float, default=2.0e-2)
    parser.add_argument("--minimum-eos-margin", type=float, default=0.2)
    parser.add_argument("--tune-steps", type=int, default=256)
    parser.add_argument("--integration-steps", type=int, default=8192)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gradient_blend_width <= 0.0:
        raise ValueError("--gradient-blend-width must be positive")
    if args.top_alpha <= 0.0:
        raise ValueError("--top-alpha must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eos = AthenaEosTable(args.eos_table)
    opacity = OpacityTable(args.opacity_table)
    builder = ProfileBuilder(args, eos, opacity)
    logrho_top = builder.find_top_density()
    fine = builder.integrate(logrho_top, args.zmin, args.integration_steps)
    profile = sample_profile(builder, fine, args)
    data_path = args.output_dir / "solar_initial_profile.dat"
    plot_path = args.output_dir / "solar_initial_profile.png"
    write_profile(data_path, profile, args, logrho_top)
    plot_profile(plot_path, profile, args)
    report(profile, args, eos, opacity, logrho_top)
    print(f"Wrote {data_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
