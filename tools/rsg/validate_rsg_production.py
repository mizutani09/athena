#!/usr/bin/env python3
"""Validate the material-EOS RSG production assets and their provenance.

This is intentionally independent of an Athena++ executable.  It checks that
the input file selects the production physics, that all sidecar checksums still
match, that the EOS/opacity domains contain the initial profile, and that the
tabulated entropy forward/inverse paths agree along that profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from make_hse_model import AthenaEos, MSUN, RSUN


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_input(path: Path) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    block = ""
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            blocks.setdefault(block, {})
        elif block and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            blocks[block][key] = value
    return blocks


def as_bool(value: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError(f"expected true/false, got {value!r}")
    return value == "true"


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--input", type=Path,
        default=root / "inputs/rsg/athinput.st35gm03n07_fld")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--samples", type=int, default=257)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    blocks = read_input(args.input)
    checks: list[str] = []

    hydro, fld, nr, problem = (
        blocks[name] for name in ("hydro", "fld", "nrfld", "problem"))
    require(as_bool(fld["is_couple"]), "matter-radiation coupling enabled", checks)
    require(as_bool(fld["include_radiation_force"]), "radiation force enabled", checks)
    require(not as_bool(fld["radiation_pressure_in_riemann"]),
            "radiation force uses source-only Riemann mode", checks)
    require(not as_bool(fld["fixed_flux_limiter"]),
            "nonlinear FLD limiter enabled", checks)
    require(fld["opacity_table_domain_policy"] == "error",
            "opacity out-of-domain policy is fatal", checks)
    require(not as_bool(hydro["eos_table_clamp"]),
            "EOS table clamping disabled", checks)
    require(problem["radiation_boundary"] == "marshak_vacuum",
            "six-face Marshak-vacuum boundary selected", checks)
    require(nr["linear_solver"] == "fgmres"
            and nr["coarse_operator"] == "galerkin",
            "FGMRES with Galerkin MG preconditioner selected", checks)
    require(as_bool(nr["backward_error_convergence"])
            and as_bool(nr["exact_local_gas_solve"]),
            "backward-error and exact local gas solves enabled", checks)

    eos_path = root / hydro["eos_file_name"]
    opacity_path = root / fld["opacity_table_file"]
    profile_path = root / problem["hse_model_file"]
    eos_meta = json.loads(Path(str(eos_path) + ".json").read_text())
    opacity_meta = json.loads(Path(str(opacity_path) + ".json").read_text())
    profile_meta = json.loads(Path(str(profile_path) + ".json").read_text())

    require(sha256(eos_path) == eos_meta["output_sha256"],
            "EOS checksum matches sidecar", checks)
    require(sha256(opacity_path) == opacity_meta["output_sha256"],
            "opacity checksum matches sidecar", checks)
    require(sha256(profile_path) == profile_meta["profile_sha256"],
            "HSE profile checksum matches sidecar", checks)
    require(profile_meta["eos_sha256"] == eos_meta["output_sha256"]
            and profile_meta["opacity_sha256"] == opacity_meta["output_sha256"],
            "HSE profile was built from the selected EOS and opacity", checks)
    require(eos_meta["schema"] == "athena-general-eos-v7"
            and eos_meta["material_only"]
            and not eos_meta["photon_pressure_energy_entropy"],
            "EOS is the seven-field material-only schema", checks)
    require(eos_meta["checks"]["pass"], "EOS builder audit passed", checks)

    profile = np.loadtxt(profile_path)
    finite_columns = np.r_[0:7, 8:11]
    require(profile.ndim == 2 and profile.shape[1] >= 11
            and np.isfinite(profile[:, finite_columns]).all()
            and np.all(np.isnan(profile[:, 7]) | (profile[:, 7] >= 0.0)),
            "HSE profile has eleven valid columns (interior tau is undefined)",
            checks)
    require(np.all(profile[:, 0] > 0.0)
            and np.all(np.diff(profile[:, 0]) > 0.0)
            and np.all(profile[:, 1:6] > 0.0),
            "HSE radius is monotone and thermodynamic fields are positive", checks)

    logrho = np.log10(profile[:, 1])
    logtemp = np.log10(profile[:, 4])
    erho = eos_meta["grid"]["logrho"][:2]
    etemp = eos_meta["grid"]["logT"][:2]
    require(logrho.min() >= erho[0] and logrho.max() <= erho[1]
            and logtemp.min() >= etemp[0] and logtemp.max() <= etemp[1],
            "initial profile lies inside the EOS rho-T domain", checks)
    with h5py.File(opacity_path, "r") as handle:
        orho = np.asarray(handle["axes/log10_density"])
        otemp = np.asarray(handle["axes/log10_temperature"])
        for name in ("log10_planck", "log10_rosseland"):
            require(np.isfinite(handle[f"kappa/{name}"][:]).all(),
                    f"opacity {name} is finite", checks)
    require(logrho.min() >= orho[0] and logrho.max() <= orho[-1]
            and logtemp.min() >= otemp[0] and logtemp.max() <= otemp[-1],
            "initial profile lies inside the opacity rho-T domain", checks)

    radius = float(problem["radius_rsun"]) * RSUN
    at_surface = np.searchsorted(profile[:, 0], radius)
    mass = profile[at_surface, 6] / MSUN
    target_mass = float(problem["gas_mass_msun"])
    tolerance = float(problem["grid_mass_tolerance"])
    require(abs(mass / target_mass - 1.0) <= tolerance,
            "spherical envelope mass meets the configured tolerance", checks)
    require(abs(profile[at_surface, 4]
                / float(problem["surface_temperature"]) - 1.0) < 2.0e-3,
            "photospheric temperature matches the configured Teff", checks)
    require(profile_meta["hse_relative"]["max_abs_resolved_interior"] < 0.02,
            "resolved-interior HSE residual is below two percent", checks)

    eos = AthenaEos(eos_path)
    require(eos.nvar == 7 and np.isfinite(eos.fields).all()
            and np.all(eos.fields > -90.0),
            "EOS has seven finite fields and no sentinel holes", checks)
    errors = []
    indices = np.linspace(0, profile.shape[0] - 1,
                          min(args.samples, profile.shape[0]), dtype=int)
    for index in indices:
        rho, energy, temperature, pressure = profile[index, [1, 2, 4, 5]]
        logrho_i = np.log10(rho)
        logu = np.log10(energy / rho)
        entropy = 10.0 ** eos.interp(
            eos.fields[4], logu + np.log10(eos.ratios[4]), logrho_i)
        pressure_i, energy_i, temperature_i, _ = eos.state_from_rho_entropy(
            rho, entropy)
        errors.append((abs(np.log10(energy_i / energy)),
                       abs(np.log10(temperature_i / temperature)),
                       abs(np.log10(pressure_i / pressure))))
    roundtrip = np.max(np.asarray(errors), axis=0)
    require(np.all(roundtrip < 2.0e-3),
            "entropy round trip is below 0.002 dex along the profile", checks)

    result = {
        "pass": True,
        "input": str(args.input.resolve()),
        "checks": checks,
        "assets": {
            "eos_sha256": sha256(eos_path),
            "opacity_sha256": sha256(opacity_path),
            "profile_sha256": sha256(profile_path),
        },
        "spherical_mass_msun": float(mass),
        "resolved_interior_hse_max":
            profile_meta["hse_relative"]["max_abs_resolved_interior"],
        "entropy_roundtrip_max_dex": {
            "energy": float(roundtrip[0]),
            "temperature": float(roundtrip[1]),
            "pressure": float(roundtrip[2]),
        },
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
