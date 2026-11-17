#!/usr/bin/env python3
"""Join Semenov dust opacity to an OPTAB solar-composition gas table.

The low-temperature source includes dust evaporation and is retained below a
configurable transition.  The high-temperature source supplies gas absorption
and transport opacity after the dust has evaporated.  Interpolation and the
transition are performed in log10 opacity.  Density lookups in the Semenov
source are held at its nearest qualified row outside its published range;
OPTAB is never extrapolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    with h5py.File(path) as handle:
        density = np.asarray(handle["axes/log10_density"], dtype=float)
        temperature = np.asarray(handle["axes/log10_temperature"], dtype=float)
        planck = np.asarray(handle["kappa/log10_planck"], dtype=float)
        rosseland = np.asarray(handle["kappa/log10_rosseland"], dtype=float)
    if (planck.shape != (density.size, temperature.size)
            or rosseland.shape != planck.shape
            or not np.all(np.diff(density) > 0.0)
            or not np.all(np.diff(temperature) > 0.0)
            or not np.isfinite((planck, rosseland)).all()):
        raise ValueError(f"invalid opacity table: {path}")
    return density, temperature, planck, rosseland


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def final_dust_evaporation(density):
    """Reproduce Semenov opacity.f's final composite-dust transition."""
    source_density = np.array([1e-18, 1e-16, 1e-14, 1e-12,
                               1e-10, 1e-8, 1e-6, 1e-4])
    # opacity.f sets T(5) to min(max(Tiron,Tpyroxene),
    #                            max(Tpyroxene,Tolivine)).
    source_temperature = np.array([902.0, 980.0, 1049.0, 1129.0,
                                   1230.0, 1395.0, 1570.0, 1774.0])
    return np.interp(density, source_density, source_temperature)


def error_summary(values):
    absolute = np.abs(values)
    return {
        "samples": int(absolute.size),
        "median_abs_dex": float(np.median(absolute)),
        "p99_abs_dex": float(np.percentile(absolute, 99.0)),
        "max_abs_dex": float(np.max(absolute)),
    }


def source_agreement(target_rho, target_temp, hybrid, source_axes,
                     transition_offset, use_below):
    """Measure the error introduced by regridding and a second interpolation."""
    source_rho, source_temp, source_values = source_axes
    rho_low = max(target_rho[0], source_rho[0], np.log10(2.0e-18))
    rho_high = min(target_rho[-1], source_rho[-1], -7.0)
    temp_low = max(target_temp[0], source_temp[0])
    temp_high = min(target_temp[-1], source_temp[-1])
    audit_rho = np.linspace(rho_low, rho_high, 181)
    audit_temp = np.linspace(temp_low, temp_high, 351)
    points_rho, points_temp = np.meshgrid(audit_rho, audit_temp,
                                           indexing="ij")
    boundary = np.log10(final_dust_evaporation(10.0**points_rho)
                        + transition_offset)
    mask = points_temp <= boundary if use_below else points_temp >= boundary
    points = np.column_stack((points_rho[mask], points_temp[mask]))
    hybrid_interp = RegularGridInterpolator(
        (target_rho, target_temp), hybrid, bounds_error=True)
    source_interp = RegularGridInterpolator(
        (source_rho, source_temp), source_values, bounds_error=True)
    return error_summary(hybrid_interp(points) - source_interp(points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semenov", type=Path, required=True)
    parser.add_argument("--optab", type=Path, required=True)
    parser.add_argument("--optab-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--logrho-min", type=float, default=-21.0)
    parser.add_argument("--logrho-max", type=float, default=-2.0)
    parser.add_argument("--nrho", type=int, default=256)
    parser.add_argument("--logt-min", type=float, default=np.log10(5.0))
    parser.add_argument("--logt-max", type=float, default=5.0)
    parser.add_argument("--ntemp", type=int, default=2048)
    parser.add_argument("--transition-start-offset", type=float, default=100.0)
    parser.add_argument("--transition-end-offset", type=float, default=300.0)
    args = parser.parse_args()
    if (args.nrho < 4 or args.ntemp < 4
            or args.transition_start_offset < 100.0
            or args.transition_end_offset <= args.transition_start_offset):
        parser.error("invalid grid or transition")

    optab_meta = json.loads(args.optab_metadata.read_text())
    composition = optab_meta.get("composition", {}).get("composition", {})
    definitions = optab_meta.get("mean_definitions", {})
    expected_composition = {"X": 0.7, "Y": 0.28, "Z": 0.02}
    if (set(composition) != set(expected_composition)
            or any(not np.isclose(composition[key], expected, rtol=0.0,
                                  atol=1.0e-12)
                   for key, expected in expected_composition.items())):
        raise ValueError(f"unexpected OPTAB composition: {composition}")
    if (definitions.get("Planck")
            != "line + continuum absorption, excluding pure scattering"
            or definitions.get("Rosseland")
            != "line + continuum absorption + scattering"):
        raise ValueError(f"unexpected OPTAB mean definitions: {definitions}")

    sr, st, sp, sx = load(args.semenov)
    ogr, ogt, op, ox = load(args.optab)
    target_rho = np.linspace(args.logrho_min, args.logrho_max, args.nrho)
    target_temp = np.linspace(args.logt_min, args.logt_max, args.ntemp)
    if (target_rho[0] < ogr[0] or target_rho[-1] > ogr[-1]
            or np.log10(902.0 + args.transition_start_offset) < ogt[0]
            or target_temp[-1] > ogt[-1]
            or target_temp[0] < st[0]
            or np.log10(1774.0 + args.transition_end_offset) > st[-1]):
        raise ValueError("target grid or transition lies outside a required source")

    points_rho, points_temp = np.meshgrid(target_rho, target_temp, indexing="ij")
    sem_points = np.stack((np.clip(points_rho, sr[0], sr[-1]),
                           np.minimum(points_temp, st[-1])), axis=-1)
    opt_points = np.stack((points_rho, np.maximum(points_temp, ogt[0])), axis=-1)
    output = []
    for sem_values, opt_values in ((sp, op), (sx, ox)):
        sem = RegularGridInterpolator((sr, st), sem_values,
                                      bounds_error=True)(sem_points)
        opt = RegularGridInterpolator((ogr, ogt), opt_values,
                                      bounds_error=True)(opt_points)
        evaporation = final_dust_evaporation(10.0**points_rho)
        transition_low = np.log10(evaporation + args.transition_start_offset)
        transition_high = np.log10(evaporation + args.transition_end_offset)
        fraction = smoothstep((points_temp - transition_low)
                              / (transition_high - transition_low))
        output.append((1.0 - fraction) * sem + fraction * opt)
    planck, rosseland = output
    if not np.isfinite((planck, rosseland)).all():
        raise ValueError("hybrid opacity contains a nonfinite value")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("axes/log10_density", data=target_rho)
        handle.create_dataset("axes/log10_temperature", data=target_temp)
        handle.create_dataset("kappa/log10_planck", data=planck)
        handle.create_dataset("kappa/log10_rosseland", data=rosseland)
        handle.attrs["units"] = "kappa: cm^2/g; density: g/cm^3; temperature: K"
        handle.attrs["construction"] = (
            "Semenov below transition; cubic smoothstep in log10(kappa); OPTAB above"
        )

    evaporation_axis = final_dust_evaporation(10.0**target_rho)
    agreement = {}
    for name, hybrid, sem_values, opt_values in (
            ("planck", planck, sp, op),
            ("rosseland", rosseland, sx, ox)):
        agreement[name] = {
            "semenov_below_transition_start": source_agreement(
                target_rho, target_temp, hybrid, (sr, st, sem_values),
                args.transition_start_offset, True),
            "optab_above_transition_end": source_agreement(
                target_rho, target_temp, hybrid, (ogr, ogt, opt_values),
                args.transition_end_offset, False),
        }
    audit = {
        "sources": {
            "semenov": {"path": str(args.semenov.resolve()),
                        "sha256": sha256(args.semenov),
                        "qualified_density_g_cm3": [2.0e-18, 1.0e-7]},
            "optab": {"path": str(args.optab.resolve()),
                      "sha256": sha256(args.optab),
                      "metadata": str(args.optab_metadata.resolve()),
                      "metadata_sha256": sha256(args.optab_metadata),
                      "composition": composition,
                      "mean_definitions": definitions,
                      "commit": optab_meta.get("provenance", {}).get("optab_commit")},
        },
        "grid": {"log10_density": [float(target_rho[0]), float(target_rho[-1]),
                                     target_rho.size],
                 "log10_temperature": [float(target_temp[0]), float(target_temp[-1]),
                                         target_temp.size]},
        "transition": {
            "method": "density-dependent final dust evaporation from Semenov opacity.f",
            "final_dust_evaporation_K": [float(evaporation_axis.min()),
                                          float(evaporation_axis.max())],
            "start_offset_K": args.transition_start_offset,
            "end_offset_K": args.transition_end_offset,
            "pure_optab_above_K": [float((evaporation_axis
                                           + args.transition_end_offset).min()),
                                    float((evaporation_axis
                                           + args.transition_end_offset).max())],
        },
        "semenov_density_hold": {
            "below_source_rows": int(np.count_nonzero(target_rho < sr[0])),
            "above_source_rows": int(np.count_nonzero(target_rho > sr[-1])),
            "policy": "nearest source density; only active below transition-high"},
        "ranges_cm2_g": {
            "planck": [float(10.0**planck.min()), float(10.0**planck.max())],
            "rosseland": [float(10.0**rosseland.min()), float(10.0**rosseland.max())]},
        "runtime_interpolation_source_agreement": {
            "method": ("deterministic off-grid audit; differences include source "
                       "regridding and Athena-style second bilinear interpolation"),
            **agreement,
        },
        "checks": {"finite": True, "positive": True,
                   "strict_density_axis": bool(np.all(np.diff(target_rho) > 0.0)),
                   "strict_temperature_axis": bool(np.all(np.diff(target_temp) > 0.0))},
        "limitations": [
            "Semenov density is held at its nearest row outside its source range.",
            "The transition width is a modeling choice and requires sensitivity tests.",
            "The table has not yet passed a collapse restart or conservation gate."
        ],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    metadata = args.metadata or args.output.with_suffix(args.output.suffix + ".json")
    metadata.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
