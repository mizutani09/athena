#!/usr/bin/env python3
"""Build a diagnostic Planck-absorption variant from Semenov monochromatic data.

The Semenov mean-opacity code labels its Planck output mean extinction. The
public monochromatic NRM composite-aggregate files have columns for transport
extinction, total extinction, absorption, and scattering. We infer the column
roles from total extinction = absorption + scattering and from independent
Planck integration agreeing with the author's mean-opacity table away from
evaporation transitions. The trial scales only the Planck table by the
absorption/transport-extinction ratio. It is not a validated replacement for
the author's high-temperature gas opacity.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np


SOURCE_TABLE_SHA = "6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94"
MAC_ARCHIVE_SHA = "a7d2358280c5f4598c4bb0b931e036af2c481f572c75c4b52b87436b9b32a945"


def planck_ratio(data, temperature):
    wavelength = data[:, 0] * 1e-4  # published micrometres -> cm
    exponent = 1.438776877 / (temperature[:, None] * wavelength[None, :])
    weight = np.zeros_like(exponent)
    valid = exponent < 700.0
    weight[valid] = np.broadcast_to(wavelength[None, :]**-5,
                                   exponent.shape)[valid] / np.expm1(exponent[valid])
    transport = np.trapezoid(weight * data[None, :, 1], wavelength, axis=1)
    absorption = np.trapezoid(weight * data[None, :, 3], wavelength, axis=1)
    return absorption / transport


def transition(current, next_ratio, temperature, boundary, half_width):
    phase = np.clip((temperature[None, :] - boundary[:, None]) / half_width,
                    -1.0, 1.0)
    weight = 0.5 * (1.0 + np.sin(0.5 * np.pi * phase))
    return current * (1.0 - weight) + next_ratio[None, :] * weight


def build(original, archive, output):
    if hashlib.sha256(original.read_bytes()).hexdigest() != SOURCE_TABLE_SHA:
        raise ValueError("the mean-opacity table is not the verified WOR-92 input")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != MAC_ARCHIVE_SHA:
        raise ValueError("the monochromatic archive is not the verified author data")
    with h5py.File(original) as source:
        density = 10.0**source["axes/log10_density"][:]
        temperature = 10.0**source["axes/log10_temperature"][:]
        original_planck = source["kappa/log10_planck"][:]

    ratios = []
    validations = []
    with zipfile.ZipFile(archive) as source:
        for region in range(1, 6):
            text = source.read(f"nrm/Comp_aggregates/{region}.dat").decode()
            data = np.loadtxt(io.StringIO(text.replace("D", "E")))
            if data.shape[1] != 5 or not np.allclose(
                    data[:, 2], data[:, 3] + data[:, 4], rtol=1e-12, atol=1e-11):
                raise ValueError(f"unrecognized monochromatic columns in region {region}")
            if np.any(data[:, 1] < data[:, 3]) or np.any(data[:, 1] > data[:, 2]):
                raise ValueError(f"transport extinction outside absorption/total bounds in region {region}")
            ratios.append(planck_ratio(data, temperature))
            validations.append({"region": region,
                                "ratio_at_14K": float(planck_ratio(data, np.array([14.0]))[0]),
                                "ratio_at_431K": float(planck_ratio(data, np.array([431.0]))[0])})
    ratios = np.asarray(ratios)
    # These are the source opacity.f evaporation temperatures. For the first
    # core, only the first three boundaries are visited (Tgas <= 431.3 K).
    knots = np.array([1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
    ice = np.interp(np.maximum(density, 2e-18), knots,
                    [109, 118, 129, 143, 159, 180, 207, 244])
    iron = np.interp(np.maximum(density, 2e-18), knots,
                     [835, 908, 994, 1100, 1230, 1395, 1612, 1908])
    pyroxene = np.interp(np.maximum(density, 2e-18), knots,
                         [902, 980, 1049, 1129, 1222, 1331, 1462, 1621])
    olivine = np.interp(np.maximum(density, 2e-18), knots,
                        [929, 997, 1076, 1168, 1277, 1408, 1570, 1774])
    final_dust = np.minimum(np.maximum(iron, pyroxene),
                            np.maximum(pyroxene, olivine))
    correction = np.broadcast_to(ratios[0], original_planck.shape).copy()
    for next_region, boundary, width in (
            (1, ice, 5.0), (2, np.full_like(density, 275.0), 5.0),
            (3, np.full_like(density, 425.0), 15.0),
            (4, np.full_like(density, 680.0), 5.0),
            (5, final_dust, 100.0)):
        next_ratio = (ratios[next_region] if next_region < 5
                      else np.ones_like(temperature))
        correction = transition(correction, next_ratio, temperature,
                                boundary, width)
    if not (np.all(np.isfinite(correction)) and np.all(correction > 0)
            and np.all(correction <= 1.0 + 1e-12)):
        raise ValueError("invalid absorption correction")

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(original) as source, h5py.File(output, "w") as destination:
        for name in source:
            source.copy(name, destination)
        for name, value in source.attrs.items():
            destination.attrs[name] = value
        destination["kappa/log10_planck"][:] = original_planck + np.log10(correction)
        destination.attrs["planck_mean_definition"] = (
            "experimental absorption estimate from Semenov NRM composite aggregates MAC")
        destination.attrs["scope"] = "WOR-92 first core, T<=431.3 K only"

    def value_at(temp, rho=1e-10):
        lr, lt = np.log10(density), np.log10(temperature)
        i = np.searchsorted(lr, np.log10(rho)) - 1
        j = np.searchsorted(lt, np.log10(temp)) - 1
        a = (np.log10(rho) - lr[i]) / (lr[i+1] - lr[i])
        b = (np.log10(temp) - lt[j]) / (lt[j+1] - lt[j])
        log_ratio = np.log(correction)
        return float(np.exp((1-a)*(1-b)*log_ratio[i, j]
                            + a*(1-b)*log_ratio[i+1, j]
                            + (1-a)*b*log_ratio[i, j+1]
                            + a*b*log_ratio[i+1, j+1]))

    metadata = {
        "source_table_sha256": SOURCE_TABLE_SHA,
        "monochromatic_archive_sha256": MAC_ARCHIVE_SHA,
        "source_url": "https://semenov.www3.mpia.de/Opacities/MAC/nrm.zip",
        "method": "author mean Planck extinction multiplied by monochromatic Planck absorption/transport-extinction ratio; sinusoidal evaporation transition weights",
        "monochromatic_columns_inferred": ["wavelength_um", "transport_extinction", "total_extinction", "absorption", "scattering"],
        "region_ratios": validations,
        "correction_minmax": [float(correction.min()), float(correction.max())],
        "correction_near_14_200_350_431K_at_rho1e-10": {
            str(temp): value_at(temp) for temp in (14, 200, 350, 431)},
        "valid_scope": "first core <=431.3 K; gas-opacity absorption above dust evaporation is not independently validated",
        "trial_table_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("monochromatic_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.original, args.monochromatic_archive,
                           args.output), indent=2))
