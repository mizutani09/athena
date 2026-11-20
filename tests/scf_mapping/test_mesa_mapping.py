#!/usr/bin/env python3
"""WOR-177 gates for mapping the MESA-fit SCF equilibrium into Athena++."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/"data/ic/rsg21_mesa_scf_omega0_nr16384.csv"
METADATA = ROOT/"data/ic/rsg21_mesa_scf_omega0_nr16384.json"
TAPER_PROFILE = ROOT/(
    "data/ic/rsg21_mesa_scf_omega0_nr16384_taper_rho3e-9_gamma5_3.csv")
TAPER_METADATA = ROOT/(
    "data/ic/rsg21_mesa_scf_omega0_nr16384_taper_rho3e-9_gamma5_3.json")
INPUT = ROOT/"inputs/rsg/athinput.mesa_scf_rsg_map"
N_MU = 16
ROOT_HALF_WIDTH_CM = 7.6527e13
MEAN_MOLECULAR_WEIGHT = 0.6353437771420501
GAMMA = 5.0/3.0


def piecewise_linear_spherical_integral(
        rows: list[dict[str, str]], field: str) -> float:
    ray = rows[::N_MU]
    result = 0.0
    for lower, upper in zip(ray[:-1], ray[1:]):
        r0 = float(lower["radius_cm"])
        r1 = float(upper["radius_cm"])
        f0 = float(lower[field])
        f1 = float(upper[field])
        slope = (f1-f0)/(r1-r0)
        intercept = f0-slope*r0
        result += 4.0*math.pi*(
            intercept*(r1**3-r0**3)/3.0
            +slope*(r1**4-r0**4)/4.0)
    return result


class MesaMappingTests(unittest.TestCase):
    athena: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="wor177-mesa-map-")
        cls.directory = Path(cls.tempdir.name)
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        with PROFILE.open(encoding="utf-8") as handle:
            cls.profile_rows = list(csv.DictReader(handle))
        cls.profile_unit = cls.directory/"test_profile"
        cls.audit = cls.directory/"audit"
        cls.taper = cls.directory/"surface_taper"
        common = [
            "g++", "-O2", "-std=c++11", "-Wall", "-Wextra",
            "-Wpedantic", "-Werror",
        ]
        subprocess.run(common + [
            str(ROOT/"tests/scf_mapping/test_scf_axisymmetric_profile.cpp"),
            "-o", str(cls.profile_unit)], check=True)
        subprocess.run(common + [
            str(ROOT/"tools/rsg/scf_smr_map_audit.cpp"),
            "-o", str(cls.audit)], check=True)
        subprocess.run(common + [
            str(ROOT/"tools/rsg/scf_surface_taper.cpp"),
            "-o", str(cls.taper)], check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_profile_loader_and_source_resolution(self) -> None:
        subprocess.run([
            str(self.profile_unit), str(PROFILE), "16384", str(N_MU),
        ], check=True, capture_output=True, text=True)
        mass = piecewise_linear_spherical_integral(
            self.profile_rows, "density_g_cm3")
        pressure_integral = piecewise_linear_spherical_integral(
            self.profile_rows, "pressure_dyn_cm2")
        self.assertLess(abs(mass/self.metadata["gas_mass_g"]-1.0), 7.0e-4)
        self.assertLess(abs(pressure_integral
                            /self.metadata["pressure_volume_integral_erg"]
                            -1.0), 1.3e-2)
        self.assertLess(self.metadata["scalar_virial_relative_error"], 1.0e-3)
        self.assertLess(self.metadata["force_residual_max_relative"], 1.0e-6)

    def test_hydrostatic_surface_taper_is_reproducible(self) -> None:
        output = self.directory/"surface_taper.csv"
        audit = self.directory/"surface_taper_audit.csv"
        metadata = json.loads(TAPER_METADATA.read_text(encoding="utf-8"))
        subprocess.run([
            str(self.taper), str(PROFILE), str(output), str(audit),
            str(metadata["match_density_g_cm3"]),
            str(metadata["polytropic_gamma"]),
        ], check=True, capture_output=True, text=True)
        digest = hashlib.sha256()
        with output.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), metadata["profile_sha256"])
        with audit.open(encoding="utf-8") as handle:
            row = {key: float(value)
                   for key, value in next(csv.DictReader(handle)).items()}
        self.assertLess(abs(row["mass_relative_change"]), 2.0e-4)
        self.assertLess(abs(row["pressure_volume_relative_change"]), 1.0e-5)
        self.assertLess(row["maximum_anchor_density_relative_error"], 5.0e-4)
        self.assertLess(row["maximum_anchor_enthalpy_relative_change"], 4.0e-2)
        subprocess.run([
            str(self.profile_unit), str(TAPER_PROFILE), "16384", str(N_MU),
        ], check=True, capture_output=True, text=True)

    def test_root32_l9_conservative_map(self) -> None:
        output = self.directory/"audit.csv"
        slice_output = self.directory/"slice.csv"
        meta = self.metadata
        subprocess.run([
            str(self.audit), str(PROFILE), str(output), str(slice_output),
            "32", str(ROOT_HALF_WIDTH_CM), "9", "3", "0",
            str(GAMMA), "1e-14", "1e4", str(MEAN_MOLECULAR_WEIGHT),
            str(meta["gas_mass_g"]), "0",
            str(meta["pressure_volume_integral_erg"]), "0",
            "mesa_zero_nr16384_root32_l9",
        ], check=True, capture_output=True, text=True)
        with output.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        value = {key: float(item) for key, item in row.items()
                 if key != "label"}
        self.assertLess(abs(value["mass_relative_error"]), 1.0e-3)
        self.assertEqual(value["exact_J_relative_error"], 0.0)
        self.assertEqual(value["discrete_J_relative_error"], 0.0)
        self.assertLess(abs(value["energy_relative_error"]), 2.0e-2)
        self.assertLess(value["pressure_recovery_l1_relative"], 1.0e-5)
        self.assertLess(value["force_l2_relative"], 0.1)

    def test_athena_l8_initialization(self) -> None:
        if self.athena is None:
            self.skipTest("pass --athena to run the compiled pgen map gate")
        run_directory = self.directory/"athena"
        run_directory.mkdir()
        subprocess.run([
            str(self.athena), "-i", str(INPUT), "-d", str(run_directory),
            "output2/dt=-1", "output3/dt=-1",
        ], check=True, capture_output=True, text=True)
        history = run_directory/"mesa_scf_rsg_map.hst"
        lines = history.read_text(encoding="utf-8").splitlines()
        header = next(line for line in lines if line.startswith("# [1]="))
        names = re.findall(r"\[\d+\]=([^ ]+)", header)
        values = [float(value) for value in next(
            line for line in reversed(lines)
            if line and not line.startswith("#")).split()]
        initial = dict(zip(names, values))
        meta = self.metadata
        expected_energy = meta["pressure_volume_integral_erg"]/(GAMMA-1.0)
        self.assertLess(abs(initial["mapped_stellar_mass_g"]
                            /meta["gas_mass_g"]-1.0), 1.0e-3)
        self.assertEqual(initial["mapped_Jz_g_cm2_s"], 0.0)
        self.assertLess(abs(initial["mapped_E_erg"]/expected_energy-1.0),
                        2.0e-2)
        self.assertLess(math.sqrt(initial["phi_error2"]
                                  /initial["phi_norm2"]), 1.0e-2)
        self.assertLess(math.sqrt(initial["force_error2"]
                                  /initial["force_norm2"]), 0.1)
        self.assertGreater(initial["meshblocks_L8"], 0.0)


def parse_args() -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--athena", type=Path)
    args, remaining = parser.parse_known_args()
    executable = args.athena.resolve() if args.athena else None
    return executable, [__file__, *remaining]


if __name__ == "__main__":
    MesaMappingTests.athena, unittest_args = parse_args()
    unittest.main(argv=unittest_args)
