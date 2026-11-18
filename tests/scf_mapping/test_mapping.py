#!/usr/bin/env python3
"""Automated C++ unit, conservative-remap, and Athena++ smoke tests for WOR-160."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/"data/ic/rsg21_scf_omega4e-8.csv"
METADATA = ROOT/"data/ic/rsg21_scf_omega4e-8.json"
INPUT = ROOT/"inputs/rsg/athinput.scf_rsg_map"


class MappingTests(unittest.TestCase):
    athena: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="wor160-map-")
        cls.directory = Path(cls.tempdir.name)
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        cls.unit = cls.directory/"test_profile"
        cls.gravity_unit = cls.directory/"test_core_gravity"
        cls.audit = cls.directory/"audit"
        common = ["g++", "-O2", "-std=c++11", "-Wall", "-Wextra", "-Wpedantic"]
        subprocess.run(common + [
            str(ROOT/"tests/scf_mapping/test_scf_axisymmetric_profile.cpp"),
            "-o", str(cls.unit)], check=True)
        subprocess.run(common + [
            str(ROOT/"tools/rsg/scf_smr_map_audit.cpp"),
            "-o", str(cls.audit)], check=True)
        subprocess.run(common + [
            str(ROOT/"tests/scf_mapping/test_scf_core_gravity.cpp"),
            "-o", str(cls.gravity_unit)], check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_profile_loader_and_cell_quadrature(self) -> None:
        subprocess.run([str(self.unit), str(PROFILE)], check=True,
                       capture_output=True, text=True)

    def test_core_spline_and_isothermal_atmosphere(self) -> None:
        subprocess.run([str(self.gravity_unit)], check=True,
                       capture_output=True, text=True)

    def test_recommended_smr_mapping_closes_conserved_quantities(self) -> None:
        output = self.directory/"audit.csv"
        slice_output = self.directory/"slice.csv"
        meta = self.metadata
        subprocess.run([
            str(self.audit), str(PROFILE), str(output), str(slice_output),
            "64", "7.6527e13", "5", "3", "4e-8",
            "1.6666666666666667", "1e-14", "1e4", "0.6353437771420501",
            str(meta["gas_mass_g"]), str(meta["angular_momentum_g_cm2_s"]),
            str(meta["pressure_volume_integral_erg"]),
            str(meta["rotational_kinetic_energy_erg"]), "acceptance",
        ], check=True, capture_output=True, text=True)
        with output.open(encoding="utf-8") as handle:
            result = next(csv.DictReader(handle))
        value = {key: float(item) for key, item in result.items()
                 if key != "label"}
        self.assertLess(abs(value["mass_relative_error"]), 5.0e-5)
        self.assertLess(abs(value["discrete_J_relative_error"]), 1.0e-4)
        self.assertLess(abs(value["energy_relative_error"]), 5.0e-3)
        self.assertLess(value["pressure_recovery_l1_relative"], 1.0e-5)
        self.assertLess(value["density_center_l1_relative"], 1.0e-3)
        self.assertLess(value["temperature_center_l1_relative"], 3.0e-3)
        self.assertLess(value["force_l2_relative"], 0.1)

    def test_athena_small_grid_initialization(self) -> None:
        if self.athena is None:
            self.skipTest("pass --athena to run the compiled pgen smoke test")
        run_directory = self.directory/"athena"
        run_directory.mkdir()
        subprocess.run([
            str(self.athena), "-i", str(INPUT), "-d", str(run_directory),
            f"problem/scf_profile_file={PROFILE}",
            "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32",
            "time/nlim=0", "output2/dt=-1", "output3/dt=-1",
        ], check=True, capture_output=True, text=True)
        lines = (run_directory/"scf_rsg_map.hst").read_text(
            encoding="utf-8").splitlines()
        header = next(line for line in lines if line.startswith("# [1]="))
        names = re.findall(r"\[\d+\]=([^ ]+)", header)
        values = [float(value) for value in next(
            line for line in lines if line and not line.startswith("#")).split()]
        initial = dict(zip(names, values))
        mapped_mass = initial["mapped_mass_g"]
        mapped_stellar_mass = initial["mapped_stellar_mass_g"]
        mapped_j = initial["mapped_Jz_g_cm2_s"]
        mapped_energy = initial["mapped_E_erg"]
        meta = self.metadata
        expected_energy = (meta["pressure_volume_integral_erg"]
                           /(5.0/3.0-1.0)
                           +meta["rotational_kinetic_energy_erg"])
        self.assertLess(abs(mapped_stellar_mass/meta["gas_mass_g"]-1.0), 5.0e-5)
        self.assertLess(abs(mapped_j/meta["angular_momentum_g_cm2_s"]-1.0),
                        1.0e-4)
        self.assertLess(abs(mapped_energy/expected_energy-1.0), 5.0e-3)
        # Total mass includes the separately tracked atmosphere.
        self.assertGreaterEqual(mapped_mass, mapped_stellar_mass)
        self.assertGreater(initial["atmosphere_mass_g"], 0.0)
        self.assertLess(math.sqrt(initial["phi_error2"]
                                  /initial["phi_norm2"]), 2.0e-3)
        self.assertLess(math.sqrt(initial["force_error2"]
                                  /initial["force_norm2"]), 0.25)


def parse_args() -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--athena", type=Path)
    args, remaining = parser.parse_known_args()
    executable = args.athena.resolve() if args.athena else None
    return executable, [__file__, *remaining]


if __name__ == "__main__":
    MappingTests.athena, unittest_args = parse_args()
    unittest.main(argv=unittest_args)
