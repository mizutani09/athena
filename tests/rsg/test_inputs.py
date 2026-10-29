import subprocess
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RsgInputTest(unittest.TestCase):
    def test_nearest_mg_compatible_cell_width(self):
        text = (ROOT/"inputs/rsg/athinput.st35gm03n07_fld").read_text()
        self.assertIn("nx1 = 256", text)
        self.assertAlmostEqual(2021.0/235, 8.6)
        self.assertAlmostEqual(2021.0/256, 7.89453125)
        self.assertIn("luminosity_lsun = 91932.0", text)

    def test_variants_materialize(self):
        for n, block in ((128, 32), (512, 32)):
            source = ROOT/f"inputs/rsg/athinput.st35gm03n07_fld_{n}"
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/"athinput"
                subprocess.run(["python3", str(ROOT/"scripts/materialize_rsg_input.py"),
                                str(source), str(output)], check=True)
                text = output.read_text()
            for axis in (1, 2, 3):
                self.assertIn(f"nx{axis} = {n}", text)
                self.assertIn(f"nx{axis} = {block}", text)

    def test_historical_variants_are_rejected_before_runtime(self):
        for n in (117, 470):
            source = ROOT/f"inputs/rsg/athinput.st35gm03n07_fld_{n}"
            with tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    ["python3", str(ROOT/"scripts/materialize_rsg_input.py"),
                     str(source), str(Path(tmp)/"athinput")],
                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("power-of-two", result.stderr)


if __name__ == "__main__":
    unittest.main()
