#!/usr/bin/env python3
"""Unit tests for the per-rank resource wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rank_resource_wrapper as wrapper  # noqa: E402


class RankResourceWrapperTests(unittest.TestCase):
    def test_mpi_payload_preserves_inherited_descriptors(self):
        completed = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            argv = ["rank_resource_wrapper.py", str(output), "athena", "-i", "input"]
            with mock.patch.dict(os.environ, {"PMI_RANK": "7"}, clear=False), \
                    mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(wrapper.subprocess, "run",
                                      return_value=completed) as run:
                self.assertEqual(wrapper.main(), 0)
            run.assert_called_once_with(["athena", "-i", "input"], check=False,
                                        close_fds=False)
            record = json.loads((output / "resource_rank7.json").read_text())
            self.assertEqual(record["rank"], 7)
            self.assertEqual(record["returncode"], 0)
            self.assertGreaterEqual(record["max_rss_kib"], 0)


if __name__ == "__main__":
    unittest.main()
