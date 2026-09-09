from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from baseline import _relative_to_root


class BaselinePathTests(unittest.TestCase):
    def test_absolute_lightning_checkpoint_is_made_relative_to_relative_run_root(self) -> None:
        cwd = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(dir=cwd) as directory:
            run_root = Path(directory).resolve()
            relative_root = run_root.relative_to(cwd)
            checkpoint = run_root / "blobs" / "baseline_model.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
            self.assertEqual(
                _relative_to_root(checkpoint, relative_root),
                "blobs/baseline_model.ckpt",
            )


if __name__ == "__main__":
    unittest.main()
