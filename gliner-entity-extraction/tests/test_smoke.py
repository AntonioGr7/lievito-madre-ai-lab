"""End-to-end smoke test: real GLiNER, real Trainer, tiny fixture.

Gated by RUN_GLINER_SMOKE=1 so the default `pytest` run stays fast. Verifies
the full pipeline produces a loadable checkpoint and a non-empty
`test_metrics.json`. Runs in ~30-60s on CPU.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_GLINER_SMOKE", "").strip(),
    reason="set RUN_GLINER_SMOKE=1 to run the end-to-end smoke test",
)


def test_train_gliner_end_to_end(tmp_path, gliner_tiny_path):
    project_root = Path(__file__).resolve().parents[1]
    config = project_root / "examples/pii/configs/smoke.yaml"
    cwd = tmp_path

    # The processed_dir in the YAML is relative; the test must run from a
    # cwd where that path resolves. Copy the fixture into tmp.
    # gliner_tiny_path (conftest) builds the fixture on first use if absent.
    fixture_src = gliner_tiny_path
    fixture_dst = cwd / "tests/fixtures/gliner_tiny"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture_src, fixture_dst)

    result = subprocess.run(
        [sys.executable, str(project_root / "scripts/train_gliner.py"),
         "--config", str(config),
         "--max-train-samples", "6", "--max-eval-samples", "2", "--max-test-samples", "2"],
        cwd=cwd, capture_output=True, text=True,
        # PYTHONIOENCODING=utf-8 keeps non-ASCII print() output (arrows,
        # ellipses) from crashing the child under Windows cp1252.
        env={**os.environ, "PYTHONPATH": str(project_root), "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, (
        f"smoke training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    final_dir = cwd / "outputs/_smoke/smoke/final"
    assert final_dir.exists(), f"expected final dir at {final_dir}"

    metrics_path = final_dir / "test_metrics.json"
    assert metrics_path.exists(), "test_metrics.json not written"
    metrics = json.loads(metrics_path.read_text())
    assert any(k.startswith("test_closed_") for k in metrics), (
        f"expected test_closed_* keys in metrics; got: {list(metrics.keys())}"
    )
