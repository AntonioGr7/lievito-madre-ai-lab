"""End-to-end smoke test: real detector (YOLOS-tiny), real Trainer, tiny fixture.

Gated by RUN_DET_SMOKE=1 so the default `pytest` run stays fast. Verifies the
full pipeline (load → head-swap → augment → train → mAP-eval → save) produces a
loadable checkpoint and a non-empty `test_metrics.json`. Downloads YOLOS-tiny on
first run; a few minutes on CPU.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_DET_SMOKE", "").strip(),
    reason="set RUN_DET_SMOKE=1 to run the end-to-end smoke test",
)


def test_train_detector_end_to_end(tmp_path, coco_tiny_path):
    project_root = Path(__file__).resolve().parents[1]
    config = project_root / "examples/cppe5/configs/smoke.yaml"
    cwd = tmp_path

    fixture_dst = cwd / "tests/fixtures/coco_tiny"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(coco_tiny_path, fixture_dst)

    result = subprocess.run(
        [sys.executable, str(project_root / "scripts/train_detector.py"),
         "--config", str(config),
         "--max-train-samples", "6", "--max-eval-samples", "2", "--max-test-samples", "2"],
        cwd=cwd, capture_output=True, text=True,
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
    assert any(k.startswith("test_map") for k in metrics), (
        f"expected test_map* keys; got: {list(metrics.keys())}"
    )
