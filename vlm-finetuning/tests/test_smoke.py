"""End-to-end smoke test: real VLM (SmolVLM-256M), real Trainer, tiny fixture.

Gated by RUN_VLM_SMOKE=1 so the default `pytest` run stays fast. Verifies the
full pipeline (load → LoRA → train → generate-based eval → save) produces a
loadable checkpoint and a non-empty `test_metrics.json`. Downloads the 256M
model on first run; takes a few minutes on CPU.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_VLM_SMOKE", "").strip(),
    reason="set RUN_VLM_SMOKE=1 to run the end-to-end smoke test",
)


def _run_smoke(tmp_path, *, config_rel: str, fixture_src: Path, fixture_rel: str, final_rel: str):
    """Run a smoke config end-to-end from an isolated cwd and assert it produced
    a final checkpoint with test_* metrics. Returns the parsed metrics dict.

    The ``processed_dir`` in each YAML is relative, so we run from a cwd where it
    resolves, copying the (conftest-built) fixture into place first.
    """
    project_root = Path(__file__).resolve().parents[1]
    config = project_root / config_rel
    cwd = tmp_path

    fixture_dst = cwd / fixture_rel
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture_src, fixture_dst)

    result = subprocess.run(
        [sys.executable, str(project_root / "scripts/train_vlm.py"),
         "--config", str(config),
         "--max-train-samples", "6", "--max-eval-samples", "2", "--max-test-samples", "2"],
        cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(project_root), "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, (
        f"smoke training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    final_dir = cwd / final_rel
    assert final_dir.exists(), f"expected final dir at {final_dir}"

    metrics_path = final_dir / "test_metrics.json"
    assert metrics_path.exists(), "test_metrics.json not written"
    metrics = json.loads(metrics_path.read_text())
    assert any(k.startswith("test_") for k in metrics), (
        f"expected test_* keys in metrics; got: {list(metrics.keys())}"
    )
    return metrics


def test_train_vlm_end_to_end(tmp_path, vlm_tiny_path):
    """Grounding (task: box) path — boxes/points, scored by mAP-style precision/recall."""
    _run_smoke(
        tmp_path,
        config_rel="examples/grounding/configs/smoke.yaml",
        fixture_src=vlm_tiny_path,
        fixture_rel="tests/fixtures/vlm_tiny",
        final_rel="outputs/_smoke/smoke/final",
    )


def test_train_vlm_text_end_to_end(tmp_path, vlm_text_tiny_path):
    """Generic free-text (task: text) path — image→string, scored by exact-match / token-F1."""
    metrics = _run_smoke(
        tmp_path,
        config_rel="examples/json_extraction/configs/smoke.yaml",
        fixture_src=vlm_text_tiny_path,
        fixture_rel="tests/fixtures/vlm_text_tiny",
        final_rel="outputs/_smoke_text/smoke_text/final",
    )
    # Sanity-check we exercised the text metrics, not the grounding ones.
    assert "test_token_f1" in metrics, (
        f"expected text-path metrics (test_token_f1); got: {list(metrics.keys())}"
    )
