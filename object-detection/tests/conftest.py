"""Shared pytest fixtures.

The tiny COCO fixture under ``tests/fixtures/coco_tiny/`` is generated output,
rebuilt on demand from ``tests/fixtures/build_coco_tiny.py``. The ``coco_tiny_path``
fixture builds it if missing, so a fresh clone runs the smoke test with no extra
step. The build is lazy and ``test_smoke`` is gated behind ``RUN_DET_SMOKE=1``,
so the fast suite never imports ``datasets`` or touches the fixture.
"""
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "coco_tiny"


@pytest.fixture(scope="session")
def coco_tiny_path() -> Path:
    if not (FIXTURE_DIR / "dataset_dict.json").exists():
        import importlib.util

        builder = Path(__file__).parent / "fixtures" / "build_coco_tiny.py"
        spec = importlib.util.spec_from_file_location("build_coco_tiny", builder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    return FIXTURE_DIR
