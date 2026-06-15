"""Shared pytest fixtures.

The tiny GLiNER dataset under ``tests/fixtures/gliner_tiny/`` is generated
output, not committed source — it is rebuilt on demand from
``tests/fixtures/build_gliner_tiny.py``. The ``gliner_tiny_path`` fixture below
builds it if missing, so a fresh clone runs the smoke test with no extra step.
The build is lazy: ``test_smoke`` is gated behind ``RUN_GLINER_SMOKE=1``, and
pytest only sets up a fixture for tests that actually run, so the normal
(fast) suite never imports ``datasets`` or touches the fixture.
"""
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gliner_tiny"


@pytest.fixture(scope="session")
def gliner_tiny_path() -> Path:
    """Path to the tiny GLiNER DatasetDict, building it on first use if absent."""
    if not (FIXTURE_DIR / "dataset_dict.json").exists():
        import importlib.util

        builder = Path(__file__).parent / "fixtures" / "build_gliner_tiny.py"
        spec = importlib.util.spec_from_file_location("build_gliner_tiny", builder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    return FIXTURE_DIR
