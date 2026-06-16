"""Shared pytest fixtures.

The tiny datasets under ``tests/fixtures/`` are generated output, not committed
source — each is rebuilt on demand from its ``tests/fixtures/build_*.py`` script.
The ``*_path`` fixtures below build them if missing, so a fresh clone runs the
smoke tests with no extra step. The build is lazy: ``test_smoke`` is gated behind
``RUN_VLM_SMOKE=1``, and pytest only sets up a fixture for tests that actually
run, so the normal (fast) suite never imports ``datasets`` or touches a fixture.

- ``vlm_tiny`` — grounding (boxes), built by ``build_vlm_tiny.py``.
- ``vlm_text_tiny`` — generic free-text SFT, built by ``build_vlm_text_tiny.py``.
"""
import importlib.util
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_if_absent(name: str) -> Path:
    """Path to a generated fixture dir, running its build_<name>.py if absent."""
    fixture_dir = FIXTURES / name
    if not (fixture_dir / "dataset_dict.json").exists():
        builder = FIXTURES / f"build_{name}.py"
        spec = importlib.util.spec_from_file_location(f"build_{name}", builder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    return fixture_dir


@pytest.fixture(scope="session")
def vlm_tiny_path() -> Path:
    """Path to the tiny grounding DatasetDict, building it on first use if absent."""
    return _build_if_absent("vlm_tiny")


@pytest.fixture(scope="session")
def vlm_text_tiny_path() -> Path:
    """Path to the tiny free-text DatasetDict, building it on first use if absent."""
    return _build_if_absent("vlm_text_tiny")
