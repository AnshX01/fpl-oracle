"""A server import is insufficient: validate the solver API and real binary solve."""

import importlib.util
from pathlib import Path

import pytest


def load_check():
    path = Path(__file__).resolve().parents[1] / "scripts/check_solver_runtime.py"
    spec = importlib.util.spec_from_file_location("solver_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_bundled_cbc_binary_solve():
    assert load_check().check_solver_runtime() == "3.3.2"


def test_rejects_pulp4_before_legacy_api_call(monkeypatch):
    module = load_check()
    monkeypatch.setattr(module.pulp, "__version__", "4.0.0")
    with pytest.raises(RuntimeError, match="requires pulp==3.3.2"):
        module.check_solver_runtime()


def test_dependency_and_existing_windows_env_are_pinned():
    root = Path(__file__).resolve().parents[1]
    assert '"pulp==3.3.2"' in (root / "pyproject.toml").read_text()
    startup = (root / "start.ps1").read_text()
    assert startup.index("Installing tested PuLP") > startup.index("# Keep existing environments")
    assert startup.index("scripts/check_solver_runtime.py") < startup.index("if ($SetupOnly)")
