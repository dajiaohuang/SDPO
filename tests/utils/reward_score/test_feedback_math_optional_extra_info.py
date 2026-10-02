import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def feedback_math(monkeypatch):
    math_verify = types.ModuleType("math_verify")
    math_verify.parse = lambda value: value
    math_verify.verify = lambda _gold, _prediction: False
    monkeypatch.setitem(sys.modules, "math_verify", math_verify)

    module_path = Path(__file__).resolve().parents[3] / "verl/utils/reward_score/feedback/math.py"
    spec = importlib.util.spec_from_file_location("feedback_math_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_score_accepts_default_none_extra_info(feedback_math):
    result = feedback_math.compute_score(r"\boxed{42}", "42")

    assert result["score"] == 1.0
    assert result["pred"] == "42"
