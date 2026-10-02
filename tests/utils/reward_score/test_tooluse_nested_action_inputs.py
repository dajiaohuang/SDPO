import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "verl/utils/reward_score/feedback/tooluse.py"
SPEC = importlib.util.spec_from_file_location("tooluse_feedback", MODULE_PATH)
TOOLUSE_FEEDBACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLUSE_FEEDBACK)


def test_extract_action_inputs_with_nested_objects_and_multiple_calls():
    solution = (
        'Action Input: {"query": {"term": "axolotl", "filters": {"color": "wild"}}}\n'
        'Action Input: {"page": 2}'
    )

    assert TOOLUSE_FEEDBACK.extract_action_inputs(solution) == {
        "query": {"term": "axolotl", "filters": {"color": "wild"}},
        "page": 2,
    }
