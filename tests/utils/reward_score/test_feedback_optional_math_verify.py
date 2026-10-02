from verl.utils.reward_score.feedback import compute_score


def test_non_math_feedback_scoring_does_not_require_math_verify():
    result = compute_score(data_source="sciknoweval", solution_str="A", ground_truth="A")

    assert result["score"] == 1.0
