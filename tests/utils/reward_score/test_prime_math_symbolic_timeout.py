from verl.utils.reward_score.prime_math.grader import math_equal


def test_math_equal_compares_symbolically_with_timeout():
    assert math_equal("x + 1", "1 + x", timeout=10.0)
