# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import verl.utils.reward_score.math_verify as math_verify_reward


class FakeTimeoutException(Exception):
    pass


def test_timeout_uses_configured_score(monkeypatch):
    class FakeExtractionConfig:
        pass

    def timeout_metric(**_kwargs):
        def verify(_ground_truth, _prediction):
            raise FakeTimeoutException

        return verify

    monkeypatch.setattr(math_verify_reward, "TimeoutException", FakeTimeoutException, raising=False)
    monkeypatch.setattr(math_verify_reward, "math_metric", timeout_metric, raising=False)
    monkeypatch.setattr(math_verify_reward, "LatexExtractionConfig", FakeExtractionConfig, raising=False)
    monkeypatch.setattr(math_verify_reward, "ExprExtractionConfig", FakeExtractionConfig, raising=False)

    assert math_verify_reward.compute_score("response", "answer", timeout_score=0.25) == 0.25
