# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest

from verl.experimental.reward_loop.reward_loop import RewardLoopManager


class FakeWorker:
    class ComputeScoreBatch:
        remote = staticmethod(lambda chunk: chunk)

    compute_score_batch = ComputeScoreBatch()


class FakeData:
    def chunk(self, count):
        return [object() for _ in range(count)]


def test_reward_models_sleep_when_remote_scoring_fails(monkeypatch):
    lifecycle = []
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.reward_model_manager = SimpleNamespace(
        wake_up=lambda: lifecycle.append("wake"), sleep=lambda: lifecycle.append("sleep")
    )
    manager.reward_loop_workers = [FakeWorker()]

    def fail_get(_):
        raise RuntimeError("worker failed")

    monkeypatch.setattr("verl.experimental.reward_loop.reward_loop.ray.get", fail_get)

    with pytest.raises(RuntimeError, match="worker failed"):
        manager.compute_rm_score(FakeData())

    assert lifecycle == ["wake", "sleep"]
