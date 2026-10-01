from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.experimental.reward_loop.reward_loop import RewardLoopManager


class _RemoteMethod:
    def remote(self, chunk):
        return chunk


@pytest.mark.parametrize("sleep_fails", [False, True])
def test_compute_rm_score_sleeps_after_worker_failure(monkeypatch, sleep_fails):
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.reward_model_manager = Mock()
    if sleep_fails:
        manager.reward_model_manager.sleep.side_effect = OSError("cleanup failed")
    manager.reward_loop_workers = [SimpleNamespace(compute_score_batch=_RemoteMethod())]
    data = SimpleNamespace(chunk=lambda count: ["chunk"])

    def fail_ray_get(refs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr("verl.experimental.reward_loop.reward_loop.ray.get", fail_ray_get)

    with pytest.raises(RuntimeError, match="worker failed"):
        manager.compute_rm_score(data)

    manager.reward_model_manager.wake_up.assert_called_once_with()
    manager.reward_model_manager.sleep.assert_called_once_with()


def test_compute_rm_score_keeps_success_result_and_sleeps(monkeypatch):
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.reward_model_manager = Mock()
    manager.reward_loop_workers = [SimpleNamespace(compute_score_batch=_RemoteMethod())]

    class FakeData:
        batch = {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        }

        def chunk(self, count):
            return ["chunk"]

        def __len__(self):
            return 1

    data = FakeData()

    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.ray.get",
        lambda refs: [[{"reward_score": 0.5, "reward_extra_info": {}}]],
    )

    result = manager.compute_rm_score(data)

    assert result.batch["rm_scores"].tolist() == [[0.0, 0.5, 0.0]]
    manager.reward_model_manager.wake_up.assert_called_once_with()
    manager.reward_model_manager.sleep.assert_called_once_with()
