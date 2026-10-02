from torch import nn

from verl.workers.rollout.hf_rollout import HFRollout


def test_hf_rollout_can_be_initialized_for_sync_generation():
    module = nn.Linear(2, 2)
    config = {"response_length": 8}

    rollout = HFRollout(module, config)

    assert rollout.module is module
    assert rollout.config is config
