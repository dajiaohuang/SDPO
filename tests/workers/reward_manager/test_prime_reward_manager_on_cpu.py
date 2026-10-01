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

import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.reward_manager.prime import PrimeRewardManager


class DummyTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["response"] * len(sequences)


def test_prime_reward_manager_uses_custom_reward_key():
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        },
        non_tensors={
            "task_source": np.array(["math"]),
            "reward_model": np.array([{"ground_truth": "answer"}], dtype=object),
        },
    )
    manager = PrimeRewardManager(DummyTokenizer(), num_examine=0, reward_fn_key="task_source")
    manager.verify = lambda _data: [0.75]

    rewards = manager(data)

    assert rewards.shape == (1, 2)
    assert rewards[0, 1].item() == 0.75
