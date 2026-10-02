from types import SimpleNamespace

import torch

from verl.workers.reward_manager.dapo import DAPORewardManager


class FakeData:
    def __init__(self):
        self.batch = {
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        self.non_tensor_batch = {"data_source": ["example"]}
        self.item = SimpleNamespace(
            batch={"prompts": self.batch["prompts"][0], "responses": self.batch["responses"][0],
                   "attention_mask": self.batch["attention_mask"][0]},
            non_tensor_batch={"reward_model": {"ground_truth": "answer"}, "data_source": "example"},
        )

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return self.item


class FakeTokenizer:
    eos_token = "<eos>"

    def decode(self, _tokens, skip_special_tokens=True):
        return "response"


def test_dapo_manager_allows_omitted_overlong_buffer_config():
    manager = DAPORewardManager(
        tokenizer=FakeTokenizer(), num_examine=0, compute_score=lambda **_kwargs: 1.0, overlong_buffer_cfg=None
    )

    reward = manager(FakeData())

    assert reward.tolist() == [[1.0]]
