import pytest
import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": torch.tensor([[11, 12, 13]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}


class FakeDataFrame:
    class Row:
        def to_dict(self):
            return {"messages": [{"role": "assistant", "content": "answer"}]}

    @property
    def iloc(self):
        return self

    def __getitem__(self, _index):
        return self.Row()


def make_dataset(truncation):
    dataset = object.__new__(MultiTurnSFTDataset)
    dataset.pad_mode = DatasetPadMode.NO_PADDING
    dataset.truncation = truncation
    dataset.max_length = 2
    dataset.messages_key = "messages"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.tools = None
    dataset.enable_thinking = None
    dataset.apply_chat_template_kwargs = {}
    dataset.tokenizer = FakeTokenizer()
    dataset.processor = None
    dataset.system_prompt = ""
    dataset.generation_prompt = ""
    dataset.ignore_input_ids_mismatch = False
    dataset.dataframe = FakeDataFrame()
    return dataset


def test_no_padding_error_truncation_raises():
    with pytest.raises(ValueError, match="sequence_length=3 is larger than self.max_length=2"):
        make_dataset("error")[0]


def test_no_padding_left_truncation_keeps_final_tokens():
    assert make_dataset("left")[0]["input_ids"].tolist() == [12, 13]
