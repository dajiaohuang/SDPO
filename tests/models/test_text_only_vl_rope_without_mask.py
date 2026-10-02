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

import torch

from verl.models.transformers.qwen2_vl import get_rope_index as qwen2_get_rope_index
from verl.models.transformers.qwen3_vl import get_rope_index as qwen3_get_rope_index


def _processor():
    return SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=1),
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: hash(token) % 1000),
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
    )


def test_qwen2_vl_text_only_rope_without_attention_mask():
    input_ids = torch.tensor([11, 12, 13])

    position_ids = qwen2_get_rope_index(_processor(), input_ids)

    assert torch.equal(position_ids, torch.tensor([[0, 1, 2]]).expand(3, -1))


def test_qwen3_vl_text_only_rope_without_attention_mask():
    input_ids = torch.tensor([11, 12, 13])

    position_ids = qwen3_get_rope_index(_processor(), input_ids)

    assert torch.equal(position_ids, torch.tensor([[0, 1, 2]]).expand(3, -1))
