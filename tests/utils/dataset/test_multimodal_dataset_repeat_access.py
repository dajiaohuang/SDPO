from io import BytesIO

import pytest
from PIL import Image

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset.vision_utils import process_image


def _png_bytes():
    image = Image.new("RGB", (2, 2), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("dataset_cls", "messages_key", "image_key"),
    [
        (RLHFDataset, "prompt", "images"),
        (MultiTurnSFTDataset, "messages", "images"),
    ],
)
def test_build_messages_does_not_mutate_stored_rows(dataset_cls, messages_key, image_key):
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.prompt_key = "prompt"
    dataset.messages_key = "messages"
    dataset.image_key = image_key
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.image_patch_size = 14

    stored_messages = [{"role": "user", "content": "Look at <image>"}]
    image = Image.new("RGB", (2, 2))
    for _ in range(2):
        row = {messages_key: stored_messages, image_key: [image], "videos": []}
        dataset._build_messages(row)

    assert stored_messages[0]["content"] == "Look at <image>"


def test_process_image_does_not_mutate_byte_mapping():
    image_data = {"bytes": _png_bytes()}

    result = process_image(image_data)

    assert result.mode == "RGB"
    assert result.size == (2, 2)
    assert "image" not in image_data
