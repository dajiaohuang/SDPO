import torch

from verl.models.mcore.qwen2_5_vl.rope_utils import get_rope_index


def test_mcore_video_rope_uses_processor_frame_interval():
    input_ids = torch.tensor([[151652, 151656, 151656, 151653, 1]])
    video_grid_thw = torch.tensor([[2, 2, 2]])

    timed_position_ids, _ = get_rope_index(
        input_ids,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=torch.tensor([0.5]),
    )
    default_position_ids, _ = get_rope_index(input_ids, video_grid_thw=video_grid_thw)

    assert timed_position_ids[0, 0, 1:3].tolist() == [1, 2]
    assert default_position_ids[0, 0, 1:3].tolist() == [1, 3]
