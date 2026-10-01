from types import SimpleNamespace

import torch

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def test_fsdp_merger_preserves_non_floating_state_tensor_dtype(tmp_path):
    checkpoint = {
        "position_ids": torch.tensor([300, 513], dtype=torch.long),
        "weight": torch.tensor([1.25, 2.5], dtype=torch.float32),
    }
    torch.save(checkpoint, tmp_path / "model_world_size_1_rank_0.pt")

    merger = object.__new__(FSDPModelMerger)
    merger.config = SimpleNamespace(local_dir=str(tmp_path))

    merged = merger._load_and_merge_state_dicts(
        world_size=1,
        total_shards=1,
        mesh_shape=(1,),
        mesh_dim_names=("fsdp",),
    )

    assert merged["position_ids"].dtype == torch.long
    assert torch.equal(merged["position_ids"], checkpoint["position_ids"])
    assert merged["weight"].dtype == torch.bfloat16
