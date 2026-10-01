import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from verl.utils import megatron_peft_utils


class _AdapterModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.block = nn.Module()
        self.block.adapter = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.block.adapter.weight.fill_(value)


def _install_megatron_stubs(monkeypatch):
    mpu = SimpleNamespace(
        get_data_parallel_rank=lambda: 0,
        get_tensor_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
    )
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.mpu = mpu
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    megatron_utils = ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", megatron_utils)


def test_virtual_pipeline_adapter_states_roundtrip_independently(tmp_path, monkeypatch):
    _install_megatron_stubs(monkeypatch)
    model_path_prefix = tmp_path / "mp_rank_00_000"
    monkeypatch.setattr(megatron_peft_utils, "_get_rank_checkpoint_path", lambda _: str(model_path_prefix))

    source_models = [_AdapterModel(1.0), _AdapterModel(2.0)]
    megatron_peft_utils.save_adapter_checkpoint(source_models, str(tmp_path / "adapter_checkpoint"))

    checkpoint = torch.load(f"{model_path_prefix}_adapter.pt", map_location="cpu", weights_only=True)
    assert len(checkpoint["adapter_state_dicts"]) == 2

    restored_models = [_AdapterModel(0.0), _AdapterModel(0.0)]
    _install_megatron_stubs(monkeypatch)
    megatron_peft_utils.load_adapter_checkpoint(restored_models, str(tmp_path / "adapter_checkpoint"))

    for source, restored in zip(source_models, restored_models, strict=True):
        torch.testing.assert_close(source.block.adapter.weight, restored.block.adapter.weight)


def test_loads_legacy_single_chunk_adapter_checkpoint(tmp_path, monkeypatch):
    _install_megatron_stubs(monkeypatch)
    model_path_prefix = tmp_path / "mp_rank_00_000"
    monkeypatch.setattr(megatron_peft_utils, "_get_rank_checkpoint_path", lambda _: str(model_path_prefix))
    source_model = _AdapterModel(3.0)
    adapter_state = megatron_peft_utils.get_adapter_state_dict(source_model)
    torch.save({"adapter_state_dict": adapter_state}, f"{model_path_prefix}_adapter.pt")

    restored_model = _AdapterModel(0.0)
    _install_megatron_stubs(monkeypatch)
    megatron_peft_utils.load_adapter_checkpoint(restored_model, str(tmp_path / "adapter_checkpoint"))

    torch.testing.assert_close(source_model.block.adapter.weight, restored_model.block.adapter.weight)
