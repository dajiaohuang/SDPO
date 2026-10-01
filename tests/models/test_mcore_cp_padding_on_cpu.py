import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _PackedSeqParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_preprocessor(monkeypatch, cp_size: int, cp_rank: int):
    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: cp_rank,
    )
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.parallel_state = mpu
    packed_seq_params = ModuleType("megatron.core.packed_seq_params")
    packed_seq_params.PackedSeqParams = _PackedSeqParams
    core.packed_seq_params = packed_seq_params
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)

    model = ModuleType("verl.utils.model")
    model.CausalLMOutputForPPO = type("CausalLMOutputForPPO", (), {})
    monkeypatch.setitem(sys.modules, "verl.utils.model", model)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_cpu_test", util_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_thd_no_padding


def _run_preprocessor(monkeypatch, values: list[int], cp_size: int, cp_rank: int, need_roll: bool):
    preprocess = _load_preprocessor(monkeypatch, cp_size, cp_rank)
    input_ids = torch.nested.as_nested_tensor([torch.tensor(values)], layout=torch.jagged)
    output, _ = preprocess(input_ids, pre_process=True, need_roll=need_roll)
    return output[0].tolist()


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_short_two_token_sequence_rolls_labels_across_cp_ranks(monkeypatch, cp_rank):
    output = _run_preprocessor(monkeypatch, [10, 11], cp_size=2, cp_rank=cp_rank, need_roll=True)

    expected_valid_labels = [11, 10]
    assert output[0] == expected_valid_labels[cp_rank]


def test_shorter_than_front_chunk_has_no_assignment_or_boundary_error(monkeypatch):
    rank_zero = _run_preprocessor(monkeypatch, [10], cp_size=2, cp_rank=0, need_roll=True)
    rank_one = _run_preprocessor(monkeypatch, [10], cp_size=2, cp_rank=1, need_roll=True)

    assert rank_zero[0] == 10
    assert rank_one == [0, 0]


@pytest.mark.parametrize("cp_rank,expected_valid_labels", [(0, [11, 12]), (1, [13, 14, 10])])
def test_cp_padding_preserves_next_token_mapping_for_partial_final_chunk(monkeypatch, cp_rank, expected_valid_labels):
    output = _run_preprocessor(monkeypatch, [10, 11, 12, 13, 14], cp_size=2, cp_rank=cp_rank, need_roll=True)

    assert output[: len(expected_valid_labels)] == expected_valid_labels
