import torch

from verl.checkpoint_engine.base import synchronize_device


def test_synchronize_device_only_calls_cuda_for_cuda_devices(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(True))

    synchronize_device("cpu")
    assert calls == []

    synchronize_device("cuda")
    assert calls == [True]
