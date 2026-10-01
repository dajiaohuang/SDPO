import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_rendezvous_raises_when_id_store_does_not_appear(monkeypatch):
    ray = types.ModuleType("ray")
    ray.__path__ = []
    ray.remote = lambda cls: cls
    ray.get = Mock()
    ray.get_actor = Mock()

    ray_util = types.ModuleType("ray.util")
    ray_util.list_named_actors = Mock()

    cupy = types.ModuleType("cupy")
    cupy.__path__ = []
    cupy_cuda = types.ModuleType("cupy.cuda")
    cupy_cuda.__path__ = []
    cupy_nccl = types.ModuleType("cupy.cuda.nccl")
    cupy_nccl.NcclCommunicator = Mock()
    cupy_nccl.get_unique_id = Mock()

    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cupy_cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nccl", cupy_nccl)

    source = Path(__file__).parents[2] / "verl" / "utils" / "rendezvous" / "ray_backend.py"
    spec = importlib.util.spec_from_file_location("ray_backend_timeout_test", source)
    ray_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ray_backend)

    monkeypatch.setattr(ray_backend, "get_nccl_id_store_by_name", Mock(return_value=None))
    monkeypatch.setattr(ray_backend.time, "sleep", Mock())

    with pytest.raises(TimeoutError, match="Timed out waiting for NCCL ID store 'test-group' after 2 attempts"):
        ray_backend.create_nccl_communicator_in_ray(rank=1, world_size=2, group_name="test-group", max_retries=2)
