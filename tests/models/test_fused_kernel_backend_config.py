from types import SimpleNamespace

import pytest

from verl.models.transformers.monkey_patch import patch_forward_with_backends


def test_fused_kernel_backend_defaults_to_torch():
    model_type = type("DummyModel", (), {"forward": lambda self: None})
    model = model_type()
    model.config = SimpleNamespace(model_type="llama")

    patch_forward_with_backends(model, use_fused_kernels=True)

    assert model_type.forward.__module__ == "verl.models.transformers.dense_common"
    assert model_type.forward.__name__ == "forward_with_torch_backend"


def test_fused_kernel_backend_rejects_unsupported_value():
    model = object()

    with pytest.raises(ValueError, match="Unsupported fused_kernels_backend: typo"):
        patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="typo")
