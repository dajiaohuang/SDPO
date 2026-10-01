import torch

from verl.utils.activation_offload import _get_unique_tensor_key


def test_activation_offload_keys_distinguish_views_with_different_shapes():
    storage = torch.arange(6)

    short_view = storage[:3]
    long_view = storage[:4]

    assert _get_unique_tensor_key(short_view) != _get_unique_tensor_key(long_view)


def test_activation_offload_keys_distinguish_views_with_different_strides():
    storage = torch.arange(9).reshape(3, 3)

    row_major_view = storage
    transposed_view = storage.t()

    assert row_major_view.shape == transposed_view.shape
    assert _get_unique_tensor_key(row_major_view) != _get_unique_tensor_key(transposed_view)


def test_activation_offload_keys_match_identical_views():
    storage = torch.arange(9).reshape(3, 3)

    first_view = storage[:, 1:]
    same_view = storage[:, 1:]

    assert _get_unique_tensor_key(first_view) == _get_unique_tensor_key(same_view)
