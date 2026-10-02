import copy

import pytest
import torch
import torch.nn as nn

from verl.models.transformers.tiled_mlp import TiledMLP, _mlp_forward_fn


class SimpleSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.up_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)
        self.act_fn = nn.SiLU()


@pytest.mark.parametrize("sequence_length", [1, 3, 4, 5])
def test_tiled_mlp_matches_reference_when_sequence_has_fewer_chunks(sequence_length):
    torch.manual_seed(0)
    reference = SimpleSwiGLU()
    tiled = copy.deepcopy(reference)
    reference_input = torch.randn(2, sequence_length, 8, requires_grad=True)
    tiled_input = reference_input.detach().clone().requires_grad_()
    grad_output = torch.randn_like(reference_input)

    reference_output = _mlp_forward_fn(reference, reference_input)
    tiled_output = TiledMLP.apply(_mlp_forward_fn, tiled, tiled_input, 4, list(tiled.parameters()))

    reference_output.backward(grad_output)
    tiled_output.backward(grad_output)

    torch.testing.assert_close(tiled_output, reference_output)
    torch.testing.assert_close(tiled_input.grad, reference_input.grad)
    for reference_parameter, tiled_parameter in zip(reference.parameters(), tiled.parameters(), strict=True):
        torch.testing.assert_close(tiled_parameter.grad, reference_parameter.grad)
