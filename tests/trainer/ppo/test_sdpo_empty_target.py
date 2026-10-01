from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.sdpo_utils import should_skip_empty_sdpo_update, validate_sdpo_strategy


def _actor_config(*, loss_mode="sdpo", entropy_coeff=0, use_kl_loss=False):
    return SimpleNamespace(
        policy_loss=SimpleNamespace(loss_mode=loss_mode),
        entropy_coeff=entropy_coeff,
        use_kl_loss=use_kl_loss,
    )


def _batch(target_mask, response_mask=None):
    if response_mask is None:
        response_mask = torch.ones((len(target_mask), 3))
    return SimpleNamespace(
        batch={
            "self_distillation_mask": torch.tensor(target_mask, dtype=torch.float32),
            "response_mask": response_mask,
        }
    )


def test_empty_sdpo_targets_skip_optimizer_only_update():
    assert should_skip_empty_sdpo_update(_batch([0, 0]), _actor_config())


def test_valid_sdpo_target_keeps_actor_update():
    assert not should_skip_empty_sdpo_update(_batch([0, 1]), _actor_config())


def test_empty_response_tokens_skip_sdpo_update():
    assert should_skip_empty_sdpo_update(_batch([1], response_mask=torch.zeros((1, 3))), _actor_config())


def test_other_actor_objectives_keep_update_for_empty_sdpo_targets():
    batch = _batch([0, 0])
    assert not should_skip_empty_sdpo_update(batch, _actor_config(entropy_coeff=0.01))
    assert not should_skip_empty_sdpo_update(batch, _actor_config(use_kl_loss=True))
    assert not should_skip_empty_sdpo_update(batch, _actor_config(loss_mode="vanilla"))


def test_megatron_sdpo_fails_with_a_supported_backend_message():
    with pytest.raises(ValueError, match="not implemented for the Megatron actor strategy"):
        validate_sdpo_strategy("megatron", self_distillation_enabled=True)


def test_supported_sdpo_backends_and_non_sdpo_megatron_pass_validation():
    validate_sdpo_strategy("fsdp", self_distillation_enabled=True)
    validate_sdpo_strategy("fsdp2", self_distillation_enabled=True)
    validate_sdpo_strategy("megatron", self_distillation_enabled=False)
