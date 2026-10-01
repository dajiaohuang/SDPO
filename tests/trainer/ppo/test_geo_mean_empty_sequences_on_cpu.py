from types import SimpleNamespace

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_geo_mean


def test_geo_mean_loss_ignores_fully_masked_sequences():
    config = SimpleNamespace(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2)
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.tensor([[0.1]])
    advantages = torch.tensor([[2.0]])
    response_mask = torch.ones(1, 1)

    single_loss, _ = compute_policy_loss_geo_mean(
        old_log_prob, log_prob, advantages, response_mask, config=config
    )
    mixed_loss, _ = compute_policy_loss_geo_mean(
        old_log_prob=torch.cat((old_log_prob, torch.zeros_like(old_log_prob))),
        log_prob=torch.cat((log_prob, torch.zeros_like(log_prob))),
        advantages=torch.cat((advantages, torch.full_like(advantages, 9.0))),
        response_mask=torch.cat((response_mask, torch.zeros_like(response_mask))),
        config=config,
    )

    torch.testing.assert_close(mixed_loss, single_loss)
