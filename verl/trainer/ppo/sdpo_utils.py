"""Small helpers for SDPO trainer control flow."""


def should_skip_empty_sdpo_update(batch, actor_config) -> bool:
    """Avoid optimizer-only updates when SDPO has no valid teacher targets."""
    if actor_config.policy_loss.loss_mode != "sdpo" or actor_config.entropy_coeff != 0 or actor_config.use_kl_loss:
        return False

    if "self_distillation_mask" not in batch.batch or "response_mask" not in batch.batch:
        return False

    valid_target_tokens = batch.batch["response_mask"] * batch.batch["self_distillation_mask"].unsqueeze(-1)
    return not bool(valid_target_tokens.any().item())
