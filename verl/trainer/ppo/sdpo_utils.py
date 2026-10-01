"""Small helpers for SDPO trainer control flow."""


import os


def should_skip_empty_sdpo_update(batch, actor_config) -> bool:
    """Avoid optimizer-only updates when SDPO has no valid teacher targets."""
    if actor_config.policy_loss.loss_mode != "sdpo" or actor_config.entropy_coeff != 0 or actor_config.use_kl_loss:
        return False

    if "self_distillation_mask" not in batch.batch or "response_mask" not in batch.batch:
        return False

    valid_target_tokens = batch.batch["response_mask"] * batch.batch["self_distillation_mask"].unsqueeze(-1)
    return not bool(valid_target_tokens.any().item())


def validate_sdpo_strategy(strategy: str, self_distillation_enabled: bool) -> None:
    """Reject actor backends without the SDPO loss and teacher integration."""
    if self_distillation_enabled and strategy == "megatron":
        raise ValueError("SDPO is not implemented for the Megatron actor strategy; use FSDP or FSDP2.")


def save_sdpo_teacher_checkpoint(checkpoint_manager, local_path, global_step, max_ckpt_to_keep=None):
    if checkpoint_manager is None or local_path is None:
        return
    checkpoint_manager.save_checkpoint(
        local_path=os.path.join(local_path, "sdpo_teacher"),
        hdfs_path=None,
        global_step=global_step,
        max_ckpt_to_keep=max_ckpt_to_keep,
    )


def load_sdpo_teacher_checkpoint(checkpoint_manager, local_path, del_local_after_load=False):
    if checkpoint_manager is None or local_path is None:
        return
    teacher_path = os.path.join(local_path, "sdpo_teacher")
    if not teacher_path.startswith("hdfs") and not os.path.exists(teacher_path):
        raise FileNotFoundError(
            "EMA SDPO checkpoint is missing `sdpo_teacher`; it cannot be resumed faithfully. "
            "Use a checkpoint saved with EMA teacher state or start a new run."
        )
    checkpoint_manager.load_checkpoint(
        local_path=teacher_path,
        hdfs_path=None,
        del_local_after_load=del_local_after_load,
    )
