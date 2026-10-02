# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _apply_chat_template_for_ids(tokenizer, messages, add_generation_prompt, apply_chat_template_kwargs):
    kwargs = {
        **apply_chat_template_kwargs,
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
    }
    return tokenizer.apply_chat_template(messages, **kwargs)


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = _apply_chat_template_for_ids(
        tokenizer, [{"role": "user", "content": ""}], False, apply_chat_template_kwargs
    )
    token2 = _apply_chat_template_for_ids(
        tokenizer, [{"role": "user", "content": ""}] * 2, False, apply_chat_template_kwargs
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = _apply_chat_template_for_ids(
        tokenizer, [{"role": "user", "content": ""}], False, apply_chat_template_kwargs
    )
    token2 = _apply_chat_template_for_ids(
        tokenizer, [{"role": "user", "content": ""}] * 2, False, apply_chat_template_kwargs
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = _apply_chat_template_for_ids(
        tokenizer, [{"role": "user", "content": ""}], True, apply_chat_template_kwargs
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
