import asyncio
from types import SimpleNamespace

import pytest

try:
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
except ImportError as error:
    if "AutoModelForVision2Seq" in str(error):
        pytest.skip("installed Transformers is missing AutoModelForVision2Seq", allow_module_level=True)
    raise


class FakeInteraction:
    def __init__(self):
        self.generated_with = []
        self.finalized = []

    async def start_interaction(self, instance_id, **kwargs):
        return "interaction-instance"

    async def generate_response(self, instance_id, messages, **kwargs):
        self.generated_with.append(instance_id)
        return True, "interaction response", 0.0, {}

    async def finalize_interaction(self, instance_id, **kwargs):
        self.finalized.append(instance_id)


class FakeParser:
    async def extract_tool_calls(self, response_ids):
        return "", []


class FakeServerManager:
    def __init__(self, fail=False):
        self.fail = fail

    async def generate(self, **kwargs):
        if self.fail:
            raise RuntimeError("generation failed")
        return SimpleNamespace(token_ids=[2], log_probs=[], routed_experts=None)


def make_loop(interaction, fail_generation=False):
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.interaction_config_file = "configured"
    loop.interaction_map = {"demo": interaction}
    loop.tool_schemas = []
    loop.tool_parser = FakeParser()
    loop.tool_parser_name = "hermes"
    loop.server_manager = FakeServerManager(fail=fail_generation)
    loop.response_length = 16
    loop.max_assistant_turns = 0
    loop.max_user_turns = 0
    loop.loop = asyncio.get_running_loop()
    loop.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "assistant response")

    async def process_vision_info(messages):
        return {}

    async def apply_chat_template(messages, **kwargs):
        return [1]

    loop.process_vision_info = process_vision_info
    loop.apply_chat_template = apply_chat_template
    return loop


def test_run_finalizes_returned_interaction_instance_id():
    async def run():
        interaction = FakeInteraction()
        loop = make_loop(interaction)

        await loop.run(
            {},
            raw_prompt=[{"role": "user", "content": "question"}],
            extra_info={"interaction_kwargs": {"name": "demo"}},
        )

        assert interaction.generated_with == ["interaction-instance"]
        assert interaction.finalized == ["interaction-instance"]

    asyncio.run(run())


def test_run_finalizes_interaction_when_generation_fails():
    async def run():
        interaction = FakeInteraction()
        loop = make_loop(interaction, fail_generation=True)

        with pytest.raises(RuntimeError, match="generation failed"):
            await loop.run(
                {},
                raw_prompt=[{"role": "user", "content": "question"}],
                extra_info={"interaction_kwargs": {"name": "demo"}},
            )

        assert interaction.finalized == ["interaction-instance"]

    asyncio.run(run())
