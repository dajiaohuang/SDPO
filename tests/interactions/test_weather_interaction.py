import pytest

from verl.interactions.weather_interaction import WeatherInteraction


@pytest.mark.asyncio
async def test_weather_interaction_retries_when_tool_call_is_missing():
    interaction = WeatherInteraction({})
    instance_id = await interaction.start_interaction()

    should_terminate, response, reward, _ = await interaction.generate_response(
        instance_id, [{"role": "assistant", "content": "I do not know."}]
    )

    assert should_terminate is False
    assert response == "Please use the weather tool to get the weather information."
    assert reward == 0.0

    should_terminate, response, reward, _ = await interaction.generate_response(
        instance_id, [{"role": "tool", "content": '{"temperature": 20}'}]
    )

    assert should_terminate is True
    assert response == "Thank you for your weather query!"
    assert reward == 1.0
