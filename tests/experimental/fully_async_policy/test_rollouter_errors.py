import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter


@pytest.mark.asyncio
async def test_streaming_generation_propagates_processor_error_and_signals_completion():
    rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    rollouter.async_rollout_manager = object()
    rollouter.current_param_version = 3
    rollouter.message_queue_client = SimpleNamespace(put_sample=AsyncMock())
    rollouter.lock = asyncio.Lock()
    rollouter.running = True

    async def feed_samples():
        await asyncio.Event().wait()

    async def process_samples():
        raise RuntimeError("worker failed")

    rollouter._feed_samples = feed_samples
    rollouter._processor_worker = process_samples

    with pytest.raises(RuntimeError, match="worker failed"):
        await rollouter._streaming_generation_main()

    assert rollouter.feed_task.cancelled()
    rollouter.message_queue_client.put_sample.assert_awaited_once_with(sample=None, param_version=3)
    assert not rollouter.running


@pytest.mark.asyncio
async def test_fit_propagates_generation_error():
    rollouter = FullyAsyncRollouter.__new__(FullyAsyncRollouter)
    rollouter.lock = asyncio.Lock()
    rollouter.paused = False
    rollouter.running = False

    async def fail_generation():
        raise RuntimeError("generation failed")

    async def monitor():
        await asyncio.Event().wait()

    rollouter._streaming_generation_main = fail_generation
    rollouter._async_monitor_loop = monitor

    with pytest.raises(RuntimeError, match="generation failed"):
        await rollouter.fit()
