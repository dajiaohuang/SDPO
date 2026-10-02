import asyncio
import importlib

import cloudpickle
import pytest

pytest.importorskip("vllm")
vLLMAsyncRollout = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout").vLLMAsyncRollout


class FakeSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.sent_two = asyncio.Event()

    async def recv(self):
        try:
            return next(self.messages)
        except StopIteration:
            await asyncio.Future()

    async def send(self, message):
        self.sent.append(cloudpickle.loads(message))
        if len(self.sent) == 2:
            self.sent_two.set()


class FakeRollout:
    _loop_forever = vLLMAsyncRollout._loop_forever

    async def _execute_method(self, method, *args, **kwargs):
        if method == "fail":
            raise ValueError("worker method failed")
        return "next request served"


@pytest.mark.asyncio
async def test_method_error_does_not_stop_zeromq_server_loop():
    rollout = FakeRollout()
    rollout.socket = FakeSocket(
        [
            cloudpickle.dumps(("fail", (), {})),
            cloudpickle.dumps(("succeed", (), {})),
        ]
    )

    task = asyncio.create_task(rollout._loop_forever())
    await asyncio.wait_for(rollout.socket.sent_two.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert isinstance(rollout.socket.sent[0], ValueError)
    assert str(rollout.socket.sent[0]) == "worker method failed"
    assert rollout.socket.sent[1] == "next request served"
