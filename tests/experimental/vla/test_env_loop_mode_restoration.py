import asyncio

import pytest

from verl.experimental.vla.env_loop import EnvLoop


class ResetFuture:
    def get(self):
        return object()


class RolloutWorkerGroup:
    def __init__(self):
        self.modes = []

    def switch_to_rollout(self):
        self.modes.append("rollout")

    def switch_to_train(self):
        self.modes.append("train")


class AsyncLoop:
    def run_until_complete(self, coroutine):
        return asyncio.run(coroutine)


def test_generate_sequences_restores_train_mode_when_rollout_fails(monkeypatch):
    env_loop = EnvLoop.__new__(EnvLoop)
    env_loop.rollout_wg = RolloutWorkerGroup()

    async def fail_run(prompts, reset_results):
        raise RuntimeError("rollout failed")

    env_loop.run = fail_run
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: AsyncLoop())

    with pytest.raises(RuntimeError, match="rollout failed"):
        env_loop.generate_sequences(object(), ResetFuture())

    assert env_loop.rollout_wg.modes == ["rollout", "train"]
