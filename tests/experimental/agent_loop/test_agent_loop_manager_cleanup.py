import pytest

try:
    from verl.experimental.agent_loop import agent_loop as agent_loop_module
except ImportError as error:
    if "AutoModelForVision2Seq" in str(error):
        pytest.skip("installed Transformers is missing AutoModelForVision2Seq", allow_module_level=True)
    raise


class FakePrompts:
    def chunk(self, count):
        return [object() for _ in range(count)]


class FakeWorker:
    class GenerateSequences:
        @staticmethod
        def remote(chunk):
            return object()

    generate_sequences = GenerateSequences()


class FakeRewardModelManager:
    def __init__(self, events):
        self.events = events

    def wake_up(self):
        self.events.append("reward_wake")

    def sleep(self):
        self.events.append("reward_sleep")


def test_generation_failure_sleeps_rollout_and_reward_workers(monkeypatch):
    events = []
    manager = agent_loop_module.AgentLoopManager.__new__(agent_loop_module.AgentLoopManager)
    manager.agent_loop_workers = [FakeWorker()]
    manager.reward_model_manager = FakeRewardModelManager(events)
    manager.wake_up = lambda: events.append("rollout_wake")
    manager.sleep = lambda: events.append("rollout_sleep")

    def fail_get(object_refs):
        raise RuntimeError("worker generation failed")

    monkeypatch.setattr(agent_loop_module.ray, "get", fail_get)

    with pytest.raises(RuntimeError, match="worker generation failed"):
        manager.generate_sequences(FakePrompts())

    assert events == ["rollout_wake", "reward_wake", "rollout_sleep", "reward_sleep"]
