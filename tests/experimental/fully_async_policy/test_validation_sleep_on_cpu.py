# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer


@pytest.mark.asyncio
async def test_rollout_manager_sleeps_after_validation_failure():
    lifecycle = []
    trainer = SimpleNamespace(
        config=SimpleNamespace(async_training=SimpleNamespace(use_trainer_do_validate=True)),
        async_rollout_manager=SimpleNamespace(
            wake_up=lambda: _record(lifecycle, "wake"),
            sleep=lambda: _record(lifecycle, "sleep"),
        ),
        _validate=lambda _: (_ for _ in ()).throw(RuntimeError("validation failed")),
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        await FullyAsyncTrainer.__ray_actor_class__._validate_process(trainer)

    assert lifecycle == ["wake", "sleep"]


@pytest.mark.asyncio
async def test_rollout_manager_sleeps_after_partial_wake_failure():
    lifecycle = []

    async def fail_wake():
        lifecycle.append("wake")
        raise RuntimeError("wake failed")

    trainer = SimpleNamespace(
        config=SimpleNamespace(async_training=SimpleNamespace(use_trainer_do_validate=True)),
        async_rollout_manager=SimpleNamespace(wake_up=fail_wake, sleep=lambda: _record(lifecycle, "sleep")),
        _validate=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="wake failed"):
        await FullyAsyncTrainer.__ray_actor_class__._validate_process(trainer)

    assert lifecycle == ["wake", "sleep"]


async def _record(values, event):
    values.append(event)
