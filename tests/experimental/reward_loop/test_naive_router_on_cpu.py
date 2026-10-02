# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.experimental.reward_loop.router.naive_router import NaiveRouter


class TimeoutClient:
    def request(self, *args, **kwargs):
        raise TimeoutError


class Request:
    method = "POST"
    headers = {}

    async def body(self):
        return b"{}"


@pytest.mark.asyncio
async def test_worker_count_is_released_after_all_retries_fail():
    router = NaiveRouter(["http://worker"], max_attempts=2, retry_delay=0)
    router.client = TimeoutClient()

    with pytest.raises(RuntimeError, match="Failed to complete async request"):
        await router._make_async_request(Request(), "generate")

    assert router.request_counts == {"http://worker": 0}
