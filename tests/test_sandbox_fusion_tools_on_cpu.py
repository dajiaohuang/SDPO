# Copyright 2026
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

from verl.tools.sandbox_fusion_tools import ExecutionWorker


def test_execution_worker_without_global_rate_limit():
    worker = ExecutionWorker(enable_global_rate_limit=False)

    assert worker.execute(lambda left, right: left + right, 2, 3) == 5
