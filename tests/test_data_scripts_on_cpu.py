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

import json

from data.split_tests import main as split_tests


def test_split_tests_creates_output_directory(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "new" / "split"
    tests = {"inputs": ["1", "2"], "outputs": ["1", "2"]}
    input_path.write_text(json.dumps({"tests": json.dumps(tests)}) + "\n", encoding="utf-8")

    split_tests(str(input_path), str(output_dir))

    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    assert train_path.is_file()
    assert test_path.is_file()
    train_record = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    test_record = json.loads(test_path.read_text(encoding="utf-8").splitlines()[0])
    assert len(json.loads(train_record["tests"])["inputs"]) == 1
    assert len(json.loads(test_record["tests"])["inputs"]) == 2
