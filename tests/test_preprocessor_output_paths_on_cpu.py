import runpy
import sys
from pathlib import Path

import pytest

import datasets


class _FakeDataset:
    def map(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def train_test_split(self, *args, **kwargs):
        return {"train": self, "test": self}

    def to_parquet(self, path):
        output = Path(path)
        assert output.parent.is_dir()
        output.write_bytes(b"test parquet")

    def __getitem__(self, index):
        assert isinstance(index, int)
        return {}


class _FakeDatasetDict:
    def __getitem__(self, split):
        return _FakeDataset()


@pytest.mark.parametrize(
    "script_name",
    [
        "aime2024_multiturn_w_tool.py",
        "dapo_multiturn_w_tool.py",
        "geo3k.py",
        "geo3k_multiturn_w_tool.py",
        "gsm8k.py",
        "gsm8k_multiturn_sft.py",
        "gsm8k_multiturn_w_interaction.py",
        "gsm8k_multiturn_w_tool.py",
        "gsm8k_tool_agent_loop.py",
        "hellaswag.py",
        "math_dataset.py",
        "pokemon.py",
    ],
)
def test_preprocessor_expands_and_creates_output_dir(monkeypatch, tmp_path, script_name):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: _FakeDatasetDict())
    script = Path("examples/data_preprocess") / script_name
    monkeypatch.setattr(sys, "argv", [str(script)])

    runpy.run_path(str(script), run_name="__main__")

    files = list(tmp_path.rglob("*.parquet"))
    assert files
    assert all(path.is_file() for path in files)
    assert all(path.is_relative_to(tmp_path) for path in files)
