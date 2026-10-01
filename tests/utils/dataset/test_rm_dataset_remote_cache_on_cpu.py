import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


def test_rm_dataset_uses_collision_safe_local_hdfs_cache(tmp_path, monkeypatch, request):
    repo_root = Path(__file__).parents[3]

    verl_package = types.ModuleType("verl")
    verl_package.__path__ = [str(repo_root / "verl")]
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = [str(repo_root / "verl" / "utils")]
    utils_package.hf_tokenizer = Mock()
    dataset_package = types.ModuleType("verl.utils.dataset")
    dataset_package.__path__ = [str(repo_root / "verl" / "utils" / "dataset")]

    hdfs_io = types.ModuleType("hdfs_io")
    hdfs_io.copy = Mock()
    hdfs_io.exists = Mock()
    hdfs_io.makedirs = Mock()

    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizer = object

    monkeypatch.setitem(sys.modules, "verl", verl_package)
    monkeypatch.setitem(sys.modules, "verl.utils", utils_package)
    monkeypatch.setitem(sys.modules, "verl.utils.dataset", dataset_package)
    monkeypatch.setitem(sys.modules, "hdfs_io", hdfs_io)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    fs = importlib.import_module("verl.utils.fs")
    request.addfinalizer(lambda: sys.modules.pop("verl.utils.fs", None))

    copied_sources = []

    def fake_copy(src: str, dst: str):
        destination = Path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(src, encoding="utf-8")
        copied_sources.append(src)

    monkeypatch.setattr(fs, "copy", fake_copy)

    module_path = repo_root / "verl" / "utils" / "dataset" / "rm_dataset.py"
    spec = importlib.util.spec_from_file_location("verl.utils.dataset.rm_dataset", module_path)
    rm_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm_dataset)

    sources = ["hdfs://bucket-a/partition/data.parquet", "hdfs://bucket-b/partition/data.parquet"]
    dataset = rm_dataset.RMDataset.__new__(rm_dataset.RMDataset)
    dataset.cache_dir = str(tmp_path / "cache")
    dataset.parquet_files = sources.copy()
    dataset._download()

    assert [Path(path).read_text(encoding="utf-8") for path in dataset.parquet_files] == sources
    assert len(set(dataset.parquet_files)) == 2
    assert copied_sources == sources

    second_dataset = rm_dataset.RMDataset.__new__(rm_dataset.RMDataset)
    second_dataset.cache_dir = dataset.cache_dir
    second_dataset.parquet_files = sources.copy()
    second_dataset._download()

    assert second_dataset.parquet_files == dataset.parquet_files
    assert copied_sources == sources
