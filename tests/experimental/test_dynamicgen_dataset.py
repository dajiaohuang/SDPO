import pytest

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset import DynamicGenDataset
except ImportError as error:
    if "AutoModelForVision2Seq" in str(error):
        pytest.skip("installed Transformers is missing AutoModelForVision2Seq", allow_module_level=True)
    raise


def test_initial_dataset_generation_does_not_require_batch(monkeypatch):
    dataset = DynamicGenDataset.__new__(DynamicGenDataset)
    generated = object()
    appended = []

    class Generator:
        def generate(self, source_dataset):
            assert source_dataset is dataset
            return generated

    dataset.data_generator = Generator()
    monkeypatch.setattr(dataset, "append_dataframe", appended.append)

    dataset.on_batch_end()

    assert appended == [generated]
