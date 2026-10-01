from omegaconf import OmegaConf

from verl.experimental.dynamic_dataset import dynamicgen_dataset


def test_dynamicgen_dataset_runs_initial_generation_without_batch(monkeypatch):
    generated_datasets = []

    class Generator(dynamicgen_dataset.AbstractDataGenerator):
        def generate(self, dataset):
            assert dataset.dataframe == "initial"
            return "generated"

    def fake_rlhf_init(self, *args, **kwargs):
        self.dataframe = "initial"

    monkeypatch.setattr(dynamicgen_dataset.RLHFDataset, "__init__", fake_rlhf_init)
    monkeypatch.setattr(dynamicgen_dataset, "load_extern_object", lambda path, name: Generator)
    monkeypatch.setattr(
        dynamicgen_dataset.DynamicGenDataset,
        "append_dataframe",
        lambda self, dataframe: generated_datasets.append(dataframe),
    )

    config = OmegaConf.create({"datagen": {"path": "test.module", "name": "Generator"}})
    dynamicgen_dataset.DynamicGenDataset(data_files="unused", tokenizer=None, config=config)

    assert generated_datasets == ["generated"]
