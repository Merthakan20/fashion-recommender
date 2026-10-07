import pytest

from fashion_rec import config
from fashion_rec.data import list_images


@pytest.fixture(autouse=True)
def restore_dataset():
    before = config.DATASET
    yield
    config.set_dataset(before)


def test_set_dataset_switches_all_dataset_paths(tmp_path, monkeypatch):
    config.set_dataset("hm")
    assert config.DATA_DIR == config.DATASETS["hm"]
    assert config.IMAGE_BACKBONE == "clip"
    assert config.CLIP_CACHE_PATH.parent.name == "hm" and config.COPURCHASE_PATH.parent.name == "hm"
    config.set_dataset("women")
    assert config.IMAGE_BACKBONE == "vgg16" and config.CACHE_PATH.parent.name == "women"


def test_functions_read_dataset_at_call_time(data_dir, monkeypatch):
    monkeypatch.setitem(config.DATASETS, "hm", data_dir)
    config.set_dataset("hm")
    assert [p.parent for p in list_images()] == [data_dir] * len(list_images())


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError):
        config.set_dataset("zara")
