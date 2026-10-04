import numpy as np
import pytest

import fashion_rec.features as features_module
from fashion_rec.features import build_index, cached_features, extract_features

from .conftest import COLORS, make_image


def test_features_are_l2_normalized(data_dir, model):
    paths = sorted(data_dir.iterdir())
    feats = extract_features(model, paths)
    assert feats.shape == (len(paths), 4 * 4 * 3)
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, rtol=1e-6)


def test_extract_features_uses_batches(data_dir, model, monkeypatch):
    monkeypatch.setattr(features_module, "BATCH_SIZE", 2)
    feats = extract_features(model, sorted(data_dir.iterdir()))
    assert model.calls == 3  # 5 görsel / 2'şerli = 3 batch
    assert len(feats) == len(COLORS)


def test_build_index_writes_cache(data_dir, cache_path, model):
    paths, feats = build_index(data_dir, cache_path, model=model)
    assert cache_path.exists()
    assert [p.name for p in paths] == sorted(COLORS)
    assert feats.shape[0] == len(COLORS)


def test_second_call_uses_cache(data_dir, cache_path, model):
    _, first = build_index(data_dir, cache_path, model=model)
    calls = model.calls
    _, second = build_index(data_dir, cache_path, model=model)
    assert model.calls == calls
    np.testing.assert_array_equal(first, second)


def test_cache_without_model_does_not_load_model(data_dir, cache_path, model):
    build_index(data_dir, cache_path, model=model)
    # model verilmedi: önbellek geçerli olduğu için load_model çağrılmamalı
    build_index(data_dir, cache_path)


def test_new_image_invalidates_cache(data_dir, cache_path, model):
    build_index(data_dir, cache_path, model=model)
    calls = model.calls
    make_image(data_dir / "yellow.jpg", (255, 255, 0))
    paths, feats = build_index(data_dir, cache_path, model=model)
    assert model.calls > calls
    assert len(paths) == len(feats) == len(COLORS) + 1


def test_force_recomputes(data_dir, cache_path, model):
    build_index(data_dir, cache_path, model=model)
    calls = model.calls
    build_index(data_dir, cache_path, model=model, force=True)
    assert model.calls > calls


def test_model_key_change_invalidates_cache(data_dir, cache_path, model):
    paths = sorted(data_dir.iterdir())
    calls = []

    def compute():
        calls.append(1)
        return np.ones((len(paths), 3))

    cached_features(paths, cache_path, compute, key="model-a")
    cached_features(paths, cache_path, compute, key="model-a")
    assert len(calls) == 1
    cached_features(paths, cache_path, compute, key="model-b")
    assert len(calls) == 2


def test_cache_without_key_is_valid_for_empty_key(data_dir, cache_path):
    paths = sorted(data_dir.iterdir())
    np.savez(cache_path, features=np.ones((len(paths), 3)), names=np.array([p.name for p in paths]))
    result = cached_features(paths, cache_path, lambda: pytest.fail("önbellek kullanılmalıydı"))
    assert result.shape == (len(paths), 3)
