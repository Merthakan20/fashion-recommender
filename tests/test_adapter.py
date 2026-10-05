import numpy as np

import fashion_rec.text_search as text_search_module
from fashion_rec.adapter import Adapter, current_adapter
from fashion_rec.text_search import build_text_index, clip_key


def identity(dim=3):
    return Adapter(np.eye(dim), np.zeros(dim), np.eye(dim), np.zeros(dim))


def test_identity_adapter_only_normalizes():
    x = np.array([[3.0, 0.0, 4.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(identity().image(x), x / np.linalg.norm(x, axis=1, keepdims=True), rtol=1e-6)


def test_adapter_transforms_image_and_text_separately():
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    adapter = Adapter(swap, np.zeros(3), np.eye(3), np.zeros(3))
    x = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(adapter.image(x), [[0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(adapter.text(x), [[1, 0, 0]], atol=1e-6)


def test_save_load_roundtrip_and_signature(tmp_path):
    path = tmp_path / "adapter.npz"
    identity().save(path)
    loaded = Adapter.load(path)
    np.testing.assert_array_equal(loaded.w_image, np.eye(3))
    assert len(loaded.signature) == 12
    assert current_adapter(tmp_path / "yok.npz") is None


def test_adapter_changes_clip_cache_key(data_dir, tmp_path, clip_model, monkeypatch):
    cache = tmp_path / "clip.npz"
    base_key = clip_key()
    build_text_index(data_dir, cache, model=clip_model)
    calls = clip_model.image_calls

    adapter = identity()
    adapter.signature = "abc123"
    monkeypatch.setattr(text_search_module, "current_adapter", lambda: adapter)
    assert clip_key() == base_key + "/adapter-abc123"
    build_text_index(data_dir, cache, model=clip_model)  # anahtar değişti: yeniden hesaplanmalı
    assert clip_model.image_calls == calls + 1
