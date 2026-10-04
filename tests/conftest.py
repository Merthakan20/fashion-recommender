"""Testler için ortak fixture'lar.

Gerçek VGG16 yerine hafif bir sahte model kullanılır; böylece testler
TensorFlow yüklemeden saniyeler içinde çalışır.
"""

import importlib

import numpy as np
import pytest
from PIL import Image

import fashion_rec.features as features_module
import fashion_rec.text_search as text_search_module

# `fashion_rec.recommend` paket seviyesinde fonksiyonu gösterir; modülün kendisi gerekiyor
recommend_module = importlib.import_module("fashion_rec.recommend")

# Düz renkli test görselleri. Kırmızı–turuncu ve mavi–camgöbeği birbirine
# benzer, kırmızı ile mavi ise hiç benzemez (cosine = 0).
COLORS = {
    "red.jpg": (255, 0, 0),
    "orange.png": (255, 128, 0),
    "blue.jpeg": (0, 0, 255),
    "cyan.webp": (0, 200, 255),
    "green.jpg": (0, 255, 0),
}


class FakeModel:
    """Görsel piksellerini olduğu gibi 'özellik' olarak döndürür ve çağrıları sayar."""

    def __init__(self):
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        return batch


class FakeClipModel:
    """Görselin ortalama rengini, metindeki renk kelimesinin RGB değerini vektör yapar."""

    WORDS = {"red": (255, 0, 0), "orange": (255, 128, 0), "blue": (0, 0, 255),
             "cyan": (0, 200, 255), "green": (0, 255, 0)}

    def __init__(self):
        self.image_calls = 0

    def encode_images(self, paths):
        self.image_calls += 1
        return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
                         for p in paths])

    def encode_text(self, texts):
        return np.array([next(self.WORDS[w] for w in t.lower().split() if w in self.WORDS)
                         for t in texts], dtype=np.float32)


def fake_preprocess(path):
    img = Image.open(path).convert("RGB").resize((4, 4))
    return np.asarray(img, dtype=np.float32)


def make_image(path, color):
    Image.new("RGB", (32, 32), color).save(path)


@pytest.fixture(autouse=True)
def no_tensorflow(monkeypatch):
    """TensorFlow'a giden yolları sahteleriyle değiştirir."""
    monkeypatch.setattr(features_module, "preprocess_image", fake_preprocess)

    def fail_load_model():
        raise AssertionError("testlerde gerçek model yüklenmemeli")

    monkeypatch.setattr(features_module, "load_model", fail_load_model)
    monkeypatch.setattr(recommend_module, "load_model", fail_load_model)
    monkeypatch.setattr(text_search_module, "load_clip_model", fail_load_model)


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name, color in COLORS.items():
        make_image(d / name, color)
    return d


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "features.npz"


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def clip_model():
    return FakeClipModel()
