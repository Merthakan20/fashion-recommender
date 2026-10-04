"""Testler için ortak fixture'lar.

Gerçek VGG16 yerine hafif bir sahte model kullanılır; böylece testler
TensorFlow yüklemeden saniyeler içinde çalışır.
"""

import importlib

import numpy as np
import pytest
from PIL import Image

import fashion_rec.features as features_module

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
