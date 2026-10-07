"""Kıyafetin renk dağılımından renk vektörü çıkarma ve önbellekleme."""

from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .config import COLOR_BINS, COLOR_CROP
from .data import list_images
from .features import cached_features


def color_histogram(path: Path) -> np.ndarray:
    """Görselin orta bölgesinin HSV histogramı; karekökü alınıp L2 normalize edilir.

    Orta bölge arka planın etkisini azaltır; karekök sayesinde cosine
    benzerliği Hellinger benzerliğine eşit olur.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    left, top, right, bottom = COLOR_CROP
    img = img.crop((int(left * w), int(top * h), int(right * w), int(bottom * h))).resize((64, 64))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 256
    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=COLOR_BINS, range=[(0, 1)] * 3)
    v = np.sqrt(hist.ravel())
    return v / np.linalg.norm(v)


def extract_color_features(paths: list[Path]) -> np.ndarray:
    return np.stack([color_histogram(p) for p in paths])


def build_color_index(data_dir: Path | None = None, cache_path: Path | None = None,
                      force: bool = False) -> tuple[list[Path], np.ndarray]:
    """Veri setinin (görsel yolları, renk vektörleri) ikilisini döndürür; önbellek kuralları `build_index` ile aynı."""
    paths = list_images(data_dir)
    return paths, cached_features(paths, cache_path or config.COLOR_CACHE_PATH, lambda: extract_color_features(paths), force,
                                  key=f"hsv{COLOR_BINS}-crop{COLOR_CROP}")
