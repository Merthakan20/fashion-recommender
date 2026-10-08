"""Ürünü tam göstermeyen (kırpık / yakın çekim) fotoğrafları tanıma.

Bazı ürünlerin tek fotoğrafı ürünün yalnız bir kısmını gösteriyor (ör. H&M'de
yalnız topuğu görünen ayakkabı, kumaş dokusu yakın çekimi); veri setinde
başka açıdan fotoğraf yok. Stüdyo çekimlerinde ürün tamamen görünüyorsa
fotoğrafın kenarları açık renkli fondur; kırpık çekimde ürün kenarlara taşar.
Ölçüt: kenar şeridindeki fon olmayan piksel oranı. H&M'de görsellerin %96'sı
0–0,05 aralığında; 0,2 üstü (~%3) gözle kontrolde kırpık / yakın çekim.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .features import cached_features, report_progress

PARTIAL_THRESHOLD = 0.2


def border_ratio(path: Path) -> float:
    """Kenar şeridinde fon (açık ve renksiz) olmayan piksel oranı (0 = ürün kenarlara değmiyor)."""
    a = np.asarray(Image.open(path).convert("RGB").resize((48, 72)), dtype=np.float32)
    border = np.concatenate([a[:2].reshape(-1, 3), a[-2:].reshape(-1, 3),
                             a[:, :2].reshape(-1, 3), a[:, -2:].reshape(-1, 3)])
    background = (border.mean(axis=1) > 200) & ((border.max(axis=1) - border.min(axis=1)) < 25)
    return float(1 - background.mean())


def build_photo_index(paths: list[Path], cache_path: Path | None = None) -> np.ndarray:
    """Her görselin kenar oranı (önbellekli)."""
    def compute():
        out = []
        for i, p in enumerate(paths):
            if i % 16 == 0:
                report_progress(i + 16, len(paths), "fotoğraf kontrolü")
            out.append(border_ratio(p))
        return np.array(out, dtype=np.float32)

    return cached_features(paths, cache_path or config.PHOTO_CACHE_PATH, compute, key="border-v1")


def partial_names(paths: list[Path], ratios: np.ndarray, threshold: float = PARTIAL_THRESHOLD) -> set[str]:
    """Kırpık / yakın çekim fotoğraflı ürünlerin dosya adları."""
    return {p.name for p, r in zip(paths, ratios) if r > threshold}
