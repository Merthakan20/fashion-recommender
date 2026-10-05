"""Görsel benzerliğine dayalı öneri."""

from pathlib import Path

import numpy as np

from .color import color_histogram
from .features import extract_features, load_model


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def recommend(image_path: Path, paths: list[Path], features: np.ndarray,
              top_n: int = 5, model=None, color_features: np.ndarray | None = None,
              color_weight: float = 0.0) -> list[tuple[Path, float]]:
    """Girdi görseline en benzer `top_n` görseli (yol, skor) olarak döndürür.

    Sonuçlar azalan benzerlikle sıralıdır; girdi görseli veri setindeyse
    sonuçlardan çıkarılır ve vektörü önbellekten alınır (model yüklenmez).

    `color_weight` > 0 ise renk benzerliği de hesaba katılır: VGG16 ve renk
    benzerlikleri z-skoruna çevrilip ağırlıklı toplanır (skorlar artık 0–1
    aralığında değildir). Bunun için `color_features` gerekir.
    """
    if not 0 <= color_weight <= 1:
        raise ValueError("color_weight 0 ile 1 arasında olmalı")
    if color_weight > 0 and color_features is None:
        raise ValueError("color_weight > 0 için color_features gerekli")

    image_path = Path(image_path).resolve()
    resolved = [p.resolve() for p in paths]
    in_dataset = image_path in resolved

    if in_dataset:
        query = features[resolved.index(image_path)]
    else:
        query = extract_features(model or load_model(), [image_path])[0]

    # vektörler L2 normalize olduğu için nokta çarpımı = cosine benzerliği
    similarities = features @ query
    if color_weight > 0:
        color_query = color_features[resolved.index(image_path)] if in_dataset else color_histogram(image_path)
        similarities = (1 - color_weight) * zscore(similarities) + color_weight * zscore(color_features @ color_query)
    ranked = np.argsort(similarities)[::-1]

    results = [(paths[i], float(similarities[i])) for i in ranked
               if resolved[i] != image_path]
    return results[:top_n]
