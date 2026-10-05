"""Görsel benzerliğine dayalı öneri."""

from pathlib import Path

import numpy as np

from .color import color_histogram
from .data import catalog_index
from .features import extract_features, load_model


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def recommend(image_path: Path, paths: list[Path], features: np.ndarray,
              top_n: int = 5, model=None, color_features: np.ndarray | None = None,
              color_weight: float = 0.0, encode=None) -> list[tuple[Path, float]]:
    """Girdi görseline en benzer `top_n` görseli (yol, skor) olarak döndürür.

    Sonuçlar azalan benzerlikle sıralıdır; girdi görseli veri setindeyse
    sonuçlardan çıkarılır ve vektörü önbellekten alınır (model yüklenmez).

    `color_weight` > 0 ise renk benzerliği de hesaba katılır: VGG16 ve renk
    benzerlikleri z-skoruna çevrilip ağırlıklı toplanır (skorlar artık 0–1
    aralığında değildir). Bunun için `color_features` gerekir.

    Girdi katalogda yoksa vektörü `encode(yollar) -> L2 normalize vektörler`
    ile (ör. CLIP), verilmemişse VGG16 (`model`) ile hesaplanır.
    """
    if not 0 <= color_weight <= 1:
        raise ValueError("color_weight 0 ile 1 arasında olmalı")
    if color_weight > 0 and color_features is None:
        raise ValueError("color_weight > 0 için color_features gerekli")

    query_index = catalog_index(paths)[1].get(Path(image_path).resolve())
    in_dataset = query_index is not None

    if in_dataset:
        query = features[query_index]
    else:
        query = encode([image_path])[0] if encode else extract_features(model or load_model(), [image_path])[0]

    # vektörler L2 normalize olduğu için nokta çarpımı = cosine benzerliği
    similarities = features @ query
    if color_weight > 0:
        color_query = color_features[query_index] if in_dataset else color_histogram(image_path)
        similarities = (1 - color_weight) * zscore(similarities) + color_weight * zscore(color_features @ color_query)
    ranked = np.argsort(similarities)[::-1]

    results = [(paths[i], float(similarities[i])) for i in ranked if i != query_index]
    return results[:top_n]
