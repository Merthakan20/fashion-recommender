"""Görsel benzerliğine dayalı öneri."""

from pathlib import Path

import numpy as np

from .features import extract_features, load_model


def recommend(image_path: Path, paths: list[Path], features: np.ndarray,
              top_n: int = 5, model=None) -> list[tuple[Path, float]]:
    """Girdi görseline en benzer `top_n` görseli (yol, skor) olarak döndürür.

    Sonuçlar azalan benzerlikle sıralıdır; girdi görseli veri setindeyse
    sonuçlardan çıkarılır ve vektörü önbellekten alınır (model yüklenmez).
    """
    image_path = Path(image_path).resolve()
    resolved = [p.resolve() for p in paths]

    if image_path in resolved:
        query = features[resolved.index(image_path)]
    else:
        query = extract_features(model or load_model(), [image_path])[0]

    # vektörler L2 normalize olduğu için nokta çarpımı = cosine benzerliği
    similarities = features @ query
    ranked = np.argsort(similarities)[::-1]

    results = [(paths[i], float(similarities[i])) for i in ranked
               if resolved[i] != image_path]
    return results[:top_n]
