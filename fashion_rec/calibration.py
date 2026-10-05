"""Ham benzerlik skorlarını anlaşılır yüzdelere çevirme.

Ham skorlar (cosine ya da z-skoru toplamı) yönteme göre farklı aralıklardadır
ve doğrudan yüzde değildir: CLIP'te metin–görsel benzerliği 0,30 çok iyi bir
eşleşmedir. Bu yüzden skor, aynı yöntemle üretilmiş referans sorgularının ilk
`TOP_K` sonucundan oluşan bir dağılımla karşılaştırılır: %80, "tipik en iyi
eşleşmelerin %80'inden daha benzer" demektir.

Tüm çiftler yerine yalnızca en iyi eşleşmelerin kullanılması yüzdeyi katalog
büyüklüğünden bağımsız kılar: 44 bin üründe ilk birkaç sonuç tüm çiftlerin
%99,99'undan iyidir ve yüzde bir şey söylemez. Referans sabit olduğundan
anlamsız bir sorgu yüksek yüzde almaz.
"""

from pathlib import Path

import numpy as np

from .recommend import recommend
from .text_search import combined_search, search

TOP_K = 10  # her referans sorgusunun ilk kaç sonucu referansa girer


class PercentScale:
    """Skoru, referans skorlar içindeki yüzdelik sırasına (0–100) çevirir."""

    def __init__(self, reference):
        self.reference = np.sort(np.asarray(reference, dtype=float).ravel())
        if not len(self.reference):
            raise ValueError("referans skor listesi boş olamaz")

    def __call__(self, score: float) -> float:
        return 100.0 * np.searchsorted(self.reference, score, side="right") / len(self.reference)


def sample_paths(paths: list[Path], n: int, seed: int = 0) -> list[Path]:
    """Tekrarlanabilir rastgele örnek (katalog büyüdükçe referansı hızlı tutmak için)."""
    if n >= len(paths):
        return list(paths)
    idx = np.random.default_rng(seed).choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def _scores(results_lists, top_k: int = TOP_K) -> list[float]:
    """Her sonuç listesinin (azalan sıralı) ilk `top_k` skoru."""
    return [score for results in results_lists for _, score in results[:top_k]]


def image_scale(paths: list[Path], features: np.ndarray, color_features: np.ndarray | None = None,
                color_weight: float = 0.0, sample: int = 100) -> PercentScale:
    """Görselden öneri için referans: örnek katalog görsellerinin en iyi eşleşmelerinin skorları."""
    return PercentScale(_scores(
        recommend(q, paths, features, top_n=TOP_K, color_features=color_features, color_weight=color_weight)
        for q in sample_paths(paths, sample)))


def text_scale(paths: list[Path], clip_features: np.ndarray, model, queries: list[str]) -> PercentScale:
    """Metinden arama için referans: genel kıyafet tariflerinin en iyi eşleşmelerinin skorları."""
    return PercentScale(_scores(search(q, paths, clip_features, top_n=TOP_K, model=model) for q in queries))


def combined_scale(paths: list[Path], clip_features: np.ndarray, model, queries: list[str],
                   text_weight: float = 0.5, sample: int = 20) -> PercentScale:
    """Görsel + metin araması için referans: örnek görsel × genel tarif çiftlerinin en iyi eşleşmeleri."""
    return PercentScale(_scores(
        combined_search(image, q, paths, clip_features, top_n=TOP_K, text_weight=text_weight, model=model)
        for image in sample_paths(paths, sample) for q in queries))
