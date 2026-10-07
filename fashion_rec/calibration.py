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

from .data import catalog_index
from .diversity import mmr
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


def _dedupe(results, paths, features, query=None):
    """Arayüzdeki gibi kopyaları ele (sorgunun başka ilanları, aynı ürünün iki ilanı) ve skora göre sırala.

    Referans kopyaları içerseydi (benzerlik ≈ 1) çıta yapay olarak yükselir ve
    gerçekten farklı ürünler düşük yüzde alırdı.
    """
    vector = None
    if query is not None:
        row = catalog_index(paths)[1].get(Path(query).resolve())
        vector = features[row] if row is not None else None
    kept = mmr(results, paths, features, TOP_K, diversity=0.0, query_vector=vector, drop_near_duplicates=True)
    return sorted(kept, key=lambda r: -r[1])


def image_scale(paths: list[Path], features: np.ndarray, color_features: np.ndarray | None = None,
                color_weight: float = 0.0, sample: int = 100, subset: list[Path] | None = None) -> PercentScale:
    """Görselden öneri için referans: örnek katalog görsellerinin en iyi eşleşmelerinin skorları.

    `subset` verilirse (ör. bir bölümün ürünleri) sorgular oradan seçilir ve
    sonuçlar oraya sınırlanır; öneriler bölümle sınırlıyken yüzde de o
    bölümün tipik eşleşmeleriyle kıyaslanır.
    """
    allowed = set(subset) if subset is not None else None
    pool = list(subset) if subset is not None else paths
    return PercentScale(_scores(
        _dedupe([r for r in recommend(q, paths, features, top_n=len(paths), color_features=color_features,
                                      color_weight=color_weight) if allowed is None or r[0] in allowed],
                paths, features, q)
        for q in sample_paths(pool, sample)))


def text_scale(paths: list[Path], clip_features: np.ndarray, model, queries: list[str]) -> PercentScale:
    """Metinden arama için referans: genel kıyafet tariflerinin en iyi eşleşmelerinin skorları."""
    return PercentScale(_scores(_dedupe(search(q, paths, clip_features, top_n=100, model=model), paths, clip_features)
                                for q in queries))


def combined_scale(paths: list[Path], clip_features: np.ndarray, model, queries: list[str],
                   text_weight: float = 0.5, sample: int = 20, subset: list[Path] | None = None) -> PercentScale:
    """Görsel + metin araması için referans: örnek görsel × genel tarif çiftlerinin en iyi eşleşmeleri.

    `subset` için bkz. `image_scale`.
    """
    allowed = set(subset) if subset is not None else None
    pool = list(subset) if subset is not None else paths
    return PercentScale(_scores(
        _dedupe([r for r in combined_search(image, q, paths, clip_features, top_n=len(paths) if allowed else 100,
                                            text_weight=text_weight, model=model) if allowed is None or r[0] in allowed],
                paths, clip_features, image)
        for image in sample_paths(pool, sample) for q in queries))
