"""Satın alma kayıtlarından "birlikte alınanlar" (bkz. scripts/build_copurchase.py).

Sepet = aynı müşterinin aynı gün aldıkları. H&M'de son 3 ayın sepetleriyle
yapılan değerlendirmede (notebooks/evaluate_copurchase.py) kombin
yuvalarında gerçekten birlikte alınan parçayı ilk 10'a getirme oranı:
kurallar 0,026, çok satanlar 0,043, birlikte alınma 0,087; geçmişi olan
ürünlerde birlikte alınma kurallardan ~5 kat iyi. Yeni ürünlerde (geçmişi
yok) kurallar en iyi yedek olduğundan kombin önerisi ikisini birleştirir.
"""

from pathlib import Path

import numpy as np
from scipy import sparse

from .catalog import Metadata, display_name
from . import config
from .outfit import Weights

# kombin önerisinde kurallara eklenen ağırlıklar (değerlendirmede MRR ile seçildi)
PURCHASE_WEIGHTS = Weights(copurchase=8, popularity=1)


def load_copurchase(paths: list[Path], path: Path | None = None):
    """Birlikte alınma matrisi; dosya yoksa veya katalogla uyuşmuyorsa None."""
    path = path or config.COPURCHASE_PATH
    if not Path(path).exists():
        return None
    matrix = sparse.load_npz(path).tocsr()
    return matrix if matrix.shape == (len(paths), len(paths)) else None


def base_name(path: Path, metadata: Metadata) -> str:
    """Renk eki olmadan ürün adı ("Strap top - Black" → "Strap top"); aynı ürünün renklerini tanımak için."""
    return display_name(path, metadata).split(" - ")[0].strip().lower()


def bought_together(path: Path, paths: list[Path], matrix, metadata: Metadata,
                    k: int = 8) -> list[tuple[Path, int]]:
    """`path` ile aynı sepette en sık alınan `k` ürün (yol, sepet sayısı).

    Aynı ürünün başka renkleri (aynı ad) çıkarılır: onlar tamamlayıcı değil, alternatiftir.
    """
    index = {p: i for i, p in enumerate(paths)}
    if matrix is None or path not in index:
        return []
    row = matrix[index[path]]
    order = np.argsort(-row.data)
    own = base_name(path, metadata)
    out = []
    for j in order:
        candidate = paths[row.indices[j]]
        if base_name(candidate, metadata) != own:
            out.append((candidate, int(row.data[j])))
        if len(out) == k:
            break
    return out


MAX_BASKET = 30  # toptan alımlar gibi çok büyük sepetler çiftleri şişirmesin


def baskets_matrix(customer: np.ndarray, day: np.ndarray, item: np.ndarray, n_items: int):
    """(sepet × ürün) ikili seyrek matris; sepet = (müşteri, gün), 2–30 farklı ürünlü sepetler."""
    keys = np.unique(np.stack([customer, day], axis=1), axis=0, return_inverse=True)[1].ravel()
    m = sparse.csr_matrix((np.ones(len(item), dtype=np.float32), (keys, item)), shape=(keys.max() + 1, n_items))
    m.data[:] = 1  # aynı sepette aynı ürün birden çok kez alınmış olabilir
    sizes = np.asarray(m.sum(axis=1)).ravel()
    return m[(sizes >= 2) & (sizes <= MAX_BASKET)]


def copurchase_matrix(baskets) -> sparse.csr_matrix:
    """(ürün × ürün) birlikte alınma sayıları: aynı sepette kaç kez birlikte geçtikleri."""
    cooc = (baskets.T @ baskets).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    return cooc

