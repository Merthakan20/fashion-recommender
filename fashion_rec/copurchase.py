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
from .config import COPURCHASE_PATH
from .outfit import Weights

# kombin önerisinde kurallara eklenen ağırlıklar (değerlendirmede MRR ile seçildi)
PURCHASE_WEIGHTS = Weights(copurchase=8, popularity=1)


def load_copurchase(paths: list[Path], path: Path = COPURCHASE_PATH):
    """Birlikte alınma matrisi; dosya yoksa veya katalogla uyuşmuyorsa None."""
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
