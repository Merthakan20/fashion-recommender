from pathlib import Path

import numpy as np
from scipy import sparse

from fashion_rec.copurchase import bought_together, load_copurchase
from fashion_rec.outfit import OutfitCatalog, Weights

NAMES = ["shirt.jpg", "jeans_a.jpg", "jeans_b.jpg", "shirt_red.jpg"]
META = {
    "shirt.jpg": {"name": "Oxford shirt - Blue", "subCategory": "Topwear", "gender": "Men", "usage": "Casual"},
    "jeans_a.jpg": {"name": "Slim jeans - Black", "subCategory": "Bottomwear", "gender": "Men", "usage": "Casual"},
    "jeans_b.jpg": {"name": "Loose jeans - Blue", "subCategory": "Bottomwear", "gender": "Men", "usage": "Casual"},
    "shirt_red.jpg": {"name": "Oxford shirt - Red", "subCategory": "Topwear", "gender": "Men", "usage": "Casual"},
}


def matrix():
    # shirt: jeans_b ile 5, jeans_a ile 1, kendi kırmızı rengiyle 9 sepette
    m = np.zeros((4, 4))
    m[0, 2] = m[2, 0] = 5
    m[0, 1] = m[1, 0] = 1
    m[0, 3] = m[3, 0] = 9
    return sparse.csr_matrix(m)


def test_bought_together_sorted_and_skips_same_product_colours():
    paths = [Path(n) for n in NAMES]
    result = bought_together(paths[0], paths, matrix(), META)
    assert [(p.name, c) for p, c in result] == [("jeans_b.jpg", 5), ("jeans_a.jpg", 1)]  # kırmızı gömlek yok
    assert bought_together(Path("yok.jpg"), paths, matrix(), META) == []


def test_load_copurchase_checks_shape(tmp_path):
    path = tmp_path / "co.npz"
    sparse.save_npz(path, matrix())
    assert load_copurchase([Path(n) for n in NAMES], path).shape == (4, 4)
    assert load_copurchase([Path(n) for n in NAMES[:3]], path) is None  # katalog değişmiş
    assert load_copurchase([], tmp_path / "yok.npz") is None


def test_copurchase_weight_changes_outfit_order():
    paths = [Path(n) for n in NAMES]
    feats = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [1.0, 0.0]])
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    catalog = OutfitCatalog(paths, feats, META, copurchase=matrix())
    plain = catalog.suggest(META["shirt.jpg"], feats[0], per_slot=2, exclude=paths[0])
    assert plain["alt"][0][0].name == "jeans_a.jpg"  # stil benzerliği
    bought = catalog.suggest(META["shirt.jpg"], feats[0], per_slot=2, weights=Weights(copurchase=8), exclude=paths[0])
    assert bought["alt"][0][0].name == "jeans_b.jpg"  # birlikte daha çok alınan
