from pathlib import Path

import numpy as np
import pytest

from fashion_rec.outfit import (OutfitCatalog, Weights, colour_harmony, gender_ok, infer_row, name_conflicts,
                                role_of)


def item(name, sub, gender="Men", usage="Casual", colour="Black", season="Summer", article=""):
    return {"name": name, "subCategory": sub, "gender": gender, "usage": usage, "baseColour": colour, "season": season,
            "articleType": article}


ROWS = {
    "shirt.jpg": item("Men Navy Shirt", "Topwear", colour="Navy Blue"),
    "jeans.jpg": item("Men Blue Jeans", "Bottomwear", colour="Blue", article="Jeans"),
    "formal_trousers.jpg": item("Men Formal Trousers", "Bottomwear", usage="Formal", colour="Grey"),
    "women_skirt.jpg": item("Women Red Skirt", "Bottomwear", gender="Women", colour="Red"),
    "red_shorts.jpg": item("Men Red Shorts", "Bottomwear", colour="Red"),
    "sneakers.jpg": item("Men White Sneakers", "Shoes", colour="White"),
    "watch.jpg": item("Men Steel Watch", "Watches", colour="Steel"),
    "unisex_cap.jpg": item("Unisex Black Cap", "Headwear", gender="Unisex"),
    "fake_shoe.jpg": item("Men Solid White Shirts", "Flip Flops", colour="White"),  # etiket hatası
    "perfume.jpg": item("Men Perfume", "Fragrance"),
}


@pytest.fixture
def catalog():
    paths = [Path(n) for n in ROWS]
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(len(paths), 8))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return OutfitCatalog(paths, feats, ROWS), paths, feats


def names(items):
    return [p.name for p, _ in items]


def test_roles_and_rules():
    assert role_of(ROWS["shirt.jpg"]) == "üst"
    assert role_of(ROWS["perfume.jpg"]) is None
    assert colour_harmony("Navy Blue", "Red") == 1.0   # nötr + canlı
    assert colour_harmony("Red", "Red") == -0.5        # tekdüze canlı
    assert colour_harmony("Red", "Green") == -1.0      # çatışan canlı
    assert colour_harmony("Red", "Denim") == 1.0       # kot nötr
    assert gender_ok("Men", "Unisex") and not gender_ok("Men", "Women")
    assert name_conflicts(ROWS["fake_shoe.jpg"]) and not name_conflicts(ROWS["jeans.jpg"])


def test_outfit_slots_gender_usage_and_label_errors(catalog):
    oc, paths, feats = catalog
    outfit = oc.suggest(ROWS["shirt.jpg"], feats[0], per_slot=5, exclude=paths[0])
    assert list(outfit) == ["alt", "ayakkabı", "aksesuar"]
    assert "women_skirt.jpg" not in names(outfit["alt"])          # farklı cinsiyet
    assert "formal_trousers.jpg" not in names(outfit["alt"])      # resmi, günlük gömlekle uyumsuz
    assert names(outfit["ayakkabı"]) == ["sneakers.jpg"]          # adı "Shirts" olan terlik elendi
    assert set(names(outfit["aksesuar"])) == {"watch.jpg", "unisex_cap.jpg"}


def test_colour_harmony_affects_order(catalog):
    oc, paths, feats = catalog
    row = item("Men Red Tshirt", "Topwear", colour="Red")
    # stil sinyali olmadan (sıfır vektör) yalnız renk: kırmızı üste kot (denim = nötr), kırmızı şorttan önce
    plain = oc.suggest(row, np.zeros(feats.shape[1]), per_slot=2, weights=Weights(usage=0, season=0))
    assert names(plain["alt"])[0] == "jeans.jpg"


def test_visual_roles_exclude_mismatched_images(catalog):
    _, paths, feats = catalog
    visual = np.array(["üst", "üst", "alt", "alt", "alt", "ayakkabı", "aksesuar", "aksesuar", "ayakkabı", ""])
    oc = OutfitCatalog(paths, feats, ROWS, visual_roles=visual)  # kot pantolonun görseli "üst" gibi görünüyor
    assert "jeans.jpg" not in names(oc.suggest(ROWS["shirt.jpg"], feats[0], per_slot=5)["alt"])


def test_no_outfit_for_unsupported_role(catalog):
    oc, _, feats = catalog
    assert oc.suggest(ROWS["perfume.jpg"], feats[0]) == {}


def test_infer_row_majority():
    neighbours = [(Path("shirt.jpg"), 0.9), (Path("jeans.jpg"), 0.8), (Path("sneakers.jpg"), 0.7),
                  (Path("watch.jpg"), 0.6)]
    row = infer_row(neighbours, {**ROWS, "jeans.jpg": item("x", "Topwear")})
    assert row["subCategory"] == "Topwear" and row["gender"] == "Men"
