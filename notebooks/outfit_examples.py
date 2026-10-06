"""Kombin önerisinin vekil ölçütleri ve görsel örnekleri.

    python notebooks/outfit_examples.py

Doğru cevap ("birlikte giyilir" etiketi) olmadığından iki kontrol yapılır:
1. Vekil ölçütler (500 sorgu): kullanım birebir / uyumlu oranı, renk uyumu
   (nötr renk içeren veya canlı renk çatışmasız çift oranı). Sezgisel bu
   kuralları kullandığı için bu ölçütler yalnızca kuralların işlediğini
   gösterir, kombinin iyi olduğunu kanıtlamaz. Karşılaştırma: yuvadan rastgele
   (aynı cinsiyet) ve yalnız CLIP benzerliği.
2. Görsel örnekler: notebooks/outfit_examples/ altına sorgu + yuva başına en
   iyi parça; asıl değerlendirme gözle yapılır.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fashion_rec.backbone import build_image_index  # noqa: E402
from fashion_rec.catalog import load_metadata  # noqa: E402
from fashion_rec.config import ROLE_CACHE_PATH  # noqa: E402
from fashion_rec.outfit import (USAGE_COMPATIBLE, OutfitCatalog, Weights, clip_roles, colour_harmony,  # noqa: E402
                                outfit_colour, role_of)

OUT_DIR = Path(__file__).resolve().parent / "outfit_examples"
EXAMPLES = ["Turtle Check Men Navy Blue Shirt", "Peter England Men Party Blue Jeans", "Titan Women Silver Watch",
            "Manchester United Men Solid Black Track Pants"]


def main():
    paths, feats = build_image_index()
    metadata = load_metadata()
    catalog = OutfitCatalog(paths, feats, metadata, visual_roles=clip_roles(paths, feats, ROLE_CACHE_PATH))
    rng = np.random.default_rng(0)
    eligible = np.flatnonzero(catalog.role != "")
    queries = rng.choice(eligible, 500, replace=False)

    methods = {
        "rastgele (aynı cinsiyet)": None,
        "yalnız CLIP": Weights(usage=0, colour=0, season=0),
        "sezgisel (varsayılan)": Weights(),
    }
    print(f"{len(queries)} sorgu, yuva başına en iyi parça\n")
    print(f"{'yöntem':<26}{'kullanım aynı':>15}{'kullanım uyumlu':>17}{'renk uyumlu':>13}")
    for label, weights in methods.items():
        same = ok = colour = n = 0
        for q in queries:
            row = metadata[paths[q].name]
            if weights is None:  # yuvadaki uygun adaylardan rastgele: CLIP puanı yerine gürültü
                outfit = catalog.suggest(row, rng.normal(size=feats.shape[1]), per_slot=1,
                                         weights=Weights(usage=0, colour=0, season=0), exclude=paths[q])
            else:
                outfit = catalog.suggest(row, feats[q], per_slot=1, weights=weights, exclude=paths[q])
            for items in outfit.values():
                cand = metadata[items[0][0].name]
                same += cand["usage"] == row["usage"]
                ok += cand["usage"] in USAGE_COMPATIBLE.get(row["usage"], {row["usage"]})
                colour += colour_harmony(outfit_colour(row), outfit_colour(cand)) > 0  # kot = nötr
                n += 1
        print(f"{label:<26}{same / n:>15.2f}{ok / n:>17.2f}{colour / n:>13.2f}")

    OUT_DIR.mkdir(exist_ok=True)
    by_name = {metadata[p.name]["name"]: i for i, p in enumerate(paths)}
    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(10, 3.2 * len(EXAMPLES)))
    for r, name in enumerate(EXAMPLES):
        q = by_name[name]
        row = metadata[paths[q].name]
        outfit = catalog.suggest(row, feats[q], per_slot=1, exclude=paths[q])
        cells = [("seçilen", paths[q])] + [(slot, items[0][0]) for slot, items in outfit.items()]
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(cells):
                slot, path = cells[c]
                ax.imshow(Image.open(path).convert("RGB"))
                ax.set_title(f"{slot}\n{metadata[path.name]['name'][:30]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "examples.png", dpi=80)
    print(f"\nGörsel örnekler: {OUT_DIR / 'examples.png'} (roller: {[role_of(metadata[paths[by_name[n]].name]) for n in EXAMPLES]})")


if __name__ == "__main__":
    main()
