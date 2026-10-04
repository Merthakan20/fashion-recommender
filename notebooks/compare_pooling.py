"""VGG16 özellik vektörü için flatten / avg / max havuzlama karşılaştırması.

    python notebooks/compare_pooling.py

Veri setinde etiket olmadığı için dosya adlarından üç kaba ölçüt çıkarılır:
- etnik: etnik kıyafet sorgularında ilk 5 önerinin etnik olma oranı
- renk: adında renk geçen sorgularda ilk 5 önerinin aynı rengi taşıma oranı
- eş: bilinen eş görsel çiftlerinde eşin ortalama sırası (1 = en iyi)
Ayrıca birkaç sorgu için önerileri yan yana gösteren görseller kaydedilir.
"""

import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fashion_rec.config import BATCH_SIZE  # noqa: E402
from fashion_rec.data import list_images  # noqa: E402
from fashion_rec.features import load_model, preprocess_image  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "pooling_comparison"
TOP_N = 5

ETHNIC = ("saree", "anarkali", "kurta", "salwar")
COLORS = ("black", "white", "red", "blue", "green", "pink", "yellow", "beige",
          "burgundy", "purple", "silver", "cream", "orange", "teal", "peach", "brown")
PAIRS = [
    ("sparkling, fitted dress.jpeg", "sparkling, fitted dress 2.jpeg"),
    ("tight-fitting, off-the-shoulder white dress.webp", "tight-fitting, off-the-shoulder white dress 2.jpg"),
    ("dress with a classic black and white houndstooth pattern.jpg",
     "fitted dress with a classic black and white houndstooth pattern.webp"),
    ("long, flowing dress with a full skirt.jpg", "Latest-Party-Wear-One-Piece-Gown-For-Women.webp"),
]
SHOWCASE = ["black floral saree.jpg", "red velvet sheath midi dress.jpg",
            "sparkling black dress.jpg", "black top, white trousers.jpg"]


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def raw_features(paths):
    """VGG16'nın havuzlanmamış (N, 7, 7, 512) çıktısı."""
    model = load_model()
    out = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = np.stack([preprocess_image(p) for p in paths[start:start + BATCH_SIZE]])
        out.append(model.predict(batch, verbose=0))
    return np.concatenate(out)


def rankings(features):
    """Her görsel için kendisi hariç, azalan benzerlikle sıralı indeksler."""
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)


def words(name):
    return set(re.findall(r"[a-z]+", name.lower()))


def ethnic_score(names, ranks):
    queries = [i for i, n in enumerate(names) if words(n) & set(ETHNIC)]
    hits = [sum(bool(words(names[j]) & set(ETHNIC)) for j in ranks[i, :TOP_N]) for i in queries]
    return np.mean(hits) / TOP_N, len(queries)


def color_score(names, ranks):
    scores = []
    for i, n in enumerate(names):
        colors = words(n) & set(COLORS)
        if colors:
            scores.append(np.mean([bool(words(names[j]) & colors) for j in ranks[i, :TOP_N]]))
    return np.mean(scores), len(scores)


def pair_rank(names, ranks):
    index = {n: i for i, n in enumerate(names)}
    positions = []
    for a, b in PAIRS:
        for q, t in ((a, b), (b, a)):
            positions.append(int(np.where(ranks[index[q]] == index[t])[0][0]) + 1)
    return np.mean(positions), positions


def save_showcase(paths, names, all_ranks, out_dir=OUT_DIR):
    out_dir.mkdir(exist_ok=True)
    for query in SHOWCASE:
        qi = names.index(query)
        fig, axes = plt.subplots(len(all_ranks), TOP_N + 1, figsize=(2.2 * (TOP_N + 1), 2.6 * len(all_ranks)))
        for row, (method, ranks) in enumerate(all_ranks.items()):
            for col, idx in enumerate([qi, *ranks[qi, :TOP_N]]):
                ax = axes[row, col]
                ax.imshow(Image.open(paths[idx]).convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(names[idx][:24], fontsize=7)
                if col == 0:
                    ax.set_ylabel(method, fontsize=11, fontweight="bold")
        fig.suptitle(f"Sorgu: {query}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{Path(query).stem}.png", dpi=80)
        plt.close(fig)


def main():
    paths = list_images()
    names = [p.name for p in paths]

    start = time.perf_counter()
    raw = raw_features(paths)
    print(f"{len(paths)} görsel, ham çıktı {raw.shape[1:]}, {time.perf_counter() - start:.1f} sn\n")

    variants = {
        "flatten": normalize(raw.reshape(len(raw), -1)),
        "avg": normalize(raw.mean(axis=(1, 2))),
        "max": normalize(raw.max(axis=(1, 2))),
    }
    all_ranks = {m: rankings(f) for m, f in variants.items()}

    print(f"{'yöntem':<8} {'boyut':>6} {'etnik@5':>8} {'renk@5':>7} {'eş sırası':>10}  eş sıraları")
    for method, feats in variants.items():
        ranks = all_ranks[method]
        eth, n_eth = ethnic_score(names, ranks)
        col, n_col = color_score(names, ranks)
        pr, positions = pair_rank(names, ranks)
        print(f"{method:<8} {feats.shape[1]:>6} {eth:>8.2f} {col:>7.2f} {pr:>10.1f}  {positions}")
    print(f"\n(etnik sorgu: {n_eth}, renkli sorgu: {n_col}, eş çifti: {len(PAIRS)})")

    base = all_ranks["flatten"][:, :TOP_N]
    for method in ("avg", "max"):
        overlap = np.mean([len(set(base[i]) & set(all_ranks[method][i, :TOP_N])) for i in range(len(names))])
        print(f"flatten ile ortak ilk-{TOP_N} ({method}): {overlap:.2f} / {TOP_N}")

    save_showcase(paths, names, all_ranks)
    print(f"\nKarşılaştırma görselleri: {OUT_DIR}")


if __name__ == "__main__":
    main()
