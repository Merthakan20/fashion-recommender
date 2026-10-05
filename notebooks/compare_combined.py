"""Görsel + metin birlikte arama ("bunun gibi ama kırmızı") için yöntem karşılaştırması.

    python notebooks/compare_combined.py

Her örnekte bir sorgu görseli ve kısa bir Türkçe değişiklik metni vardır.
Doğru cevap, hem görselin özelliğini hem değişikliği taşıyan görsellerdir
(dosya adındaki kelime gruplarıyla tanımlanır; sorgu görselinin kendisi hariç).

Yöntemler:
- vektör: normalize(görsel + w · metin) ile arama
- skor:   (1 - a) · z(görsel benzerliği) + a · z(metin benzerliği)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from compare_multilingual import matches  # noqa: E402
from fashion_rec.text_search import build_text_index, load_clip_model, normalize  # noqa: E402

SEQUIN = ("sequin", "sparkl")
# (sorgu görseli, değişiklik, doğru cevabın dosya adında geçmesi gereken kelime grupları)
CASES = [
    ("sparkling black dress.jpg", "kırmızı", [("red",), SEQUIN]),
    ("bright red, sequined dress with thin shoulder straps.jpg", "siyah", [("black",), SEQUIN]),
    ("black, sequined dress with thin shoulder straps.jpg", "kırmızı", [("red",), SEQUIN]),
    ("sparkling black dress.jpg", "mavi", [("blue",), SEQUIN]),
    ("pink, knee-length dress that sparkles.jpg", "mor", [("purple",), SEQUIN]),
    ("red velvet sheath midi dress.jpg", "siyah", [("black",), ("midi", "mid-calf", "slip")]),
    ("black floral saree.jpg", "renkli", [("saree",), ("colorful",)]),
    ("black top, white trousers.jpg", "bej takım", [("beige", "brown"), ("suit", "blazer")]),
    ("black double-breasted blazer with gold buttons, paired with a mid-length skirt.jpg", "bej",
     [("beige", "brown"), ("suit", "blazer")]),
    ("royal blue off-shoulder dress.jpg", "yeşil", [("green",), ("off-shoulder", "off-the-shoulder")]),
    ("royal blue off-shoulder dress.jpg", "siyah", [("black",), ("off-shoulder", "off-the-shoulder")]),
    ("red dress with a pattern of small white flowers.jpg", "mavi", [("blue",), ("floral", "flower")]),
    ("anarkali suit with lavender in color with intricate white patterns throughout the fabric.jpg", "sarı",
     [("yellow",), ("anarkali",)]),
    ("classic black slip dress with a midi length.jpg", "kırmızı", [("red",), ("midi", "sheath", "fitted")]),
]


def zscore(x):
    return (x - x.mean()) / x.std()


def evaluate(names, rank_fn):
    """rank_fn(sorgu indeksi, metin vektörü) -> skorlar; hit@1, hit@3, MRR döner."""
    hit1 = hit3 = rr = 0.0
    firsts = []
    for (query, _, groups), text in zip(CASES, TEXTS):
        qi = names.index(query)
        scores = rank_fn(qi, text)
        scores[qi] = -np.inf
        order = np.argsort(-scores)
        first = [matches(names[i], groups) for i in order].index(True) + 1
        firsts.append(first)
        hit1 += first == 1
        hit3 += first <= 3
        rr += 1 / first
    n = len(CASES)
    return hit1 / n, hit3 / n, rr / n, firsts


def main():
    global TEXTS
    model = load_clip_model()
    paths, feats = build_text_index(model=model)
    names = [p.name for p in paths]
    for query, _, groups in CASES:
        targets = [n for n in names if matches(n, groups) and n != query]
        assert targets, f"doğru cevap yok: {query}"
    TEXTS = normalize(model.encode_text([c[1] for c in CASES]))

    print(f"{len(CASES)} örnek, {len(paths)} görsel\n")
    print(f"{'yöntem':<14} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}  ilk doğru sıraları")

    def show(label, fn):
        h1, h3, mrr, firsts = evaluate(names, fn)
        print(f"{label:<14} {h1:>6.2f} {h3:>6.2f} {mrr:>6.2f}  {firsts}")
        return mrr

    show("yalnız görsel", lambda qi, t: feats @ feats[qi])
    show("yalnız metin", lambda qi, t: feats @ t)
    print()
    best = {}
    for w in (0.5, 1, 1.5, 2, 3, 4):
        best[("vektör", w)] = show(f"vektör w={w}", lambda qi, t, w=w: feats @ normalize((feats[qi] + w * t)[None])[0])
    print()
    for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        best[("skor", a)] = show(f"skor a={a}",
                                 lambda qi, t, a=a: (1 - a) * zscore(feats @ feats[qi]) + a * zscore(feats @ t))
    (method, param), mrr = max(best.items(), key=lambda kv: kv[1])
    print(f"\nEn iyi: {method} {param} (MRR {mrr:.2f})")


if __name__ == "__main__":
    main()
