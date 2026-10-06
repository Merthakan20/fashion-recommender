"""Çeşitlilik (MMR) ağırlığının öneri kalitesine etkisi.

    python notebooks/compare_diversity.py

1.000 sorgu, öneriler sorgunun bölümüyle sınırlı (uygulamanın varsayılanı).
İlgi: ilk 5 önerinin aynı ürün türü / renk oranı. Çeşitlilik: ilk 5
önerinin kendi aralarındaki ortalama görsel benzerlik (düşük = çeşitli) ve
neredeyse kopya (sorguya ya da başka bir öneriye benzerliği > 0,97) öneri
içeren sorgu oranı.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fashion_rec.backbone import build_image_index  # noqa: E402
from fashion_rec.catalog import load_metadata, segments  # noqa: E402
from fashion_rec.diversity import mmr  # noqa: E402

DIVERSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
QUERIES, TOP_N, NEAR_DUPLICATE = 1000, 5, 0.97


def main():
    paths, feats = build_image_index()
    metadata = load_metadata()
    seg = np.array([s or "" for s in segments(paths, metadata)])
    labels = {f: np.array([metadata[p.name][f] for p in paths]) for f in ("articleType", "baseColour")}
    rng = np.random.default_rng(0)
    queries = rng.choice(len(paths), QUERIES, replace=False)

    # her sorgu için bölüm içi ilk 100 aday (MMR havuzu)
    candidates = []
    for q in queries:
        members = np.flatnonzero(seg == seg[q])
        sims = feats[members] @ feats[q]
        sims[members == q] = -np.inf
        order = members[np.argsort(-sims)[:100]]
        candidates.append((q, [(paths[i], float(feats[i] @ feats[q])) for i in order]))

    index = {p: i for i, p in enumerate(paths)}
    print(f"{QUERIES} sorgu, bölüm içi, ilk {TOP_N} öneri\n")
    print(f"{'çeşitlilik':<12}{'tür':>7}{'renk':>7}{'öneriler arası benzerlik':>27}{'kopya içeren sorgu':>21}")
    for d in DIVERSITIES:
        type_hits, colour_hits, ils, dup = [], [], [], 0
        for q, results in candidates:
            top = [index[p] for p, _ in mmr(results, paths, feats, TOP_N, diversity=d, query_vector=feats[q])]
            type_hits.append(np.mean(labels["articleType"][top] == labels["articleType"][q]))
            colour_hits.append(np.mean(labels["baseColour"][top] == labels["baseColour"][q]))
            sub = feats[top] @ feats[top].T
            ils.append(sub[np.triu_indices(len(top), 1)].mean())
            dup += bool((feats[top] @ feats[q] > NEAR_DUPLICATE).any() or (sub[np.triu_indices(len(top), 1)] > NEAR_DUPLICATE).any())
        print(f"{d:<12}{np.mean(type_hits):>7.3f}{np.mean(colour_hits):>7.3f}{np.mean(ils):>27.3f}{dup / QUERIES:>21.0%}")


if __name__ == "__main__":
    main()
