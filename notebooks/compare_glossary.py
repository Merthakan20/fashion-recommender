"""Türkçe moda sözlüğünün metinden aramaya etkisi.

    FASHION_DATASET=products python notebooks/compare_glossary.py

Yöntemler: sözlüksüz (özgün sorgu), yalnız çeviri, özgün + çeviri vektör
ortalaması. Yöntem geliştirme setinde (evaluate_products.TEXT_QUERIES)
seçilir; ayrık set (turkish_queries.HELD_OUT_QUERIES) sözlük yazılırken
kullanılmadı ve yalnızca raporlanır. Ölçüt: tüm katalogda ilk 10 isabet.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from evaluate_products import TEXT_QUERIES  # noqa: E402
from fashion_rec.catalog import load_metadata  # noqa: E402
from fashion_rec.glossary import translate  # noqa: E402
from fashion_rec.text_search import build_text_index, load_clip_model, normalize  # noqa: E402
from turkish_queries import HELD_OUT_QUERIES  # noqa: E402

FIELDS = ["articleType", "baseColour", "gender", "masterCategory"]
DEV = [(q, e) for q, e in TEXT_QUERIES if q != "red dress"]


def hits(vectors, feats, labels, queries, k=10):
    out = []
    for v, (_, exp) in zip(vectors, queries):
        top = np.argsort(-(feats @ v))[:k]
        out.append(np.mean([all(labels[f][i] in vals for f, vals in exp.items()) for i in top]))
    return np.array(out)


def main():
    model = load_clip_model()
    paths, feats = build_text_index(model=model)
    metadata = load_metadata()
    labels = {f: np.array([metadata[p.name][f] for p in paths]) for f in FIELDS}

    results = {}
    for set_name, queries in (("geliştirme", DEV), ("ayrık", HELD_OUT_QUERIES)):
        original = normalize(model.encode_text([q for q, _ in queries]))
        translated_text = [translate(q) or q for q, _ in queries]
        translated = normalize(model.encode_text(translated_text))
        methods = {"sözlüksüz": original, "yalnız çeviri": translated,
                   "ortalama": normalize(original + translated)}
        results[set_name] = {m: hits(v, feats, labels, queries) for m, v in methods.items()}
        print(f"\n{set_name} seti ({len(queries)} sorgu), ilk 10 isabet ortalaması:")
        for m, h in results[set_name].items():
            print(f"  {m:<14} {h.mean():.3f}")
        print("  sorgu bazında (sözlüksüz → çeviri → ortalama):")
        for i, (q, _) in enumerate(queries):
            row = [results[set_name][m][i] for m in methods]
            mark = "  ↑" if row[2] > row[0] else ("  ↓" if row[2] < row[0] else "")
            print(f"    {q:<28} {row[0]:.1f} → {row[1]:.1f} → {row[2]:.1f}   ({translated_text[i]}){mark}")


if __name__ == "__main__":
    main()
