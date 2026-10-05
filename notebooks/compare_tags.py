"""CLIP ile sıfır örnekli (zero-shot) renk / tür etiketlemenin doğruluğu.

    python notebooks/compare_tags.py

Her görsel, etiket metinlerinin CLIP vektörlerine en yakın olan etiketi alır.
Doğruluk, dosya adından kesin çıkarılabilen etiketlerle ölçülür (adında tek
bir renk kelimesi geçen görseller; türü adından belli olan görseller).
Birkaç istem (prompt) kalıbı karşılaştırılır.
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fashion_rec.text_search import build_text_index, load_clip_model, normalize  # noqa: E402

# etiket: (CLIP'e verilen İngilizce tarif, dosya adında bu etiketi gösteren kelimeler)
COLORS = {
    "siyah": ("black", ("black",)),
    "beyaz": ("white", ("white", "cream")),
    "kırmızı": ("red", ("red", "burgundy")),
    "mavi": ("blue", ("blue", "teal", "turquoise")),
    "yeşil": ("green", ("green", "olive")),
    "pembe": ("pink", ("pink", "peach")),
    "sarı": ("yellow", ("yellow", "mustard")),
    "mor": ("purple", ("purple", "lavender")),
    "bej / kahverengi": ("beige or brown", ("beige", "brown")),
    "gri / gümüş": ("grey or silver", ("silver", "grey", "gray")),
}
TYPES = {
    "elbise": ("a dress", ("dress", "gown")),
    "etnik": ("traditional Indian clothing such as a saree, kurta or anarkali suit",
              ("saree", "anarkali", "kurta", "salwar")),
    "pantolonlu": ("an outfit with trousers", ("trousers", "jumpsuit")),
    "etekli": ("a top with a skirt", ("skirt",)),
    "ceket / üst": ("a jacket, coat, blazer or sweater", ("jacket", "coat", "blazer", "sweater")),
}
# türü belirlerken öncelik: etnik > pantolonlu > etekli > ceket > elbise (ör. "anarkali ... dress")
TYPE_PRIORITY = ["etnik", "pantolonlu", "etekli", "ceket / üst", "elbise"]
TEMPLATES = ["{}", "a photo of {}", "a photo of a woman wearing {}"]
COLOR_TEMPLATES = ["{}", "a {} outfit", "a photo of a woman wearing {} clothes"]


def words(name):
    return set(re.findall(r"[a-z]+", name.lower()))


def true_color(name):
    found = {label for label, (_, keys) in COLORS.items() if words(name) & set(keys)}
    return found.pop() if len(found) == 1 else None


def true_type(name):
    w = words(name)
    for label in TYPE_PRIORITY:
        if w & set(TYPES[label][1]):
            return label
    return None


def classify(model, feats, labels, prompts):
    text = normalize(model.encode_text(prompts))
    return [labels[i] for i in np.argmax(feats @ text.T, axis=1)]


def accuracy(pred, truth):
    pairs = [(p, t) for p, t in zip(pred, truth) if t is not None]
    errors = Counter((t, p) for p, t in pairs if p != t)
    return np.mean([p == t for p, t in pairs]), len(pairs), errors


def main():
    model = load_clip_model()
    paths, feats = build_text_index(model=model)
    names = [p.name for p in paths]

    for group, table, templates, truth_fn in (("Renk", COLORS, COLOR_TEMPLATES, true_color),
                                              ("Tür", TYPES, TEMPLATES, true_type)):
        labels = list(table)
        truth = [truth_fn(n) for n in names]
        print(f"\n{group}:")
        best = None
        for template in templates:
            pred = classify(model, feats, labels, [template.format(table[label][0]) for label in labels])
            acc, n, errors = accuracy(pred, truth)
            print(f"  {template!r:<42} doğruluk {acc:.2f} ({n} görsel)")
            if best is None or acc > best[0]:
                best = (acc, template, pred, errors)
        acc, template, pred, errors = best
        print(f"  en iyi: {template!r}; dağılım: {dict(Counter(pred))}")
        print(f"  en sık hatalar (doğru → tahmin): {errors.most_common(5)}")


if __name__ == "__main__":
    main()
