"""Türkçe metinden arama için çok dilli CLIP modellerinin karşılaştırması.

    python notebooks/compare_multilingual.py

"multilingual-st" modeli için ayrıca `pip install sentence-transformers` gerekir
(projenin bağımlılıklarında yok).

Her sorgunun Türkçe ve İngilizce hali vardır; doğru cevap, dosya adında
geçmesi gereken kelime gruplarıyla tanımlanır (her gruptan en az biri).
Ölçütler: hit@1, hit@3 ve ilk doğru sonucun ters sırası (MRR).
"""

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fashion_rec.data import list_images  # noqa: E402
from fashion_rec.text_search import ClipModel, normalize  # noqa: E402

# (Türkçe, İngilizce, dosya adında aranacak kelime grupları)
QUERIES = [
    ("kırmızı payetli elbise", "red sequined dress", [("red",), ("sequin", "sparkl")]),
    ("sari", "saree", [("saree",)]),
    ("siyah dantel elbise", "black lace dress", [("black",), ("lace",)]),
    ("beyaz pantolon", "white trousers", [("white",), ("trousers",)]),
    ("mavi çiçekli elbise", "blue floral dress", [("blue",), ("floral",)]),
    ("kazayağı desenli elbise", "houndstooth dress", [("houndstooth",)]),
    ("bej takım elbise", "beige suit", [("beige",), ("suit",)]),
    ("pembe elbise", "pink dress", [("pink",)]),
    ("anarkali takım", "anarkali suit", [("anarkali",)]),
    ("omzu açık siyah elbise", "black off-shoulder dress", [("black",), ("off-shoulder", "off-the-shoulder")]),
    ("sarı elbise", "yellow dress", [("yellow",)]),
    ("siyah blazer ceket", "black blazer", [("blazer",)]),
    ("tüylü kazak", "fluffy sweater", [("sweater",)]),
    ("yırtmaçlı uzun elbise", "long dress with a high slit", [("slit",)]),
    ("kareli elbise", "checkered dress", [("checkered", "gingham", "plaid")]),
    ("gümüş rengi elbise", "silver dress", [("silver",)]),
    ("mor payetli elbise", "purple sequined dress", [("purple",)]),
    ("yeşil elbise", "green dress", [("green",)]),
    ("krem rengi palto", "cream coat", [("coat",)]),
    ("kırmızı kurta", "red kurta", [("kurta", "salwar")]),
]


def matches(name, groups):
    name = name.lower()
    return all(any(word in name for word in group) for group in groups)


def score(text_feats, image_feats, names, lang):
    hit1 = hit3 = rr = 0.0
    misses = []
    for (tr, en, groups), t in zip(QUERIES, text_feats):
        order = np.argsort(-(image_feats @ t))
        ok = [matches(names[i], groups) for i in order]
        first = ok.index(True) + 1
        hit1 += first == 1
        hit3 += first <= 3
        rr += 1 / first
        if first > 3:
            misses.append(f"{tr if lang == 'tr' else en!r} → {names[order[0]][:45]} (ilk doğru: {first}.)")
    n = len(QUERIES)
    return hit1 / n, hit3 / n, rr / n, misses


class MultilingualST:
    """OpenAI CLIP ViT-B/32 görsel kodlayıcısı + çok dilli metin kodlayıcısı."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
        self.clip = ClipModel()

    def encode_images(self, paths):
        return self.clip.encode_images(paths)

    def encode_text(self, texts):
        return self.text_model.encode(texts)


class OpenClipModel:
    def __init__(self, arch, pretrained):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    def encode_images(self, paths):
        with torch.no_grad():
            return self.model.encode_image(
                torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths])).numpy()

    def encode_text(self, texts):
        with torch.no_grad():
            return self.model.encode_text(self.tokenizer(texts)).numpy()


def main():
    paths = list_images()
    names = [p.name for p in paths]
    models = {
        "clip-b32 (mevcut)": ClipModel,
        "multilingual-st": MultilingualST,
        "xlmr-b32-laion": lambda: OpenClipModel("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    }

    print(f"{len(QUERIES)} sorgu, {len(paths)} görsel\n")
    print(f"{'model':<20} {'dil':<3} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    all_misses = {}
    for label, factory in models.items():
        model = factory()
        image_feats = normalize(model.encode_images(paths))
        for lang, idx in (("tr", 0), ("en", 1)):
            text_feats = normalize(model.encode_text([q[idx] for q in QUERIES]))
            h1, h3, mrr, misses = score(text_feats, image_feats, names, lang)
            all_misses[(label, lang)] = misses
            print(f"{label:<20} {lang:<3} {h1:>6.2f} {h3:>6.2f} {mrr:>6.2f}")

    print("\nİlk 3'te doğru sonuç olmayan sorgular:")
    for (label, lang), misses in all_misses.items():
        print(f"  {label} [{lang}]: {len(misses)}")
        for m in misses:
            print(f"      {m}")


if __name__ == "__main__":
    main()
