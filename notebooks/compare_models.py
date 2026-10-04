"""VGG16 (flatten) ile CLIP görsel vektörlerinin karşılaştırması.

    python notebooks/compare_models.py

Ölçütler ve yan yana görseller compare_pooling.py ile aynıdır. CLIP için
ayrıca metinden arama denenir.
"""

import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from compare_pooling import (TOP_N, color_score, ethnic_score, normalize, pair_rank,
                             rankings, raw_features, save_showcase)
from fashion_rec.data import list_images

OUT_DIR = Path(__file__).resolve().parent / "model_comparison"
CLIP_MODELS = {"clip-b32": "ViT-B-32-quickgelu", "clip-b16": "ViT-B-16-quickgelu"}
TEXT_QUERIES = ["a red sequined dress", "a saree", "a black lace dress",
                "white trousers", "a blue floral dress", "a houndstooth dress"]


def clip_features(arch, paths):
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer(arch)
    with torch.no_grad():
        images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
        feats = model.encode_image(images).numpy()
    return normalize(feats), model, tokenizer


def text_search(model, tokenizer, features, names):
    with torch.no_grad():
        text = normalize(model.encode_text(tokenizer(TEXT_QUERIES)).numpy())
    for query, sims in zip(TEXT_QUERIES, text @ features.T):
        top = np.argsort(-sims)[:3]
        print(f'  "{query}"')
        for i in top:
            print(f"      {sims[i]:.3f}  {names[i]}")


def main():
    paths = list_images()
    names = [p.name for p in paths]

    variants, timings, clip = {}, {}, {}
    start = time.perf_counter()
    raw = raw_features(paths)
    variants["vgg16"] = normalize(raw.reshape(len(raw), -1))
    timings["vgg16"] = time.perf_counter() - start
    for method, arch in CLIP_MODELS.items():
        start = time.perf_counter()
        variants[method], *clip[method] = clip_features(arch, paths)
        timings[method] = time.perf_counter() - start

    all_ranks = {m: rankings(f) for m, f in variants.items()}

    print(f"\n{'yöntem':<9} {'boyut':>6} {'süre':>6} {'etnik@5':>8} {'renk@5':>7} {'eş sırası':>10}  eş sıraları")
    for method, feats in variants.items():
        ranks = all_ranks[method]
        eth, _ = ethnic_score(names, ranks)
        col, _ = color_score(names, ranks)
        pr, positions = pair_rank(names, ranks)
        print(f"{method:<9} {feats.shape[1]:>6} {timings[method]:>5.1f}s {eth:>8.2f} {col:>7.2f} {pr:>10.1f}  {positions}")

    for method, (model, tokenizer) in clip.items():
        print(f"\nMetinden arama ({method}):")
        text_search(model, tokenizer, variants[method], names)

    save_showcase(paths, names, all_ranks, OUT_DIR)
    print(f"\nKarşılaştırma görselleri: {OUT_DIR}  (ilk {TOP_N} öneri)")


if __name__ == "__main__":
    main()
