"""Renk duyarlı benzerlik: VGG16 + renk histogramı karşılaştırması.

    python notebooks/compare_color.py

Renk vektörü HSV histogramıdır (karekökü alınıp L2 normalize; cosine =
Hellinger benzerliğine yakın). Arka planın etkisini azaltmak için tüm
görsel ile orta bölge karşılaştırılır. VGG16 ile birleştirme z-skoru +
ağırlıkla yapılır: (1 - c) · z(vgg) + c · z(renk).
Ölçütler compare_pooling.py ile aynıdır (renk@5, etnik@5, eş sırası).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from compare_pooling import color_score, ethnic_score, pair_rank, save_showcase  # noqa: E402
from fashion_rec.features import build_index  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "color_comparison"
BINS = (12, 3, 3)  # ton, doygunluk, parlaklık
FULL, CENTER = (0.0, 0.0, 1.0, 1.0), (0.25, 0.15, 0.75, 0.85)  # (sol, üst, sağ, alt) oranları
VARIANTS = {  # ad: (kırpma, arka plan eşiği)
    "orta": (CENTER, None),
    "tüm+maske40": (FULL, 40),
    "tüm+maske60": (FULL, 60),
    "orta+maske40": (CENTER, 40),
}


def background_mask(img, threshold):
    """Kenar şeridinin medyan rengine yakın pikselleri arka plan sayar (True = kıyafet)."""
    rgb = np.asarray(img.resize((64, 64)), dtype=np.float32)
    border = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                             rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    return np.linalg.norm(rgb - bg, axis=2) > threshold


def histogram(path, crop, mask_threshold=None):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    l, t, r, b = crop
    img = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    hsv = np.asarray(img.resize((64, 64)).convert("HSV"), dtype=np.float32) / 256
    pixels = hsv.reshape(-1, 3)
    if mask_threshold is not None:
        keep = background_mask(img, mask_threshold).ravel()
        if keep.mean() > 0.05:  # neredeyse her şey arka plan sayıldıysa maskeyi kullanma
            pixels = pixels[keep]
    hist, _ = np.histogramdd(pixels, bins=BINS, range=[(0, 1)] * 3)
    v = np.sqrt(hist.ravel())
    return v / np.linalg.norm(v)


def zscore_rows(sim):
    """Her sorgu satırını kendi içinde z-skoruna çevirir (köşegen hariç)."""
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    return (sim - np.nanmean(off, axis=1, keepdims=True)) / np.nanstd(off, axis=1, keepdims=True)


def rank_from_sim(sim):
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)


def main():
    paths, vgg = build_index()
    names = [p.name for p in paths]
    vgg_z = zscore_rows(vgg @ vgg.T)

    print(f"{'yöntem':<16} {'renk@5':>7} {'etnik@5':>8} {'eş sırası':>10}  eş sıraları")

    def show(label, sim):
        ranks = rank_from_sim(sim)
        col, _ = color_score(names, ranks)
        eth, _ = ethnic_score(names, ranks)
        pr, positions = pair_rank(names, ranks)
        print(f"{label:<16} {col:>7.2f} {eth:>8.2f} {pr:>10.1f}  {positions}")
        return ranks

    showcase = {"vgg16": show("vgg16", vgg @ vgg.T)}
    for crop_name, (crop, threshold) in VARIANTS.items():
        hist = np.stack([histogram(p, crop, threshold) for p in paths])
        color_z = zscore_rows(hist @ hist.T)
        print()
        show(f"renk ({crop_name})", hist @ hist.T)
        for c in (0.3, 0.4, 0.5):
            ranks = show(f"  + vgg c={c}", (1 - c) * vgg_z + c * color_z)
            if crop_name in ("orta", "tüm+maske40") and c == 0.5:
                showcase[f"{crop_name} {c}"] = ranks

    save_showcase(paths, names, showcase, OUT_DIR)
    print(f"\nKarşılaştırma görselleri: {OUT_DIR}")


if __name__ == "__main__":
    main()
