"""Fashion Product Images üzerinde gerçek etiketlerle model değerlendirmesi.

    FASHION_DATASET=products python notebooks/evaluate_products.py

Görselden öneri: 5.000 ürünlük rastgele bir alt katalogda her yöntem için
1.000 sorgunun ilk 5 önerisinde aynı ürün türü / renk / cinsiyet / ana
kategori oranı (precision@5) ölçülür. Tüm yöntemler aynı alt katalogda
karşılaştırılır (VGG16 flatten 44 bin üründe bellekte ~4,4 GB tutar).
Metinden arama: etiketlerden doğruluğu bilinen Türkçe / İngilizce sorgularda
ilk 10 sonucun isabet oranı, tüm katalogda.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from fashion_rec.catalog import load_metadata  # noqa: E402
from fashion_rec.color import extract_color_features  # noqa: E402
from fashion_rec.config import BATCH_SIZE, DATASET  # noqa: E402
from fashion_rec.features import load_model, preprocess_image, report_progress  # noqa: E402
from fashion_rec.text_search import build_text_index, load_clip_model, normalize  # noqa: E402

SUBSET, QUERIES, TOP_K = 5000, 1000, 5
FIELDS = ["articleType", "baseColour", "gender", "masterCategory"]
CACHE = Path(__file__).resolve().parent / "products_eval_cache.npz"

# (sorgu, beklenen etiketler: alan -> kabul edilen değerler)
TEXT_QUERIES = [
    ("kırmızı elbise", {"articleType": {"Dresses"}, "baseColour": {"Red"}}),
    ("red dress", {"articleType": {"Dresses"}, "baseColour": {"Red"}}),
    ("siyah ayakkabı", {"masterCategory": {"Footwear"}, "baseColour": {"Black"}}),
    ("erkek gömleği", {"articleType": {"Shirts"}, "gender": {"Men"}}),
    ("mavi kot pantolon", {"articleType": {"Jeans"}, "baseColour": {"Blue", "Navy Blue"}}),
    ("kol saati", {"articleType": {"Watches"}}),
    ("kadın çantası", {"articleType": {"Handbags"}, "gender": {"Women"}}),
    ("güneş gözlüğü", {"articleType": {"Sunglasses"}}),
    ("beyaz spor ayakkabı", {"articleType": {"Sports Shoes", "Casual Shoes"}, "baseColour": {"White"}}),
    ("kız çocuk elbisesi", {"articleType": {"Dresses"}, "gender": {"Girls"}}),
    ("erkek çocuk tişört", {"articleType": {"Tshirts"}, "gender": {"Boys"}}),
    ("sari", {"articleType": {"Sarees"}}),
    ("kemer", {"articleType": {"Belts"}}),
    ("pembe tişört", {"articleType": {"Tshirts"}, "baseColour": {"Pink"}}),
    ("yeşil kurta", {"articleType": {"Kurtas"}, "baseColour": {"Green"}}),
    ("cüzdan", {"articleType": {"Wallets"}}),
    ("topuklu ayakkabı", {"articleType": {"Heels"}}),
    ("şort", {"articleType": {"Shorts"}}),
    ("parfüm", {"articleType": {"Perfume and Body Mist", "Fragrance Gift Set", "Deodorant"}}),
    ("küpe", {"articleType": {"Earrings"}}),
]


def vgg_raw(paths):
    model = load_model()
    out = []
    for start in range(0, len(paths), BATCH_SIZE):
        report_progress(start + BATCH_SIZE, len(paths), "VGG16")
        batch = np.stack([preprocess_image(p) for p in paths[start:start + BATCH_SIZE]])
        out.append(model.predict(batch, verbose=0).astype(np.float16))
    return np.concatenate(out)


def zscore_rows(sim):
    return (sim - sim.mean(axis=1, keepdims=True)) / sim.std(axis=1, keepdims=True)


def precision_at_k(sim, query_idx, labels):
    """sim: (sorgu, katalog). Her alan için ilk K komşunun etiket eşleşme oranı."""
    sim = sim.copy()
    sim[np.arange(len(query_idx)), query_idx] = -np.inf  # sorgunun kendisi hariç
    top = np.argpartition(-sim, TOP_K, axis=1)[:, :TOP_K]
    return {f: float(np.mean(labels[f][top] == labels[f][query_idx][:, None])) for f in FIELDS}


def main():
    assert DATASET == "products", "FASHION_DATASET=products ile çalıştırın"
    clip = load_clip_model()
    paths, clip_all = build_text_index(model=clip)
    metadata = load_metadata()
    labels_all = {f: np.array([metadata.get(p.name, {}).get(f, "") for p in paths]) for f in FIELDS}
    print(f"{len(paths)} ürün\n")

    # --- metinden arama (tüm katalog) ---
    text = normalize(clip.encode_text([q for q, _ in TEXT_QUERIES]))
    print(f"Metinden arama (CLIP, tüm katalog), ilk 10 isabet oranı:")
    hits = []
    for (query, expected), t in zip(TEXT_QUERIES, text):
        top = np.argsort(-(clip_all @ t))[:10]
        ok = [all(labels_all[f][i] in vals for f, vals in expected.items()) for i in top]
        hits.append(np.mean(ok))
        print(f"  {np.mean(ok):.1f}  {query}")
    print(f"  ortalama: {np.mean(hits):.2f}\n")

    # --- görselden öneri (5.000 ürünlük alt katalog) ---
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(paths), size=SUBSET, replace=False))
    sub_paths = [paths[i] for i in sub]
    labels = {f: v[sub] for f, v in labels_all.items()}
    queries = np.sort(rng.choice(SUBSET, size=QUERIES, replace=False))

    if CACHE.exists() and list(np.load(CACHE)["names"]) == [p.name for p in sub_paths]:
        cache = np.load(CACHE)
        raw, colors = cache["raw"], cache["colors"]
    else:
        start = time.perf_counter()
        raw = vgg_raw(sub_paths)
        print(f"VGG16: {time.perf_counter() - start:.0f} sn")
        colors = extract_color_features(sub_paths)
        np.savez(CACHE, raw=raw, colors=colors, names=np.array([p.name for p in sub_paths]))
    raw = raw.astype(np.float32)

    variants = {
        "vgg16 flatten": normalize(raw.reshape(len(raw), -1)),
        "vgg16 avg": normalize(raw.mean(axis=(1, 2))),
        "vgg16 max": normalize(raw.max(axis=(1, 2))),
        "clip görsel": clip_all[sub],
    }
    sims = {m: f[queries] @ f.T for m, f in variants.items()}
    color_sim = zscore_rows(colors[queries] @ colors.T)
    for base in ("vgg16 flatten", "clip görsel"):
        for c in (0.3,):
            sims[f"{base} + renk {c}"] = (1 - c) * zscore_rows(sims[base]) + c * color_sim

    print(f"Görselden öneri: {SUBSET} ürünlük alt katalog, {QUERIES} sorgu, precision@{TOP_K}")
    print(f"{'yöntem':<24}" + "".join(f"{f:>16}" for f in FIELDS))
    for method, sim in sims.items():
        p = precision_at_k(sim, queries, labels)
        print(f"{method:<24}" + "".join(f"{p[f]:>16.3f}" for f in FIELDS))


if __name__ == "__main__":
    main()
