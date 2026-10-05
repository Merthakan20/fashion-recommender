"""CLIP vektörleri üzerine doğrusal adaptör eğitir (görsel ↔ ürün adı eşleştirmesi).

    FASHION_DATASET=products python scripts/train_adapter.py

CLIP dondurulur; görsel ve metin vektörleri birer doğrusal katmandan
(birim matrisle başlar) geçirilir ve her ürünün görseli kendi adıyla
(`metadata.csv` → name) eşleşecek şekilde karşıtlıklı (InfoNCE) kayıpla
eğitilir. Aynı adı taşıyan ürünler birbirinin pozitifi sayılır. Katmanları
birim matrise yakın tutan ceza (λ), İngilizce adlarla eğitilirken CLIP'in
Türkçe anlama yeteneğinin bozulmasını önler.

Ürünlerin %20'si test için tamamen ayrılır; λ, eğitim verisinden ayrılan
doğrulama diliminde seçilir. Sonuçlar raporlandıktan sonra adaptör tüm
veriyle yeniden eğitilip `config.CLIP_ADAPTER_PATH` dosyasına yazılır ve
CLIP önbelleği adaptörlü vektörlerle güncellenir.
"""

import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate_products import TEXT_QUERIES  # noqa: E402
from fashion_rec.adapter import Adapter  # noqa: E402
from fashion_rec.catalog import load_metadata  # noqa: E402
from fashion_rec.config import CACHE_DIR, CLIP_ADAPTER_PATH, CLIP_CACHE_PATH, CLIP_MODEL, CLIP_PRETRAINED  # noqa: E402
from fashion_rec.data import list_images  # noqa: E402
from fashion_rec.features import cached_features  # noqa: E402
from fashion_rec.text_search import ClipModel, clip_key, normalize  # noqa: E402

RAW_KEY = f"{CLIP_MODEL}/{CLIP_PRETRAINED}"
RAW_IMAGE_CACHE = CACHE_DIR / "clip_features_raw.npz"
RAW_NAME_CACHE = CACHE_DIR / "name_text_raw.npz"
LAMBDAS = (0.01, 0.1, 1.0)
EPOCHS, BATCH, LR, SEED = 15, 1024, 5e-4, 0
FIELDS = ["articleType", "baseColour", "gender"]

_raw_model = None


def raw_model():
    global _raw_model
    if _raw_model is None:
        _raw_model = ClipModel(adapter=None)
    return _raw_model


def raw_image_features(paths):
    """Adaptörsüz CLIP görsel vektörleri; mevcut önbellek hamsa kopyalanır (44 bin görseli yeniden kodlamamak için)."""
    if not RAW_IMAGE_CACHE.exists() and CLIP_CACHE_PATH.exists() and str(np.load(CLIP_CACHE_PATH)["key"]) == RAW_KEY:
        shutil.copy(CLIP_CACHE_PATH, RAW_IMAGE_CACHE)
    return cached_features(paths, RAW_IMAGE_CACHE, lambda: normalize(raw_model().encode_images(paths)), key=RAW_KEY)


def raw_name_features(paths, names):
    def compute():
        out = []
        for start in range(0, len(names), 512):
            print(f"ürün adları: {start}/{len(names)}", file=sys.stderr, flush=True)
            out.append(raw_model().encode_text(names[start:start + 512]))
        return normalize(np.concatenate(out))
    return cached_features(paths, RAW_NAME_CACHE, compute, key=RAW_KEY + "/names")


def train(images, texts, name_ids, lam, epochs=EPOCHS):
    """Görsel/metin doğrusal katmanlarını eğitir; Adapter döndürür."""
    torch.manual_seed(SEED)
    dim = images.shape[1]
    w_i, w_t = (torch.nn.Parameter(torch.eye(dim)) for _ in range(2))
    b_i, b_t = (torch.nn.Parameter(torch.zeros(dim)) for _ in range(2))
    log_scale = torch.nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))
    opt = torch.optim.Adam([w_i, w_t, b_i, b_t, log_scale], lr=LR)
    x_img, x_txt, ids = torch.from_numpy(images), torch.from_numpy(texts), torch.from_numpy(name_ids)
    eye = torch.eye(dim)
    for _ in range(epochs):
        for batch in torch.randperm(len(x_img)).split(BATCH):
            zi = torch.nn.functional.normalize(x_img[batch] @ w_i.T + b_i, dim=1)
            zt = torch.nn.functional.normalize(x_txt[batch] @ w_t.T + b_t, dim=1)
            logits = log_scale.exp() * zi @ zt.T
            positives = (ids[batch][:, None] == ids[batch][None, :]).float()
            targets = positives / positives.sum(1, keepdim=True)  # aynı adlı ürünler de pozitif
            loss = (torch.sum(-targets * logits.log_softmax(1), 1).mean()
                    + torch.sum(-targets * logits.log_softmax(0).T, 1).mean()) / 2
            loss = loss + lam * ((w_i - eye).pow(2).sum() + (w_t - eye).pow(2).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return Adapter(w_i.detach().numpy(), b_i.detach().numpy(), w_t.detach().numpy(), b_t.detach().numpy())


def label_queries(labels, idx, n=50):
    """Değerlendirme kataloğunda en sık 50 (renk, ürün türü) çifti için İngilizce sorgu."""
    pairs = Counter(zip(labels["baseColour"][idx], labels["articleType"][idx]))
    return [(f"{c} {a}".lower(), {"baseColour": {c}, "articleType": {a}})
            for (c, a), k in pairs.most_common(n) if c and a and k >= 5]


def text_precision(img, txt, labels, idx, queries, k=10):
    sims = txt @ img[idx].T
    top = np.argsort(-sims, axis=1)[:, :k]
    return float(np.mean([[all(labels[f][idx][j] in vals for f, vals in exp.items()) for j in row]
                          for row, (_, exp) in zip(top, queries)]))


def image_precision(img, labels, idx, n_queries=1000, k=5):
    rng = np.random.default_rng(SEED)
    q = rng.choice(len(idx), size=min(n_queries, len(idx)), replace=False)
    sims = img[idx][q] @ img[idx].T
    sims[np.arange(len(q)), q] = -np.inf
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return {f: float(np.mean(labels[f][idx][top] == labels[f][idx][q][:, None])) for f in FIELDS}


def evaluate(adapter, img_raw, labels, idx, query_sets):
    img = adapter.image(img_raw) if adapter else img_raw
    out = {}
    for name, queries in query_sets.items():
        txt = normalize(raw_model().encode_text([q for q, _ in queries]))
        txt = adapter.text(txt) if adapter else txt
        out[name] = text_precision(img, txt, labels, idx, queries)
    out.update({f"görsel {f}": v for f, v in image_precision(img, labels, idx).items()})
    return out


def text_pairs(source, idx, name_txt, name_ids, tmpl_txt, tmpl_ids):
    """Eğitim çiftleri için (görsel sırası, metin vektörü, metin kimliği): ad, şablon veya ikisi."""
    if source == "ad":
        return idx, name_txt[idx], name_ids[idx]
    if source == "şablon":
        return idx, tmpl_txt[idx], tmpl_ids[idx]
    offset = name_ids.max() + 1  # iki metin türünün kimlikleri çakışmasın
    return (np.concatenate([idx, idx]), np.concatenate([name_txt[idx], tmpl_txt[idx]]),
            np.concatenate([name_ids[idx], tmpl_ids[idx] + offset]))


def main():
    paths = list_images()
    metadata = load_metadata()
    assert metadata, "Bu script metadata.csv'li bir veri seti ister (FASHION_DATASET=products)"
    names = [metadata.get(p.name, {}).get("name", "") or p.stem for p in paths]
    labels = {f: np.array([metadata.get(p.name, {}).get(f, "") for p in paths]) for f in FIELDS + ["masterCategory"]}
    name_ids = np.unique(names, return_inverse=True)[1].astype(np.int64)
    # etiketlerden kurulan genel tarif, ör. "Women Silver Watches": marka / model kelimeleri olmadan
    templates = [" ".join(v for v in (labels["gender"][i], labels["baseColour"][i], labels["articleType"][i]) if v)
                 for i in range(len(paths))]
    unique_templates, tmpl_ids = np.unique(templates, return_inverse=True)

    img_raw = raw_image_features(paths).astype(np.float32)
    name_txt = raw_name_features(paths, names).astype(np.float32)
    tmpl_txt = normalize(raw_model().encode_text(list(unique_templates))).astype(np.float32)[tmpl_ids]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(paths))
    n_test, n_val = len(paths) // 5, len(paths) // 10
    test, val, train_idx = np.sort(perm[:n_test]), np.sort(perm[n_test:n_test + n_val]), np.sort(perm[n_test + n_val:])
    manual_tr = [(q, e) for q, e in TEXT_QUERIES if q != "red dress"]  # tek İngilizce sorgu hariç hepsi Türkçe
    print(f"{len(paths)} ürün: eğitim {len(train_idx)}, doğrulama {len(val)}, test {len(test)}; "
          f"{len(unique_templates)} farklı şablon\n")

    # metin kaynağı + λ seçimi doğrulama diliminde (etiket sorguları + Türkçe sorgular)
    val_sets = {"etiket sorguları": label_queries(labels, val), "elle (tr)": manual_tr}

    def val_score(r):
        return (r["etiket sorguları"] + r["elle (tr)"]) / 2

    base_val = evaluate(None, img_raw, labels, val, val_sets)
    print(f"{'CLIP (adaptörsüz)':<20} doğrulama: etiket {base_val['etiket sorguları']:.3f}, "
          f"elle tr {base_val['elle (tr)']:.3f}, görsel türü {base_val['görsel articleType']:.3f}")
    best = None
    for source in ("ad", "şablon", "ikisi"):
        for lam in LAMBDAS:
            idx, txt, ids = text_pairs(source, train_idx, name_txt, name_ids, tmpl_txt, tmpl_ids)
            adapter = train(img_raw[idx], txt, ids, lam)
            r = evaluate(adapter, img_raw, labels, val, val_sets)
            print(f"{source:<7} λ={lam:<10} doğrulama: etiket {r['etiket sorguları']:.3f}, "
                  f"elle tr {r['elle (tr)']:.3f}, görsel türü {r['görsel articleType']:.3f}")
            if best is None or val_score(r) > best[0]:
                best = (val_score(r), source, lam, adapter)
    score, source, lam, adapter = best
    print(f"seçilen: {source}, λ = {lam} (doğrulama {score:.3f}, adaptörsüz {val_score(base_val):.3f})\n")

    test_sets = {"etiket sorguları": label_queries(labels, test), "elle (tr)": manual_tr}
    base, tuned = evaluate(None, img_raw, labels, test, test_sets), evaluate(adapter, img_raw, labels, test, test_sets)
    print(f"Test ({len(test)} hiç görülmemiş ürün):")
    print(f"  {'ölçüt':<22}{'CLIP':>8}{'adaptör':>10}")
    for key in base:
        print(f"  {key:<22}{base[key]:>8.3f}{tuned[key]:>10.3f}")

    if "--no-save" in sys.argv:
        return
    if score <= val_score(base_val):
        print("\nAdaptör doğrulamada CLIP'i geçemedi; kaydedilmedi.")
        return
    idx, txt, ids = text_pairs(source, np.arange(len(paths)), name_txt, name_ids, tmpl_txt, tmpl_ids)
    final = train(img_raw[idx], txt, ids, lam)
    final.save(CLIP_ADAPTER_PATH)
    adapted = final.image(img_raw)
    cached_features(paths, CLIP_CACHE_PATH, lambda: adapted, force=True, key=clip_key())
    print(f"\nAdaptör (tüm veriyle) kaydedildi: {CLIP_ADAPTER_PATH}; CLIP önbelleği güncellendi ({clip_key()})")


if __name__ == "__main__":
    main()
