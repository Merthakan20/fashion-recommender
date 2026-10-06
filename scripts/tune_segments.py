"""Bölüm (Kadın / Erkek / Kız Çocuk / Erkek Çocuk / Aksesuar) bazında öneri ayarlarını seçer.

    python scripts/tune_segments.py            # varsayılan veri seti (products)

Her bölüm için gerçek etiketlerle görselden öneri ölçülür (ilk 5 önerinin
aynı ürün türü / renk / cinsiyeti taşıma oranı):
- tüm katalogdan öneri ile aynı bölümden öneri karşılaştırılır,
- "renk önemi" 0–0,6 arasında taranır. Sorguların yarısında hedef
  (ürün türü ve renk isabetinin ortalaması) en yüksek olan değer seçilir,
  diğer yarısında raporlanır.
Seçilen değerler `fashion_rec/segment_settings.json` dosyasına yazılır;
uygulama bunları bölümün varsayılanı olarak kullanır.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from fashion_rec.backbone import build_image_index  # noqa: E402
from fashion_rec.catalog import SEGMENTS, corrected_gender, load_metadata, segments  # noqa: E402
from fashion_rec.color import build_color_index  # noqa: E402
from fashion_rec.config import DATASET  # noqa: E402
from fashion_rec.segment_settings import SETTINGS_PATH  # noqa: E402

COLOR_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
QUERIES, TOP_K, SEED = 1000, 5, 0
FIELDS = ["articleType", "baseColour", "gender"]


def zscore_rows(sim):
    return (sim - sim.mean(axis=1, keepdims=True)) / sim.std(axis=1, keepdims=True)


def precision(sim, query_pos, labels_q, labels_c):
    """sim: (sorgu, aday); query_pos: sorgunun aday listesindeki yeri (kendisi hariç tutulur)."""
    sim = sim.copy()
    sim[np.arange(len(query_pos)), query_pos] = -np.inf
    top = np.argpartition(-sim, TOP_K, axis=1)[:, :TOP_K]
    return {f: float(np.mean(labels_c[f][top] == labels_q[f][:, None])) for f in FIELDS}


def objective(p):
    return (p["articleType"] + p["baseColour"]) / 2


def main():
    paths, feats = build_image_index()
    _, colors = build_color_index()
    metadata = load_metadata()
    assert metadata, "Bu script metadata.csv'li bir veri seti ister"
    seg = np.array([s or "" for s in segments(paths, metadata)])
    labels = {f: np.array([metadata[p.name][f] for p in paths]) for f in FIELDS}
    labels["gender"] = np.array([corrected_gender(metadata[p.name]) for p in paths])  # adı çocuk ürünü diyenler düzeltilmiş
    rng = np.random.default_rng(SEED)

    print(f"{DATASET}: {len(paths)} ürün; ilk {TOP_K} öneride aynı ürün türü / renk / cinsiyet oranı (test yarısı)\n")
    header = f"{'bölüm':<12}{'ürün':>7}  {'ayar':<26}{'tür':>7}{'renk':>7}{'cinsiyet':>10}"
    print(header)
    print("-" * len(header))
    settings = {}
    for name in SEGMENTS:
        members = np.flatnonzero(seg == name)
        if len(members) < 50:
            continue
        queries = rng.choice(members, size=min(QUERIES, len(members)), replace=False)
        tune, test = np.array_split(rng.permutation(queries), 2)

        def evaluate(q, c, restrict):
            cands = members if restrict else np.arange(len(paths))
            pos = np.searchsorted(cands, q)  # cands sıralı; sorgu adaylar arasında
            sim = feats[q] @ feats[cands].T
            if c > 0:
                sim = (1 - c) * zscore_rows(sim) + c * zscore_rows(colors[q] @ colors[cands].T)
            return precision(sim, pos, {f: v[q] for f, v in labels.items()}, {f: v[cands] for f, v in labels.items()})

        best_c = max(COLOR_WEIGHTS, key=lambda c: objective(evaluate(tune, c, True)))
        rows = [("tüm katalog, renk 0", evaluate(test, 0.0, False)),
                ("aynı bölüm, renk 0", evaluate(test, 0.0, True)),
                (f"aynı bölüm, renk {best_c} (seçilen)", evaluate(test, best_c, True))]
        for i, (label, p) in enumerate(rows):
            print(f"{name if i == 0 else '':<12}{len(members) if i == 0 else '':>7}  {label:<26}"
                  f"{p['articleType']:>7.3f}{p['baseColour']:>7.3f}{p['gender']:>10.3f}")
        settings[name] = {"color_weight": best_c}

    all_settings = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    all_settings[DATASET] = settings
    SETTINGS_PATH.write_text(json.dumps(all_settings, ensure_ascii=False, indent=2) + "\n")
    print(f"\nAyarlar yazıldı: {SETTINGS_PATH}\n{json.dumps(settings, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
