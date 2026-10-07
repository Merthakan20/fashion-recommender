"""Kombin önerisinin gerçek satın alma verisiyle değerlendirmesi (H&M).

    FASHION_DATASET=hm python scripts/build_copurchase.py --cutoff 2020-06-22
    FASHION_DATASET=hm python notebooks/evaluate_copurchase.py

Öğrenme: 2020-06-22 öncesi sepetler (cache/hm/copurchase_train.npz).
Test: sonraki 3 ayın sepetlerinde birlikte alınmış tamamlayıcı çiftler
(q, b): b, q'nun kombin yuvalarından birinin adayıdır (ör. üst → alt).
Her yöntem yuvadaki tüm adayları sıralar; b'nin sırasından isabet@10,
isabet@50 ve MRR hesaplanır. Ağırlıklar çiftlerin yarısında seçilir,
diğer yarısında raporlanır.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from fashion_rec.backbone import build_image_index  # noqa: E402
from fashion_rec.catalog import load_metadata  # noqa: E402
from fashion_rec.config import CACHE_DIR, DATASET, ROLE_CACHE_PATH  # noqa: E402
from fashion_rec.outfit import SLOTS, OutfitCatalog, Weights, clip_roles, role_of  # noqa: E402

PAIRS, SEED = 4000, 0


def main():
    assert DATASET == "hm", "FASHION_DATASET=hm ile çalıştırın"
    paths, feats = build_image_index()
    metadata = load_metadata()
    train = sparse.load_npz(CACHE_DIR / "copurchase_train.npz")
    test = sparse.load_npz(CACHE_DIR / "test_baskets.npz").tocsr()
    assert train.shape[0] == len(paths), "copurchase matrisi katalogla uyuşmuyor; build_copurchase.py'yi yeniden çalıştırın"
    catalog = OutfitCatalog(paths, feats, metadata, visual_roles=clip_roles(paths, feats, ROLE_CACHE_PATH),
                            copurchase=train)
    rows = [metadata[p.name] for p in paths]

    # test sepetlerinden tamamlayıcı çiftler
    rng = np.random.default_rng(SEED)
    pairs = []
    for basket in rng.permutation(test.shape[0]):
        items = test.indices[test.indptr[basket]:test.indptr[basket + 1]]
        for q in items:
            role = catalog.role[q]
            if not role:
                continue
            for b in items:
                if b != q and catalog.role[b] in SLOTS.get(role, []):
                    pairs.append((q, b))
        if len(pairs) >= PAIRS * 3:
            break
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    evaluable = []
    for q, b in pairs:
        idx = catalog.candidates(rows[q], catalog.role[b], exclude=paths[q])
        pos = np.flatnonzero(idx == b)
        if len(pos):
            evaluable.append((q, b, idx, int(pos[0])))
        if len(evaluable) >= PAIRS:
            break
    cold = np.mean([train[q].nnz == 0 for q, *_ in evaluable])
    print(f"{len(evaluable)} test çifti (sorgunun öğrenme döneminde hiç birlikte alınmadığı oran: {cold:.0%}); "
          f"yuva başına ortalama {np.mean([len(i) for *_, i, _ in evaluable]):.0f} aday\n")

    def evaluate(weights, subset, random=False):
        ranks = []
        for q, b, idx, pos in subset:
            if random:
                score = rng.random(len(idx))
            else:
                score = catalog.scores(rows[q], feats[q], idx, weights, paths[q])
            # eşit puanlılar rastgele sıralanmış sayılır (yarısı önde): yeni ürünlerin hepsi 0 popülerlik
            # aldığında eşitlikleri hedefin lehine saymak popülerlik / birlikte alınmayı olduğundan iyi gösterir
            ties = int(np.sum(score == score[pos])) - 1
            ranks.append(1 + int(np.sum(score > score[pos])) + ties / 2)
        ranks = np.array(ranks)
        return np.mean(ranks <= 10), np.mean(ranks <= 50), np.mean(1 / ranks)

    tune, report = evaluable[::2], evaluable[1::2]
    candidates = {
        "rastgele": None,
        "popülerlik (çok satanlar)": [Weights(style=0, usage=0, colour=0, season=0, popularity=1)],
        "kurallar (mevcut)": [Weights()],
        "birlikte alınma": [Weights(style=0, usage=0, colour=0, season=0, copurchase=1, popularity=0.01)],
        "kurallar + popülerlik": [Weights(popularity=w) for w in (0.25, 0.5, 1, 2)],
        "kurallar + birlikte alınma + popülerlik": [Weights(copurchase=c, popularity=w)
                                                     for c in (2, 4, 8, 16) for w in (0.5, 1, 2)],
    }
    warm = [e for e in report if train[e[0]].nnz > 0]
    cold_set = [e for e in report if train[e[0]].nnz == 0]
    print(f"{'yöntem':<42}{'isabet@10':>10}{'isabet@50':>11}{'MRR':>8}{'@10 geçmişli':>14}{'@10 geçmişsiz':>15}  seçilen")
    for label, grid in candidates.items():
        if grid is None:
            best, chosen = None, ""
        else:
            best = max(grid, key=lambda w: evaluate(w, tune)[2])  # MRR ile seçim
            chosen = f"birlikte={best.copurchase}, popülerlik={best.popularity}" if len(grid) > 1 else ""
        h10, h50, mrr = evaluate(best, report, random=best is None)
        w10 = evaluate(best, warm, random=best is None)[0]
        c10 = evaluate(best, cold_set, random=best is None)[0]
        print(f"{label:<42}{h10:>10.3f}{h50:>11.3f}{mrr:>8.3f}{w10:>14.3f}{c10:>15.3f}  {chosen}")
    print(f"\n(geçmişli: sorgu öğrenme döneminde en az bir kez birlikte alınmış, {len(warm)} çift; geçmişsiz: {len(cold_set)})")


if __name__ == "__main__":
    main()
